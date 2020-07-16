import torch.utils.data
from utils import *
from torch import optim
from eval import eval_net
from tqdm import tqdm
from networks import *


class _TrainBase(Visdom):
    def __init__(self, **kwargs):
        self.save_weight_path = kwargs["save_weight_path"]
        self.epochs = kwargs["epochs"]
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.need_vis = kwargs["vis"]
        self.batch_size = kwargs["batch_size"]
        self.plot_size = kwargs["plot_size"]
        ori_paths = self.gather_path(kwargs["train_paths"], "ori")
        gt_paths = self.gather_path(kwargs["train_paths"], "9")

        data_loader = CellImageLoad(
            ori_paths, gt_paths, time_late=kwargs["time_late"]
        )
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()
        if kwargs["val_paths"] is not None:
            ori_paths = self.gather_path(kwargs["val_paths"], "ori")
            gt_paths = self.gather_path(kwargs["val_paths"], "9")
            data_loader = CellImageLoad(
                ori_paths, gt_paths, time_late=kwargs["time_late"]
            )
            self.val_loader = torch.utils.data.DataLoader(
                data_loader,
                batch_size=kwargs["batch_size"],
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_loader = None
        self.loss_flag = False
        self.criterion = self.loss_function(kwargs["criterion"])
        self.optimizer = optim.Adam(self.net.parameters(), lr=kwargs["lr"])
        self.iteration = 0
        self.decay = 0.1
        self.vis_env = kwargs["vis_env"]
        # loss counters
        self.loc_loss = 0
        self.conf_loss = 0
        self.epoch_loss = 0
        self.bad = 0
        self.losses = []
        self.evals = []
        self.val_losses = []

    def gather_path(self, train_paths, mode):
        ori_path = []
        for train_path in train_paths:
            ori_path.extend(sorted(train_path.joinpath(mode).glob("*.tif")))

        return ori_path

    def loss_function(self, loss_type):
        if isinstance(loss_type, nn.MSELoss):
            self.loss_flag = True
        else:
            self.loss_flag = False
        return loss_type


class TrainNet(_TrainBase):
    def main(self):
        self.vis_init(self.vis_env)
        for epoch in range(self.epochs):
            self.net.train()
            pbar = tqdm(total=self.number_of_traindata)
            for iteration, data in enumerate(self.train_dataset_loader):
                img = data["image"]
                target = data["gt"]

                if self.gpu:
                    img = img.cuda()
                    target = target.cuda()

                pred_img = self.net(img)

                loss1 = self.criterion(pred_img[0], target[:, :1])
                loss2 = self.criterion(pred_img[1], target[:, 1:2])
                loss = loss1 + loss2

                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.need_vis:
                    self.update_vis_plot(
                        iteration, [loss.item()], self.iter_plot, None, "append"
                    )
                if self.need_vis:
                    self.update_vis_show(img[:, :1, :, :].cpu(), self.ori_view)
                    self.update_vis_show(img[:, 1:, :, :].cpu(), self.ori_view2)

                    pred_img1 = (pred_img[0] - pred_img[0].min()) / (
                        pred_img[0].max() - pred_img[0].min()
                    )
                    self.update_vis_show(pred_img1, self.img_view)
                    pred_img2 = (pred_img[1] - pred_img[1].min()) / (
                        pred_img[1].max() - pred_img[1].min()
                    )
                    self.update_vis_show(pred_img2, self.img2_view)
                    self.update_vis_show(target[:, :1, :, :].cpu(), self.gt_view)
                    self.update_vis_show(target[:, 1:, :, :].cpu(), self.gt_view2)

                    self.iteration += 1

                    if self.iteration % 10000 == 0:
                        torch.save(
                            self.net.state_dict(),
                            str(
                                self.save_weight_path.parent.joinpath(
                                    "epoch_weight/{:05d}.pth".format(epoch)
                                )
                            ),
                        )

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.decay * param_group["lr"]
                    if self.iteration >= 40000:
                        torch.save(
                            self.net.state_dict(),
                            str(
                                self.save_weight_path.parent.joinpath(
                                    "temp.pth".format(epoch)
                                )
                            ),
                        )
                        print("stop running")
                        break
                pbar.update(self.batch_size)
            pbar.close()

            if self.val_loader is not None:
                self.validation(iteration, epoch)
            else:
                torch.save(
                    self.net.state_dict(),
                    str(
                        self.save_weight_path.parent.joinpath("temp.pth".format(epoch))
                    ),
                )
                if epoch % 100 == 0:
                    torch.save(
                        self.net.state_dict(),
                        str(
                            self.save_weight_path.parent.joinpath(
                                "epoch_weight/{:05d}.pth".format(epoch)
                            )
                        ),
                    )

            if self.bad >= 100:
                print("stop running")
                break

    def validation(self, number_of_train_data, epoch):
        loss = self.epoch_loss / (number_of_train_data + 1)
        print("Epoch finished ! Loss: {}".format(loss))
        torch.save(
            self.net.state_dict(),
            str(
                self.save_weight_path.parent.joinpath(
                    "epoch_weight/{:05d}.pth".format(epoch)
                )
            ),
        )
        val_loss = eval_net(
            self.net,
            self.val_loader,
            self.gpu,
            self.loss_flag,
            self.vis,
            self.img_view_val,
            self.gt_view_val,
            self.pred_view_val,
            self.criterion,
        )

        print("val_loss: {}".format(val_loss))
        try:
            if min(self.val_losses) > val_loss:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
                self.bad = 0
                print("update bad")
                with self.save_weight_path.parent.joinpath("best.txt").open("w") as f:
                    f.write("{}".format(epoch))
                pass
            else:
                self.bad += 1
                print("bad ++")
        except ValueError:
            torch.save(self.net.state_dict(), str(self.save_weight_path))
        self.val_losses.append(val_loss)

        if self.need_vis:
            self.update_vis_plot(
                iteration=epoch,
                loss=loss,
                val=[loss, val_loss],
                window1=self.iter_plot,
                window2=self.epoch_plot,
                update_type="append",
            )
        print("bad = {}".format(self.bad))
        self.epoch_loss = 0


sequence_list = [2, 9, 17, 18]
if __name__ == "__main__":
    for time_late in [1, 5, 9]:
        torch.cuda.set_device(0)
        plot_size = 9
        train_paths = [Path(f"/home/kazuya/main/ECCV/correlation_test/images/sequ9")]
        val_paths = [Path(f"/home/kazuya/main/ECCV/correlation_test/images/sequ16")]

        net = UNet3(n_channels=1, n_classes=1, sig=False)
        net.cuda()

        save_weights_path = Path(f"./weights/C2C12_9_{time_late}/best.pth")
        save_weights_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        save_weights_path.parent.mkdir(parents=True, exist_ok=True)
        save_weights_path.parent.chmod(0o777)

        args = {
            "gpu": True,
            "batch_size": 8,
            "epochs": 1500,
            "lr": 1e-3,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "save_weight_path": save_weights_path,
            "net": net,
            "vis": True,
            "plot_size": plot_size,
            "criterion": nn.MSELoss(),
            "vis_env": net.__class__.__name__,
            "time_late": time_late,
        }

        train = TrainNet(**args)
        train.main()
