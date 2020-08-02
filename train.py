import torch.utils.data
from utils import *
from torch import optim
import torch.nn as nn
from eval import eval_net
from tqdm import tqdm
from networks import CoDetectionCNN
import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="./data/train",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--val_path",
        dest="val_path",
        help="validation data path",
        default="./data/val",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="./weight",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, type=bool,
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=16, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=500, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--vis",
        dest="vis",
        help="visdom",
        default=True,
        type=bool,
    )

    args = parser.parse_args()
    return args


class _TrainBase(Visdom):
    def __init__(self, **kwargs):
        self.save_weight_path = kwargs["save_weight_path"]
        self.epochs = kwargs["epochs"]
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.need_vis = kwargs["vis"]
        if self.need_vis:
            self.vis_env = kwargs["vis_env"]

        self.batch_size = kwargs["batch_size"]
        ori_paths = self.gather_path(kwargs["train_paths"], "img")
        gt_paths = self.gather_path(kwargs["train_paths"], "like")

        data_loader = CellImageLoad(
            ori_paths, gt_paths, time_late=kwargs["time_late"]
        )
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()
        if kwargs["val_paths"] is not None:
            ori_paths = self.gather_path(kwargs["val_paths"], "img")
            gt_paths = self.gather_path(kwargs["val_paths"], "like")
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
        # loss counters
        self.loc_loss = 0
        self.conf_loss = 0
        self.epoch_loss = 0
        self.bad = 0
        self.losses = []
        self.evals = []
        self.val_losses = []

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
                pbar.update(self.batch_size)
            pbar.close()

            if self.val_loader is not None:
                self.validation(iteration, epoch)
                if self.bad >= 100:
                    print("stop running")
                    break
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


if __name__ == "__main__":
    args = parse_args()

    for time_late in [1, 5, 9]:
        train_paths = [Path(args.train_path)]
        val_paths = [Path(args.val_path)]

        net = CoDetectionCNN(n_channels=1, n_classes=1, sig=False)
        net.cuda()

        save_weights_path = Path(args.weight_path).joinpath(f"{time_late}/best.pth")
        save_weights_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        save_weights_path.parent.mkdir(parents=True, exist_ok=True)

        args_list = {
            "gpu": args.gpu,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.learning_rate,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "save_weight_path": save_weights_path,
            "net": net,
            "vis": args.vis,
            "criterion": nn.MSELoss(),
            "vis_env": net.__class__.__name__,
            "time_late": time_late,
        }

        train = TrainNet(**args_list)
        train.main()
