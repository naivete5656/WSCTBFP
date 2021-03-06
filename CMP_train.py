import torch
import torch.utils.data
from pathlib import Path
import argparse
import torch.nn as nn
from utils import *
from torch import optim
from eval import eval_net1
from tqdm import tqdm
from networks import *

DATASET = "Elmer"


class _TrainBase(Visdom):
    def __init__(self, **kwargs):
        self.save_weight_path = kwargs["save_weight_path"]
        self.epochs = kwargs["epochs"]
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.need_vis = kwargs["vis"]
        self.batch_size = kwargs["batch_size"]

        ori_paths = self.gather_path_ori(kwargs["train_paths"], "img")

        data_loader = CMPLoad(ori_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=kwargs["batch_size"], shuffle=True, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()

        self.criterion = kwargs["criterion"]

        self.optimizer = optim.Adam(self.net.parameters(), lr=kwargs["lr"])
        self.iteration = 1
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

    def gather_path_ori(self, train_paths, mode):
        ori_paths = np.zeros((0, 2))
        for train_path in train_paths:
            ori_path = []
            ori_path.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
            ori_path = np.array(ori_path)

            for time_late in [1, 5, 9]:
                temp_paths = ori_path.copy().reshape(-1, 1)
                temp_paths = np.append(temp_paths, np.full((temp_paths.shape[0], 1), time_late), axis=1)
                temp_paths = temp_paths[:-time_late]
                ori_paths = np.append(ori_paths, temp_paths, axis=0)
        return ori_paths

    def gather_path_gt(self, train_paths, mode):
        ori_paths = np.zeros(0)
        for train_path in train_paths:
            ori_path = []
            if "sequ9" in train_path.name:
                ori_path.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
            else:
                ori_path.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
            ori_path = np.array(ori_path)
            time_late1 = ori_path.copy()[:-1]
            time_late5 = ori_path.copy()[:-5]
            time_late9 = ori_path.copy()[:-9]

            time_late7 = np.append(time_late1, time_late5, axis=0)
            time_late9 = np.append(time_late7, time_late9, axis=0)
            ori_paths = np.append(ori_paths, time_late9, axis=0)
        return ori_paths


class TrainNet(_TrainBase):
    def main(self):
        self.net.train()
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

                loss = self.criterion(pred_img, target)

                self.epoch_loss += loss.item()

                if loss.data < 50:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.need_vis:
                    self.update_vis_plot(
                        iteration, [loss.item()], self.iter_plot, None, "append"
                    )
                    self.update_vis_show(img[:, :1, :, :].cpu(), self.ori_view)

                    self.update_vis_show(
                        pred_img.clamp(min=0, max=1).cpu(), self.img_view
                    )
                    target[target == 255] = 1
                    self.update_vis_show(target.clamp(min=0, max=1).cpu(), self.gt_view)

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
                if self.iteration >= 30000:
                    torch.save(
                        self.net.state_dict(),
                        str(self.save_weight_path.parent.joinpath("final.pth")),
                    )
                    print("stop running")
                    break
                pbar.update(self.batch_size)
            pbar.close()
            if self.iteration >= 30000:
                torch.save(
                    self.net.state_dict(),
                    str(self.save_weight_path.parent.joinpath("final.pth")),
                )
                print("stop running")
                break

            torch.save(
                self.net.state_dict(),
                str(self.save_weight_path.parent.joinpath("temp.pth")),
            )


if __name__ == "__main__":

    train_paths = [Path(f"./data")]

    net = UNet(n_channels=2, n_classes=3, sig=False)
    net.cuda()
    save_weights_path = Path(f"./weights/CMP/best.pth")

    save_weights_path.parent.joinpath("epoch_weight").mkdir(parents=True, exist_ok=True)
    save_weights_path.parent.mkdir(parents=True, exist_ok=True)

    args = {
        "gpu": True,
        "batch_size": 10,
        "epochs": 1000,
        "lr": 1e-3,
        "train_paths": train_paths,
        "save_weight_path": save_weights_path,
        "net": net,
        "vis": True,
        "criterion": RMSE_Q_NormLoss(),
        # "criterion": IgnoreMSELoss(),
        "vis_env": "2",
    }

    train = TrainNet(**args)
    train.main()
