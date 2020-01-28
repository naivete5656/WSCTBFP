# import ptvsd
# import time
# import os

# print("Waiting to attach")

# address = ("0.0.0.0", 3000)
# ptvsd.enable_attach(address)
# ptvsd.wait_for_attach()

# time.sleep(2)

# print("attached")

import argparse
from pathlib import Path
import torch
import torch.utils.data
from utils import CellImageLoadTest
from networks import *
import cv2
import numpy as np
import multiprocessing
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

multiprocessing.set_start_method("spawn", True)


def gather_path(train_paths, mode):
    ori_paths = []
    for train_path in train_paths:
        if "sequ9" in train_path.name:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
        else:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
    return ori_paths


class Predict(object):
    def __init__(self, **kwargs):
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.vis = kwargs["vis"]
        self.env = kwargs["vis_env"]
        self.batch_size = 1
        self.save_path = kwargs["save_path"]

        ori_paths = gather_path(kwargs["data_paths"], "ori")[kwargs["start"]:]
        gt_paths = gather_path(kwargs["data_paths"], "9")[kwargs["start"]:]
        if kwargs["mask_path"] is not None:
            bg_paths = gather_path(kwargs["data_paths"], "bg")[kwargs["start"]:]
            data_loader = CellImageLoadTest(
                ori_paths, gt_paths, time_late=kwargs["time_late"], bg_path=bg_paths
            )
        else:
            data_loader = CellImageLoadTest(
                ori_paths, gt_paths, time_late=kwargs["time_late"]
            )

        self.dataset = torch.utils.data.DataLoader(
            data_loader, 1, shuffle=False, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()

        if kwargs["mask_path"] is not None:
            self.mask_paths = sorted(kwargs["mask_path"].glob("*/*.npz"))[
                              kwargs["start"]:
                              ]

        save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("ori").chmod(0o777)
        save_path.joinpath("gt").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("gt").chmod(0o777)
        save_path.joinpath("pred").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("pred").chmod(0o777)

    def vis_init(self, env):
        import visdom

        HOSTNAME = "localhost"
        PORT = 8097

        self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env=env)
        self.ori_view = self.create_vis_show()
        self.ori2_view = self.create_vis_show()
        self.gt_view = self.create_vis_show()
        self.gt2_view = self.create_vis_show()
        self.pre_view = self.create_vis_show()
        self.pre2_view = self.create_vis_show()

    def create_vis_show(self):
        return self.vis.images(torch.ones((1, 1, 256, 256)), self.batch_size)

    def update_vis_show(self, images, window1):
        self.vis.images(images, self.batch_size, win=window1)

    def __call__(self, *args, **kwargs):
        self.net.eval()
        if self.vis:
            self.vis_init(env=self.env)

        for iteration, data in enumerate(self.dataset):
            img_ori = data["image"]
            target = data["gt"]

            if kwargs["each"]:
                bg = data["bg"]
                gbs = np.load(str(self.mask_paths[iteration]))["gb"]

                gbs[0] = 0
                gbs_normt = [gbs[0, 1]]
                for gb in gbs[1:]:
                    gb = gb[0]
                    x = gb[gb > 0.01]
                    gb = gb / sorted(x)[round(len(x) * 0.99)]
                    gbs_normt.append(gb)

                gbs_normt1 = [gbs[0, 1]]

                for gb in gbs[1:]:
                    gb = gb[0]
                    x = gb[gb > 0.01]
                    gb = gb / sorted(x)[round(len(x) * 0.99)]
                    gbs_normt1.append(gb)

                gbs_normt = np.array(gbs_normt).reshape(
                    gbs.shape[0], 1, gbs.shape[2], gbs.shape[3]
                )
                gbs_normt1 = np.array(gbs_normt1).reshape(
                    gbs.shape[0], 1, gbs.shape[2], gbs.shape[3]
                )

                gbs = np.concatenate([gbs_normt, gbs_normt1], axis=1)

                region_label1 = np.argmax(gbs[:, 0], axis=0)
                gbs_sum1 = gbs[:, 0].max(0)
                region_label1[gbs_sum1 < 0.01] = 0

                region_label2 = np.argmax(gbs[:, 1], axis=0)
                gbs_sum2 = gbs[:, 1].max(0)
                region_label2[gbs_sum2 < 0.01] = 0

                for i in range(1, region_label1.max() + 1):

                    gb1 = np.zeros_like(region_label1)
                    gb1[region_label1 == i] = 1

                    gb1 = gb1.reshape((1, 1, img_ori.shape[2], img_ori.shape[3]))
                    gb1 = torch.tensor(gb1)

                    gb2 = np.zeros_like(region_label2)
                    gb2[region_label2 == i] = 1

                    gb2 = gb2.reshape((1, 1, img_ori.shape[2], img_ori.shape[3]))
                    gb2 = torch.tensor(gb2)

                    bg1 = bg[:, :1]
                    bg2 = bg[:, 1:]

                    img = torch.zeros_like(img_ori)
                    img[:, :1] = bg1 + img[:, :1]
                    img[:, 1:] = bg2 + img[:, 1:]

                    img[:, :1][gb1 == 1] = img_ori[:, :1][gb1 == 1]
                    img[:, 1:][gb2 == 1] = img_ori[:, 1:][gb2 == 1]

                    # x_y = gbs[i][0]
                    # x, y = np.where(x_y > (0.99 * x_y))
                    # x = int(x.mean())
                    # x = min(max(0, x - 20), img_ori.shape[2] - 40)
                    # y = int(y.mean())
                    # y = min(max(0, y - 20), img_ori.shape[3] - 40)
                    # img = bg.clone()
                    # img[:, :, x - 20 : x + 20, y - 10 : y + 30] = img_ori[
                    #     :, :, x - 20 : x + 20, y - 10 : y + 30
                    # ]

                    # x, y = np.where(x_y > (0.99 * x_y))
                    # x = int(x.mean())
                    # x = min(max(0, x - 128), img_ori.shape[2] - 256)
                    # y = int(y.mean())
                    # y = min(max(0, y - 128), img_ori.shape[3] - 256)
                    # img_patch = img[:, :, x: x + 256, y: y + 256]


                    if self.gpu:
                        img = img.cuda()

                    pred_img = self.net(img)
                    pred1 = pred_img[0].detach().cpu().clamp(min=0, max=1).numpy()
                    pred2 = pred_img[1].detach().cpu().clamp(min=0, max=1).numpy()

                    # pred1_full = np.zeros((bg.shape[2], bg.shape[3]))
                    # pred1_full[x: x + 256, y: y + 256] = pred1
                    # pred2_full = np.zeros((bg.shape[2], bg.shape[3]))
                    # pred2_full[x: x + 256, y: y + 256] = pred2

                    # pred1_full = pred1[0, 0]
                    # pred2_full = pred2[0, 0]

                    del pred_img
                    torch.cuda.empty_cache()

                    if self.save_path is not None:
                        cv2.imwrite(
                            str(
                                self.save_path
                                / Path(f"ori/{iteration:04d}_{i:04d}_1.png")
                            ),
                            ((img / img.max()) * 255)
                                .cpu()
                                .numpy()
                                .astype(np.uint8)[0, 0, :, :],
                        )
                        cv2.imwrite(
                            str(
                                self.save_path
                                / Path(f"ori/{iteration:04d}_{i:04d}_2.png")
                            ),
                            ((img / img.max()) * 255)
                                .cpu()
                                .numpy()
                                .astype(np.uint8)[0, 1, :, :],
                        )
                        cv2.imwrite(
                            str(
                                self.save_path
                                / Path(f"pred/{iteration:04d}_{i:04d}_1.png")
                            ),
                            ((pred1 / pred1.max()) * 255)
                                .astype(np.uint8)[0, 0, :, :],
                        )
                        cv2.imwrite(
                            str(
                                self.save_path
                                / Path(f"pred/{iteration:04d}_{i:04d}_2.png")
                            ),
                            ((pred2 / pred2.max()) * 255)
                                .astype(np.uint8)[0, 0, :, :],
                        )
                        cv2.imwrite(
                            str(
                                self.save_path
                                / Path(f"gt/{iteration:04d}_{i:04d}_1.png")
                            ),
                            ((target / target.max()) * 255)
                                .numpy()
                                .astype(np.uint8)[0, 0, :, :],
                        )
                        cv2.imwrite(
                            str(
                                self.save_path
                                / Path(f"gt/{iteration:04d}_{i:04d}_2.png")
                            ),
                            ((target / target.max()) * 255)
                                .numpy()
                                .astype(np.uint8)[0, 1, :, :],
                        )
            else:
                img = img_ori
                if self.gpu:
                    img = img.cuda()

                pred_img = self.net(img)

                pred1 = pred_img[0].clamp(min=0, max=1)
                pred2 = pred_img[1].clamp(min=0, max=1)
                pred1 = pred1.cpu()
                pred2 = pred2.cpu()
                del pred_img
                torch.cuda.empty_cache()

                if self.save_path is not None:
                    cv2.imwrite(
                        str(self.save_path / Path(f"ori/{iteration:04d}_1.png")),
                        ((img / img.max()) * 255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)[0, 0, :, :],
                    )
                    cv2.imwrite(
                        str(self.save_path / Path(f"gt/{iteration:04d}_1.png")),
                        ((target / target.max()) * 255)
                            .numpy()
                            .astype(np.uint8)[0, 0, :, :],
                    )

                    cv2.imwrite(
                        str(self.save_path / Path(f"ori/{iteration:04d}_2.png")),
                        ((img / img.max()) * 255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)[0, 1, :, :],
                    )
                    cv2.imwrite(
                        str(self.save_path / Path(f"pred/{iteration:04d}_1.png")),
                        ((pred1 / pred1.max()) * 255)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)[0, 0, :, :],
                    )
                    cv2.imwrite(
                        str(self.save_path / Path(f"pred/{iteration:04d}_2.png")),
                        ((pred2 / pred2.max()) * 255)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)[0, 0, :, :],
                    )
                    cv2.imwrite(
                        str(self.save_path / Path(f"gt/{iteration:04d}_2.png")),
                        ((target / target.max()) * 255)
                            .numpy()
                            .astype(np.uint8)[0, 1, :, :],
                    )


if __name__ == "__main__":
    num = 1
    torch.cuda.set_device(num)
    seqs = [9, 2, 16, 17, 18]
    seq = 9
    time_lates = [1, 5, 9]
    time_late = 1
    # for time_late in time_lates:
    # for seq in seqs:
    net = UNet3(n_channels=1, n_classes=1, sig=False)

    mask_path = Path(
        f"./output/guid_out/C2C12_9_{time_late}/sequ{seq}"
    )
    # mask_path = None

    if mask_path is not None:
        each = True
    else:
        each = False

    data_paths = [Path(f"/home/kazuya/main/correlation_test/images/sequ{seq}")]

    weight_path = Path(
        f"/home/kazuya/file_server2/CVPR_tracking/weight/C2C12_9_{time_late}/temp.pth"
    )

    net.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
    net.cuda()

    if mask_path is not None:
        save_path = Path(
            f"./output/detection/{weight_path.parent.name}_mask/sequ{seq}"
        )
    else:
        save_path = Path(
            f"./output/detection/C2C12_9_{time_late}/sequ{seq}"
        )
    save_path.mkdir(parents=True, exist_ok=True)

    args = {
        "net": net,
        "gpu": True,
        "data_paths": data_paths,
        "vis": False,
        "vis_env": "val",
        # "save_path": None,
        "save_path": save_path,
        "time_late": time_late,
        "start": 0,
        "mask_path": mask_path,
    }
    pre = Predict(**args)
    pre(each=each)
