import torch
import numpy as np

# import h5py
from scipy.ndimage.interpolation import rotate
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import math


def RandomFlipper(seed, ori, target):
    if seed == 0:
        return ori, target
    elif seed == 1:
        oriH = ori[:, :, ::-1].copy()
        targetH = target[:, :, ::-1].copy()
        targetH[1] = -targetH[1]
        return oriH, targetH
    elif seed == 2:
        oriV = ori[:, ::-1, :].copy()
        targetV = target[:, ::-1, :].copy()
        targetV[0] = -targetV[0]
        return oriV, targetV
    else:
        oriHV = ori[:, ::-1, ::-1].copy()
        targetHV = target[:, ::-1, ::-1].copy()
        targetHV[:2] = -targetHV[:2]
        return oriHV, targetHV


class CMPLoad(object):
    def __init__(self, ori_path, crop_size=(256, 256), crop=(0, 0), mode="", mask_rad=""):
        self.ori_paths = ori_path
        # self.gt_paths = gt_path
        self.crop_size = crop_size
        self.crop = crop
        self.mode = mode
        self.mask_rad = mask_rad

    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(30, h - self.crop_size[0] - 30)
        left = np.random.randint(30, w - self.crop_size[1] - 30)
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id, 0]
        time_late = self.ori_paths[data_id, 1]
        CMP_frame = int(img_name.stem[-3:])

        img_name2 = img_name.parent.joinpath(
            f"{int(img_name.stem[-3:]) + int(time_late):05d}.tif"
        )
        img = cv2.imread(str(img_name), -1)[
            self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
        ]
        # img = img / 13132
        img = img / 4096
        # img = img[self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512]
        try:
            img2 = cv2.imread(str(img_name2), -1)[
                self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
            ]
        except:
            print(img_name2)
        # img2 = img2 / 13132
        img2 = img2 / 4096
        # img2 = img2[
        #     self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
        # ]

        gt_name = img_name.parent.parent.joinpath(
            f"CMP_6{self.mode}_{int(time_late)}/{CMP_frame:05d}.npy"
        )
        gt = np.load(str(gt_name)).astype(np.float32)

        mask_name = img_name.parent.parent.joinpath(
            f"mask{self.mode}{self.mask_rad}_{int(time_late)}/{CMP_frame:05d}.tif"
        )
        mask = cv2.imread(str(mask_name), 0)

        gt[mask == 255] = 255

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]
        gt = gt[top:bottom, left:right]

        img = np.concatenate(
            [
                img.reshape(1, self.crop_size[0], self.crop_size[1]),
                img2.reshape(1, self.crop_size[0], self.crop_size[1]),
            ],
            axis=0,
        )

        # img = np.concatenate(
        #     [img.reshape(1, 512, 512), img2.reshape(1, 512, 512)], axis=0
        # )
        gt = gt.transpose(2, 0, 1)

        rand_value = np.random.randint(0, 4)
        # img, gt = RandomFlipper(rand_value, img, gt)

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img, "gt": gt}

        return datas


class CMPLoadGT(object):
    def __init__(self, ori_path, crop_size=(512, 512)):
        self.ori_paths = ori_path
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(30, h - self.crop_size[0] - 30)
        left = np.random.randint(30, w - self.crop_size[1] - 30)
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id, 0]
        time_late = self.ori_paths[data_id, 1]

        CMP_frame = int(img_name.stem[-3:]) - 1

        img_name2 = img_name.parent.joinpath(
            img_name.stem[:-3]
            + f"{int(img_name.stem[-3:]) + int(time_late) -1:03d}.tif"
        )
        img = cv2.imread(str(img_name), -1)
        # img = img / 4096
        img = img / 13133

        img2 = cv2.imread(str(img_name2), -1)
        img2 = img2 / 13133

        # img2 = img2 / 4096

        # gt_name = img_name.parent.parent.joinpath(
        # f"CMP_6_{int(time_late)}/{CMP_frame:05d}.npy"
        # )
        gt_name = img_name.parent.parent.joinpath(
            f"CMP_gt/{int(time_late)}/{CMP_frame:05d}.npy"
        )
        try:
            gt = np.load(str(gt_name)).astype(np.float32)
        except:
            print(gt_name)
        # data augumentation
        # top, bottom, left, right = self.random_crop_param(img.shape)
        # top, bottom, left, right = self.random_crop_param((512, 512))
        top, bottom, left, right = (0, 512, 0, 512)

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]
        gt = gt[top:bottom, left:right]

        img = np.concatenate(
            [
                img.reshape(1, self.crop_size[0], self.crop_size[1]),
                img2.reshape(1, self.crop_size[0], self.crop_size[1]),
            ],
            axis=0,
        )
        gt = gt.transpose(2, 0, 1)

        rand_value = np.random.randint(0, 4)
        # img, gt = RandomFlipper(rand_value, img, gt)

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img, "gt": gt}

        return datas


class CMFLoad(object):
    def __init__(self, ori_path, crop_size=(256, 256), crop=(300, 160)):
        self.ori_paths = ori_path
        # self.gt_paths = gt_path
        self.crop_size = crop_size
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(30, h - self.crop_size[0] - 30)
        left = np.random.randint(30, w - self.crop_size[1] - 30)
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id, 0]
        time_late = self.ori_paths[data_id, 1]
        # time_late = 9

        CMP_frame = int(img_name.stem[-2:])

        img = cv2.imread(str(img_name), -1)[
            self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
        ]
        img = img / 4096

        img_name2 = img_name.parent.joinpath(
            img_name.stem[:-2] + f"{int(img_name.stem[-2:]) + int(time_late):02d}.tif"
        )
        img2 = cv2.imread(str(img_name2), -1)[
            self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
        ]
        img2 = img2 / 4096

        gt_name = img_name.parent.parent.joinpath(
            f"CMF_6_{int(time_late)}/{CMP_frame:05d}.npy"
        )
        gt = np.load(str(gt_name)).astype(np.float32)

        mask_name = img_name.parent.parent.joinpath(
            f"mask_{int(time_late)}/{CMP_frame:05d}.tif"
        )
        if mask_name.parent.is_dir():

            mask = cv2.imread(str(mask_name), 0)

            # gt[mask == 255] = 255
            gt_save = gt.copy()
            gt[0, :, :, 0][mask == 255] = 255
            gt[0, :, :, 1][mask == 255] = 255
            gt_save[0, :, :, 0][mask == 255] = 1
            gt_save[0, :, :, 1][mask == 255] = 1

        # data augumentation
        # top, bottom, left, right = self.random_crop_param(img.shape)

        # img = img[top:bottom, left:right]
        # img2 = img2[top:bottom, left:right]

        # gt = gt[0, top:bottom, left:right, :]
        # gt_save = gt_save[0, top:bottom, left:right, :]

        # img = np.concatenate(
        #     [
        #         img.reshape(1, self.crop_size[0], self.crop_size[1]),
        #         img2.reshape(1, self.crop_size[0], self.crop_size[1]),
        #     ],
        #     axis=0,
        # )
        img = np.concatenate(
            [img.reshape(1, 512, 512), img2.reshape(1, 512, 512)], axis=0
        )

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        gt = gt[0].permute(2, 0, 1)

        datas = {"image": img, "gt": gt}

        return datas


class CMFLoadGT(object):
    def __init__(self, ori_path, crop_size=(256, 256), crop=(300, 160)):
        self.ori_paths = ori_path
        # self.gt_paths = gt_path
        self.crop_size = crop_size
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(30, h - self.crop_size[0] - 30)
        left = np.random.randint(30, w - self.crop_size[1] - 30)
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id, 0]
        time_late = self.ori_paths[data_id, 1]
        # time_late = 9

        CMP_frame = int(img_name.stem[-3:])

        img = cv2.imread(str(img_name), -1)
        img = img / 4096

        img_name2 = img_name.parent.joinpath(
            img_name.stem[:-3] + f"{int(img_name.stem[-3:]) + int(time_late):03d}.tif"
        )
        img2 = cv2.imread(str(img_name2), -1)
        img2 = img2 / 4096

        gt_name = img_name.parent.parent.joinpath(
            f"CMF_6_{int(time_late)}/{CMP_frame:05d}.npy.npz"
        )
        gt = np.load(str(gt_name))["arr_0"].astype(np.float32)

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]

        gt = gt[0, top:bottom, left:right, :]

        img = np.concatenate(
            [
                img.reshape(1, self.crop_size[0], self.crop_size[1]),
                img2.reshape(1, self.crop_size[0], self.crop_size[1]),
            ],
            axis=0,
        )

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        gt = gt.permute(2, 0, 1)

        datas = {"image": img, "gt": gt}

        return datas


class LikeLoad(object):
    def __init__(self, ori_path, gt_path, crop_size=(256, 256), crop=(300, 160)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.crop_size = crop_size
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        CMP_frame = int(img_name.stem[-3:])

        img = cv2.imread(str(img_name), -1)
        img = img / 4096
        # img = img[self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512]

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / gt.max()
        # mask_name = img_name.parent.parent.joinpath(f"mask_1/{CMP_frame:05d}.tif")
        # if mask_name.parent.is_dir():
        #     mask = cv2.imread(str(mask_name), 0)

        #     gt[mask == 255] = 255

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        gt = gt[top:bottom, left:right]

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        img = img.unsqueeze(0)
        gt = gt.unsqueeze(0)

        datas = {"image": img, "gt": gt}

        return datas


class CMFTest(object):
    def __init__(self, ori_path, crop_size=(256, 256), crop=(300, 160), time_late=1):
        self.ori_paths = ori_path
        # self.gt_paths = gt_path
        self.crop_size = crop_size
        self.crop = crop
        self.time_late = time_late

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        time_late = self.time_late
        # time_late = 9

        CMP_frame = int(img_name.stem[-2:])

        img = cv2.imread(str(img_name), -1)[
            self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
        ]

        img = img / 4096
        # img = (img - img.min()) / (img.max() - img.min())

        # img = img[self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512]

        img_name2 = img_name.parent.joinpath(
            img_name.stem[:-3] + f"{int(img_name.stem[-3:]) + int(time_late):03d}.tif"
        )
        try:
            img2 = cv2.imread(str(img_name2), -1)[
                self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
            ]
        except:
            print(img_name2)
        img2 = img2 / 4096
        # img2 = img2 / img2.max()
        # img2 = (img2 - img2.min()) / (img2.max() - img2.min())

        # img2 = img2[
        #     self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512
        # ]

        # gt_name = img_name.parent.parent.joinpath(
        #     f"CMF_6_{int(time_late)}/{CMP_frame:05d}.npy"
        # )
        # gt = np.load(str(gt_name)).astype(np.float32)[0]

        # mask_name = img_name.parent.parent.joinpath(
        #     f"mask_{int(time_late)}/{CMP_frame:05d}.tif"
        # )
        # if mask_name.parent.is_dir():

        #     mask = cv2.imread(str(mask_name), 0)

        #     # gt[mask == 255] = 255
        #     gt_save = gt.copy()
        #     gt[:, :, 0][mask == 255] = 255
        #     gt[:, :, 1][mask == 255] = 255

        img = np.concatenate(
            [img.reshape(1, 512, 512), img2.reshape(1, 512, 512)], axis=0
        )

        img = torch.from_numpy(img.astype(np.float32))
        # gt = torch.from_numpy(gt.astype(np.float32))

        # img = img.unsqueeze(0)
        # gt = gt.permute(2, 0, 1)

        datas = {"image": img}
        # datas = {"image": img, "gt": gt}

        return datas
