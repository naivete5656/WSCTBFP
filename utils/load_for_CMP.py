import torch
import numpy as np

# import h5py
from scipy.ndimage.interpolation import rotate
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import random


class CMPLoad(object):
    def __init__(self, ori_path, crop_size=(256, 256)):
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

    @classmethod
    def flip_and_rotate(cls, img, mpm, seed):
        img = rotate(img, 90 * (seed % 4))
        mpm = rotate(mpm, 90 * (seed % 4))
        # process for MPM
        ## seed = 1 or 5: 90 degrees counterclockwise
        if seed % 4 == 1:
            mpm[:, :, 1] = -mpm[:, :, 1]
            mpm = mpm[:, :, [1, 0, 2]]
        ## seed = 2 or 6: 180 degrees counterclockwise
        if seed % 4 == 2:
            mpm[:, :, :2] = -mpm[:, :, :2]
        ## seed = 3 or 7: 270 degrees counterclockwise
        if seed % 4 == 3:
            mpm[:, :, 0] = -mpm[:, :, 0]
            mpm = mpm[:, :, [1, 0, 2]]
        ## flip horizontal (4 or more)
        if seed > 3:
            img = np.fliplr(img).copy()
            mpm = np.fliplr(mpm).copy()
            mpm[:, :, 1] = -mpm[:, :, 1]
        return img, mpm

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id, 0]
        time_late = self.ori_paths[data_id, 1]
        CMP_frame = int(img_name.stem[-3:])

        img_name2 = img_name.parent.joinpath(
            f"{int(img_name.stem[-3:]) + int(time_late):05d}.tif"
        )

        img = cv2.imread(str(img_name), -1)
        img2 = cv2.imread(str(img_name2), -1)
        img = img / 255
        img2 = img2 / 255

        gt_name = img_name.parent.parent.joinpath(
            f"cmp/{int(time_late)}/{CMP_frame:05d}.npy"
        )
        gt = np.load(str(gt_name)).astype(np.float32)

        mask_name = img_name.parent.parent.joinpath(
            f"cmp/mask_{int(time_late)}/{CMP_frame:05d}.tif"
        )
        mask = cv2.imread(str(mask_name), 0)
        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]
        gt = gt[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        img = np.concatenate(
            [
                img.reshape(self.crop_size[0], self.crop_size[1], 1),
                img2.reshape(self.crop_size[0], self.crop_size[1], 1),
            ],
            axis=2,
        )

        seed = random.randrange(8)
        img, gt = self.flip_and_rotate(img, gt, seed)

        gt[mask == 255] = 255

        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img, "gt": gt}

        return datas



