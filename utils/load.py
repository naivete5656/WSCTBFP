import torch
import numpy as np

# import h5py
from scipy.ndimage.interpolation import rotate
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import math


# from torchvision.transforms import (
#     Compose,
#     RandomCrop,
#     RandomRotation,
#     CenterCrop,
#     Normalize,
#     ToTensor,
# )
class OriCrop(object):
    def __init__(self, ori_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ori_paths)


class GT(OriCrop):
    def __init__(self, ori_path, gt_path, crop_size=(256, 256), time_late=1):
        super().__init__(ori_path, crop_size)
        self.gt_paths = gt_path
        self.time_late = time_late

    def __len__(self):
        return len(self.ori_paths) - self.time_late


class RandomCropClass(object):
    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def cropping(self, img, img2, gt, gt2):
        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]
        gt = gt[top:bottom, left:right]
        gt2 = gt2[top:bottom, left:right]

        return img, img2, gt, gt2


def flip(img, img2, gt, gt2):
    rand_value = np.random.randint(0, 4)
    if rand_value == 1:
        img = np.flipud(img)
        img2 = np.flipud(img2)
        gt = np.flipud(gt)
        gt2 = np.flipud(gt2)
    elif rand_value == 2:
        img = np.fliplr(img)
        img2 = np.fliplr(img2)
        gt = np.fliplr(gt)
        gt2 = np.fliplr(gt2)
    elif rand_value == 3:
        img = np.flipud(img)
        img2 = np.flipud(img2)
        gt = np.flipud(gt)
        gt2 = np.flipud(gt2)
        img = np.fliplr(img)
        img2 = np.fliplr(img2)
        gt = np.fliplr(gt)
        gt2 = np.fliplr(gt2)
    return img, img2, gt, gt2


def load_img(img_name, img_name2, gt_name, gt_name2):
    img = cv2.imread(str(img_name), -1)
    img = img / 4096
    img2 = cv2.imread(str(img_name2), -1)
    img2 = img2 / 4096
    gt = cv2.imread(str(gt_name), -1)
    gt = gt / 255
    gt2 = cv2.imread(str(gt_name2), -1)
    gt2 = gt2 / 255
    return img, img2, gt, gt2


class CellImageLoadTime(RandomCropClass, GT):
    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id, 0]
        time_late = self.ori_paths[data_id, 1]

        img_name2 = img_name.parent.joinpath(
            img_name.stem[:-5] + f"{int(img_name.stem[-5:]) + int(time_late):05d}.tif"
        )

        gt_name = self.gt_paths[data_id]
        gt_name2 = gt_name.parent.joinpath(
            gt_name.stem[:-5] + f"{int(gt_name.stem[-5:]) + int(time_late):05d}.tif"
        )

        img, img2, gt, gt2 = load_img(img_name, img_name2, gt_name, gt_name2)

        img, img2, gt, gt2 = self.cropping(img, img2, gt, gt2)

        img, img2, gt, gt2 = flip(img, img2, gt, gt2)

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        gt2 = torch.from_numpy(gt2.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])
        gt = torch.cat([gt.unsqueeze(0), gt2.unsqueeze(0)])

        datas = {"image": img, "gt": gt}

        return datas


class CellImageLoad(RandomCropClass, GT):

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_name2 = self.ori_paths[data_id + self.time_late]

        gt_name = self.gt_paths[data_id]
        gt_name2 = self.gt_paths[data_id + self.time_late]

        img, img2, gt, gt2 = load_img(img_name, img_name2, gt_name, gt_name2)

        img, img2, gt, gt2 = self.cropping(img, img2, gt, gt2)

        img, img2, gt, gt2 = flip(img, img2, gt, gt2)

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        gt2 = torch.from_numpy(gt2.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])
        gt = torch.cat([gt.unsqueeze(0), gt2.unsqueeze(0)])

        datas = {"image": img, "gt": gt}

        return datas


class CellImageLoadTest(GT):
    def __init__(
            self, ori_path, gt_path, crop_size=(512, 512), time_late=1, bg_path=None, crop=(500, 300)
    ):
        super().__init__(ori_path, gt_path, crop_size=(512, 512), time_late=time_late)
        self.bg_paths = bg_path
        self.crop = (crop[0], crop[0] + 512, crop[1], crop[1] + 512)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_name2 = self.ori_paths[data_id + self.time_late]

        gt_name = self.gt_paths[data_id]
        gt_name2 = self.gt_paths[data_id + self.time_late]

        img, img2, gt, gt2 = load_img(img_name, img_name2, gt_name, gt_name2)

        if self.bg_paths is not None:
            bg_name = self.bg_paths[data_id]
            bg1 = cv2.imread(str(bg_name), 0)
            bg1 = bg1 / 255
            bg_name = self.bg_paths[data_id + self.time_late]
            bg2 = cv2.imread(str(bg_name), 0)
            bg2 = bg2 / 255

        # data augumentation
        # top, bottom, left, right = (134, 646, 159, 671)
        top, bottom, left, right = self.crop

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]
        gt = gt[top:bottom, left:right]
        gt2 = gt2[top:bottom, left:right]

        if self.bg_paths is not None:
            bg1 = bg1[top:bottom, left:right]
            bg2 = bg2[top:bottom, left:right]
            bg1 = torch.from_numpy(bg1.astype(np.float32))
            bg2 = torch.from_numpy(bg2.astype(np.float32))
            bg = torch.cat([bg1.unsqueeze(0), bg2.unsqueeze(0)])

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        gt2 = torch.from_numpy(gt2.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])
        gt = torch.cat([gt.unsqueeze(0), gt2.unsqueeze(0)])

        if self.bg_paths is not None:
            datas = {"image": img, "gt": gt, "bg": bg}
        else:
            datas = {"image": img, "gt": gt}

        return datas


class CellImageLoadElmer(object):
    def __init__(self, ori_path, gt_paths, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_paths
        self.crop_size = crop_size

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
        img = cv2.imread(str(img_name), -1)
        img = (img - img.min()) / (img.max() - img.min())

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), -1)
        gt = gt / gt.max()

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        gt = gt[top:bottom, left:right]

        rand_value = np.random.randint(0, 4)
        if rand_value == 1:
            img = np.flipud(img)
            gt = np.flipud(gt)
        elif rand_value == 2:
            img = np.fliplr(img)
            gt = np.fliplr(gt)
        elif rand_value == 3:
            img = np.flipud(img)
            gt = np.flipud(gt)
            img = np.fliplr(img)
            gt = np.fliplr(gt)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        img = img.unsqueeze(0)
        gt = gt.unsqueeze(0)

        datas = {"image": img, "gt": gt}

        return datas


class CellImageLoadTestElmer(object):
    def __init__(self, ori_path):
        self.ori_paths = ori_path

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), -1)
        img = (img - img.min()) / (img.max() - img.min())
        img = img[:512, :512]

        img = torch.from_numpy(img.astype(np.float32))

        img = img.unsqueeze(0)

        datas = {"image": img}

        return datas


class CellImageLoadTestsingleCha(object):
    def __init__(self, ori_path, gt_paths, crop):
        self.ori_paths = ori_path
        self.gt_paths = gt_paths
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), -1)[
              self.crop[0]: self.crop[0] + 512, self.crop[1]: self.crop[1] + 512
              ]
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 4096
        # img = img / img.max()
        # img = img / 255

        # img = img[self.crop[0] : self.crop[0] + 512, self.crop[1] : self.crop[1] + 512]
        # img = np.pad(img, [(0, 0), (24, 24)], "edge")

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), -1)
        gt = gt / gt.max()
        gt = gt[self.crop[0]: self.crop[0] + 512, self.crop[1]: self.crop[1] + 512]

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        img = img.unsqueeze(0)
        gt = gt.unsqueeze(0)

        datas = {"image": img, "gt": gt}

        return datas


class CellImageProp(object):
    def __init__(self, ori_path, time_late=1, crop=(0, 0)):
        self.ori_paths = ori_path
        self.time_late = time_late
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths) - self.time_late

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_name2 = self.ori_paths[data_id + self.time_late]
        img = cv2.imread(str(img_name), -1)
        img = img / 4095
        img = img[self.crop[0]: self.crop[0] + 512, self.crop[1]: self.crop[1] + 512]

        img2 = cv2.imread(str(img_name2), -1)
        img2 = img2 / 4095
        img2 = img2[
               self.crop[0]: self.crop[0] + 512, self.crop[1]: self.crop[1] + 512
               ]

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])

        datas = {"image": img}

        return datas


class CellImagePropChallenge(object):
    def __init__(self, ori_path, time_late=1, crop=(0, 0), dataset=None):
        self.ori_paths = ori_path
        self.time_late = time_late
        self.crop = crop
        self.dataset = dataset

    def __len__(self):
        return len(self.ori_paths) - self.time_late

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_name2 = self.ori_paths[data_id + self.time_late]
        img = cv2.imread(str(img_name), -1)
        img = img / 255
        if self.dataset == "PhC-C2DH-U373":
            img = np.pad(img, [(0, 0), (4, 4)], "edge")
            img = img[4:516]
        else:
            img = np.pad(img, [(0, 0), (24, 24)], "edge")

        img2 = cv2.imread(str(img_name2), -1)
        img2 = img2 / 255
        if self.dataset == "PhC-C2DH-U373":
            img2 = np.pad(img2, [(0, 0), (4, 4)], "edge")
            img2 = img2[4:516]
        else:
            img2 = np.pad(img2, [(0, 0), (24, 24)], "edge")

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])

        datas = {"image": img}

        return datas


class CellImageLoadCMF(object):
    def __init__(self, ori_path, gt_path, mask_path, crop_size=(256, 256), time_late=1):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.mask_paths = mask_path
        self.crop_size = crop_size
        self.time_late = time_late

    def __len__(self):
        return len(self.ori_paths) - self.time_late

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id + self.time_late]
        img_name2 = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), -1)[134:646, 159:671]
        img = img / img.max()
        img2 = cv2.imread(str(img_name2), -1)[134:646, 159:671]
        img2 = img2 / img2.max()

        gt_name = self.gt_paths[data_id]
        gt = np.load(str(gt_name)).astype(np.float32)[:512, :512]

        mask_name = self.mask_paths[data_id]
        mask = cv2.imread(str(mask_name), 0)

        gt[mask == 255] = 255

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]
        gt = gt[top:bottom, left:right]

        # rand_value = np.random.randint(0, 4)
        # img = rotate(img, 90 * rand_value, mode="nearest")
        # img2 = rotate(img2, 90 * rand_value, mode="nearest")
        # gt = rotate(gt, 90 * rand_value)

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])
        gt = gt.permute(2, 0, 1)

        datas = {"image": img, "gt": gt}

        return datas


class CellImageLoadCMFTest(object):
    def __init__(self, ori_path, gt_path, time_late=1):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.time_late = time_late

    def __len__(self):
        return len(self.ori_paths) - self.time_late

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_name2 = self.ori_paths[data_id + self.time_late]
        # img = cv2.imread(str(img_name), -1)[134:646, 159:671]
        img = cv2.imread(str(img_name), -1)

        img = img / img.max()
        img = img[500:1012, 300: 300 + 512]
        # img = img[:512, :512]

        # img2 = cv2.imread(str(img_name2), -1)[134:646, 159:671]
        img2 = cv2.imread(str(img_name2), -1)

        img2 = img2 / img2.max()
        img2 = img2[500:1012, 300: 300 + 512]
        # img2 = img2[:512, :512]

        gt_name = self.gt_paths[data_id]
        gt = np.load(str(gt_name)).astype(np.float32)[:512, :512]

        # data augumentation

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])
        gt = gt.permute(2, 0, 1)

        # datas = {"image": img, "gt": gt}
        datas = {"image": img, "gt": gt}

        return datas


class CellImageLoadBg(object):
    def __init__(self, ori_path, time_late=1, bg_path=None, crop=(0, 0)):
        self.ori_paths = ori_path
        self.crop = crop
        self.time_late = time_late
        self.bg_paths = bg_path
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths) - self.time_late

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img_name2 = self.ori_paths[data_id + self.time_late]
        img = cv2.imread(str(img_name), -1)
        img = img / img.max()
        img2 = cv2.imread(str(img_name2), -1)
        img2 = img2 / img2.max()

        bg_name = self.bg_paths[data_id]
        bg1 = cv2.imread(str(bg_name), 0)
        bg1 = bg1 / 255
        bg_name = self.bg_paths[data_id + self.time_late]
        bg2 = cv2.imread(str(bg_name), 0)
        bg2 = bg2 / 255
        dif = img.mean() - bg1.mean()
        bg1 = bg1 + dif
        dif = img2.mean() - bg2.mean()
        bg2 = bg2 + dif

        # data augumentation
        top, bottom, left, right = (
            self.crop[0],
            self.crop[0] + 512,
            self.crop[1],
            self.crop[1] + 512,
        )

        img = img[top:bottom, left:right]
        img2 = img2[top:bottom, left:right]

        if self.bg_paths is not None:
            bg1 = bg1[top:bottom, left:right]
            bg2 = bg2[top:bottom, left:right]
            bg1 = torch.from_numpy(bg1.astype(np.float32))
            bg2 = torch.from_numpy(bg2.astype(np.float32))
            bg = torch.cat([bg1.unsqueeze(0), bg2.unsqueeze(0)])

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))

        img = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])

        datas = {"image": img, "bg": bg}

        return datas
