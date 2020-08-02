from pathlib import Path
import torch.utils.data
from utils import CellImageLoadForward, gather_path
from networks import *
import cv2
import numpy as np
import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default="./data",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./output/forward",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mask_path",
        dest="mask_path",
        help="mask path",
        default="./output/backward",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        # default="./weight",
        default="./weights",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true"
    )

    args = parser.parse_args()
    return args


class ForwardPropagation(object):
    def __init__(self, **kwargs):
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.batch_size = 1
        self.save_path = kwargs["save_path"]

        ori_paths = gather_path(kwargs["data_paths"], "img")
        gt_paths = gather_path(kwargs["data_paths"], "like")
        mask_paths = sorted(kwargs["mask_path"].glob("*/*.npz"))
        bg_paths = gather_path(kwargs["data_paths"], "bg")

        assert len(ori_paths) > 0, print(kwargs["input_path"])
        assert len(gt_paths) > 0, print(kwargs["input_path"])

        data_loader = CellImageLoadForward(
            ori_paths, gt_paths, bg_paths, mask_paths, kwargs["time_late"]
        )

        self.dataset = torch.utils.data.DataLoader(
            data_loader, 1, shuffle=False, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()

        self.save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
        self.save_path.joinpath("pred").mkdir(parents=True, exist_ok=True)

    def __call__(self, *args, **kwargs):
        self.net.eval()
        for iteration, data in enumerate(self.dataset):
            img_ori = data["image"]
            target = data["gt"]
            bg = data["bg"]
            masks = data["mask"]

            masks = masks.numpy()[0]
            region_label1 = np.argmax(masks[:, 0], axis=0)
            masks_sum1 = masks[:, 0].max(0)
            region_label1[masks_sum1 < 0.01] = 0

            region_label2 = np.argmax(masks[:, 1], axis=0)
            masks_sum2 = masks[:, 1].max(0)
            region_label2[masks_sum2 < 0.01] = 0

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

                if self.gpu:
                    img = img.cuda()

                pred_img = self.net(img)
                pred1 = pred_img[0].detach().cpu().clamp(min=0, max=1).numpy()
                pred2 = pred_img[1].detach().cpu().clamp(min=0, max=1).numpy()

                del pred_img
                torch.cuda.empty_cache()

                img = (img * 255).cpu().numpy().astype(np.uint8)
                pred1 = (pred1 / pred1.max() * 255).astype(np.uint8)
                pred2 = (pred2 / pred2.max() * 255).astype(np.uint8)

                cv2.imwrite(str(self.save_path / Path(f"img/{iteration:04d}_{i:04d}_1.png")), img[0, 0, :, :])
                cv2.imwrite(str(self.save_path / Path(f"img/{iteration:04d}_{i:04d}_2.png")), img[0, 1, :, :])
                cv2.imwrite(str(self.save_path / Path(f"pred/{iteration:04d}_{i:04d}_1.png")), pred1[0, 0, :, :])
                cv2.imwrite(str(self.save_path / Path(f"pred/{iteration:04d}_{i:04d}_2.png")), pred2[0, 0, :, :])


if __name__ == "__main__":
    args = parse_args()
    time_lates = [1, 5, 9]
    for time_late in time_lates:
        net = CoDetectionCNN(n_channels=1, n_classes=1, sig=False)

        input_path = [Path(args.input_path)]

        mask_path = Path(args.mask_path).joinpath(f"{time_late}")

        weight_path = Path(args.weight_path).joinpath(f"C2C12_9_{time_late}/temp.pth")

        net.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
        if args.gpu:
            net.cuda()

        save_path = Path(args.output_path).joinpath(f"{time_late}")

        save_path.mkdir(parents=True, exist_ok=True)

        args_list = {
            "net": net,
            "gpu": args.gpu,
            "data_paths": input_path,
            "save_path": save_path,
            "time_late": time_late,
            "mask_path": mask_path,
        }
        pre = ForwardPropagation(**args_list)
        pre()
