from pathlib import Path
import torch.utils.data
from utils import CellImageLoadTest
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
        default="./output/detection",
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


def gather_path(train_paths, mode):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
    return ori_paths


class Predict(object):
    def __init__(self, **kwargs):
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.batch_size = 1
        self.save_path = kwargs["save_path"]

        ori_paths = gather_path(kwargs["input_path"], "img")
        gt_paths = gather_path(kwargs["input_path"], "like")

        assert len(ori_paths) > 0, print(kwargs["input_path"])
        assert len(gt_paths) > 0, print(kwargs["input_path"])

        data_loader = CellImageLoadTest(
            ori_paths, gt_paths, time_late=kwargs["time_late"],
        )

        self.dataset = torch.utils.data.DataLoader(
            data_loader, 1, shuffle=False, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()

        self.save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
        self.save_path.joinpath("gt").mkdir(parents=True, exist_ok=True)
        self.save_path.joinpath("pred").mkdir(parents=True, exist_ok=True)

    def __call__(self, *args, **kwargs):
        self.net.eval()

        for iteration, data in enumerate(self.dataset):
            img = data["image"]
            target = data["gt"]

            if self.gpu:
                img = img.cuda()

            pred_img = self.net(img)

            pred1 = pred_img[0].clamp(min=0, max=1)
            pred2 = pred_img[1].clamp(min=0, max=1)
            pred1 = pred1.cpu()
            pred2 = pred2.cpu()
            del pred_img
            torch.cuda.empty_cache()

            img = (img * 255).cpu().numpy().astype(np.uint8)
            target = (target * 255).cpu().numpy().astype(np.uint8)
            pred1 = (pred1 * 255).detach().cpu().numpy().astype(np.uint8)
            pred2 = (pred2 * 255).detach().cpu().numpy().astype(np.uint8)

            cv2.imwrite(
                str(self.save_path / Path(f"img/{iteration:04d}_1.png")),
                img[0, 0, :, :]
            )
            cv2.imwrite(
                str(self.save_path / Path(f"img/{iteration:04d}_2.png")),
                img[0, 1, :, :],
            )

            cv2.imwrite(
                str(self.save_path / Path(f"gt/{iteration:04d}_1.png")),
                target[0, 0, :, :],
            )
            cv2.imwrite(
                str(self.save_path / Path(f"gt/{iteration:04d}_2.png")),
                target[0, 1, :, :],
            )

            cv2.imwrite(
                str(self.save_path / Path(f"pred/{iteration:04d}_1.png")),
                pred1[0, 0, :, :],
            )
            cv2.imwrite(
                str(self.save_path / Path(f"pred/{iteration:04d}_2.png")),
                pred2[0, 0, :, :],
            )


if __name__ == "__main__":
    args = parse_args()

    time_lates = [1, 5, 9]
    for time_late in time_lates:
        net = CoDetectionCNN(n_channels=1, n_classes=1, sig=False)

        input_path = [Path(args.input_path)]

        weight_path = Path(args.weight_path).joinpath(f"C2C12_9_{time_late}/temp.pth")

        net.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
        if args.gpu:
            net.cuda()

        save_path = Path(args.output_path).joinpath(f"{time_late}")
        save_path.mkdir(parents=True, exist_ok=True)

        args_list = {
            "net": net,
            "gpu": args.gpu,
            "input_path": input_path,
            "save_path": save_path,
            "time_late": time_late,
        }
        pre = Predict(**args_list)
        pre()
