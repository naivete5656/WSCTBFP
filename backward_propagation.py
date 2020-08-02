from guided_model import GuidedModel
import torch.utils.data
from utils import *
import multiprocessing
from pathlib import Path
import torch
from networks import CoDetectionCNN
from utils import gather_path

import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="data path")
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
        default="./output/backward",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        # default="./weights/best.pth",
        default="./weights",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true"
    )

    args = parser.parse_args()
    return args


class Backwardprop(object):
    def __init__(
            self, input_path, output_path, net, gpu=True, time_late=1
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)
        ori_paths = gather_path(input_path, "img")
        data_loader = CellImageProp(ori_paths, time_late=time_late)

        self.dataset = torch.utils.data.DataLoader(
            data_loader, 1, shuffle=False, num_workers=0
        )

        self.number_of_data = data_loader.__len__()

        self.gpu = gpu
        # network load
        self.net = net
        self.net.eval()
        if self.gpu:
            self.net.cuda()

        self.back_model = GuidedModel(self.net)
        self.back_model.inference()

    def main(self):
        for iteration, data in enumerate(self.dataset):
            img = data["image"]
            if self.gpu:
                img = img.cuda()

            self.output_path_each = self.output_path.joinpath(
                f"{iteration:05d}"
            )
            self.output_path_each.mkdir(parents=True, exist_ok=True)

            module = self.back_model
            gbs = module(img, self.output_path_each)

            gbs = np.array(gbs)

            cv2.imwrite(
                str(self.output_path_each.joinpath("original.tif")),
                (img.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8),
            )
            np.savez_compressed(
                str(self.output_path_each.joinpath(f"{iteration:05d}")), gb=gbs
            )

    def unet_pred(self, img, save_path=None):
        # throw unet
        if self.gpu:
            img = img.cuda()
        pre_img = self.net(img)
        pre_img = pre_img.detach().cpu().numpy()[0, 0].clamp(min=0, max=1)
        pre_img = (pre_img) * 255
        if save_path is not None:
            cv2.imwrite(str(save_path), (pre_img * 255).astype(np.uint8))
        return pre_img


if __name__ == "__main__":
    args = parse_args()
    time_lates = [1, 5, 9]
    for time_late in time_lates:
        net = CoDetectionCNN(n_channels=1, n_classes=1, sig=False)

        input_path = [Path(args.input_path)]

        weight_path = Path(args.weight_path).joinpath(f"C2C12_9_{time_late}/temp.pth")

        output_path = Path(args.output_path).joinpath(f"{time_late}")

        net.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
        net.cuda()

        bp = Backwardprop(
            input_path, output_path, net, time_late=time_late,
        )
        bp.main()
