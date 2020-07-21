from guided_model import GuidedModel
import torch.utils.data
from utils import *
import multiprocessing
from pathlib import Path
import torch
from networks import UNet, UNet2, UNet3


def gather_path(paths, mode):
    ori_paths = []
    for path in paths:
        ori_paths.extend(sorted(path.joinpath(mode).glob("*.tif")))
    return ori_paths


class Backwardprop(object):
    def __init__(
            self, input_path, output_path, net, gpu=True, time_late=1, t_or_n=1
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)
        output_path.chmod(0o777)
        ori_paths = gather_path(input_path, "ori")
        if seq == 10:
            crop_init = (300, 160)
        elif (seq== 6) or (seq == 15):
            crop_init = (0, 0)
        elif seq == 13:
            crop_init = (528, 0)
        else:
            crop_init = (500, 300)
        data_loader = CellImageProp(ori_paths, time_late=time_late, crop=crop_init)
        self.dataset = torch.utils.data.DataLoader(
            data_loader, 1, shuffle=False, num_workers=1
        )
        self.number_of_data = data_loader.__len__()

        self.gpu = gpu
        # network load
        self.net = net
        self.net.eval()
        if self.gpu:
            self.net.cuda()
        self.t_or_n = t_or_n

        self.back_model = GuidedModel(self.net)
        self.back_model.inference()

    def main(self):
        for iteration, data in enumerate(self.dataset):
            img = data["image"]
            if self.gpu:
                img = img.cuda()

            self.output_path_each = self.output_path.joinpath(
                "{:05d}".format(iteration)
            )
            self.output_path_each.mkdir(parents=True, exist_ok=True)
            self.output_path_each.chmod(0o777)

            module = self.back_model
            gbs = module(img, self.output_path_each, t_or_n=self.t_or_n)

            gbs = np.array(gbs)
            gbs_coloring = self.coloring(gbs)
            gbs_coloring = np.array(gbs_coloring)

            cv2.imwrite(
                str(self.output_path_each.joinpath("original.tif")),
                (img.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8),
            )
            np.savez_compressed(
                str(self.output_path_each.joinpath("{:05d}".format(iteration))), gb=gbs
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

    def coloring(self, gbs):
        # coloring
        r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
        gbs_coloring = []
        for peak_i, gb in enumerate(gbs):
            gb = gb / gb.max() * 255
            gb = gb.clip(0, 255).astype(np.uint8)
            result = np.ones((gb.shape[0], gb.shape[1], gb.shape[2], 3))
            result = gb[..., np.newaxis] * result
            peak_i = peak_i % 8
            result[..., 0][result[..., 0] != 0] = r[peak_i] * gb[gb != 0]
            result[..., 1][result[..., 1] != 0] = g[peak_i] * gb[gb != 0]
            result[..., 2][result[..., 2] != 0] = b[peak_i] * gb[gb != 0]
            gbs_coloring.append(result)
        return gbs_coloring


if __name__ == "__main__":
    torch.cuda.set_device(0)
    time_lates = [1, 5, 9]
    for time_late in time_lates:
        net = UNet3(n_channels=1, n_classes=1, sig=False)

        input_path = [Path(f"/home/kazuya/main/ECCV/correlation_test/images/sequ9")]

        weight_path = Path(
            f"./weights/C2C12_9_{time_late}/best.pth"
        )

        output_path = Path(f"./outputs/test/{weight_path.parent.name}/sequ9")

        net.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
        net.cuda()

        bp = Backwardprop(
            input_path, output_path, net, time_late=time_late, t_or_n=1
        )
        bp.main()
