from types import MethodType
import torch.nn as nn
from .guided_parts import guide_relu
from scipy.io import savemat
import numpy as np
import cv2
from utils import local_maxim, gaus_filter
import torch
import matplotlib.pyplot as plt


class GuidedModel(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.inferencing = False
        self.shape = None

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module._original_forward = module.forward
                module.forward = MethodType(guide_relu, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.ReLU) and hasattr(module, "_original_forward"):
                module.forward = module._original_forward

    def forward(
        self,
        img,
        root_path,
        t_or_n=1,
        peak=None,
        class_threshold=0,
        peak_threshold=30,
        retrieval_cfg=None,
        dataset=None,
    ):
        assert img.dim() == 4, "PeakResponseMapping layer only supports batch mode."
        if self.inferencing:
            img.requires_grad_()

        # classification network forwarding
        class_response_maps = super().forward(img)

        pre_img = (
            class_response_maps[t_or_n].detach().clamp(min=0, max=1).cpu().numpy()[0, 0]
        )
        self.shape = pre_img.shape
        if peak is None:
            cv2.imwrite(
                str(root_path.joinpath("detection.tif")),
                (pre_img * 255).astype(np.uint8),
            )
        # peak
        peaks = local_maxim((pre_img * 255).astype(np.uint8), 125, 2).astype(np.int)

        if dataset == "PhC-C2DL-PSC":
            peaks = peaks[(peaks[:, 0] > 24) & (peaks[:, 0] < 744)]
        region = np.zeros(self.shape)
        for label, peak in enumerate(peaks):
            region = cv2.circle(
                region, (int(peak[0]), int(peak[1])), 18, label + 1, thickness=-1
            )

        gbs = []
        # each propagate
        peaks = np.insert(peaks, 0, [0, 0], axis=0)
        with open(str(root_path.joinpath("peaks.txt")), mode="w") as f:
            f.write("ID,x,y\n")
            for i in range(int(region.max() + 1)):
                if img.grad is not None:
                    img.grad.zero_()
                # f.write(f"{i},{peaks[i, 0]},{peaks[i ,1]}\n")
                f.write("{},{},{}\n".format(i, peaks[i, 0], peaks[i, 1]))
                mask = np.zeros((1, self.shape[0], self.shape[1]), dtype=np.float32)
                mask[0][region == i] = 1
                mask = mask.reshape([1, 1, self.shape[0], self.shape[1]])
                mask = torch.from_numpy(mask)
                mask = mask.cuda()

                # class_response_maps[1].backward(mask, retain_graph=True)
                class_response_maps[t_or_n].backward(mask, retain_graph=True)
                result = img.grad.detach().clone().clamp(min=0).cpu().numpy()[0]
                gbs.append(result)
        return gbs

    def train(self, mode=True):
        super().train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super().train(False)
        self._patch()
        self.inferencing = True
        return self
