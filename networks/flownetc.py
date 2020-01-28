import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import math
import numpy as np

from .util import conv, predict_flow, deconv, crop_like, correlate

"Parameter count , 39,175,298 "


class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetC, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 1, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:, :1]
        x2 = x[:, 1:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, concat2, out_conv1a, flow3, flow4, flow5, flow6
        else:
            return flow2, concat2, out_conv1a, flow3, flow4, flow5, flow6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


class FlowNetC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flownetc = FlowNetC()
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.predict_flow1 = predict_flow(130)
        self.predict_flow = predict_flow(66)
        self.deconv1 = deconv(194, 64)
        self.deconv = deconv(130, 64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        flow2, concat2, out_conv1a, flow3, flow4, flow5, flow6 = self.flownetc(x)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1a, out_deconv1, flow2_up), 1)

        flow1 = self.predict_flow1(concat1)

        flow_up = self.upsampled_flow1_to_0(flow1)
        out_deconv = self.deconv(concat1)

        concat0 = torch.cat((out_deconv, flow_up), 1)
        flow = self.predict_flow(concat0)
        # flow = self.sigmoid(flow)

        return flow, flow1, flow2, flow3, flow4, flow5, flow6


class FlowNetCA(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super().__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 1, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.predict_flow1 = predict_flow(130)
        self.predict_flow = predict_flow(66)
        self.deconv1 = deconv(194, 64)
        self.deconv = deconv(130, 64)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:, :1]
        x2 = x[:, 1:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv1 = torch.cat([out_conv1a, out_conv1b])
        out_conv2 = torch.cat([out_conv2a, out_conv2b])
        out_conv3 = torch.cat([out_conv3a, out_conv3b])

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)

        flow1 = self.predict_flow1(concat1)

        flow_up = self.upsampled_flow1_to_0(flow1)
        out_deconv = self.deconv(concat1)

        concat0 = torch.cat((out_deconv, flow_up), 1)
        flow = self.predict_flow(concat0)

        return flow, flow1, flow2, flow3, flow4, flow5, flow6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args = args.parse_args()
    args.fp16 = False

    model = FlowNetC2(args)
    x = torch.randn(10, 2, 256, 256)
    pred = model(x)
# class FlowNetC(nn.Module):
#     def __init__(self, args, batchNorm=True, div_flow=20):
#         super(FlowNetC, self).__init__()

#         self.batchNorm = batchNorm
#         self.div_flow = div_flow

#         self.conv1 = conv(self.batchNorm, 1, 64, kernel_size=7, stride=2)
#         self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
#         self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
#         self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

#         if args.fp16:
#             self.corr = nn.Sequential(
#                 tofp32(),
#                 Correlation(
#                     pad_size=20,
#                     kernel_size=1,
#                     max_displacement=20,
#                     stride1=1,
#                     stride2=2,
#                     corr_multiply=1,
#                 ),
#                 tofp16(),
#             )
#         else:
#             self.corr = Correlation(
#                 pad_size=20,
#                 kernel_size=1,
#                 max_displacement=20,
#                 stride1=1,
#                 stride2=2,
#                 corr_multiply=1,
#             )

#         # self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
#         self.corr_activation = nn.ReLU(inplace=True)
#         self.conv3_1 = conv(self.batchNorm, 473, 256)
#         self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
#         self.conv4_1 = conv(self.batchNorm, 512, 512)
#         self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
#         self.conv5_1 = conv(self.batchNorm, 512, 512)
#         self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
#         self.conv6_1 = conv(self.batchNorm, 1024, 1024)

#         self.deconv5 = deconv(1024, 512)
#         self.deconv4 = deconv(1026, 256)
#         self.deconv3 = deconv(770, 128)
#         self.deconv2 = deconv(386, 64)

#         self.predict_flow6 = predict_flow(1024)
#         self.predict_flow5 = predict_flow(1026)
#         self.predict_flow4 = predict_flow(770)
#         self.predict_flow3 = predict_flow(386)
#         self.predict_flow2 = predict_flow(194)

#         self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
#         self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
#         self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
#         self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)

#             if isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)
#                 # init_deconv_bilinear(m.weight)
#         self.upsample1 = nn.Upsample(scale_factor=4, mode="bilinear")

#     def forward(self, x):
#         x1 = x[:, 0:1, :, :]
#         x2 = x[:, 1::, :, :]

#         out_conv1a = self.conv1(x1)
#         out_conv2a = self.conv2(out_conv1a)
#         out_conv3a = self.conv3(out_conv2a)

#         # FlownetC bottom input stream
#         out_conv1b = self.conv1(x2)
#         out_conv2b = self.conv2(out_conv1b)
#         out_conv3b = self.conv3(out_conv2b)

#         # Merge streams
#         out_corr = self.corr(out_conv3a, out_conv3b)  # False
#         out_corr = self.corr_activation(out_corr)

#         # Redirect top input stream and concatenate
#         out_conv_redir = self.conv_redir(out_conv3a)

#         in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

#         # Merged conv layers
#         out_conv3_1 = self.conv3_1(in_conv3_1)

#         out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

#         out_conv5 = self.conv5_1(self.conv5(out_conv4))
#         out_conv6 = self.conv6_1(self.conv6(out_conv5))

#         flow6 = self.predict_flow6(out_conv6)
#         flow6_up = self.upsampled_flow6_to_5(flow6)
#         out_deconv5 = self.deconv5(out_conv6)

#         concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

#         flow5 = self.predict_flow5(concat5)
#         flow5_up = self.upsampled_flow5_to_4(flow5)
#         out_deconv4 = self.deconv4(concat5)
#         concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

#         flow4 = self.predict_flow4(concat4)
#         flow4_up = self.upsampled_flow4_to_3(flow4)
#         out_deconv3 = self.deconv3(concat4)
#         concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

#         flow3 = self.predict_flow3(concat3)
#         flow3_up = self.upsampled_flow3_to_2(flow3)
#         out_deconv2 = self.deconv2(concat3)
#         concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

#         flow2 = self.predict_flow2(concat2)

#         return flow2, concat2, out_conv1a, flow3, flow4, flow5, flow6
