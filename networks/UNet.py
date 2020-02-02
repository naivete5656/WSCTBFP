from .network_parts import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super(UNet, self).__init__()
        filter_channel = [64, 128, 256, 512]
        self.inc = Inconv(n_channels, 64)
        self.down = nn.ModuleList([])
        for i in range(3):
            self.down.append(Down(filter_channel[i], filter_channel[i + 1]))
        self.down.append(Down(1024, 512))

        self.up_t = nn.ModuleList([Up(1024, 256)])
        self.up_tn = nn.ModuleList([Up(1024, 256)])
        for i in range(3):
            self.up_t.append(
                Up(filter_channel[-(i + 1)], int(filter_channel[-(i + 1)] / 4))
            )
            self.up_tn.append(
                Up(filter_channel[-(i + 1)], int(filter_channel[-(i + 1)] / 4))
            )
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, t_dec, tn_dec = ([0] * 4 for _ in range(4))

        t_enc[0] = self.inc(x_inp1)
        tn_enc[0] = self.inc(x_inp2)
        for i in range(3):
            t_enc[i + 1] = self.down[i](t_enc[i])
            tn_enc[i + 1] = self.down[i](tn_enc[i])
        enc = torch.cat([t_enc[-1], tn_enc[-1]], dim=1)
        enc = self.down[-1](enc)
        t_dec[0] = self.up_t[0](enc, t_enc[-1])
        tn_dec[0] = self.up_tn[0](enc, tn_enc[-1])

        for i in range(3):
            t_dec[i + 1] = self.up_t[i + 1](t_dec[i], t_enc[-(i + 2)])
            tn_dec[i + 1] = self.up_tn[i + 1](tn_dec[i], tn_enc[-(i + 2)])
        pred_t = self.outc_t(t_dec[-1])
        pred_tn = self.outc_tn(tn_dec[-1])
        return pred_t, pred_tn


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super().__init__()
        filter_channel = [128, 256, 512, 1024, 1024]
        self.inc = Inconv(n_channels, 64)
        self.down = nn.ModuleList([])
        for i in range(4):
            self.down.append(Down(filter_channel[i], filter_channel[i + 1]))

        self.up = nn.ModuleList([Up(2048, 512)])
        for i in range(3):
            self.up.append(
                Up(filter_channel[-(i + 2)], int(filter_channel[-(i + 2)] / 4))
            )
        self.conv1 = DoubleConv(64, 32)
        self.conv2 = DoubleConv(64, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc = [0] * 5
        t_dec = [0] * 4

        t = self.inc(x_inp1)
        tn = self.inc(x_inp2)
        t_enc[0] = torch.cat([t, tn], dim=1)

        for i in range(4):
            t_enc[i + 1] = self.down[i](t_enc[i])

        t_dec[0] = self.up[0](t_enc[-1], t_enc[-2])

        for i in range(3):
            t_dec[i + 1] = self.up[i + 1](t_dec[i], t_enc[-(i + 3)])
        pred_t = self.outc_t(self.conv1(t_dec[-1]))
        pred_tn = self.outc_tn(self.conv2(t_dec[-1]))
        return pred_t, pred_tn


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super().__init__()
        filter_channel = [64, 256, 512]
        self.inc = Inconv(n_channels, 64)

        self.down = nn.ModuleList([Down(64, 128)])
        self.down.append(Down(256, 256))
        self.down.append(Down(256, 512))
        self.down.append(Down(512, 512))

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3_t = Up(256, 64)
        self.up3_tn = Up(256, 64)
        self.up4_t = Up(128, 32)
        self.up4_tn = Up(128, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4

        t_enc[0] = self.inc(x_inp1)

        tn_enc[0] = self.inc(x_inp2)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn


class UNetSelfSuper(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super().__init__()
        filter_channel = [64, 256, 512]
        self.inc = Inconv(n_channels, 64)

        self.down = nn.ModuleList([Down(64, 128)])
        self.down.append(Down(256, 256))
        self.down.append(Down(256, 512))
        self.down.append(Down(512, 512))

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3_t = Up(256, 64)
        self.up3_t1 = Upself(128, 128)
        self.up3_tn = Up(256, 64)
        self.up4_t = Up(128, 32)
        self.up4_t1 = Upself(128, 32)
        self.up4_tn = Up(128, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_t1 = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4

        t_enc[0] = self.inc(x_inp1)
        tn_enc[0] = self.inc(x_inp2)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        t1_dec = self.up3_t1(dec)
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec = self.up4_t(t_dec, t_enc[-2])
        t1_dec = self.up3_t1(t1_dec)
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_t1 = self.outc_t1(tn_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn, pred_t1


class UNetOldVer(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super().__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(128, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(192, 64)
        self.outc = Outconv(64, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]

        x1_1 = self.inc(x_inp1)
        x1_2 = self.inc(x_inp2)
        x1 = torch.cat([x1_1, x1_2], dim=1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, x1


class UNetOldVer2(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super(UNet2, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(1024, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 128)
        self.outc = Outconv(128, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]

        x1_1 = self.inc(x_inp1)
        x1_2 = self.inc(x_inp2)
        x1 = torch.cat([x1_1, x1_2], dim=1)
        x2_1 = self.down1(x1_1)
        x2_2 = self.down1(x1_2)
        x2 = torch.cat([x2_1, x2_2], dim=1)
        x3_1 = self.down2(x2_1)
        x3_2 = self.down2(x2_2)
        x3 = torch.cat([x3_1, x3_2], dim=1)
        x4_1 = self.down3(x3_1)
        x4_2 = self.down3(x3_2)
        x4 = torch.cat([x4_1, x4_2], dim=1)
        x5_1 = self.down4(x4_1)
        x5_2 = self.down4(x4_2)
        x5 = torch.cat([x5_1, x5_2], dim=1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, x1


class UNet_2d(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super().__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(64, n_classes, sig=sig)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == "__main__":
    import torch

    x = torch.rand((10, 2, 256, 256))
    net = UNet(n_channels=1, n_classes=1)
    net(x)
