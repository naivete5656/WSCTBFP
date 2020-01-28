import torch.nn as nn
import torch
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    def __init__(self, alfa=0.25):
        super().__init__()
        self.alfa = alfa

    def forward(self, x, y):
        diff = y - x
        return (
            torch.max((self.alfa - 1) * diff, self.alfa * diff)
        ).sum() / x.data.nelement()


class IgnoreMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x[(y != 255) & (y != -255)]
        y = y[(y != 255) & (y != -255)]
        loss = F.mse_loss(x, y)
        if loss.data > 20:
            print(1)
        return F.mse_loss(x, y)


class RMSE_Q_NormLoss(nn.Module):
    def __init__(self, q=0.8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.q = q

    def forward(self, x, y):
        x_norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt()
        x_norm = x_norm[(y[:, :1] != 255) & (y[:, :1] != -255)]
        y_norm = y_norm[(y[:, :1] != 255) & (y[:, :1] != -255)]
        x = x[(y != 255) & (y != -255)]
        y = y[(y != 255) & (y != -255)]
        dis = y_norm - x_norm
        dis_q = torch.max((self.q - 1) * dis, self.q * dis)
        dis_q_mse = torch.mean((dis_q) ** 2)
        if self.mse(x, y) + dis_q_mse > 20:
            print(1)
        return self.mse(x, y) + dis_q_mse


class IgnoreRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        x = x[y != 255]
        y = y[y != 255]
        return torch.sqrt(self.mse(x, y))
