import torch.nn as nn
import torch


def eval_net(
    net,
    dataset,
    gpu=True,
    loss_flag=True,
    vis=None,
    vis_im=None,
    vis_gt=None,
    vis_pred=None,
    loss=nn.MSELoss(),
):
    criterion = loss
    net.eval()
    losses = 0
    torch.cuda.empty_cache()
    for iteration, data in enumerate(dataset):
        img = data["image"]
        target = data["gt"]

        if gpu:
            img = img.cuda()
            target = target.cuda()

        pred_img = net(img)



        loss1 = criterion(pred_img[0], target[:, :1])
        loss2 = criterion(pred_img[1], target[:, 1:])
        loss = loss1 + loss2

        losses += loss.data
        if vis is not None:
            vis.images(img[:, :1, :, :].cpu(), 1, win=vis_im)
            pred_img = (pred_img[0] - pred_img[0].min()) / (
                pred_img[0].max() - pred_img[0].min()
            )
            vis.images(pred_img[:, :1, :, :].cpu(), 1, win=vis_pred)
            vis.images(target[:, :1, :, :], 1, win=vis_gt)

    return losses / iteration


def eval_net1(
    net,
    dataset,
    gpu=True,
    vis=None,
    vis_im=None,
    vis_gt=None,
    vis_pred=None,
    loss=nn.MSELoss(),
):
    criterion = loss
    net.eval()
    losses = 0
    torch.cuda.empty_cache()
    for iteration, data in enumerate(dataset):
        img = data["image"]
        target = data["gt"]
        if gpu:
            img = img.cuda()
            target = target.cuda()

        pred_img = net(img)

        loss = criterion(pred_img, target)

        losses += loss.data
        if vis is not None:
            vis.images(img[:, :1, :, :].cpu(), 1, win=vis_im)
            pred_img = pred_img.clamp(min=0, max=1)
            vis.images(pred_img.cpu(), 1, win=vis_pred)
            vis.images(target.cpu(), 1, win=vis_gt)

    return losses / iteration


def eval_celltracking(net, dataset, gpu, vis, vis_im, vis_pred, vis_gt, criterion):
    net.eval()
    torch.cuda.empty_cache()
    for iteration, data in enumerate(dataset):
        img = data["image"]
        target = data["gt"]
        if gpu:
            img = img.cuda()
            target = target.cuda()

        pred_img = net(img)

        vis.images(img[:, :1, :, :].cpu(), 1, win=vis_im)
        pred_img = pred_img[0].clamp(min=0, max=1)
        vis.images(pred_img[:, :1, :, :].cpu(), 1, win=vis_pred)
        vis.images(target[:, :1, :, :], 1, win=vis_gt)
