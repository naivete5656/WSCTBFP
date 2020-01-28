import torch


class Visdom(object):
    def vis_init(self, env="main"):
        if self.need_vis:
            import visdom

            HOSTNAME = "localhost"
            PORT = 8097
            self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env=env)
            vis_title = "CVPR work shop"
            vis_legend = ["MSE Loss", "TVLoss", "PeakLoss"]
            vis_epoch_legend = ["Loss", "Val Loss"]
            self.iter_plot = self.create_vis_plot(
                "Iteration", "Loss", vis_title, vis_legend
            )
            self.epoch_plot = self.create_vis_plot(
                "Epoch", "Loss", vis_title, vis_epoch_legend
            )
            self.ori_view = self.create_vis_show()
            self.ori_view2 = self.create_vis_show()
            self.img_view = self.create_vis_show()
            self.img2_view = self.create_vis_show()
            self.gt_view = self.create_vis_show()
            self.gt_view2 = self.create_vis_show()
            self.img_view_val = self.create_vis_show()
            self.gt_view_val = self.create_vis_show()
            self.pred_view_val = self.create_vis_show()
        else:
            self.vis = None
            self.ori_view = None
            self.img_view = None
            self.img_view2 = None
            self.gt_view = None
            self.img_view_val = None
            self.gt_view_val = None
            self.pred_view_val = None

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
        )

    def update_vis_plot(self, iteration, loss, window1, window2, update_type, val=None):
        try:
            self.vis.line(
                X=torch.ones((1)).cpu() * iteration,
                Y=torch.Tensor(loss).unsqueeze(0).cpu(),
                win=window1,
                update=update_type,
            )
        except TypeError:
            pass
        # initialize epoch plot on first iteration
        if window2 is not None:
            self.vis.line(
                X=torch.ones((1,)).cpu() * iteration,
                Y=torch.Tensor(val).unsqueeze(0).cpu(),
                win=window2,
                update=update_type,
            )

    def create_vis_show(self):
        return self.vis.images(
            torch.ones((self.batch_size, 1, 256, 256)), self.batch_size
        )

    def update_vis_show(self, images, window1):
        self.vis.images(images, self.batch_size, win=window1)
