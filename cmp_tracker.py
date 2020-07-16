import torch
import numpy as np
import cv2
from networks import UNet_2d
import matplotlib.pyplot as plt
from utils import (
    chw_to_hwc,
    getImageTable,
    getSyntheticImage,
    get3chImage,
    getVIDEO,
)
import os
from scipy.ndimage.filters import gaussian_filter
import colorsys
import random
import glob
import matplotlib
from pathlib import Path

matplotlib.use("tkagg")


class CMP_TRACKER:
    """Cell Motion & Position map TRACKER
    Attributes:
        IMAGE_NAMES (list):
    """

    def __init__(self, image_names, unet_path, save_dir, mag_th, itv):
        self.IMAGE_NAMES = image_names
        self.UNET_PATH = unet_path
        self.MODEL = self.getUNet()
        self.IMAGE_DIR = save_dir
        self.image_size = (cv2.imread(self.IMAGE_NAMES[0], -1)).shape
        self.TRACK_DIR = os.path.join(save_dir, f"track{itv}")
        os.makedirs(self.TRACK_DIR, exist_ok=True)
        self.LOG_DIR = os.path.join(save_dir, "log")
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.GIF_DIR = os.path.join(save_dir, "gif")
        os.makedirs(self.GIF_DIR, exist_ok=True)
        self.MAG_TH = mag_th
        self.MAG_TH2 = mag_th * 1.5
        self.MAG_MAX = 0.6
        self.NUM_COLORS = 100
        self.APPLIED_ITV = itv
        self.id_color = np.array(
            [
                colorsys.hsv_to_rgb(h, 0.8, 0.8)
                for h in np.linspace(0, 1, self.NUM_COLORS)
            ]
        )
        random.seed(0)
        random.shuffle(self.id_color)

        self.writeDetail()

    def writeDetail(self):
        with open(os.path.join(self.LOG_DIR, "info.txt"), mode="a") as f:
            f.write("MODEL: {}\n".format(self.MODEL.__class__.__name__))
            f.write("MODEL path: {}\n".format(self.UNET_PATH))
            f.write("image directory: {}\n".format(os.path.split(self.IMAGE_DIR[0])[0]))
            f.write("applied frame interval: {}\n".format(self.APPLIED_ITV))
            f.write("magnitude threshold: {}\n".format(self.MAG_TH))

    def getUNet(self):
        model = UNet_2d(n_channels=2, n_classes=3, sig=False)
        model.cuda()
        state_dict = torch.load(self.UNET_PATH, map_location="cpu")
        #
        model.load_state_dict(state_dict)
        # model = torch.nn.DataParallel(model)
        model.eval()
        return model

    def inferenceCMP(self, names):
        imgs = []
        for name in names:
            img = cv2.imread(name, -1)
            # img = np.load(name)
            img = img / 4096
            # img = img / 13132

            # img = img[:512, :512]
            # img = (img - img.min()) / (img.max() - img.min())

            imgs.append((img).astype("float32")[None, :, :])
        img = np.concatenate(imgs, axis=0)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        output = self.MODEL(img)
        acm = output[0].cpu().detach().numpy()
        return chw_to_hwc(acm)
        # img_name = Path(names[0])
        # cmp = np.load(
        #     f"/home/kazuya/main/correlation_test/images/Elmer_phase/CMP_gt/1/{img_name.stem}.npy"
        # )[:512, :512].astype(np.float32)
        # return cmp

    def getPeaks_getIndicatedPoints(self, acm):
        mag = np.sqrt(np.sum(np.square(acm), axis=-1))
        # mag[mag.max() * 0.2 > mag] = 0
        norm = gaussian_filter(mag, sigma=5)

        mag[mag > self.MAG_MAX] = self.MAG_MAX
        map_left = np.zeros(norm.shape)
        map_right = np.zeros(norm.shape)
        map_top = np.zeros(norm.shape)
        map_bottom = np.zeros(norm.shape)
        map_left_top = np.zeros(norm.shape)
        map_right_top = np.zeros(norm.shape)
        map_left_bottom = np.zeros(norm.shape)
        map_right_bottom = np.zeros(norm.shape)
        map_left[1:, :] = norm[:-1, :]
        map_right[:-1, :] = norm[1:, :]
        map_top[:, 1:] = norm[:, :-1]
        map_bottom[:, :-1] = norm[:, 1:]
        map_left_top[1:, 1:] = norm[:-1, :-1]
        map_right_top[:-1, 1:] = norm[1:, :-1]
        map_left_bottom[1:, :-1] = norm[:-1, 1:]
        map_right_bottom[:-1, :-1] = norm[1:, 1:]
        peaks_binary = np.logical_and.reduce(
            (
                norm >= map_left,
                norm >= map_right,
                norm >= map_top,
                norm >= map_bottom,
                norm >= map_left_top,
                norm >= map_left_bottom,
                norm >= map_right_top,
                norm >= map_right_bottom,
                norm > self.MAG_TH,
            )
        )
        _, _, _, center = cv2.connectedComponentsWithStats(
            (peaks_binary * 1).astype("uint8")
        )
        center = center[1:]
        result = []
        for center_cell in center.astype("int"):
            vec = acm[center_cell[1], center_cell[0]]
            mag_value = mag[center_cell[1], center_cell[0]]
            vec = vec / np.linalg.norm(vec)
            # print(vec)
            x = 0 if vec[1] == 0 else 5 * (vec[1] / vec[2])
            y = 0 if vec[0] == 0 else 5 * (vec[0] / vec[2])
            x = int(x)
            y = int(y)
            result.append(
                [
                    center_cell[0],
                    center_cell[1],
                    center_cell[0] + x,
                    center_cell[1] + y,
                    mag_value,
                ]
            )
        return np.array(result)

    def searchPeak(self, mag, x, y):
        max_idx = -1
        try:
            first_mag_value = mag[int(y), int(x)]
            if first_mag_value < 0.1:
                return int(x), int(y), 0, 0
        except IndexError:
            return int(x), int(y), 0, 0
        while max_idx != 2:
            points = [[y + 1, x], [y - 1, x], [y, x], [y, x - 1], [y, x + 1]]
            mags = []
            for p in points:
                try:
                    mags.append(mag[int(p[0]), int(p[1])])
                except IndexError:
                    mags.append(0)
            max_idx = np.argmax(mags)
            y = points[max_idx][0]
            x = points[max_idx][1]
            if mags[max_idx] == mags[2]:
                break
        x, y = self.adjastPosition(x, y)
        return x, y, first_mag_value, mags[max_idx]

    def whoareU(self, log):
        daughters = np.unique(log[:, 1])
        parents = np.unique(log[:, 10])
        out = np.setdiff1d(parents, daughters)
        if len(out) > 1:
            print(out)
        return out

    def associateCells(self, frame, pos, pre_pos, pre_mag, new_id):
        ass_log = []
        ass_flag = np.zeros(len(pos))
        fin_flag = np.ones(len(pre_pos))

        # association part -------------------------------------------------------
        # process each position of current frame ---------------------------------
        for i, focus_pos in enumerate(pos):
            # x, y is estimated point moved to the peak
            x, y, fmag_value, mag_value = self.searchPeak(
                pre_mag, focus_pos[2], focus_pos[3]
            )
            if mag_value > self.MAG_TH:
                distance_list = [
                    np.sqrt((pospos[2] - x) ** 2 + (pospos[3] - y) ** 2)
                    for pospos in pre_pos
                ]
                if min(distance_list) < 10:
                    add_ass_log = []
                    min_idx = np.argmin(distance_list)
                    add_ass_log.extend([frame, pre_pos[min_idx][1]])
                    add_ass_log.extend(focus_pos)
                    add_ass_log.extend(
                        [
                            fmag_value,
                            mag_value,
                            min(distance_list),
                            pre_pos[min_idx][10],
                        ]
                    )
                    ass_log.append(add_ass_log)
                    fin_flag[min_idx] = 0
                    ass_flag[i] = 1
        print(f"associated cell: {len(ass_log)}")

        # appearance part --------------------------------------------------------
        new_pos = np.delete(pos, np.where(ass_flag != 0)[0], axis=0)
        app_log = []
        for focus_new_pos in new_pos:
            add_new_log = []
            add_new_log.extend([frame, new_id])
            add_new_log.extend(focus_new_pos)
            add_new_log.extend([0, 0, 0, 0])
            ass_log.append(add_new_log)
            app_log.append(add_new_log)
            new_id += 1
        ass_log = np.array(ass_log)
        print(f"appeared cell: {len(app_log)}")

        # division part ----------------------------------------------------------
        div_log = np.empty((0, 11))
        ids, count = np.unique(ass_log[:, 1], axis=0, return_counts=True)
        two_or_more_ids = ids[count > 1]
        for tm_id in two_or_more_ids:
            if tm_id == 364:
                print("stb")
            daughters = ass_log[ass_log[:, 1] == tm_id]
            daughters = daughters[np.argsort(daughters[:, 7])]
            # change not daughters id & add not daughters to app log -------------
            for d in daughters[:-2]:
                d[1] = new_id
                ass_log[:, 1][
                    (ass_log[:, 2] == d[2]) & (ass_log[:, 3] == d[3])
                    ] = new_id
                app_log.append(d.tolist())
                new_id += 1
            # change daughters id & register parents id -------------------------
            for d in daughters[-2:]:
                ass_log[:, 1][
                    (ass_log[:, 2] == d[2]) & (ass_log[:, 3] == d[3])
                    ] = new_id
                ass_log[:, 10][
                    (ass_log[:, 2] == d[2]) & (ass_log[:, 3] == d[3])
                    ] = tm_id
                new_id += 1
            div_log = np.append(div_log, ass_log[ass_log[:, 10] == tm_id], axis=0)
        print(f"division cell: {len(div_log)}")

        return ass_log, fin_flag, div_log, np.array(app_log), new_id

    def saveResultImage(self, frame, name1, name2, mag, current_log):
        pad = 20
        img1 = np.pad(cv2.imread(name1, -1), (pad, pad), "constant")
        img2 = np.pad(cv2.imread(name2, -1), (pad, pad), "constant")
        mag = mag * 255
        mag[mag > 255] = 255
        mag = mag.astype("uint8")
        mag = np.pad(mag, (pad, pad), "constant")
        plot_img2 = get3chImage(img2) / 4095 * 255
        plot_img2 = plot_img2.astype("uint8")
        plot_img1 = get3chImage(img1) / 4095 * 255
        plot_img1 = plot_img1.astype("uint8")
        track_img = plot_img2.copy().astype("uint8")
        for target in current_log:
            color = self.id_color[int(target[1]) % self.NUM_COLORS] * 255
            color = tuple(color)
            cv2.drawMarker(
                plot_img2,
                (int(target[2]) + pad, int(target[3]) + pad),
                color,
                markerType=cv2.MARKER_SQUARE,
                markerSize=3,
                thickness=3,
            )
            cv2.arrowedLine(
                plot_img2,
                (int(target[2]) + pad, int(target[3]) + pad),
                (int(target[4]) + pad, int(target[5]) + pad),
                color=color,
                thickness=2,
            )
            cv2.drawMarker(
                plot_img1,
                (int(target[4]) + pad, int(target[5]) + pad),
                color,
                markerType=cv2.MARKER_SQUARE,
                markerSize=3,
                thickness=3,
            )
            cv2.arrowedLine(
                plot_img1,
                (int(target[2]) + pad, int(target[3]) + pad),
                (int(target[4]) + pad, int(target[5]) + pad),
                color=color,
                thickness=2,
            )
            cv2.drawMarker(
                track_img,
                (int(target[2]) + pad, int(target[3]) + pad),
                color,
                markerType=cv2.MARKER_SQUARE,
                markerSize=3,
                thickness=3,
            )
            cv2.putText(
                track_img,
                f"{int(target[1])}",
                (int(target[2]) + pad, int(target[3]) + pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                thickness=1,
            )

        img1 = (img1 / 4095 * 255).astype("uint8")
        img2 = (img2 / 4095 * 255).astype("uint8")
        mag_red = get3chImage(mag)
        mag_red[:, :, :2] = 0
        mag_red = getSyntheticImage(img2, 0.7, mag_red, 0.5)
        getImageTable(
            [img1, mag, plot_img1, img2, mag_red, plot_img2],
            clm=3,
            save_name=os.path.join(self.IMAGE_DIR, f"{frame:04}.png"),
        )
        cv2.imwrite(os.path.join(self.TRACK_DIR, f"{frame:04}.png"), track_img)

    def adjastPosition(self, x, y):
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > self.image_size[1] - 1:
            x = self.image_size[1] - 1
        if y > self.image_size[0] - 1:
            y = self.image_size[0] - 1
        return int(x), int(y)

    def insertCell_deleteDivision(self, frame, log, not_ass_log, div_log, app_log):
        focus_not_ass_log = not_ass_log[not_ass_log[:, 0] == (frame - self.APPLIED_ITV)]
        if len(focus_not_ass_log) == 0:
            return log, not_ass_log, div_log, app_log, np.empty((0, 11))
        focus_div_log = div_log[div_log[:, 0] > (frame - self.APPLIED_ITV)]
        focus_app_log = app_log[app_log[:, 0] > (frame - self.APPLIED_ITV)]
        connect_log = np.concatenate([focus_app_log, focus_div_log], axis=0)
        if len(connect_log) == 0:
            return log, not_ass_log, div_log, app_log, np.empty((0, 11))
        ids = np.unique(connect_log[:, 1])
        vote = [[0.0] * (len(ids))] * len(focus_not_ass_log)
        vote = np.array(vote)
        ins_log = []
        # acm1 is frame of focus_not_ass_log
        acm1 = self.inferenceCMP(
            [
                self.IMAGE_NAMES[frame - self.APPLIED_ITV - 1],
                self.IMAGE_NAMES[frame - self.APPLIED_ITV],
            ]
        )
        mag1 = np.sqrt(np.sum(np.square(acm1), axis=-1))

        # [frame of focus_not_ass_log + 2] ~ current frame
        for check_frame in range(frame - self.APPLIED_ITV + 2, frame + 1):
            acm2 = self.inferenceCMP(
                [
                    self.IMAGE_NAMES[frame - self.APPLIED_ITV],
                    self.IMAGE_NAMES[check_frame],
                ]
            )
            mag2 = np.sqrt(np.sum(np.square(acm2), axis=-1))

            # vote each connect candidate
            for idx, fid in enumerate(ids):
                target = log[(log[:, 0] == check_frame) & (log[:, 1] == fid)]
                if len(target) > 0:
                    target = target[0]
                    x, y, fmag_value, mag_value = self.searchPeak(
                        mag2, target[2], target[3]
                    )
                    if fmag_value > self.MAG_TH:
                        vec = acm2[y, x]
                        vec = vec / np.linalg.norm(vec)
                        xx = 5 * (vec[1] / vec[2]) if vec[1] != 0 else 0
                        yy = 5 * (vec[0] / vec[2]) if vec[0] != 0 else 0
                        x = int(x + xx)
                        y = int(y + yy)
                        x, y, fmag_value, mag_value = self.searchPeak(mag1, x, y)
                        distance_list = [
                            np.sqrt((nad[2] - x) ** 2 + (nad[3] - y) ** 2)
                            for nad in focus_not_ass_log
                        ]
                        if min(distance_list) < 10:
                            vote[np.argmin(distance_list), idx] = (
                                    vote[np.argmin(distance_list), idx] + fmag_value
                            )

        # process each focus_not_ass_log
        for idx, v in enumerate(vote):
            if v.max() > self.MAG_TH2:
                if (focus_not_ass_log[idx][10] == 364) | (
                        focus_not_ass_log[idx][10] == 106
                ):
                    print("stb")

                # not_ass_logから削除
                not_ass_log = not_ass_log[
                    not_ass_log[:, 1] != focus_not_ass_log[idx][1]
                    ]

                connect_app = connect_log[connect_log[:, 1] == ids[np.argmax(v)]][0]

                if connect_app[1] == 364:
                    print()

                # not_ass_log
                not_ass_log[:, 10][
                    not_ass_log[:, 1] == connect_app[1]
                    ] = focus_not_ass_log[idx][10]
                not_ass_log[:, 1][
                    not_ass_log[:, 1] == connect_app[1]
                    ] = focus_not_ass_log[idx][1]
                not_ass_log[:, 10][
                    not_ass_log[:, 10] == connect_app[1]
                    ] = focus_not_ass_log[idx][1]

                # 分裂だったとき
                if connect_app[10] > 0:
                    try:
                        par_id = connect_app[10]
                        par_par_id = log[log[:, 1] == par_id][0][10]
                        dau = div_log[
                            (div_log[:, 1] != connect_app[1])
                            & (div_log[:, 10] == par_id)
                            ][0]
                        # 親情報を前フレームのIDに，IDを前半フレームに，被親情報を前フレームのIDに
                        log[:, 10][log[:, 1] == connect_app[1]] = focus_not_ass_log[
                            idx
                        ][10]
                        log[:, 1][log[:, 1] == connect_app[1]] = focus_not_ass_log[idx][
                            1
                        ]
                        log[:, 10][log[:, 10] == connect_app[1]] = focus_not_ass_log[
                            idx
                        ][1]
                        log[:, 10][log[:, 10] == par_id] = par_par_id
                        log[:, 1][log[:, 1] == dau[1]] = par_id
                        log[:, 10][log[:, 10] == dau[1]] = par_id
                        not_ass_log[:, 10][not_ass_log[:, 1] == dau[1]] = par_par_id
                        not_ass_log[:, 1][not_ass_log[:, 1] == dau[1]] = par_id
                        not_ass_log[:, 10][not_ass_log[:, 10] == dau[1]] = par_id
                        div_log = div_log[div_log[:, 1] != connect_app[1]]
                        div_log = div_log[div_log[:, 1] != dau[1]]
                        div_log[:, 10][div_log[:, 10] == dau[1]] = par_id
                        div_log[:, 10][
                            div_log[:, 10] == connect_app[1]
                            ] = focus_not_ass_log[idx][1]
                    except:
                        dau = connect_log[connect_log[:, 1] == connect_app[1]][0]
                        change_app = log[
                            (log[:, 0] == dau[0])
                            & (log[:, 2] == dau[2])
                            & (log[:, 3] == dau[3])
                            ][0]
                        log[:, 10][(log[:, 1] == change_app[1])] = focus_not_ass_log[
                            idx
                        ][10]
                        log[:, 1][(log[:, 1] == change_app[1])] = focus_not_ass_log[
                            idx
                        ][1]
                        log[:, 10][log[:, 10] == connect_app[1]] = focus_not_ass_log[
                            idx
                        ][1]
                        div_log[:, 10][
                            div_log[:, 10] == connect_app[1]
                            ] = focus_not_ass_log[idx][1]
                else:
                    # 親情報を前フレームのIDに，IDを前半フレームに，被親情報を前フレームのIDに
                    log[:, 10][log[:, 1] == connect_app[1]] = focus_not_ass_log[idx][10]
                    log[:, 1][log[:, 1] == connect_app[1]] = focus_not_ass_log[idx][1]
                    log[:, 10][log[:, 10] == connect_app[1]] = focus_not_ass_log[idx][1]
                    div_log[:, 10][
                        div_log[:, 10] == connect_app[1]
                        ] = focus_not_ass_log[idx][1]
                    app_log = app_log[app_log[:, 1] != connect_app[1]]
                # insert
                for ins_frame in range(
                        int(focus_not_ass_log[idx][0] + 1), int(connect_app[0])
                ):
                    acm = self.inferenceCMP(
                        [
                            self.IMAGE_NAMES[ins_frame],
                            self.IMAGE_NAMES[int(connect_app[0])],
                        ]
                    )
                    mag = np.sqrt(np.sum(np.square(acm), axis=-1))
                    x, y, fmag_value, mag_value = self.searchPeak(
                        mag, connect_app[2], connect_app[3]
                    )
                    try:
                        vec = acm[y, x]
                    except IndexError:
                        print()
                    vec = vec / np.linalg.norm(vec)
                    xx = 5 * (vec[1] / vec[2]) if vec[1] != 0 else 0
                    yy = 5 * (vec[0] / vec[2]) if vec[0] != 0 else 0
                    x = int(x + xx)
                    y = int(y + yy)
                    x, y = self.adjastPosition(x, y)
                    ins_data = [
                        ins_frame,
                        focus_not_ass_log[idx][1],
                        x,
                        y,
                        0,
                        0,
                        -1,
                        -1,
                        -1,
                        -1,
                        focus_not_ass_log[idx][10],
                    ]
                    ins_log.append(ins_data)
                if len(ins_log) > 0:
                    log = np.append(log, ins_log, axis=0)

        return log, not_ass_log, div_log, app_log, ins_log

    def TRACK(self):
        track_full = []  # output, shape is (,11)
        not_ass_log = np.empty((0, 11))
        pop_log = np.empty((0, 11))
        div_log = np.empty((0, 11))
        ins_log = []

        # frame0 - farme1 ------------------------------------------------------------
        print("-------0 - 1-------")
        pre_cmp = self.inferenceCMP([self.IMAGE_NAMES[0], self.IMAGE_NAMES[1]])
        pre_mag = np.sqrt(np.sum(np.square(pre_cmp), axis=-1))
        pre_pos = self.getPeaks_getIndicatedPoints(pre_cmp)
        for i, each_pos in enumerate(pre_pos):
            # register frame(0)
            add_track = []
            add_track.extend([0, i + 1])  # frame, ID
            add_track.extend(each_pos[2:4])  # x, y
            add_track.extend(
                [0, 0, 0, 0, 0, 0, 0]
            )  # xx, yy, l(t), l(t-1)vec, l(t-1)peak, distance, par-ID
            track_full.append(add_track)
            # register frame(1)
            add_track = []
            add_track.extend([1, i + 1])  # frame, ID
            add_track.extend(each_pos)  # x, y, xx, yy
            add_track.extend(
                [pre_mag[int(each_pos[1]), int(each_pos[0])], 0, 0, 0]
            )  # l(t), l(t-1)v, l(t-1)p, distance, par-ID
            track_full.append(add_track)
        self.saveResultImage(
            1, self.IMAGE_NAMES[0], self.IMAGE_NAMES[1], pre_mag, track_full
        )

        track_full = np.array(track_full)
        pre_pos = track_full[track_full[:, 0] == 1]
        new_id = len(pre_pos) + 1

        # frame2 - -------------------------------------------------------------------
        for frame in range(2, len(self.IMAGE_NAMES)):
            print(f"-------{frame - 1} - {frame}-------")
            # association ------------------------------------------------------------
            acm = self.inferenceCMP(
                [self.IMAGE_NAMES[frame - 1], self.IMAGE_NAMES[frame]]
            )
            mag = np.sqrt(np.sum(np.square(acm), axis=-1))
            pos = self.getPeaks_getIndicatedPoints(acm)
            ass_log, fin_flag, add_div_log, add_pop_log, new_id = self.associateCells(
                frame, pos, pre_pos, pre_mag, new_id
            )
            div_log = np.append(div_log, add_div_log, axis=0)
            if len(pop_log) > 0:
                pop_log = np.append(pop_log, add_pop_log, axis=0)
            not_ass_pos = np.delete(pre_pos, np.where(fin_flag == 0)[0], axis=0)
            print(f"not associated: {len(not_ass_pos)}")
            not_ass_log = np.append(not_ass_log, not_ass_pos, axis=0)
            self.saveResultImage(
                frame,
                self.IMAGE_NAMES[frame - 1],
                self.IMAGE_NAMES[frame],
                mag,
                ass_log,
            )
            pre_mag = mag.copy()
            track_full = np.append(track_full, ass_log, axis=0)

            self.whoareU(track_full)
            if frame == 162:
                print()

            # insert & change id ------------------------------------------------------
            track_full, not_ass_log, div_log, pop_log, add_ins_log = self.insertCell_deleteDivision(
                frame, track_full, not_ass_log, div_log, pop_log
            )

            if len(track_full[track_full[:, 1] == 364]):
                print()

            self.whoareU(track_full)

            pre_pos = track_full[track_full[:, 0] == frame].copy()
            ins_log.extend(add_ins_log)
            # delete ------------------------------------------------------------------
        print("Tracked !!")
        self.saveCTK(track_full, not_ass_log, div_log, ins_log)

    def reformatID(self, ids, log):
        new_ids = np.arange(1, len(ids) + 1)
        new_log = np.empty((0, 11))
        for each_log in log.copy():
            eid = each_log[1]
            pid = each_log[10]
            each_log[1] = new_ids[np.where(ids == eid)[0][0]]
            if pid != 0:
                try:
                    each_log[10] = new_ids[np.where(ids == pid)[0][0]]
                except:
                    print(pid)
            new_log = np.append(new_log, [each_log], axis=0)
        return new_log

    def convertPointList(self, log):
        tmp = log[:, :4].copy()
        converted = tmp.copy()
        converted[:, :2] = tmp[:, -2:]
        converted[:, -2:] = tmp[:, :2]
        return converted

    def saveCTK(self, log, not_ass_log, div_log, ins_log):
        log = log[np.argsort(log[:, 0])]
        ids = np.unique(log[:, 1])

        new_log = self.reformatID(ids, log)
        # new_not_ass_log = self.reformatID(ids, not_ass_log)
        # new_div_log = self.reformatID(ids, div_log)
        # new_ins_log = self.reformatID(ids, ins_log)

        save_dir = os.path.join(self.LOG_DIR, "PyLog")
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, "new_log.txt"), new_log)
        np.savetxt(os.path.join(save_dir, "log.txt"), log)
        np.savetxt(os.path.join(save_dir, "end_trajectory.txt"), not_ass_log)
        np.savetxt(os.path.join(save_dir, "mitosis_event.txt"), div_log)
        np.savetxt(os.path.join(save_dir, "insert_position.txt"), ins_log)

        np.savetxt(
            os.path.join(self.LOG_DIR, "tracking.states"), new_log[:, :4], fmt="%d"
        )
        np.savetxt(
            os.path.join(self.LOG_DIR, "end_trajectory.txt"), not_ass_log, fmt="%d"
        )
        np.savetxt(os.path.join(self.LOG_DIR, "mitosis_event.txt"), div_log, fmt="%d")
        np.savetxt(os.path.join(self.LOG_DIR, "insert_position.txt"), ins_log, fmt="%d")

        tree = np.empty((0, 2))
        parents = np.unique(new_log[:, 10])
        for nid in np.unique(new_log[:, 1]):
            par = new_log[:, 10][new_log[:, 1] == nid][0]
            tree_id = [nid, par]
            tree = np.append(tree, [tree_id], axis=0)
        np.savetxt(os.path.join(self.LOG_DIR, "tracking.tree"), tree, fmt="%d")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    for seq in [15]:
        for mode in ["_reject_new", "_reject_new1", "_reject_new2"]:
            for itv in [1]:
                demo = CMP_TRACKER(
                    image_names=sorted(glob.glob(f"/home/kazuya/main/weakly_tracking/images/sequ{seq}/ori/*.tif"))[
                        :780:itv
                    ],
                    # image_names=sorted(glob.glob(f"/home/kazuya/main/Hayashida/Elmer_phase/*.png"))[
                    #             :780:itv
                    #             ],
                    # unet_path=f"./weights/CMP6_Elmer/temp.pth",
                    unet_path=f"/home/kazuya/main/weight/CMP6_sequ{seq}{mode}/temp.pth",
                    save_dir=f"./output/cmp_track/{mode}/sequ{seq}/result{itv}",
                    mag_th=0.1,
                    itv=1,
                )
                demo.TRACK()
