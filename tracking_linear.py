from skimage.feature import peak_local_max
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pulp
from utils import gaus_filter, optimum
import shutil


def gen_gt_gaussian(plots, shape):
    gauses = []
    for label, plot in enumerate(plots):
        img = np.zeros((shape[0], shape[1]))
        img = cv2.circle(img, (int(plot[0]), int(plot[1])), 12, label + 1, thickness=-1)
        gauses.append(img)
    return gauses


def make_file(save_path):
    # path def
    save_path.joinpath(f"{frame:05d}/tp").mkdir(
        parents=True, exist_ok=True
    )
    save_path.joinpath(f"{frame:05d}/reject").mkdir(
        parents=True, exist_ok=True
    )
    save_path.joinpath(f"{frame:05d}/fn").mkdir(
        parents=True, exist_ok=True
    )
    save_path.joinpath(f"{frame:05d}/fp").mkdir(
        parents=True, exist_ok=True
    )


def load_image(frame, ori_img_path):
    # load original image
    img = cv2.imread(str(ori_img_path.joinpath(f"{frame:04d}_1.png")))
    img_tn = cv2.imread(str(ori_img_path.joinpath(f"{frame:04d}_2.png")))
    pred_im = cv2.imread(
        str(ori_img_path.parent.joinpath(f"pred/{frame:04d}_1.png")), 0
    )
    pred_tn_im = cv2.imread(
        str(ori_img_path.parent.joinpath(f"pred/{frame:04d}_2.png")), 0
    )
    return img, img_tn, pred_im, pred_tn_im


def load_image_each_cell(frame, cell_id, assoc_pred_path_root):
    asso_img = cv2.imread(
        str(assoc_pred_path_root.joinpath(f"pred/{frame:04d}_{cell_id + 1:04d}_1.png")), 0
    )
    asso_img_tn = cv2.imread(
        str(assoc_pred_path_root.joinpath(f"pred/{frame:04d}_{cell_id + 1:04d}_2.png")), 0
    )
    inp_img = cv2.imread(
        str(assoc_pred_path_root.joinpath(f"ori/{frame:04d}_{cell_id + 1:04d}_1.png")), 0
    )
    inp_img_tn = cv2.imread(
        str(assoc_pred_path_root.joinpath(f"ori/{frame:04d}_{cell_id + 1:04d}_2.png")), 0
    )

    return asso_img, asso_img_tn, inp_img, inp_img_tn


def visuarize_img(patch_window, img1, img2):
    img = np.vstack(
        [
            img1[
            patch_window[0]: patch_window[0] + 100,
            patch_window[1]: patch_window[1] + 100,
            ],
            img2[
            patch_window[0]: patch_window[0] + 100,
            patch_window[1]: patch_window[1] + 100,
            ],
        ]
    )
    return img


def output_result(pred, pred_tn, cell_id_t, cell_id_tn, frame, assoc_pred_path_root, save_mode, gt, region):
    tn_pos = pred_tn[cell_id_tn]
    t_pos = pred[cell_id_t]
    asso_img, asso_img_tn, inp_img, inp_img_tn = load_image_each_cell(frame, cell_id_tn, assoc_pred_path_root)
    patch_window = (tn_pos - 50).clip(0).astype(int)[:2][::-1]
    patch_window[0] = min(patch_window[0], inp_img.shape[0])
    patch_window[1] = min(patch_window[1], inp_img.shape[1])
    img_vis_tn = img_tn.copy()

    img_vis = img.copy()

    pred_im_vis_tn = pred_tn_im.copy()
    pred_im_vis_tn = cv2.cvtColor(pred_im_vis_tn, cv2.COLOR_GRAY2BGR)
    pred_im_vis = cv2.cvtColor(pred_im, cv2.COLOR_GRAY2BGR)
    # if save_mode == "reject":
    #     cv2.circle(img_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (255, 0, 0), 1)
    #     cv2.circle(pred_im_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (255, 0, 0), 1)
    # elif save_mode == "fp":
    #     cv2.circle(img_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 255, 0), 1)
    #     cv2.circle(pred_im_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 255, 0), 1)
    # else:
    #     cv2.circle(img_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 0, 255), 1)
    #     cv2.circle(pred_im_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 0, 255), 1)

    # if save_mode != "reject":
    #     cv2.circle(img_vis, (int(t_pos[0]), int(t_pos[1])), 5, (0, 255, 0), 1)
    #     cv2.circle(pred_im_vis, (int(t_pos[0]), int(t_pos[1])), 5, (0, 255, 0), 1)

    # if gt is not None:
    #     cv2.circle(img_vis, (int(gt[2]), int(gt[3])), 5, (0, 0, 255), 1)
    #     cv2.circle(pred_im_vis, (int(gt[2]), int(gt[3])), 5, (0, 0, 255), 1)


    img_vis = visuarize_img(patch_window, img_vis, img_vis_tn)

    temp = np.zeros_like(pred_im_vis_tn)
    temp[region == cell_id_tn + 1] = 255
    pred_im_vis_tn[temp[:, :, 0] != 0, :2] = 0
    # plt.imshow(pred_im_vis_tn), plt.show()
    pred_im_vis = visuarize_img(patch_window, pred_im_vis, pred_im_vis_tn)

    asso_img_vis = visuarize_img(patch_window, asso_img, asso_img_tn)
    asso_img_vis = cv2.cvtColor(asso_img_vis, cv2.COLOR_GRAY2BGR)
    asso_img_vis[:100, :, 1] = asso_img_vis[:100, :, 1] * 0.85
    asso_img_vis[:100, :, 2] = asso_img_vis[:100, :, 2] * 0.4
    inp_img_vis = visuarize_img(patch_window, inp_img, inp_img_tn)
    inp_img_vis = cv2.cvtColor(inp_img_vis, cv2.COLOR_GRAY2BGR)
    vis_img = np.hstack([img_vis, pred_im_vis])
    lik_img_vis = np.hstack([inp_img_vis, asso_img_vis])
    vis_img = np.hstack([vis_img, lik_img_vis])

    cv2.imwrite(str(save_path.joinpath(f"{frame:05d}/{save_mode}/{cell_id_tn}.png")), vis_img)


def local_maxima(img, threshold, dist):
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0)
    return data


def check_associate(gts, pred_tn, pred, frame, assoc_pred_path_root, img, img_tn, pred_im, pred_tn_im):
    tp = 0
    fp = 0
    reject = 0

    # set gt
    gt = gts[gts[:, 0] == (frame)]
    gt_tn = gts[gts[:, 0] == (frame + time_late)]

    gt2t = optimum(gt[:, 2:4], pred, 10)
    gtn2tn = optimum(gt_tn[:, 2:4], pred_tn, 10)

    # gt
    t_ids = set(gt[gt2t[:, 0].astype(np.int)][:, 1])
    tn_ids = set(gt_tn[gtn2tn[:, 0].astype(np.int)][:, 1])
    tn_par_ids = set(gt_tn[gtn2tn[:, 0].astype(np.int)][:, 4])
    true_ids = (t_ids & tn_ids) | (t_ids & tn_par_ids)

    gauses = []
    try:
        for peak in pred_tn:
            temp = np.zeros((512,512))
            temp[int(peak[1]), int(peak[0])] = 255
            gauses.append(gaus_filter(temp, 401, 12))
        region = np.argmax(gauses, axis=0) + 1
        likely_map = np.max(gauses, axis=0)
        region[likely_map < 0.05] = 0
        #
        # r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
    except ValueError:
        region = np.zeros((512, 512), dtype=np.uint8)
        likely_map = np.zeros((512, 512))
    # visualize
    for cell_id_tn, pre in enumerate(pred_tn):
        if pre[3] == 1:
            cell_id_t = int(pre[2])

            # get pred peak's gt id
            gt2t_assoc_index = np.where(gt2t[:, 1] == cell_id_t)[0]
            gtn2tn_assoc_index = np.where(gtn2tn[:, 1] == cell_id_tn)[0]
            if (gt2t_assoc_index.shape[0] != 0) and (
                    gtn2tn_assoc_index.shape[0] != 0
            ):
                gt_index = gt2t[gt2t_assoc_index][0, 0]
                gt_cell_id = gt[int(gt_index)][1]

                gtn_index = gtn2tn[gtn2tn_assoc_index][0, 0]
                gtn_cell_id = gt_tn[int(gtn_index)][1]

                parent_id = gt_tn[int(gtn_index)][4]
                mitosis_flag = (parent_id == gtn_cell_id)

                if (gt_cell_id == gtn_cell_id) or (mitosis_flag):
                    gt_pos = gt[int(gt_index)]
                    save_mode = "tp"
                    tp += 1
                    true_ids.remove(gt_cell_id)
                else:
                    gt_pos = gt_tn[gt_tn[:, 1] == gtn_cell_id]
                    if gt_pos.shape[0] == 0:
                        gt_pos = None
                    else:
                        gt_pos = gt_pos[0]
                    save_mode = "fp"
                    fp += 1

            else:
                if gtn2tn_assoc_index.shape[0] != 0:
                    gtn_index = gtn2tn[gtn2tn_assoc_index][0, 0]
                    gt_pos = gt_tn[[int(gtn_index)]]
                    if gt_pos.shape[0] == 0:
                        gt_pos = None
                    else:
                        gt_pos = gt_pos[0]
                else:
                    gt_pos = None
                # 端にある細胞への処理
                plot_list = np.concatenate(
                    (np.expand_dims(pred[cell_id_t][:2], axis=0), np.expand_dims(pred_tn[cell_id_tn][:2], axis=0)),
                    axis=0)
                flag = np.any((10 > plot_list) | (plot_list > img.shape[0] - 10))
                if flag:
                    save_mode = "no"
                    reject += 1
                else:
                    save_mode = "fp"
                    fp += 1
        else:
            cell_id_t = -1
            gt_pos = None
            save_mode = "reject"
            reject += 1

        output_result(pred, pred_tn, cell_id_t, cell_id_tn, frame, assoc_pred_path_root, save_mode, gt_pos, region)

    fn_ind = np.isin(gt[:, 1], list(true_ids))
    fn_pos = gt[fn_ind]
    plt.imshow(img), plt.plot(fn_pos[:, 2], fn_pos[:, 3], "rx"), plt.savefig(
        str(save_path.joinpath(f"{frame:05d}/fn/fn_cells.png"))), plt.close()

    return tp, reject, fp, fn_pos.shape[0]


class LinearAssociation(object):
    def __init__(self, pred_tn, pred, dist_thresh=20):
        self.dist_thresh = dist_thresh
        self.c_size = pred_tn.shape[0] + pred.shape[0]
        self.pred_tn_size = pred_tn.shape[0]
        self.pred_size = pred.shape[0]
        self.c = np.zeros((0, pred_tn.shape[0] + pred.shape[0]))
        self.d = []
        self.associate_id = []

    def add_cand(
            self, asso_img, cell_id_tn, gauses, pred_im, cand_idx
    ):
        # get peak from assoc
        mask = pred_im.copy()
        mask[pred_im > 50] = 1
        mask[pred_im <= 50] = 0
        asso_img = mask * asso_img

        for idx in cand_idx:
            gau = gauses[idx]
            aso = asso_img[gau > 0].copy()
            gaus = pred_im[gau > 0].copy()
            aso = aso / 255
            gaus = gaus / 255
            thresh = np.sum(np.square(gaus))
            residual = np.sum(np.square(gaus - aso)) / thresh

            self.associate_id.append([cell_id_tn, idx])
            self.d.append(1 - residual)
            c1 = np.zeros((1, self.c_size))
            c1[0, cell_id_tn] = 1
            c1[0, self.pred_tn_size + idx] = 1
            self.c = np.append(self.c, c1, axis=0)

    def associate_predict_result(
            self, pred_tn, pred, frame, assoc_pred_path, pred_im
    ):
        dist_list = []
        pred_tn_plot = pred_tn[:, :2].copy()

        gauses = gen_gt_gaussian(pred, pred_im.shape)

        for cell_id_tn, pre in enumerate(pred_tn_plot):
            asso_img = cv2.imread(str(assoc_pred_path.joinpath(f"pred/{frame:04d}_{cell_id_tn + 1:04d}_1.png")), 0)
            dist = np.sqrt(np.sum(np.square(pred[:, :2] - pre), axis=1))
            cand_idx = np.where(dist < self.dist_thresh)[0]
            # if cand_idx
            # plt.imshow(asso_img), plt.plot(pre[0], pre[1], "r2"), plt.show()
            #
            # plt.imshow(img), plt.plot(pre[0], pre[1], "r2"), plt.plot(pred[cand_idx][:,0], pred[cand_idx][:,1], 'g3'),plt.plot,plt.show()

            self.add_cand(
                asso_img, cell_id_tn, gauses, pred_im, cand_idx
            )
        self.d = np.array(self.d)

        prob = pulp.LpProblem("review", pulp.LpMaximize)

        index = list(range(self.d.shape[0]))  # type:

        x_vars = pulp.LpVariable.dict("x", index, lowBound=0, upBound=1, cat="Integer")

        # 最大化する値
        prob += sum([self.d[i] * x_vars[i] for i in range(self.d.shape[0])])

        # 条件の定義
        for j in range(self.c.shape[1]):
            prob += sum([self.c[i, j] * x_vars[i] for i in range(self.d.shape[0])]) <= 1

        # 最適化問題を解く
        prob.solve()

        # 最適化結果を抽出　=0 のデータを削除
        x_list = np.zeros(self.d.shape[0], dtype=int)
        for jj in range(self.d.shape[0]):
            x_list[jj] = int(x_vars[jj].value())

        self.associate_id = np.array(self.associate_id)

        for idx, associate_id in enumerate(self.associate_id):
            if x_list[idx]:
                if self.d[idx] > 0.5:
                    pred_tn[associate_id[0]][2] = associate_id[1]
                    pred_tn[associate_id[0]][3] = 1
                    pred[associate_id[1]][2] = associate_id[0]
                    pred[associate_id[1]][3] = 1
                else:
                    pred_tn[associate_id[0]][3] = 2
                    pred[associate_id[1]][3] = 2

        return pred_tn, pred


dis_th = {1: 20, 5: 40, 9: 60}

if __name__ == "__main__":
    mode = ""
    for time_late in [9]:
        for seq in [6]:
            ori_path_root = Path(f"./output/detection/C2C12_9_{time_late}/sequ{seq}")
            assoc_pred_path_root = Path(f"./output/detection/C2C12_9_{time_late}_mask/sequ{seq}")
            guided_path_root = Path(f"./output/guid_out/C2C12_9_{time_late}/sequ{seq}")
            gts = np.loadtxt(f"/home/kazuya/main/correlation_test/images/tracking_annotation/gt_seq_{seq}.txt",
                             delimiter=",",
                             skiprows=1)
            # save_path_root = Path(f"./output/association{mode}/C2C12_9_{time_late}/sequ{seq}")
            save_path_root = Path(f"./output/for_vis/C2C12_9_{time_late}/sequ{seq}")
            if save_path_root.is_dir():
                shutil.rmtree(save_path_root)

            tps = 0
            rejects = 0
            fps = 0
            fns = 0

            ori_path = ori_path_root.joinpath(f"ori")
            assoc_pred_path = assoc_pred_path_root
            save_path = save_path_root

            tp = 0
            fp = 0
            reject = 0
            for frame in range(0, 200 - time_late):
                guided_path = guided_path_root.joinpath(f"{frame:05d}")
                make_file(save_path)

                img, img_tn, pred_im, pred_tn_im = load_image(frame, ori_path)

                pred_tn = np.loadtxt(
                    str(guided_path.joinpath("peaks.txt")), skiprows=2, delimiter=",", ndmin=2
                )
                pred = local_maxima(pred_im, 80, 2)

                # [x, y, cell_id, state]
                pred_tn = pred_tn[:, 1:]
                pred_tn = np.insert(pred_tn, 2, [[-1], [0]], axis=1)
                # [x, y, cell_id, state]
                pred = np.insert(pred, 2, [[-1], [0]], axis=1)

                associater = LinearAssociation(pred_tn, pred, dist_thresh=dis_th[time_late])

                pred_tn, pred = associater.associate_predict_result(
                    pred_tn, pred, frame, assoc_pred_path, pred_im
                )

                np.savetxt(
                    str(save_path.joinpath(f"{frame:05d}/traject_1.txt")),
                    pred,
                    fmt="%03d",
                    header="[x, y, state, cell_id]",
                    delimiter=",",
                )

                np.savetxt(
                    str(save_path.joinpath(f"{frame:05d}/traject_2.txt")),
                    pred_tn,
                    fmt="%03d",
                    header="[x, y, state, cell_id]",
                    delimiter=",",
                )

                tp, reject, fp, fn = check_associate(gts, pred_tn, pred, frame, assoc_pred_path, img, img_tn,
                                                     pred_im, pred_tn_im)

                print(tp, reject, fp, fn)

                tps += tp
                rejects += reject
                fps += fp
                fns += fn
                print(f"{frame},precision = {tps / (tps + fps)}", f"recall={tps / (tps + fns)}")
                print(f"f1={2 * tps / (2 * tps + fps + fns)}")  # 除いた細胞の割合
                print(f"reject={(tps + fps + fns) / ((tps + fps + fns) + rejects)}")  # 除いた細胞の割合
            print(f"final result,precision = {tps / (tps + fps)}", f"recall={tps / (tps + fns)}")
            print(f"f1={2 * tps / (2 * tps + fps + fns)}")  # 除いた細胞の割合
            print(f"reject={(tps + fps + fns) / ((tps + fps + fns) + rejects)}")  # 除いた細胞の割合
            with save_path.joinpath("result.txt").open("w") as f:
                f.write(
                    f"{tps / (tps + fps)}, {tps / (tps + fns)}, {2 * tps / (2 * tps + fps + fns)}, {(tps + fps + fns) / ((tps + fps + fns) + rejects)}\n"
                )
                f.write(f"tp, fp, fn, reject\n")
                f.write(f"{tps}, {fps}, {fns}, {rejects}\n")
