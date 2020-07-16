import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import peak_local_max
import math
import pulp
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import cv2
from utils import gaus_filter, optimum


def gen_gt_gaussian(plots, shape):
    gauses = []
    for label, plot in enumerate(plots):
        img = np.zeros((shape[0], shape[1]))
        img = cv2.circle(img, (int(plot[0]), int(plot[1])), 18, label + 1, thickness=-1)
        gauses.append(img)
    return gauses


def load_image(frame, cell_id_t, root_path, inverse):
    asso_img1 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_1.png")), 0
    )
    asso_img2 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_2.png")), 0
    )
    if inverse:
        return asso_img1
    else:
        return asso_img2


def visuarize_img(pre, frame, cell_id_t, root_path, img1, img2, pred1, pred2):

    inp_img1 = cv2.imread(
        str(root_path.joinpath(f"ori/{frame:04d}_{cell_id_t + 1:04d}_1.png")), 0
    )
    inp_img2 = cv2.imread(
        str(root_path.joinpath(f"ori/{frame:04d}_{cell_id_t + 1:04d}_2.png")), 0
    )

    asso_img1 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_1.png")), 0
    )
    asso_img2 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_2.png")), 0
    )
    patch_window = (pre[1:3] - 50).clip(0, 512 - 100).astype(int)[::-1]
    x = img2.copy()
    # x = cv2.drawMarker(
    #     x,
    #     (int(pre[1]), int(pre[2])),
    #     0,
    #     markerType=cv2.MARKER_CROSS,
    #     markerSize=10,
    #     thickness=1,
    #     line_type=cv2.LINE_8,
    # )
    img = np.hstack(
        [
            img1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            x[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    pred = np.hstack(
        [
            pred1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            pred2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )

    inp_img = np.hstack(
        [
            inp_img1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            inp_img2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )
    asso_img = np.hstack(
        [
            asso_img1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            asso_img2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )
    visual_img = np.vstack([img, inp_img])
    lik_img = np.vstack([pred, asso_img])
    visual_img = np.hstack([visual_img, lik_img])
    return visual_img


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


def decide_cell_id_tn(cell_id_t, cell_id_tn, pred1, pred2, dists, dist_list, n=1):
    if pred2[cell_id_tn][2] == 0:
        plt.imshow(img2), plt.plot(
            pred1[cell_id_t][1], pred1[cell_id_t][2], "rx"
        ), plt.plot(pred2[cell_id_tn][0], pred2[cell_id_tn][1], "b3"), plt.savefig(
            "test.png"
        ), plt.close()
        pred2[cell_id_tn][2] = 1
        pred2[cell_id_tn][3] = cell_id_t
        pred1[cell_id_t][3] = cell_id_tn
        pred1[cell_id_t][4] = 1
        # plt.imshow(img1), plt.plot(
        #     pred1[cell_id_t][1], pred1[cell_id_t][2], "rx"
        # ), plt.plot(pred2[cell_id_tn][0], pred2[cell_id_tn][1], "b3"), plt.show()
        return pred1, pred2
    else:
        cell_id_conf = int(pred2[cell_id_tn][3])
        if np.min(dists[0]) < np.min(dist_list[cell_id_conf][0]):
            plt.imshow(img2), plt.plot(
                pred1[cell_id_t][1], pred1[cell_id_t][2], "rx"
            ), plt.plot(pred2[cell_id_tn][0], pred2[cell_id_tn][1], "b3"), plt.savefig(
                "test.png"
            ), plt.close()
            pred1[cell_id_t][3] = cell_id_tn
            pred1[cell_id_t][4] = 1
            pred2[cell_id_tn][3] = cell_id_t
            dists = dist_list[cell_id_conf]
            cell_id_t = cell_id_conf
        try:
            dist_idx = np.where(dists[:, 0] == np.sort(dists[:, 0])[1])[0][0]
            cell_id_tn = int(dists[dist_idx][1])
        except IndexError:
            dist_idx = 0
            dists = [[255]]

        if dists[dist_idx][0] > 0.8:
            pred1[cell_id_t][4] = 3
            return pred1, pred2
        if n < 5:
            return decide_cell_id_tn(
                cell_id_t, cell_id_tn, pred1, pred2, dists, dist_list, n + 1
            )
        else:
            pred1[cell_id_t][4] = 3
            return pred1, pred2


def calculate_dist(asso_img, gauses, gt, cand_idx):
    dist = []
    for idx in cand_idx:
        gau = gauses[idx]
        aso = asso_img[gau > 0].copy()
        gaus = gt[gau > 0].copy()
        aso = aso / 255
        gaus = gaus / 255
        thresh = np.sum(np.square(gaus)) / gaus.size
        residual = np.sum(np.square(gaus - aso)) / gaus.size / thresh
        if 1 > residual:
            dist.append([residual, idx])
        else:
            dist.append([255, idx])
    return np.array(dist)


def update_state_by_iou(
    asso_img, cell_id_t, pred1, pred2, pre, dist_list, gauses, pred_im, cand_idx
):
    # get peak from assoc
    mask = pred2_im.copy()
    mask[pred_im > 50] = 1
    mask[pred_im <= 50] = 0
    asso_img = mask * asso_img

    dist = calculate_dist(asso_img, gauses, pred_im, cand_idx)
    try:
        min_dist = min(dist[:, 0])
    except:
        if len(dist) == 0:
            min_dist = 255
        else:
            min_dist = dist[:, 0]
    if min_dist > 0.8:
        # それ以外は除外
        pred1[cell_id_t][4] = 2
        dist_list.append([])
        return pred1, pred2, dist_list
    # get near ground truth point as assocate peak.
    dist_list.append(dist)
    cell_id_tn = int(dist[np.argmin(dist[:, 0])][1])

    pred1, pred2 = decide_cell_id_tn(
        cell_id_t, cell_id_tn, pred1, pred2, dist, dist_list
    )
    return pred1, pred2, dist_list


def associate_predict_result(
    pred1, pred2, frame, assoc_pred_path, img1, img2, inverse, pred_im
):
    dist_list = []
    pred1_plot = pred1[:, :3].copy()

    gauses = gen_gt_gaussian(pred2, pred_im.shape)
    print(pred1_plot[:, 0].max())
    for cell_id_t, pre in enumerate(pred1_plot):
        asso_img = load_image(frame, cell_id_t, assoc_pred_path, inverse)

        dist = np.sqrt(np.sum(np.square(pred2[:, :2] - pre[1:]), axis=1))
        cand_idx = np.where(dist < 20)[0]
        # if cand_idx
        pred1, pred2, dist_list = update_state_by_iou(
            asso_img, cell_id_t, pred1, pred2, pre, dist_list, gauses, pred_im, cand_idx
        )
    return pred1, pred2


if __name__ == "__main__":
    ori_path_root = Path(f"./output/detection/C2C12_9_1")
    assoc_pred_path_root = Path(f"./output/detection/C2C12_9_1_mask")
    guided_path_root = Path(f"./output/guid_out/C2C12_9_1")
    gts = np.loadtxt(f"/home/kazuya/main/correlation_test/images/tracking_annotation/gt_sequ_9.txt", delimiter=",", skiprows=1)
    save_path_root = Path("./output/association/C2C12_9_1")
    num = 1
    time_lates = [1]
    dataset = "sequ9"
    seqs = [9]
    inverse = True
    if inverse:
        direction = "fromn"
        image_num = 1
        mode = "backward"
    else:
        direction = ""
        image_num = 2
        mode = "forward"

    for time_late in time_lates:
        for seq in seqs:
            ori_img_path = ori_path_root.joinpath(f"sequ{seq}/ori")
            assoc_pred_path = assoc_pred_path_root.joinpath(f"sequ{seq}")
            save_path = save_path_root.joinpath(f"sequ{seq}")

            tp = 0
            miss = 0
            exclude = 0
            disapeared = 0
            over = 0
            for frame in range(0, 780 - time_late):
                guided_path = guided_path_root.joinpath(f"sequ{seq}/{frame:05d}")
                # path def
                save_path.joinpath(f"{frame:05d}/exclude").mkdir(
                    parents=True, exist_ok=True
                )
                save_path.joinpath(f"{frame:05d}/exclude2").mkdir(
                    parents=True, exist_ok=True
                )
                save_path.joinpath(f"{frame:05d}/miss").mkdir(
                    parents=True, exist_ok=True
                )
                save_path.joinpath(f"{frame:05d}/correct").mkdir(
                    parents=True, exist_ok=True
                )
                save_path.joinpath(f"{frame:05d}/over").mkdir(
                    parents=True, exist_ok=True
                )

                # load original image
                img1 = cv2.imread(str(ori_img_path.joinpath(f"{frame:04d}_1.png")))
                img2 = cv2.imread(str(ori_img_path.joinpath(f"{frame:04d}_2.png")))
                pred1_im = cv2.imread(
                    str(ori_img_path.parent.joinpath(f"pred/{frame:04d}_1.png")), 0
                )
                pred2_im = cv2.imread(
                    str(ori_img_path.parent.joinpath(f"pred/{frame:04d}_2.png")), 0
                )

                if inverse:
                    pred_im = pred1_im
                else:
                    pred_im = pred2_im

                # load
                pred1 = np.loadtxt(
                    str(guided_path.joinpath("peaks.txt")), skiprows=2, delimiter=",", ndmin=2
                )
                pred2 = local_maxima(pred_im, 80, 2)
                if "PhC-C2DL-PSC" == dataset:
                    pred2 = pred2[(pred2[:, 0] > 24) & (pred2[:, 0] < 744)]

                # [index, x, y, cell_id, state]
                pred1 = np.insert(pred1, 3, [[-1], [0]], axis=1)
                # [x, y, flag, cell_id]
                pred2 = np.insert(pred2, 2, [[0], [-1]], axis=1)

                pred1, pred2 = associate_predict_result(
                    pred1, pred2, frame, assoc_pred_path, img1, img2, inverse, pred_im
                )

                # set gt
                gt1 = gts[gts[:, 0] == (frame + 1)]
                gt2 = gts[gts[:, 0] == (frame + time_late + 1)]
                if inverse:
                    temp = gt1
                    gt1 = gt2
                    gt2 = temp
                gt2t = optimum(gt1[:, 2:4], pred1[:, 1:], 20)
                gtn2tn = optimum(gt2[:, 2:4], pred2, 20)

                for cell_id_t, pre in enumerate(pred1):
                    vis_img = visuarize_img(
                        pre,
                        frame,
                        cell_id_t,
                        assoc_pred_path,
                        img1,
                        img2,
                        pred1_im,
                        pred2_im,
                    )
                    if pre[4] == 2:

                        cv2.imwrite(
                            str(
                                save_path.joinpath(
                                    f"{frame:05d}/exclude/{cell_id_t}.png"
                                )
                            ),
                            vis_img,
                        )
                        exclude += 1
                    elif pre[4] == 3:

                        cv2.imwrite(
                            str(
                                save_path.joinpath(
                                    f"{frame:05d}/exclude2/{cell_id_t}.png"
                                )
                            ),
                            vis_img,
                        )
                        exclude += 1
                    elif pre[4] == 1:

                        cell_id_tn = pre[3]

                        # get pred peak's gt id
                        gt2t_assoc_index = np.where(gt2t[:, 1] == cell_id_t)[0]
                        gtn2tn_assoc_index = np.where(gtn2tn[:, 1] == cell_id_tn)[0]
                        if (gt2t_assoc_index.shape[0] != 0) and (
                            gtn2tn_assoc_index.shape[0] != 0
                        ):
                            gt_index = gt2t[gt2t_assoc_index][0, 0]
                            gt_cell_id = gt1[int(gt_index)][1]

                            gtn_index = gtn2tn[gtn2tn_assoc_index][0, 0]
                            gtn_cell_id = gt2[int(gtn_index)][1]
                            plt.imshow(img1), plt.plot(
                                gt1[int(gt_index)][2], gt1[int(gt_index)][3], "rx"
                            ), plt.plot(
                                gt2[int(gtn_index)][2], gt2[int(gtn_index)][3], "rx"
                            ), plt.savefig(
                                "test.png"
                            )
                            plt.close()
                            if inverse:
                                parent_id = gt1[int(gt_index)][4]
                                mitosis_flag = parent_id == gtn_cell_id
                            else:
                                parent_id = gt2[int(gtn_index)][4]
                                mitosis_flag = parent_id == gt_cell_id
                            if (gt_cell_id == gtn_cell_id) or (mitosis_flag):

                                cv2.imwrite(
                                    str(
                                        save_path.joinpath(
                                            f"{frame:05d}/correct/{cell_id_t}.png"
                                        )
                                    ),
                                    vis_img,
                                )
                                tp += 1

                            else:
                                cv2.imwrite(
                                    str(
                                        save_path.joinpath(
                                            f"{frame:05d}/miss/{cell_id_t}.png"
                                        )
                                    ),
                                    vis_img,
                                )
                                miss += 1
                        else:
                            cv2.imwrite(
                                str(
                                    save_path.joinpath(
                                        f"{frame:05d}/over/{cell_id_t}.png"
                                    )
                                ),
                                vis_img,
                            )
                            # plt.imshow(vis_img), plt.show()
                            over += 1

                np.savetxt(
                    str(save_path.joinpath(f"{frame:05d}/traject_1.txt")),
                    pred1,
                    fmt="%03d",
                    header="[index, x, y, cell_id, state]",
                    delimiter=",",
                )

                np.savetxt(
                    str(save_path.joinpath(f"{frame:05d}/traject_2.txt")),
                    pred2,
                    fmt="%03d",
                    header="[x, y, state, cell_id]",
                    delimiter=",",
                )
                print(f"{frame},{tp / (tp + miss +  over)}")
                print(exclude / (tp + exclude + miss + over))  # 除いた細胞の割合
            print(tp / (tp + miss + over))  # tracking accuracy
            print(exclude / (tp + exclude + miss + over))  # 除いた細胞の割合
            with save_path.joinpath("result.txt").open("w") as f:
                f.write(
                    f"{tp / (tp + miss + over)}, {exclude / (tp + exclude + miss + over)}\n"
                )
                f.write(f"tp, miss, over, exclude\n")
                f.write(f"{tp}, {miss}, {over}, {exclude}\n")
