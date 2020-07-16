import shutil
from pathlib import Path
import numpy as np
from utils import optimum
import cv2
import matplotlib.pyplot as plt


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


def output_result(pred, pred_tn, cell_id_t, cell_id_tn, frame, assoc_pred_path_root, save_mode, gt):
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
    if save_mode == "reject":
        cv2.circle(img_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (255, 0, 0), 1)
        cv2.circle(pred_im_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (255, 0, 0), 1)
    elif save_mode == "fp":
        cv2.circle(img_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 255, 0), 1)
        cv2.circle(pred_im_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 255, 0), 1)
    else:
        cv2.circle(img_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 0, 255), 1)
        cv2.circle(pred_im_vis_tn, (int(tn_pos[0]), int(tn_pos[1])), 5, (0, 0, 255), 1)

    if save_mode != "reject":
        cv2.circle(img_vis, (int(t_pos[0]), int(t_pos[1])), 5, (0, 255, 0), 1)
        cv2.circle(pred_im_vis, (int(t_pos[0]), int(t_pos[1])), 5, (0, 255, 0), 1)

    if gt is not None:
        cv2.circle(img_vis, (int(gt[2]), int(gt[3])), 5, (0, 0, 255), 1)
        cv2.circle(pred_im_vis, (int(gt[2]), int(gt[3])), 5, (0, 0, 255), 1)

    img_vis = visuarize_img(patch_window, img_vis, img_vis_tn)
    pred_im_vis = visuarize_img(patch_window, pred_im_vis, pred_im_vis_tn)
    asso_img_vis = visuarize_img(patch_window, asso_img, asso_img_tn)
    asso_img_vis = cv2.cvtColor(asso_img_vis, cv2.COLOR_GRAY2BGR)
    inp_img_vis = visuarize_img(patch_window, inp_img, inp_img_tn)
    inp_img_vis = cv2.cvtColor(inp_img_vis, cv2.COLOR_GRAY2BGR)
    vis_img = np.hstack([img_vis, pred_im_vis])
    lik_img_vis = np.hstack([inp_img_vis, asso_img_vis])
    vis_img = np.hstack([vis_img, lik_img_vis])

    cv2.imwrite(str(save_path.joinpath(f"{frame:05d}/{save_mode}/{cell_id_tn}.png")), vis_img)


def check_associate(gts, pred_tn, pred, frame, assoc_pred_path_root, img, img_tn, pred_im, pred_tn_im, time_late):
    tp = 0
    fp = 0
    reject = 0

    # set gt
    gt = gts[gts[:, 0] == (frame)]
    gt_tn = gts[gts[:, 0] == (frame + time_late)]

    gt2t = optimum(gt[:, 2:4], pred, 20)
    gtn2tn = optimum(gt_tn[:, 2:4], pred_tn, 20)

    # gen gt traject list
    # t_ids = set(gt[gt2t[:, 0].astype(np.int)][:, 1])
    # tn_ids = set(gt_tn[gtn2tn[:, 0].astype(np.int)][:, 1])
    # tn_par_ids = set(gt_tn[gtn2tn[:, 0].astype(np.int)][:, 4])
    # true_ids = (t_ids & tn_ids) | (t_ids & tn_par_ids)
    true_ids = (set(gt[:, 1]) & set(gt_tn[:, 1])) | (set(gt[:, 1]) & set(gt_tn[:, 4]))

    # visualize
    for cell_id_tn, pre in enumerate(pred_tn):
        if pre[3] == 1:
            cell_id_t = np.where(int(pre[2]) == pred[:, 2])[0][0]

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
                mitosis_flag = (parent_id == gt_cell_id)

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

        output_result(pred, pred_tn, cell_id_t, cell_id_tn, frame, assoc_pred_path_root, save_mode, gt_pos)

    fn_ind = np.isin(gt[:, 1], list(true_ids))
    fn_pos = gt[fn_ind]
    plt.imshow(img), plt.plot(fn_pos[:, 2], fn_pos[:, 3], "rx"), plt.savefig(
        str(save_path.joinpath(f"{frame:05d}/fn/fn_cells.png"))), plt.close()

    return tp, reject, fp, fn_pos.shape[0]


if __name__ == "__main__":
    mode = {1: "", 2: "_wo_reject"}
    mode = mode[1]
    time_lates = [1, 5, 9]
    with Path("../output/test/result.txt").open("w") as f:
        for time_late in time_lates:
            for seq in [11, 12, 13, 2, 6, 15]:
            # for seq in [13]:
                frame = 0
                save_path = Path(f"../output/test/sequ{seq}")
                save_path.mkdir(parents=True, exist_ok=True)

                if save_path.is_dir():
                    shutil.rmtree(save_path)

                ori_path = Path(f"../output/detection/C2C12_9_{time_late}/sequ{seq}/ori")
                assoc_pred_path = Path(f"../output/detection/C2C12_9_{time_late}_mask/sequ{seq}")
                root_paths = sorted(
                    [x for x in Path(f"../output/association{mode}/C2C12_9_{time_late}/sequ{seq}").iterdir() if x.is_dir()])
                gts = np.loadtxt(f"/home/kazuya/main/correlation_test/images/tracking_annotation/gt_seq_{seq}.txt",
                # gts = np.loadtxt(f"/home/kazuya/main/correlation_test/images/tracking_annotation/gt_sequ_13.txt",
                                 delimiter=",", skiprows=1)

                tps = 0
                rejects = 0
                fps = 0
                fns = 0

                for frame, root_path in enumerate(root_paths):
                    img, img_tn, pred_im, pred_tn_im = load_image(frame, ori_path)

                    make_file(save_path)

                    # traject_init
                    # [x, y, id, state]
                    cell_id = 1
                    traject_1_path = root_path.joinpath("traject_1.txt")
                    traject_2_path = root_path.joinpath("traject_2.txt")
                    traject_1 = np.loadtxt(str(traject_1_path), delimiter=",", skiprows=1)
                    traject_2 = np.loadtxt(str(traject_2_path), delimiter=",", skiprows=1)

                    for ind, tra in enumerate(traject_1):
                        state = tra[3]
                        if state == 1:
                            cell_ind = tra[2]
                            traject_2[int(cell_ind)][2] = cell_id
                            traject_1[ind][2] = cell_id
                            cell_id += 1
                    pred = traject_1[traject_1[:, 3] == 1]
                    pred_tn = traject_2[traject_2[:, 3] == 1]

                    tp, reject, fp, fn = check_associate(gts, pred_tn, pred, frame, assoc_pred_path, img, img_tn,
                                                         pred_im, pred_tn_im, time_late)

                    print(tp, fp, fn, reject)

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

                f.write(
                    f"{seq}, {time_late}, {tps / (tps + fps)}, {tps / (tps + fns)}, {2 * tps / (2 * tps + fps + fns)}, {(tps + fps + fns) / ((tps + fps + fns) + rejects)}\n"
                )
