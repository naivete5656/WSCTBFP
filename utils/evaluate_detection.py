from pathlib import Path
import numpy as np
import math
import pulp
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
from utils import optimum


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


def show_res(img, gt, pred, no_detected_id, over_detection_id, path=None):
    plt.figure(figsize=(2, 1), dpi=500)
    plt.imshow(img, plt.cm.gray)
    plt.plot(gt[:, 0], gt[:, 1], "y3", label="gt_annotation")
    plt.plot(pred[:, 0], pred[:, 1], "g4", label="pred")
    if no_detected_id.shape[0] > 0:
        plt.plot(
            gt[no_detected_id][:, 0],
            gt[no_detected_id][:, 1],
            "b2",
            label="no_detected",
        )
    if over_detection_id.shape[0] > 0:
        plt.plot(
            pred[over_detection_id][:, 0],
            pred[over_detection_id][:, 1],
            "k1",
            label="over_detection",
        )
    plt.legend(bbox_to_anchor=(0, 1), loc="upper left", fontsize=5, ncol=4)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(path)
    plt.close()


def remove_outside_plot(matrix, associate_id, i, window_size, window_thresh=10):
    """
    delete peak that outside
    :param matrix:target matrix
    :param associate_id:optimizeした結果
    :param i: 0 or 1 0の場合target,1の場合predを対象にする
    :param window_size: window size
    :return: removed outside plots
    """
    # delete edge plot 対応付けされなかった中で端のデータを消去
    index = np.delete(np.arange(matrix.shape[0]), associate_id[:, i])
    if index.shape[0] != 0:
        a = np.where(
            (matrix[index][:, 0] < window_thresh)
            | (matrix[index][:, 0] > window_size[1] - window_thresh)
        )[0]
        b = np.where(
            (matrix[index][:, 1] < window_thresh)
            | (matrix[index][:, 1] > window_size[0] - window_thresh)
        )[0]
        delete_index = np.unique(np.append(a, b, axis=0))
        return (
            np.delete(matrix, index[delete_index], axis=0),
            np.delete(index, delete_index),
        )
    else:
        return matrix, np.array([], dtype=np.int)


if __name__ == "__main__":
    time_lates = ["single1"]
    seqs = [11, 12, 13, 2, 6, 15]

    root_dir = Path("/home/kazuya/main/weakly_tracking/output/detection")
    root_dir.mkdir(parents=True, exist_ok=True)
    with root_dir.joinpath("detection_eval.txt").open("w") as f:
        f.write("seq, recall, precision, fmeasure\n")
        for time_late in time_lates:
            for seq in seqs:
                root_path = root_dir = Path(
                    f"/home/kazuya/main/weakly_tracking/output/detection/C2C12_9_{time_late}/sequ{seq}"
                )
                if time_late == "single1":
                    img_paths = sorted(root_path.joinpath("ori").glob("*.png"))
                    gt_paths = sorted(root_path.joinpath("gt").glob("*.png"))
                    pred_paths = sorted(root_path.joinpath("pred").glob("*.png"))
                else:
                    img_paths = sorted(root_path.joinpath("ori").glob("*_1.png"))
                    gt_paths = sorted(root_path.joinpath("gt").glob("*_1.png"))
                    pred_paths = sorted(root_path.joinpath("pred").glob("*_1.png"))

                assert len(img_paths) != 0, "no_file"

                save_path = root_path.joinpath("detection_result")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path_txt = root_path.joinpath("tp_fn_fp")
                save_path_txt.mkdir(parents=True, exist_ok=True)

                tps = 0
                fns = 0
                fps = 0
                for i, data in enumerate(zip(img_paths, gt_paths, pred_paths)):
                    img = cv2.imread(str(data[0]))
                    gt = cv2.imread(str(data[1]), -1)
                    x_plos, y_plos = np.where(gt == 255)
                    gt_plots = []
                    for x, y in zip(x_plos, y_plos):
                        gt_plots.append([y, x])
                    gt_plots = np.array(gt_plots)
                    if gt_plots.shape[0] == 0:
                        print(1)
                    pred = cv2.imread(str(data[2]), 0)
                    thresh = 125

                    min_dist = 3
                    pred_plots = local_maxima(pred, thresh, min_dist)
                    refine_plots = pred_plots.copy()
                    for current_id, pred_plot in enumerate(pred_plots):
                        dist = np.sqrt(
                            np.sum(np.square(refine_plots - pred_plot), axis=1)
                        )
                        if np.sum(dist < min_dist) > 1:
                            x = np.where(dist < min_dist)[0]
                            refine_plots = np.delete(refine_plots, x[0], axis=0)
                    pred_plots = refine_plots

                    permit_dist = 10
                    associate_id = optimum(gt_plots, pred_plots, permit_dist)

                    window_thresh = 10
                    gt_final, no_detected_id = remove_outside_plot(
                        gt_plots,
                        associate_id,
                        0,
                        img.shape,
                        window_thresh=window_thresh,
                    )
                    pred_final, overdetection_id = remove_outside_plot(
                        pred_plots,
                        associate_id,
                        1,
                        img.shape,
                        window_thresh=window_thresh,
                    )

                    show_res(
                        img,
                        gt_plots,
                        pred_plots,
                        no_detected_id,
                        overdetection_id,
                        save_path.joinpath("{:04d}.png".format(i)),
                    )

                    np.savetxt(
                        save_path_txt.joinpath(f"gt_{data[0].stem}.txt"), gt_plots
                    )
                    np.savetxt(
                        save_path_txt.joinpath(f"pred_{data[0].stem}.txt"), pred_plots
                    )
                    np.savetxt(
                        save_path_txt.joinpath(f"over_detect_{data[0].stem}.txt"),
                        pred_plots[overdetection_id],
                    )
                    np.savetxt(
                        save_path_txt.joinpath(f"no_detect_{data[0].stem}.txt"),
                        gt_plots[no_detected_id],
                    )

                    tp = associate_id.shape[0]
                    fn = gt_final.shape[0] - associate_id.shape[0]
                    fp = pred_final.shape[0] - associate_id.shape[0]
                    tps += tp
                    fns += fn
                    fps += fp
                recall = tps / (tps + fns)
                precision = tps / (tps + fps)
                f_measure = (2 * recall * precision) / (recall + precision)
                f.write(f"seq={seq}, time_late={time_late}, {recall}, {precision}, {f_measure}\n")
                # f.write(f"{tps}, {fns}, {fps}\n")

                print(recall, precision, f_measure)
                print(tps, fns, fps)
