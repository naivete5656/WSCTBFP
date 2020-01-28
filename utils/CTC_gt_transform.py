from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

datasets = {
    1: "DIC-C2DH-HeLa",
    2: "Fluo-C2DL-MSC",
    3: "Fluo-N2DH-GOWT1",
    4: "Fluo-N2DH-SIM+",
    5: "Fluo-N2DL-HeLa",
    6: "PhC-C2DH-U373",
    7: "PhC-C2DL-PSC",
}


def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(
        img, (pad_size, pad_size), "constant"
    )  # zero padding(これしないと正規化後、画像端付近の尤度だけ明るくなる)
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # filter gaussian(適宜パラメータ調整)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


def position_generate(dataset, sequence):
    tracking_gt_path = sorted(
        Path(
            f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/{sequence:02d}_GT/TRA"
        ).glob("*.tif")
    )

    original_img_path = sorted(
        Path(
            f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/{sequence:02d}"
        ).glob(f"*.tif")
    )
    if dataset == "PhC-C2DL-PSC":
        original_img_path = original_img_path[150:250]

    points = []
    for frame, (tra_path, ori_path) in enumerate(
        zip(tracking_gt_path, original_img_path)
    ):
        tra_gt = cv2.imread(str(tra_path), -1)
        ori = cv2.imread(str(ori_path))
        ids = np.unique(tra_gt)
        ids = ids[ids != 0]
        # plt.imshow(img), plt.show()
        if dataset == "PhC-C2DL-PSC":
            frame = 150 + frame
        for cell_id in ids:
            x, y = np.where(tra_gt == cell_id)
            x_centor = x.sum() / x.size
            y_centor = y.sum() / y.size
            points.append([frame, cell_id, y_centor, x_centor])

        points2 = np.array(points)
        points2 = points2[points2[:, 0] == frame]
        plt.imshow(ori), plt.plot(points2[:, 2], points2[:, 3], "rx"), plt.savefig(
            "test.png"
        ), plt.close()
    np.savetxt(
        f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/gt_plots_{sequence:02d}.txt",
        points,
        delimiter=",",
        fmt="%d",
    )


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="txt path",
        default="./sample_cell_position.txt",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./image/gt",
        type=str,
    )
    parser.add_argument(
        "-w", "--width", dest="width", help="image width", default=512, type=int
    )
    parser.add_argument(
        "-he", "--height", dest="height", help="height", default=512, type=int
    )
    parser.add_argument(
        "-g",
        "--gaussian_variance",
        dest="g_size",
        help="gaussian variance",
        default=9,
        type=int,
    )

    args = parser.parse_args()
    return args


def likelyhood_map_gen(args):
    cell_positions = np.loadtxt(args.input_path, delimiter=",")
    black = np.zeros((args.height, args.width))

    # 1013 - number of frame
    for i in range(int(cell_positions[:, 0].max() + 1)):
        # likelihood map of one input
        result = black.copy()
        cells = cell_positions[cell_positions[:, 0] == i]
        for _, _, x, y in cells:
            img_t = black.copy()  # likelihood map of one cell
            img_t[int(y)][int(x)] = 255  # plot a white dot
            img_t = gaus_filter(img_t, 301, args.g_size)
            result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype("uint8")
        # plt.imshow(result), plt.show()
        cv2.imwrite(str(args.output_path / Path("%05d.tif" % (i))), result)
        print(i + 1)
    print("finish")


if __name__ == "__main__":
    args = parse_args()
    for dataset in datasets.values():
        for sequence in [1, 2]:
            for gauses in [3, 6, 9]:
                # position_generate(dataset, sequence)
                args.input_path = f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/gt_plots_{sequence:02d}.txt"
                args.output_path = f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/{sequence:02}-{gauses}"
                Path(
                    f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/{sequence:02}-{gauses}"
                ).mkdir(parents=True, exist_ok=True)
                args.g_size = gauses
                img_path = f"/home/kazuya/main/celltrackingchallenge_dataset/{dataset}/{sequence:02d}/t000.tif"
                img = cv2.imread(img_path, 0)
                args.height, args.width = img.shape
                likelyhood_map_gen(args)
