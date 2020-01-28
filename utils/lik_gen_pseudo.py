import numpy as np
from pathlib import Path
from utils import gaus_filter
import cv2


def like_map_gen(frame, cells):
    black = np.zeros((512, 512))

    # likelihood map of one input
    result = black.copy()
    for x, y in cells[:, 1:3]:
        img_t = black.copy()  # likelihood map of one cell
        img_t[int(y)][int(x)] = 255  # plot a white dot
        img_t = gaus_filter(img_t, 51, 6)
        result = np.maximum(result, img_t)  # compare result with gaussian_img
    #  normalization
    result = 255 * result / result.max()
    result = result.astype("uint8")
    cv2.imwrite(str(save_path / Path("%05d.tif" % frame)), result)
    print(frame)


if __name__ == "__main__":
    seqs = [2]
    for seq in seqs:
        # save_path = Path(f"./images/sequ{seq}/psue_9")
        save_path = Path(f"./images/Elmer_phase/psue_9")

        save_path.mkdir(parents=True, exist_ok=True)

        # path = Path(f"./output/association/C2C12_9/1-sequ{seq}")
        path = Path(f"./output/association/Elmer/1_1")

        for frame in range(99):
            pred1 = np.loadtxt(
                str(path.joinpath(f"{frame:05d}/traject_1.txt")),
                skiprows=1,
                delimiter=",",
            )
            exclude_cells = pred1[pred1[:, 4] != 1]
            mask = np.zeros((512, 512))
            for exclude_cell in exclude_cells:
                mask = cv2.circle(
                    mask, (int(exclude_cell[1]), int(exclude_cell[2])), 9 * 3, 255, -1
                )
            pred1 = pred1[pred1[:, 4] == 1]
            like_map_gen(frame, pred1)
