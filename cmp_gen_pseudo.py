"""
outpu by npy
"""
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib


# return number of vectors
def compute_vector(
        black, pre, nxt, result, result_l, result_y, result_x, result_z, ks, zv, sgm
):
    img_l = black.copy()  # likelihood image
    img_l[nxt[1] + ks, nxt[0] + ks] = 255
    img_l = cv2.GaussianBlur(
        img_l, ksize=(int(ks * 2) + 1, int(ks * 2) + 1), sigmaX=sgm
    )
    img_l = img_l / img_l.max()
    points = np.where(img_l > 0)
    img_y = black.copy()
    img_x = black.copy()
    img_z = black.copy()
    for y, x in zip(points[0], points[1]):
        v3d = pre + [ks, ks] - [x, y]
        # v3d = np.append(v3d, 1.4434)
        # v3d = np.append(v3d, 2.7632)
        v3d = np.append(v3d, zv)
        v3d = v3d / np.linalg.norm(v3d) * img_l[y, x]
        img_y[y, x] = v3d[1]
        img_x[y, x] = v3d[0]
        img_z[y, x] = v3d[2]
    img_i = result_l - img_l
    result_y = np.where(img_i < 0, img_y, result_y)
    result_x = np.where(img_i < 0, img_x, result_x)
    result_z = np.where(img_i < 0, img_z, result_z)

    img_i = img_l.copy()
    img_i[img_i == 0] = 2
    img_i = result_l - img_i
    result[img_i == 0] += 1
    result_y += np.where(img_i == 0, img_y, 0)
    result_x += np.where(img_i == 0, img_x, 0)
    result_z += np.where(img_i == 0, img_z, 0)

    result_l = np.maximum(result_l, img_l)
    return result, result_l, result_y, result_x, result_z


# migration mean
movement = [
    [1.4434, 1.3197942503607587],
    [2.7377, 2.195258089759622],
    [4.0067, 3.0202187134378864],
    [5.2538, 3.8274910979238674],
    [6.4813, 4.618897370307621],
    [7.6874, 5.388044434257328],
    [8.884, 6.133690378378062],
    [10.0534, 6.843635116056122],
    [11.2083, 7.529895064500173],
    [12.3036, 8.172467388870148],
    [13.4689, 8.847465726604845],
    [14.4967, 9.43786499684944],
    [15.5675, 10.046110063303223],
    [16.6942, 10.678688049700735],
    [17.6606, 11.222874594749634],
    [18.7754, 11.83313712025504],
    [19.8107, 12.411461893903311],
    [20.74, 12.936127125203171],
    [21.6791, 13.45838277173517],
]

###############hyperparameter##################
KS = 50  # kernel size
SGM = 6  # sigma
Z_VALUE = 5  # temporal axis


def generate_flow(track_let, save_path, itv=1, height=1040, width=1392):
    track_let = track_let.astype(int)
    frames = np.unique(track_let[:, 0])
    ids = np.unique(track_let[:, 1])

    black = np.zeros((height + KS * 2, width + KS * 2))
    ones = np.ones((height + KS * 2, width + KS * 2))

    par_id = -1
    i = 1

    result = ones.copy()
    result_lm = black.copy()
    result_y = black.copy()
    result_x = black.copy()
    result_z = black.copy()

    for j in ids:
        index_check = len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)])
        index_chnxt = len(
            track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)]
        )
        if index_chnxt != 0:
            par_id = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][
                0, -1
            ]

        if (index_check != 0) & (index_chnxt != 0):
            data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0][2:-1]
            dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][
                   2:-1
                   ]

            result, result_lm, result_y, result_x, result_z = compute_vector(
                black,
                data,
                dnxt,
                result,
                result_lm,
                result_y,
                result_x,
                result_z,
                KS,
                Z_VALUE,
                SGM,
            )

        elif (index_check == 0) & (index_chnxt != 0) & (par_id != -1):
            if (
                    len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)])
                    != 0
            ):
                data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)][
                           0
                       ][2:-1]
                dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][
                           0
                       ][2:-1]
                result, result_lm, result_y, result_x, result_z = compute_vector(
                    black,
                    data,
                    dnxt,
                    result,
                    result_lm,
                    result_y,
                    result_x,
                    result_z,
                    KS,
                    Z_VALUE,
                    SGM,
                )
            else:
                print(
                    track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0]
                )

    result = result[KS:-KS, KS:-KS]
    print(i + 1, "to", i + itv + 1, result.max())

    result_y = result_y[KS:-KS, KS:-KS]
    result_x = result_x[KS:-KS, KS:-KS]
    result_z = result_z[KS:-KS, KS:-KS]
    result_lm = result_lm[KS:-KS, KS:-KS]

    result_x = result_x / result
    result_y = result_y / result
    result_z = result_z / result

    result_vec = np.concatenate(
        (result_y[:, :, None], result_x[:, :, None], result_z[:, :, None]), axis=-1
    )

    np.save(str(save_path), result_vec.astype("float16"))
    plt.figure(figsize=(1, 1), dpi=1000)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(result_vec)
    plt.savefig(str(save_path.parent.joinpath(f"{save_path.stem}.png")))


if __name__ == "__main__":
    for time_late in [1, 5, 9]:
        save_CMP_path = Path(f"/home/kazuya/main/WSCTBFP/data/cmp/{time_late}")
        save_CMP_path.mkdir(parents=True, exist_ok=True)

        save_mask_path = save_CMP_path.parent.joinpath(f"mask_{time_late}")
        save_mask_path.mkdir(parents=True, exist_ok=True)

        root_path = Path(f"./output/association/{time_late}")

        pred1_paths = sorted(root_path.glob("*/*_1.txt"))
        pred2_paths = sorted(root_path.glob("*/*_2.txt"))

        for frame, pred_path in enumerate(zip(pred1_paths, pred2_paths)):
            # [x, y, cell_id, state]
            pred1 = np.loadtxt(str(pred_path[0]), delimiter=",", skiprows=1)
            # [x, y, cell_id, state]
            pred2 = np.loadtxt(str(pred_path[1]), delimiter=",", skiprows=1)

            mask = np.zeros((320, 320))
            exclude_cells = pred1[(pred1[:, 3] == 2) | (pred1[:, 3] == 0)]
            for exclude_cell in exclude_cells:
                mask = cv2.circle(
                    mask,
                    (int(exclude_cell[0]), int(exclude_cell[1])),
                    SGM * 3,
                    255,
                    -1,
                )
            cv2.imwrite(
                str(save_mask_path.joinpath(f"{frame:05d}.tif")),
                mask.astype(np.uint8),
            )

            pred1_new = pred1.copy()
            pred2_new = pred2.copy()

            cell_id = 1
            for index, pre in enumerate(pred1):
                if pre[3] == 1:
                    pred1_new[index][2] = cell_id
                    pred2_new[int(pre[2])][2] = cell_id
                    cell_id += 1

            pred1 = pred1_new
            pred2 = pred2_new
            pred1 = pred1[pred1[:, 3] == 1]
            pred2 = pred2[pred2[:, 3] == 1]

            track_let = np.zeros(((pred1.shape[0] + pred2.shape[0], 5)))
            track_let[:pred1.shape[0], 0] = 1
            track_let[pred1.shape[0]:, 0] = 2
            track_let[:pred1.shape[0], 2:4] = pred1[:, :2]
            track_let[pred1.shape[0]:, 2:4] = pred2[:, :2]
            track_let[:pred1.shape[0], 1] = pred1[:, 2]
            track_let[pred1.shape[0]:, 1] = pred2[:, 2]

            generate_flow(
                track_let,
                save_CMP_path.joinpath(f"{frame:05d}.npy"),
                height=320,
                width=320,
            )

            print("finished")
