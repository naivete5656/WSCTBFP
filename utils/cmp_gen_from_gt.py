"""
出力はNPY形式
"""
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 各座標のベクトル本数とベクトルを返す
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


# F0009のfpsごとの移動平均&標準偏差(多分)
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
KS = 50  # カーネルサイズ
SGM = 6  # シグマ
Z_VALUE = 5  # 時間軸の縮尺

###############save directory###################


track_let = np.loadtxt(
    "/home/kazuya/main/correlation_test/Elmer_gt.txt", skiprows=1, delimiter=","
).astype("int")

frames = np.unique(track_let[:, 0])
ids = np.unique(track_let[:, 1])

itvs = [1, 3, 5, 7, 9]

for itv in itvs:
    save_path = Path(f"./images/Elmer_phase/CMP_gt/{itv}")
    save_path.mkdir(parents=True, exist_ok=True)
    # いろいろ使う黒画像(0行列)
    black = np.zeros((1040 + KS * 2, 1392 + KS * 2))
    ones = np.ones((1040 + KS * 2, 1392 + KS * 2))

    par_id = -1
    for idx, i in enumerate(frames[:-itv]):
        # resultは各座標に何本ベクトルがあるか(かぶっていたら2とか3とか)
        # result_y,result_x,result_z は出力ベクトル
        result = ones.copy()
        result_lm = black.copy()
        result_y = black.copy()
        result_x = black.copy()
        result_z = black.copy()

        for j in ids:
            # i+1のフレーム
            index_check = len(
                track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)]
            )
            index_chnxt = len(
                track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)]
            )
            if index_chnxt != 0:
                par_id = track_let[
                    (track_let[:, 0] == i + itv) & (track_let[:, 1] == j)
                ][0, -1]

            # 前後のframeがあるとき(dataはframe(t)の座標、dnxtはframe(t+1)の座標)
            if (index_check != 0) & (index_chnxt != 0):
                data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0][
                    2:-1
                ]
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

            # 前は無いが、親がいるとき
            elif (index_check == 0) & (index_chnxt != 0) & (par_id != -1):
                # 親細胞のframe(t)座標
                if (
                    len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)])
                    != 0
                ):
                    data = track_let[
                        (track_let[:, 0] == i) & (track_let[:, 1] == par_id)
                    ][0][2:-1]
                    dnxt = track_let[
                        (track_let[:, 0] == i + itv) & (track_let[:, 1] == j)
                    ][0][2:-1]
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
                        track_let[
                            (track_let[:, 0] == i + itv) & (track_let[:, 1] == j)
                        ][0]
                    )

        # パディングを消す
        result = result[KS:-KS, KS:-KS]
        print(i + 1, "to", i + itv + 1, result.max())

        # 0で割れないので1に
        # result_org = result.copy()
        # パディングを消す
        result_y = result_y[KS:-KS, KS:-KS]
        result_x = result_x[KS:-KS, KS:-KS]
        result_z = result_z[KS:-KS, KS:-KS]
        result_lm = result_lm[KS:-KS, KS:-KS]

        result_x = result_x / result
        result_y = result_y / result
        result_z = result_z / result

        result_vector = np.concatenate(
            (result_y[:, :, None], result_x[:, :, None], result_z[:, :, None]), axis=-1
        )

        save_path_vector = str(save_path.joinpath(f"{i:05d}"))
        np.save(save_path_vector, result_vector.astype("float16"))
        plt.figure(figsize=(1, 1), dpi=1000)

        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.imshow(result_vector[:512, :512])
        plt.savefig(str(save_path.joinpath(f"{i:05d}.png")))

print("finished")
