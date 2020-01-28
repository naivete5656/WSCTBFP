"""
Cell Motion FieldsのgroundTruth生成
・(HEIGHT,WIDTH,2)のnumpyファイル
・result_vector[:,:,0]にy、[:,:,1]にx座標
・単位ベクトル

xml:xy
Opencv:xy
NumPy:yx
"""
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path

global radius
radius = 5


def visualize_hsv(flow, name):
    flow = flow.astype("float32")

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 1] = 255

    # flowの大きさと角度を計算
    mag, ang = cv2.cartToPolar(flow[..., 1], flow[..., 0])
    # OpenCVのhueは0~180で表現
    hsv[..., 0] = ang * 180 / np.pi / 2
    # 強さを0~255で正規化
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # これないとエラー
    hsv = hsv.astype("uint8")
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(name, rgb)


# 各座標のベクトル本数とベクトルを返す
def compute_vector(black, pre, nxt, result, result_y, result_x, sgm):
    # v' = p(t+1) - p(t)
    # 単位ベクトル化(v = v'/|v'|)
    v = nxt - pre
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v)

    # 法線ベクトル上下の定義してsgm倍(sgmはハイパーパラメータ)
    up = np.array([-v[1], v[0]]) * sgm
    dw = np.array([v[1], -v[0]]) * sgm

    # (p(t)の座標 or p(t+1)の座標)と法線ベクトル2種の和
    v1 = up + nxt + radius
    v2 = dw + nxt + radius
    v3 = up + pre + radius
    v4 = dw + pre + radius

    # p(t+1)とp(t)を結ぶ線分を囲む4点
    points = np.round(
        np.array([[v1[0], v1[1]], [v2[0], v2[1]], [v4[0], v4[1]], [v3[0], v3[1]]])
    )

    img_t = black.copy()
    img_y = black.copy()
    img_x = black.copy()
    img_z = black.copy()

    # points4点で囲む領域を1に
    img_t = cv2.fillPoly(img=img_t, pts=np.int32([points]), color=1)
    # img_t = cv2.circle(img_t, (pre[0] + radius, pre[1] + radius), radius, (1), thickness=-1, lineType=cv2.LINE_4)
    # img_t = cv2.circle(img_t, (nxt[0] + radius, nxt[1] + radius), radius, (1), thickness=-1, lineType=cv2.LINE_4)

    # v = nxt - pre
    # v = np.append(v, 1)
    # v = v / np.linalg.norm(v)

    img_y[img_t != 0] = v[1]
    img_x[img_t != 0] = v[0]
    # img_z[img_t != 0] = v[2]
    # どんどんベクトル追加
    result = result + img_t
    # ベクトルもとりあえず和でOK(あとで平均取る)
    result_x = result_x + img_x
    result_y = result_y + img_y
    # result_z = result_z + img_z

    return result, result_y, result_x


############################################################################################
txt_names = sorted(
    glob.glob("/home/kazuya/Downloads/nishimura/090303_exp1_F0009_GT_full.txt")
)
sgm = 5  # CMFの幅/2の値
############################################################################################

for txt_name in txt_names:
    for itv in [1, 3, 5, 7, 9]:
        save_path = Path(f"./images/sequ9/CMF_6_{itv}")

        track_let = np.loadtxt(txt_name).astype("int")
        frames = np.unique(track_let[:, 0])
        ids = np.unique(track_let[:, 1])

        # 論文内のσ

        # いろいろ使う黒画像(0行列)
        black = np.zeros((1040 + radius * 2, 1392 + radius * 2, 1))
        save_dir = os.path.join(save_path, os.path.split(txt_name)[1][:-4])
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        par_id = -1

        for i in frames:
            # resultは各座標に何本ベクトルがあるか(かぶっていたら2とか3とか)
            # result_y,result_x は出力ベクトル
            result = black.copy()
            result_y = black.copy()
            result_x = black.copy()
            output = []
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
                    data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0]
                    dnxt = track_let[
                        (track_let[:, 0] == i + itv) & (track_let[:, 1] == j)
                    ][0]
                    pre = data[2:-1]
                    nxt = dnxt[2:-1]
                    result, result_y, result_x = compute_vector(
                        black, pre, nxt, result, result_y, result_x, sgm
                    )

                # 前は無いが、親がいるとき
                elif (index_check == 0) & (index_chnxt != 0) & (par_id != -1):
                    # 親細胞のframe(t)座標
                    if (
                        len(
                            track_let[
                                (track_let[:, 0] == i) & (track_let[:, 1] == par_id)
                            ]
                        )
                        != 0
                    ):
                        data = track_let[
                            (track_let[:, 0] == i) & (track_let[:, 1] == par_id)
                        ][0]
                        dnxt = track_let[
                            (track_let[:, 0] == i + itv) & (track_let[:, 1] == j)
                        ][0]
                        pre = data[2:-1]
                        nxt = dnxt[2:-1]
                        result, result_y, result_x = compute_vector(
                            black, pre, nxt, result, result_y, result_x, sgm
                        )
                    else:
                        print(
                            track_let[
                                (track_let[:, 0] == i + itv) & (track_let[:, 1] == j)
                            ][0]
                        )

            # パディングを消す
            result = result[radius:-radius, radius:-radius]
            print(i, "to", i + itv, result.max())

            # 0で割れないので1に
            result_org = result.copy()
            result[result == 0] = 1
            # パディングを消す
            result_y = result_y[radius:-radius, radius:-radius]
            result_x = result_x[radius:-radius, radius:-radius]
            # result_z = result_z[radius:-radius, radius:-radius]
            result_x = result_x / result
            result_y = result_y / result
            # result_z = (result_z / result)
            result_vector = np.concatenate((result_y, result_x), axis=-1)
            vis_name = str(save_path.joinpath(f"{i:04d}.png"))
            visualize_hsv(result_vector, vis_name)
            # save_npy = save_path + '/{0:03d}.npy'.format(i)
            # np.save(save_npy, result_vector.astype('float16'))
            output.append(result_vector)
            np_output = np.array(output).astype("float16")
            save_name = save_path.joinpath(f"{i:05d}.npy")
            np.savez_compressed(save_name, np_output)
