"""
可視化に関するあれこれをまとめたもの
入力は8bit画像を想定
出力は基本カラー画像(3ch)
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get3chImage(src):
    """画像を3chにする関数
    Args:
        src: 入力画像
    """
    chk = src.shape
    if len(chk) == 2:
        out = np.concatenate(
            [src[:, :, None], src[:, :, None], src[:, :, None]], axis=-1
        )
        return out
    elif chk[-1] == 1:
        out = np.concatenate([src, src, src], axis=-1)
        return out
    else:
        return src


def getSyntheticImage(src1, rate1, src2, rate2, save_name=None):
    """2枚の画像を合成する関数
    Args:
        src1, src2: 合成する2枚の画像
        rate1, rate2: どのくらいの割合か(rate1+rate2が1を超えてもOK)
    """
    src1 = get3chImage(src1)
    src2 = get3chImage(src2)
    out = src1 * rate1 + src2 * rate2
    out[out > 255] = 255
    out = out.astype("uint8")
    if save_name is not None:
        cv2.imwrite(save_name, out)
    else:
        return out


def getImageTable(srcs=[], clm=4, save_name=None):
    """与えられた画像群を指定した列数で並べる関数
    Args:
        srcs: 画像のリスト
        clm: 何列で並べるか
        save_name: 入力したらその名前で保存， Noneなら出力
    """
    white_c = np.full((srcs[0].shape[0], 3, 3), 255).astype("uint8")
    white_r = np.full((3, (srcs[0].shape[1] + 3) * clm - 3, 3), 255).astype("uint8")
    black = np.zeros(srcs[0].shape).astype("uint8")
    out = []

    for i in range(len(srcs)):
        srcs[i] = get3chImage(srcs[i])
        srcs[i] = np.hstack([srcs[i], white_c])
    for i in range(len(srcs) % clm):
        srcs.append(black)

    for l in range(int(len(srcs) / clm)):
        c_imgs = np.hstack(srcs[l * clm : l * clm + clm])
        out.append(c_imgs[:, :-3])
        out.append(white_r)
    out = np.vstack(out)

    if save_name is not None:
        cv2.imwrite(save_name, out[:-3])
    else:
        return out


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


def getVIDEO(srcs, save_name, fps=32, font=cv2.FONT_HERSHEY_SIMPLEX):
    black_bar = np.zeros((35, srcs[0].shape[1], 3)).astype("uint8")
    black_bar[-2:] = 255
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    shape = (srcs[0].shape[1], srcs[0].shape[0] + 35)
    # shape = (srcs[0].shape[1], srcs[0].shape[0])
    video = cv2.VideoWriter(save_name, fourcc, fps, shape)
    for i, src in enumerate(srcs):
        src = get3chImage(src)
        src = cv2.vconcat([black_bar, src])
        cv2.putText(
            src,
            "frame{0:04d}_fps={1:03f}".format(i, fps),
            (10, 27),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        video.write(src)
    video.release()
