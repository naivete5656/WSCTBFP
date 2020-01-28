from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def AppendTracklet(cell, track_let, track_part):
    id = int(cell.get("id"))
    if len(cell) == 1:
        track_info = track_part.copy()
        track_info[0, 1] = id
        for info in cell.find(".//ss"):
            track_info[0, 0] = int(info.get("i"))
            track_info[0, 2] = int(float(info.get("x")))
            track_info[0, 3] = int(float(info.get("y")))
            track_let = np.append(track_let, track_info, axis=0)
        return track_let
    if len(cell) == 2:
        track_info = track_part.copy()
        track_info[0, 1] = id
        for info in cell.find(".//ss"):
            track_info[0, 0] = int(info.get("i"))
            track_info[0, 2] = int(float(info.get("x")))
            track_info[0, 3] = int(float(info.get("y")))
            track_let = np.append(track_let, track_info, axis=0)
        if len(cell.find(".//as")) == 2:
            for chil_cell in cell.find(".//as"):
                track_info = track_part.copy()
                track_info[0, -1] = id
                track_let = np.append(
                    track_let,
                    AppendTracklet(
                        chil_cell, np.empty((0, 5)).astype("int32"), track_info
                    ),
                    axis=0,
                )
            return track_let
        else:
            return track_let
    else:
        print(cell)
        return track_let


if __name__ == "__main__":
    xml_path = Path(
        "/home/kazuya/main/correlation_test/images/Elmer--000-000-frame-000-100.xml"
    )
    # seq = int(xml_path.name[4:6])
    crop_init = (int(xml_path.name[11:14]), int(xml_path.name[7:10]))
    frame = int(xml_path.name[-11:-8])
    frame_init = frame
    # load xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # number of cellf
    num_cell = len(root[0][0][0].findall("a"))
    black = np.zeros((1040, 1392))
    track_let = np.empty((0, 5)).astype("int32")
    track_part = np.zeros((1, 5)).astype("int32")
    track_part[0, -1] = -1
    for par_cell in root[0][0][0].findall("a"):
        id = int(par_cell.get("id"))
        print(id)
        track_let = AppendTracklet(par_cell, track_let, track_part.copy())
    frame = 1
    frame_init = 0
    # track_let = np.loadtxt("/home/kazuya/Downloads/nishimura/seq18.txt")
    # track_let = track_let[(track_let[:, 0] > 0) & (track_let[:, 0] < 100)]
    # track_let[:, 0] = track_let[:, 0]
    track_let = track_let[track_let[:, 0] < 100]
    track_let = track_let[
        (track_let[:, 3] > crop_init[0])
        & (track_let[:, 3] < crop_init[0] + 512)
        & (track_let[:, 2] > crop_init[1])
        & (track_let[:, 2] < crop_init[1] + 512)
    ]
    track_let[:, 2] = track_let[:, 2] - crop_init[1]
    track_let[:, 3] = track_let[:, 3] - crop_init[0]

    new_track_let = np.copy(track_let)
    old_ids = np.unique(track_let[:, 1])
    for new_id, old_id in enumerate(old_ids):
        temp = new_track_let[track_let[:, 1] == old_id]
        temp[:, 1] = new_id
        new_track_let[track_let[:, 1] == old_id] = temp
        temp = new_track_let[track_let[:, 4] == old_id]
        temp[:, 4] = new_id
        new_track_let[track_let[:, 4] == old_id] = temp
    new_track_let[:, 1] = new_track_let[:, 1] + 1
    new_track_let[:, 4] = new_track_let[:, 4] + 1

    img = cv2.imread(
        f"/home/kazuya/main/correlation_test/images/Elmer_phase/ori/ois1065_C0002_T0001.tif",
        -1,
    )
    xx = track_let[track_let[:, 0] == frame - frame_init]
    plt.imshow(
        img[crop_init[0] : crop_init[0] + 512, crop_init[1] : crop_init[1] + 512]
    ), plt.plot(xx[:, 2], xx[:, 3], "rx"), plt.show()

    np.savetxt(
        f"Elmer_gt.txt",
        new_track_let,
        fmt="%03d",
        header="frame(0~100), cell_ID, x(width), y(height), parent_id",
        delimiter=",",
    )
