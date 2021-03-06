from pathlib import Path
import numpy as np

MODES = {1: "BFres", 2: "BFres_wo_reject"}
# 3: "cmp",
# 4: "cmp_wo_reject"

DATASETS = {1: "Elmer", 2: "C2C12"}


def BF_conf():

    for mode in MODES.values():
        for seq in [11, 12, 13, 2, 6, 15]:
            for time_late in [1, 5, 9]:
                if seq == 10:
                    crop_init = (300, 160)
                elif (seq == 6) or (seq == 15):
                    crop_init = (0, 0)
                elif seq == 13:
                    crop_init = (528, 0)
                else:
                    crop_init = (500, 300)
                gt = np.loadtxt(f"/home/kazuya/main/Hayashida/sequ{seq}_pre/gt_{crop_init[0]:03d}-{crop_init[1]:03d}.txt", skiprows=3)
                pred = np.loadtxt(f"/home/kazuya/main/association_accuracy/sequ{seq}/{mode}_C2C12_{time_late}.txt")

                gt = gt[(gt[:, 2] % time_late) == 0]
                gt[:, 2] = gt[:, 2] / time_late
                gt = gt[gt[:, 0] < 512]
                gt = gt[gt[:, 0] > 7]
                gt = gt[gt[:, 1] < 512]
                gt = gt[gt[:, 1] > 7]
                gt = gt[gt[:, 2] < 96]

                pred[:, 2] = pred[:, 2] / time_late
                pred = pred[pred[:, 0] < 505]
                pred = pred[pred[:, 0] > 7]
                pred = pred[pred[:, 1] < 505]
                pred = pred[pred[:, 1] > 7]
                pred = pred[pred[:, 2] < 96]

                np.savetxt(f"/home/kazuya/main/association_accuracy/sequ{seq}/gt_{time_late}.txt", gt, fmt="%d")
                np.savetxt(f"/home/kazuya/main/association_accuracy/sequ{seq}/{mode}_Elmer_{time_late}.txt", pred, fmt="%d")


if __name__ == "__main__":
    BF_conf()

    # for time_late in [1, 5, 9]:
    #     gt = np.loadtxt("/home/kazuya/main/Hayashida/sequ11_pre/gt_500-300.txt", skiprows=3)
    #     pred = np.loadtxt(f"/home/kazuya/main/Hayashida/Elmer_phase_pre/Elmer_train.txt", skiprows=3)
    #
    #     gt = gt[(gt[:, 2] % time_late) == 0]
    #     gt[:, 2] = gt[:, 2] / time_late
    #     gt = gt[gt[:, 0] < 505]
    #     gt = gt[gt[:, 0] > 7]
    #     gt = gt[gt[:, 1] < 505]
    #     gt = gt[gt[:, 1] > 7]
    #     gt = gt[gt[:, 2] < 96]
    #
    #     pred = pred[(pred[:, 2] % time_late) == 0]
    #     pred[:, 2] = pred[:, 2] / time_late
    #     pred = pred[pred[:, 0] < 505]
    #     pred = pred[pred[:, 0] > 7]
    #     pred = pred[pred[:, 1] < 505]
    #     pred = pred[pred[:, 1] > 7]
    #     pred = pred[pred[:, 2] < 96]
    #
    #     np.savetxt(f"/home/kazuya/main/association_accuracy/Elmer/Elmer_gt_{time_late}.txt", gt, fmt="%d")
    #     np.savetxt(f"/home/kazuya/main/association_accuracy/Elmer/cmp_Elmer_{time_late}.txt", pred, fmt="%d")