import cv2
from pathlib import Path
import numpy as np


# for seq in [11, 12, 13, 2, 6, 15]:
for seq in [15]:

    if (seq == 6) or (seq == 15):
        crop_init = (0, 0)
    elif seq == 13:
        crop_init = (528, 0)
    else:
        crop_init = (500, 300)

    save_path = Path(f"/home/kazuya/main/weakly_tracking/images/sequ{seq}")
    # save_path.joinpath("ori").mkdir(parents=True, exist_ok=True)
    # save_path.joinpath("3").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("6").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("9").mkdir(parents=True, exist_ok=True)
    # save_path.joinpath("12").mkdir(parents=True, exist_ok=True)
    # save_path.joinpath("bg").mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(f"/home/kazuya/main/correlation_test/images/sequ18/ori").glob("*.tif"))
    # path_6 = Path(f"/home/kazuya/main/correlation_test/images/sequ{seq}/6")
    path_6 = Path(f"/home/kazuya/main/correlation_test/images/sequ18/6")
    # path_9 = Path(f"/home/kazuya/main/correlation_test/images/sequ{seq}/9")
    path_9 = Path(f"/home/kazuya/main/correlation_test/images/sequ18/9")
    # path_bg = Path(f"/home/kazuya/main/correlation_test/images/sequ{seq}/bg")
    for img_num, ori_path in enumerate(paths):
        img = cv2.imread(str(ori_path), -1)
        gt6 = cv2.imread(str(path_6 / f"{img_num+600:05d}.tif"))
        gt9 = cv2.imread(str(path_9 / f"{img_num+600:05d}.tif"))
        # gtbg = cv2.imread(str(path_bg / f"{img_num + 1:05d}.tif"))

        img = img / 4096 * 255

        # cv2.imwrite(str(save_path / f"{img_num:05d}.tif"),
        #             img[crop_init[0]:crop_init[0] + 512, crop_init[1]:crop_init[1] + 512].astype(np.uint8))
        try:
            cv2.imwrite(str(save_path / f"6/{img_num:05d}.tif"),
                        gt6[crop_init[0]:crop_init[0] + 512, crop_init[1]:crop_init[1] + 512])
        except TypeError:
            pass
        try:
            cv2.imwrite(str(save_path / f"9/{img_num:05d}.tif"),
                        gt9[crop_init[0]:crop_init[0] + 512, crop_init[1]:crop_init[1] + 512])
        except TypeError:
            pass
        # cv2.imwrite(str(save_path / f"bg/{img_num:05d}.tif"),
        #             gtbg[crop_init[0]:crop_init[0] + 512, crop_init[1]:crop_init[1] + 512])
