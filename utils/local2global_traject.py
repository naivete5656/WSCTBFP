from pathlib import Path
import numpy as np
from utils import optimum

if __name__ == "__main__":
    mode = {1: "", 2: "_wo_reject"}
    mode = mode[2]
    time_lates = [1, 5, 9]
    for time_late in time_lates:
        for seq in [13]:
            save_path = Path(f"/home/kazuya/main/association_accuracy/sequ{seq}")
            save_path.mkdir(parents=True, exist_ok=True)
            root_path = Path(f"../output/association{mode}/C2C12_9_{time_late}/sequ{seq}/00000")

            # traject_init
            traject_1_path = root_path.joinpath("traject_1.txt")
            traject_2_path = root_path.joinpath("traject_2.txt")
            traject_1 = np.loadtxt(str(traject_1_path), delimiter=",", skiprows=1)
            traject_2 = np.loadtxt(str(traject_2_path), delimiter=",", skiprows=1)

            traject_1 = traject_1[traject_1[:, 3] == 1][:, [0, 1, 2]]
            for cell_id in range(traject_1.shape[0]):
                cell_tn_ind = traject_1[cell_id][2]
                traject_1[cell_id][2] = cell_id + 100
                traject_2[int(cell_tn_ind)][2] = cell_id + 100

            max_cell_id = cell_id + 100 + 1

            traject = np.concatenate([traject_1, np.full((traject_1.shape[0], 1), 0)], 1)

            new_cell_inds = np.where(traject_2[:, 3] != 1)
            for cell_ind in new_cell_inds[0]:
                traject_2[cell_ind][2] = max_cell_id
                max_cell_id += 1

            traject_temp = traject_2[:, [0, 1, 2]]
            traject_temp = np.concatenate([traject_temp, np.full((traject_temp.shape[0], 1), time_late)], 1)
            # [x, y, id, frame]
            traject = np.concatenate([traject, traject_temp])

            for frame in range(time_late, 200 - time_late, time_late):
                # extract match cell and id
                current_pos = traject[traject[:, 3] == frame]

                frame_path = root_path.parent.joinpath(f"{frame:05d}")
                traject_1_path = frame_path.joinpath("traject_1.txt")
                traject_2_path = frame_path.joinpath("traject_2.txt")
                traject_1 = np.loadtxt(str(traject_1_path), delimiter=",", skiprows=1)
                traject_2 = np.loadtxt(str(traject_2_path), delimiter=",", skiprows=1)

                traject_1 = traject_1[traject_1[:, 3] == 1]
                cur2tra = optimum(current_pos, traject_1, 10)

                traject_temp = []
                for ct in cur2tra.astype(int):
                    cell_id_global = current_pos[ct[0]][2]

                    cell_id_local = int(traject_1[ct[1]][2])
                    traject_2[cell_id_local][2] = cell_id_global

                new_cell_inds = np.where(traject_2[:, 3] == 0)
                for cell_ind in new_cell_inds[0]:
                    traject_2[cell_ind][2] = max_cell_id
                    max_cell_id += 1

                traject_temp = traject_2[traject_2[:, 3] != 2][:, [0, 1, 2]]
                traject_temp = np.concatenate([traject_temp, np.full((traject_temp.shape[0], 1), frame + time_late)], 1)
                traject = np.concatenate([traject, traject_temp])

            np.savetxt(str(save_path / f"BFres{mode}_C2C12_{time_late}.txt"), traject[:, [0, 1, 3, 2]], fmt="%d")
