import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(frame, cell_id_t, root_path, inverse):
    asso_img1 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_1.png")), 0
    )
    asso_img2 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_2.png")), 0
    )
    if inverse:
        return asso_img1
    else:
        return asso_img2


def visuarize_img(pre, frame, cell_id_t, root_path, img1, img2, pred1, pred2):

    inp_img1 = cv2.imread(
        str(root_path.joinpath(f"ori/{frame:04d}_{cell_id_t + 1:04d}_1.png")), 0
    )
    inp_img2 = cv2.imread(
        str(root_path.joinpath(f"ori/{frame:04d}_{cell_id_t + 1:04d}_2.png")), 0
    )
    asso_img1 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_1.png")), 0
    )
    asso_img2 = cv2.imread(
        str(root_path.joinpath(f"pred/{frame:04d}_{cell_id_t + 1:04d}_2.png")), 0
    )
    patch_window = (pre[1:3] - 50).clip(0, 512 - 100).astype(int)[::-1]

    img = np.hstack(
        [
            img1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            img2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    pred = np.hstack(
        [
            pred1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            pred2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )

    inp_img = np.hstack(
        [
            inp_img1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            inp_img2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )
    asso_img = np.hstack(
        [
            asso_img1[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
            asso_img2[
                patch_window[0] : patch_window[0] + 100,
                patch_window[1] : patch_window[1] + 100,
            ],
        ]
    )
    visual_img = np.vstack([img, inp_img])
    lik_img = np.vstack([pred, asso_img])
    visual_img = np.hstack([visual_img, lik_img])
    return visual_img


def make_3d_association_res(frame, tps, error):
    # create a 21 x 21 vertex mesh
    xx, yy = np.meshgrid(range(512), range(512))

    # create vertices for a rotated mesh (3D rotation matrix)
    X = xx
    Y = yy

    # create some dummy data (20 x 20) for the image
    data = cv2.imread(
        "/home/kazuya/main/correlation_test/output/detection/UNet3_10/sequ18/ori/0000_1.tif",
        0,
    )
    data2 = cv2.imread(
        "/home/kazuya/main/correlation_test/output/detection/UNet3_10/sequ18/ori/0000_2.tif",
        0,
    )
    # create the figure
    fig = plt.figure()

    # show the 3D rotated projection
    ax2 = fig.add_subplot(111, projection="3d")
    cset = ax2.contourf(X, Y, data, 100, zdir="z", offset=0, cmap=cm.gray)
    cset = ax2.contourf(X, Y, data2, 100, zdir="z", offset=512, cmap=cm.gray, alpha=0.2)

    for tp in tps:
        x, y = tp[2:4]
        u, v = tp[4:6] - tp[2:4]
        ax2.quiver(
            x,
            y,
            0,
            u,
            v,
            512,
            length=1,
            color="red",
            arrow_length_ratio=0.1,
            pickradius=100,
        )
    for er in error:
        x, y = er[2:4]
        u, v = er[4:6] - er[2:4]
        ax2.quiver(
            x,
            y,
            0,
            u,
            v,
            512,
            length=1,
            color="b",
            arrow_length_ratio=0.1,
            pickradius=100,
        )
    ax2.set_zlim((0.0, 512.0))
    plt.savefig(f"./output/association/{frame:05d}/test.png")
