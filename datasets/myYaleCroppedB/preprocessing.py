# https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full

import json
import numpy as np
from collections import defaultdict
import os
import re
import kagglehub
import random

# Download latest version
path = kagglehub.dataset_download("jensdhondt/extendedyaleb-cropped-full")


pat = re.compile(
    r"^yaleB(?P<s>\d+)_P(?P<p>\d+)A(?P<a>[+-]\d{3})E(?P<e>[+-]\d{2})\.pgm$")

groups = defaultdict(list)

for root, _, files in os.walk(path):
    for f in files:
        m = pat.match(f)
        if m:
            s = int(m.group("s"))
            p = int(m.group("p"))
            groups[(s, p)].append(os.path.join(root, f))

for (s, p), imgs in sorted(groups.items()):
    ref = next((x for x in imgs if "A+000E+00" in os.path.basename(x)), None)
    if ref is None:
        ref = sorted(imgs)[0]
    targets = [x for x in imgs if x != ref]
    # print(
    #     f"subj={s:02d} pose={p:02d} ref={os.path.basename(ref)} targets={len(targets)}")


def affine_from_3pts(P, Q):
    # P,Q: (3,2)
    P_aug = np.hstack([P, np.ones((3, 1), dtype=np.float32)])  # (3,3)
    A_T, _, _, _ = np.linalg.lstsq(P_aug, Q.astype(np.float32), rcond=None)
    return A_T.T.astype(np.float32)  # (2,3)


def make_one_sample(sigma, seed=123):

    rng = np.random.default_rng(seed)

    # ROI
    x1, y1, x2, y2 = 25, 40, 100, 125
    patch_w = x2 - x1 + 1
    patch_h = y2 - y1 + 1

    # template_affine (3 control points) in patch coords (1-based)
    template_affine = np.array([
        [1.0, 1.0],
        [patch_w, 1.0],
        [(patch_w) / 2.0, patch_h],
    ], dtype=np.float32)

    # target_affine (same triangle but in image coords)
    target_affine = np.array([
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x1) + (float(x2-x1)/2.0) - 0.5, float(y2)],
    ], dtype=np.float32)

    # random offsets: 3 points × (dx,dy)
    offsets = rng.normal(0.0, sigma, size=(3, 2)).astype(np.float32)

    # test_pts = GT points in image coords
    test_pts = target_affine + offsets

    # affine mapping patch->image
    A = affine_from_3pts(template_affine, test_pts)

    # warp img2 into patch
    # tmplt = cv2.warpAffine(
    #     img2, A, dsize=(patch_w, patch_h),
    #     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    #     borderMode=cv2.BORDER_REFLECT101
    # )

    # p_init: unperturbed translation (for ROI at (x1,y1))
    p_init = np.array([[0, 0, x1 - 1],
                       [0, 0, y1 - 1]], dtype=np.float32)

    return A, p_init, template_affine, test_pts


rows = []
for (s, p), imgs in sorted(groups.items()):
    ref = next((x for x in imgs if "A+000E+00" in os.path.basename(x)), None)
    if ref is None:
        ref = sorted(imgs)[0]

    targets = [x for x in imgs if x != ref]
    for t in targets:
        for sigma in range(1, 11):
            for _ in range(10):

                out = make_one_sample(
                    sigma=sigma, seed=random.randint(1, 5_000_000))

                rows.append({
                    "img": "/".join(ref.split("/")[-3:]),
                    "tmplt": "/".join(t.split("/")[-3:]),
                    "A": out[0].tolist(),
                    "p_init": out[1].tolist(),
                    "template_affine": out[2].tolist(),
                    "test_pts": out[3].tolist()
                })


with open(os.path.join("datasets/myYaleCroppedB", "ecc_dataset_index.json"), "w") as f:
    json.dump(rows, f)
