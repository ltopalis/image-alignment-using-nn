import cv2
import torch

import numpy as np
import torch.nn.functional as F

from lossFunctions import lossFunction

device = "cuda" if torch.cuda.is_available() else "cpu"


def params_to_homography(params: torch):
    """
    params: [B, 8]
    output: [B, 3, 3]
    """

    B = params.shape[0]
    H = torch.cat([params, torch.ones((B, 1), device=device)], dim=1)

    return H.view(B, 3, 3)


def wrap_features(features: torch,
                  H: torch,
                  out_size: tuple[int, int]) -> torch:
    B, C, Height, Width = features.shape
    Ho, Wo = out_size
    device = features.device

    ys = torch.arange(0, Ho, device=device)
    xs = torch.arange(0, Wo, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Ho, Wo]
    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, ones], dim=0)     # [3, Ho, Wo]
    coords = coords.view(3, -1)                   # [3, N], N = Ho * Wo
    coords = coords.unsqueeze(0).repeat(B, 1, 1)  # [B, 3, N]
    coords = coords.to(torch.float32)

    H_inv = torch.inverse(H)  # [B, 3, 3]
    coords_src = H_inv.bmm(coords)

    xs_src = coords_src[:, 0, :] / coords_src[:, 2, :]
    ys_src = coords_src[:, 1, :] / coords_src[:, 2, :]

    xs_norm = (xs_src / (Width - 1)) * 2 - 1
    ys_norm = (ys_src / (Height - 1)) * 2 - 1

    grid = torch.stack([xs_norm, ys_norm], dim=-1)
    grid = grid.view(B, Ho, Wo, 2)

    warped = F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    )

    return warped


def pixel_ecc(F_T: torch, F_I: torch, initial_motion: torch):
    pass


if __name__ == "__main__":
    ftr = torch.randint(0, 255, (8, 1, 24, 24)).float()
    H = params_to_homography(torch.rand(8, 8))

    a = lossFunction(ftr[0, 0, :, :], ftr[0, 0, :, :], "l1")
    print(a)

    a = lossFunction(ftr[0, 0, :, :], ftr[0, 0, :, :], "l2")
    print(a)

    a = lossFunction(ftr[0, 0, :, :], ftr[0, 0, :, :], "linf")
    print(a)

    img = wrap_features(ftr, H, (24, 24))
