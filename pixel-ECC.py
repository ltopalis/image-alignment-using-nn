import torch

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def params_to_homography(params):
    """
    params: [B, 8]
    output: [B, 3, 3]
    """

    B = params.shape[0]
    H = torch.cat([params, torch.ones((B, 1), device=device)], dim=1)

    return H.view(B, 3, 3)


def wrap_features(features, H, out_size):
    pass
