import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


def make_pyramid(img: torch.Tensor, levels: int, type_: str) -> list:
    gpyramid = [img]
    # lpyramid = []
    padsize = 4
    sigma = (1.0, 1.0)

    for i in range(2, levels + 1):
        prev = gpyramid[-1]

        ks = (2 ** (levels - i + 4)) + 1

        temp = F.pad(prev, (padsize, padsize, padsize, padsize),
                     mode='constant', value=1.0)

        new = gaussian_blur2d(
            temp, (ks, ks), sigma, border_type='constant')

        r0 = padsize
        r1 = -padsize if padsize > 0 else None
        g_dec = new[:, :, r0:r1:2, r0:r1:2]

        # g_blur_same = new[:, :, r0:r1, r0:r1]
        # lap = prev - g_blur_same

        gpyramid.append(g_dec)
        # lpyramid.append(lap)

    return gpyramid  # if type_ == 'gaussian' else lpyramid


if __name__ == '__main__':
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.float64

    tmplts = loadmat('myYaleCropped.mat')['tmplts']
    pyramid_mat = loadmat(
        'pixel_ecc_affine/test_files/make_pyramid/gaussian_pyramid_7.mat')['p'][0]

    img = torch.from_numpy(tmplts[:, :, 0]).to(device=dev, dtype=dt)
    img = img.unsqueeze(0).unsqueeze(0)

    pyr = make_pyramid(img, 7, 'gaussian')

    for p1, p2 in zip(pyr, pyramid_mat):
        p2 = torch.from_numpy(p2).to(device=dev, dtype=dt)
        a = (((p1 - p2).abs()) < 1e-12).sum()
        print((a == p2.numel()).item())
