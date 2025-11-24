import numpy as np
import torch


def compute_initial_motion_(img1, img2, levels=4):

    img1 = img1.squeeze()
    img2 = img2.squeeze()

    s = 2**levels
    img1 = img1[::s, ::s]
    img2 = img2[::s, ::s]

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    assert h1 >= h2 and w1 >= w2, "Template must be smaller."

    gx1, gy1 = np.gradient(img1)
    gx2, gy2 = np.gradient(img2)

    g1 = np.sqrt(gx1**2 + gy1**2)
    g2 = np.sqrt(gx2**2 + gy2**2)

    g2_norm = np.sqrt((g2**2).sum())

    best = (-np.inf, 0, 0)

    for y in range(h1 - h2 + 1):
        for x in range(w1 - w2 + 1):
            roi = g1[y:y+h2, x:x+w2]

            cor = np.sum(roi * g2)
            print(cor)
            nor = np.sqrt(np.sum(roi**2)) * g2_norm

            if nor > 1e-8:
                ncc = cor / nor
            else:
                ncc = -np.inf

            if ncc > best[0]:
                best = (ncc, y, x)

    return best[1], best[2]


def compute_initial_motion(img1, img2, levels=4, eps=1e-8):

    if img1.ndim == 2:
        img1 = img1.unsqueeze(0)
    elif img1.ndim == 4:
        img1 = img1.squeeze(1)
    if img2.ndim == 2:
        img2 = img2.unsqueeze(0)
    elif img2.ndim == 4:
        img2 = img2.squeeze(1)

    device, dtype = img1.device, img1.dtype
    B, H1, W1 = img1.shape
    _, H2, W2 = img2.shape
    assert H1 >= H2 and W1 >= W2, "Template must be smaller."

    s = 2 ** levels
    img1 = img1[:, ::s, ::s]
    img2 = img2[:, ::s, ::s]
    _, H1s, W1s = img1.shape
    _, H2s, W2s = img2.shape

    gx1, gy1 = torch.gradient(img1, dim=(1, 2))
    gx2, gy2 = torch.gradient(img2, dim=(1, 2))

    g1 = torch.sqrt(gx1**2 + gy1**2)
    g2 = torch.sqrt(gx2**2 + gy2**2)

    g2_norm = torch.sqrt((g2**2).sum(dim=(1, 2), keepdim=True))
    g2_normalized = g2 / (g2_norm + 1e-8)

    best_val = torch.full((B,), -float("inf"), device=device, dtype=dtype)
    best_y = torch.zeros((B,), device=device, dtype=torch.long)
    best_x = torch.zeros((B,), device=device, dtype=torch.long)

    for y in range(H1s - H2s + 1):
        for x in range(W1s - W2s + 1):
            roi = g1[:, y:y + H2s, x:x + W2s]
            cor = (roi * g2_normalized).sum(dim=(1, 2))
            roi_norm = torch.sqrt((roi ** 2).sum(dim=(1, 2)))
            # nor = roi_norm * g2_norm
            nor = torch.sqrt((roi**2).sum(dim=(1, 2)))

            valid = nor > eps
            ncc = torch.full_like(cor, -float("inf"))
            ncc[valid] = cor[valid] / (nor[valid] + 1e-8)

            mask = ncc > best_val
            best_val[mask] = ncc[mask]
            best_y[mask] = y
            best_x[mask] = x

    init_motion = torch.zeros((B, 2, 3), device=device, dtype=dtype)
    init_motion[:, 0, 2] = best_x
    init_motion[:, 1, 2] = best_y

    return init_motion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from skimage.transform import AffineTransform, warp

    dt = torch.float64
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    m = loadmat('data/myYaleCropped.mat')

    img_ = m['tmplts'][:, :, 0]
    tmplt = m['example_imgs'][:, :, 0]

    t = AffineTransform(translation=(5, 50), scale=1.0, rotation=0.0)
    warped = warp(img_, t, preserve_range=True)[:50, :80]
    # warped = warped * 0.8 - 90

    img = torch.from_numpy(img_).to(
        dtype=dt, device=dev).unsqueeze(0).unsqueeze(0)
    img = torch.cat([img, img], dim=0)
    warped = torch.from_numpy(warped).to(
        dtype=dt, device=dev).unsqueeze(0).unsqueeze(0)
    t = AffineTransform(translation=(40, 64), scale=1.0, rotation=0.0)
    warped2 = warp(img_, t, preserve_range=True)[:50, :80]
    warped2 = torch.from_numpy(warped2).to(
        dtype=dt, device=dev).unsqueeze(0).unsqueeze(0)
    warped = torch.cat([warped, warped2], dim=0)

    # plt.figure()

    # plt.subplot(1, 2, 1)
    # plt.imshow(img.squeeze().detach().cpu(), cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.tight_layout()

    # plt.subplot(1, 2, 2)
    # plt.imshow(warped.squeeze().detach().cpu(), cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # a = loadmat('data/s.mat')

    # warped = a['img']
    # img = a['tmplt']

    a = compute_initial_motion(img, warped, levels=4)

    print(a)
