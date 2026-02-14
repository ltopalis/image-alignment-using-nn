import torch
import torch.nn.functional as F


# --------------------- fspecial ---------------------


def fspecial(kind: str,
             size,                     # int or (kH, kW)
             *,
             sigma=None,               # for 'gaussian' / 'log'
             laplacian_ks: int = 4,    # 4 or 8 neighbors
             device=None,
             dtype=torch.float32):
    """
    Returns kernel shaped (1, 1, kH, kW) on the requested device/dtype.
    Supported kinds: 'gaussian', 'log', 'average', 'laplacian', 'sobelx', 'sobely'
    """
    if isinstance(size, int):
        kH = kW = size
    else:
        kH, kW = size
    assert kH > 0 and kW > 0 and kH % 2 == 1 and kW % 2 == 1, "Use odd kernel sizes."

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kind = kind.lower()

    if kind == "average":
        k = torch.ones((kH, kW), device=device, dtype=dtype) / (kH * kW)

    elif kind == "gaussian":
        if sigma is None:
            # MATLAB-like heuristic if not provided
            sigma = 0.3*((kH+kW)/2 - 1) + 0.8
        sy = sx = float(sigma) if not isinstance(
            sigma, (tuple, list)) else sigma
        y = torch.arange(kH, device=device, dtype=dtype) - (kH - 1)/2
        x = torch.arange(kW, device=device, dtype=dtype) - (kW - 1)/2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        k = torch.exp(-(xx**2)/(2*sx**2) - (yy**2)/(2*sy**2))
        k = k / k.sum()

    elif kind in ("log", "gaussian_log", "logg"):
        assert sigma is not None, "Provide sigma for LoG."
        sy = sx = float(sigma) if not isinstance(
            sigma, (tuple, list)) else sigma
        y = torch.arange(kH, device=device, dtype=dtype) - (kH - 1)/2
        x = torch.arange(kW, device=device, dtype=dtype) - (kW - 1)/2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r2 = xx**2 + yy**2
        g = torch.exp(-r2 / (2*sx**2))
        k = ((r2 - 2*sx**2) / (sx**4)) * g
        k -= k.mean()              # zero mean
        k /= (k.abs().sum() + 1e-12)

    elif kind == "laplacian":
        assert (kH, kW) == (3, 3), "Laplacian kernel is 3x3."
        if laplacian_ks == 4:
            k = torch.tensor([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], device=device, dtype=dtype)
        elif laplacian_ks == 8:
            k = torch.tensor([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]], device=device, dtype=dtype)
        else:
            raise ValueError("laplacian_ks must be 4 or 8")

    elif kind == "sobelx":
        assert (kH, kW) == (3, 3), "Sobel is 3x3."
        k = torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], device=device, dtype=dtype)

    elif kind == "sobely":
        assert (kH, kW) == (3, 3)
        k = torch.tensor([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], device=device, dtype=dtype)

    else:
        raise ValueError(f"Unsupported kind: {kind}")

    return k.unsqueeze(0).unsqueeze(0)  # (1,1,kH,kW)


# --------------------- filter2 ---------------------
def filter2(h: torch.Tensor,
            x: torch.Tensor,
            *,
            mode: str = "corr",      # 'corr' (MATLAB filter2) or 'conv'
            padding: str = "same",   # 'same' | 'valid' | int | (padW, padH)
            channelwise: bool = True):
    """
    MATLAB-like 2D filtering for tensors.
    h: (1,1,kH,kW) or (kH,kW)
    x: (B,C,H,W)
    Returns: y with shape (B,C,H,W) for 'same', otherwise appropriate size.
    - mode='corr' matches MATLAB filter2 (no kernel flip).
    - mode='conv' flips kernel 180°, like conv2.
    """
    assert x.ndim == 4
    B, C, H, W = x.shape
    if h.ndim == 2:
        h = h.unsqueeze(0).unsqueeze(0)
    assert h.ndim == 4 and h.shape[:2] == (1, 1)

    kH, kW = h.shape[-2], h.shape[-1]
    if mode == "conv":
        h_use = torch.flip(h, dims=(-2, -1))
    elif mode == "corr":
        h_use = h
    else:
        raise ValueError("mode must be 'corr' or 'conv'.")

    # groups = C to apply the same kernel independently per channel
    h_rep = h_use.repeat(C, 1, 1, 1)  # (C,1,kH,kW)

    # padding
    if padding == "same":
        pad = (kW // 2, kH // 2)
    elif padding == "valid":
        pad = (0, 0)
    elif isinstance(padding, int):
        pad = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad = padding
    else:
        raise ValueError("Invalid padding.")

    y = F.conv2d(x, h_rep, bias=None, stride=1, padding=pad, groups=C)
    return y


def grad(img):
    # img: (B,C,H,W)
    gx = torch.zeros_like(img)  # d/dx (στήλες, dim=3)
    gx[..., 1:-1] = (img[..., 2:] - img[..., :-2]) * 0.5
    gx[..., 0] = img[..., 1] - img[..., 0]
    gx[..., -1] = img[..., -1] - img[..., -2]

    gy = torch.zeros_like(img)  # d/dy (γραμμές, dim=2)
    gy[:, :, 1:-1, :] = (img[:, :, 2:, :] - img[:, :, :-2, :]) * 0.5
    gy[:, :, 0, :] = img[:, :, 1, :] - img[:, :, 0, :]
    gy[:, :, -1, :] = img[:, :, -1, :] - img[:, :, -2, :]

    return gx, gy


if __name__ == '__main__':
    from scipy.io import loadmat

    dt = torch.float64
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    h = fspecial('gaussian', 3, sigma=.5, dtype=dt, device=dev)
    hh = fspecial('gaussian', 21, sigma=2, dtype=dt, device=dev)

    h_ = loadmat('pixel_ecc_affine/test_files/matlab_functions/h.mat')['h']
    hh_ = loadmat('pixel_ecc_affine/test_files/matlab_functions/hh.mat')['hh']

    h_ = torch.from_numpy(h_).to(dev, dt)
    hh_ = torch.from_numpy(hh_).to(dev, dt)

    print(f'h = {(((h - h_).abs() < 1e-12).sum() == 3 * 3).item()}')
    print(f'hh = {(((hh - hh_).abs() < 1e-12).sum() == 21 * 21).item()}')

    tmplts = loadmat('myYaleCropped.mat')['tmplts']

    template = tmplts[:, :, 0]
    template = torch.from_numpy(template).unsqueeze(
        0).unsqueeze(0).to(device=dev, dtype=dt)
    w = torch.tensor([[1, 0, 20], [0, 1, 60]], dtype=dt, device=dev)
    w = w.unsqueeze(0).unsqueeze(0)

    yv = torch.arange(80-1, 200, device=dev, dtype=dt)
    xv = torch.arange(50-1, 150, device=dev, dtype=dt)

    a = filter2(h, template).squeeze()
    b = filter2(hh, template).squeeze()

    temp_h = loadmat(
        'pixel_ecc_affine/test_files/matlab_functions/temp_h.mat')['t']
    temp_hh = loadmat(
        'pixel_ecc_affine/test_files/matlab_functions/temp_hh.mat')['t']

    temp_h = torch.from_numpy(temp_h).to(dev, dt)
    temp_hh = torch.from_numpy(temp_hh).to(dev, dt)

    print(f'h = {(((temp_h - a).abs() < 1e-12).sum() == 298 * 250).item()}')
    print(f'hh = {(((temp_hh - b).abs() < 1e-12).sum() == 298 * 250).item()}')
