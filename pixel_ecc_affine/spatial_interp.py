import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=12)
torch.use_deterministic_algorithms(True)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


@torch.no_grad()
def spatial_interp(
    img: torch.Tensor,
    warp: torch.Tensor,
    method: str,
    transform: str,
    nx,
    ny
) -> torch.Tensor:
    dev, dt = img.device, img.dtype
    B, C, H, W = img.shape

    if transform in ('affine'):
        if warp.shape[-2] == 2:
            A = torch.zeros((B, C, 3, 3), dtype=dt, device=dev)
            A[:, :, :2, :] = warp.clone()
            w = A.clone()
        else:
            w = warp.clone()

    xx, yy = torch.meshgrid(nx, ny, indexing='ij')
    a = xx.permute(1, 0).reshape(-1)
    b = yy.permute(1, 0).reshape(-1)
    o = torch.ones(len(b), device=dev, dtype=dt)
    xy = torch.stack((a, b, o), dim=0)

    A = w.clone()
    A[:, :, 2, 2] = 1.

    xy_prime = A @ xy
    xy_prime = xy_prime[:, :, :2, :]

    ny_len = ny.numel() if torch.is_tensor(ny) else len(ny)
    nx_len = nx.numel() if torch.is_tensor(nx) else len(nx)
    N = ny_len * nx_len

    xq = xy_prime[:, :, 0, :].reshape(B * C, ny_len, nx_len)
    yq = xy_prime[:, :, 1, :].reshape(B * C, ny_len, nx_len)

    def to_norm_matlab(coord, size):
        if size == 1:
            return torch.zeros_like(coord)
        return ((coord + 0.5) / (size)) * 2 - 1

    grid = torch.stack([to_norm_matlab(xq, W),
                        to_norm_matlab(yq, H)], dim=-1)

    m = method.lower()
    mode = "bilinear" if m in ("linear", "bilinear") else \
           ("bicubic" if m in ("cubic", "bicubic") else "nearest")

    img_bc = img.reshape(B * C, 1, H, W).to(dt)
    out_bc = F.grid_sample(
        img_bc, grid, mode=mode, padding_mode="zeros", align_corners=False
    )
    oob = (xq < -0.5) | (xq > (W - 0.5)) | (yq < -0.5) | (yq > (H - 0.5))
    oob = oob.unsqueeze(1)
    out_bc = out_bc.masked_fill(oob, 0.0)

    out = out_bc.view(B, C, ny_len, nx_len)

    return out


if __name__ == '__main__':
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.float32

    p = loadmat('pixel_ecc_affine/test_files/spatial_interp/p.mat')['p']
    wimage = loadmat(
        'pixel_ecc_affine/test_files/spatial_interp/wimage.mat')['wimage']
    xvector = loadmat(
        'pixel_ecc_affine/test_files/spatial_interp/xvector.mat')['xvector']
    yvector = loadmat(
        'pixel_ecc_affine/test_files/spatial_interp/yvector.mat')['yvector']
    wrpd = loadmat(
        'pixel_ecc_affine/test_files/spatial_interp/wrpd.mat')['wrpd']

    p = torch.from_numpy(p).to(device=dev, dtype=dt)
    wimage = torch.from_numpy(wimage).to(device=dev, dtype=dt)
    wrpd = torch.from_numpy(wrpd).to(device=dev, dtype=dt)
    xvector = torch.from_numpy(xvector).to(device=dev, dtype=dt).squeeze()
    yvector = torch.from_numpy(yvector).to(device=dev, dtype=dt).squeeze()

    p = p.unsqueeze(0).unsqueeze(0)
    wimage = wimage.unsqueeze(0).unsqueeze(0)
    wrpd = wrpd.unsqueeze(0).unsqueeze(0)

    out = spatial_interp(wimage, p, 'linear', 'affine', xvector, yvector)

    print((((out - wrpd).abs() < 1e-6).sum() == out.numel()).item())
    print((out - wrpd).abs().min().item(), (out - wrpd).abs().max().item())

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(out.squeeze().detach().cpu(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.imshow(wrpd.squeeze().detach().cpu(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.imshow((wrpd-out).abs().squeeze().detach().cpu(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig('spa.png', dpi=200)

    # tmplts = loadmat('myYaleCropped.mat')['tmplts']
    # warped = loadmat(
    #     'pixel_ecc_affine/test_files/spatial_interp/warped_img.mat')['warped']
    # warped = torch.from_numpy(warped).to(
    #     device=dev, dtype=dt).unsqueeze(0).unsqueeze(0)

    # p = loadmat('pixel_ecc_affine/test_files/spatial_interp/p.mat')['p']
    # p = torch.from_numpy(p).to(device=dev, dtype=dt).unsqueeze(0).unsqueeze(0)

    # img = torch.from_numpy(tmplts[:, :, 0]).to(device=dev, dtype=dt)
    # img = img.unsqueeze(0).unsqueeze(0)

    # w = torch.tensor([[1, 0, 20], [0, 1, 60]], device=dev,
    #                  dtype=dt).unsqueeze(0).unsqueeze(0)

    # yv = torch.arange(80-1, 200, device=dev, dtype=dt)
    # xv = torch.arange(50-1, 150, device=dev, dtype=dt)

    # wimage = spatial_interp(img, w, 'linear', 'affine', xv, yv)

    # print((((wimage - warped).abs() < 1e-11).sum() == wimage.numel()).item())

    # plt.figure()
    # plt.imshow(wimage.squeeze().detach().cpu(), cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('plot.png', dpi=200)
