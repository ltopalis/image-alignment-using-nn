import torch

torch.set_default_dtype(torch.float32)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


def next_level(warp_in: torch.Tensor, transform: str, high_flag: bool):
    warp = warp_in.clone()
    dev, dt = warp.device, warp.dtype
    B, C, _, _ = warp.shape

    if high_flag:
        if transform == 'affine':
            warp[:, :, :2, -1] /= 2

    else:
        if transform == 'affine':
            warp[:, :, :2, -1] *= 2

    return warp


if __name__ == '__main__':

    from param_update import param_update

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.float64

    w = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], device=dev, dtype=dt)
    delt = torch.tensor([[[1, 2, 3, 4, 5, 6]]], device=dev, dtype=dt)

    out = param_update(w, delt, 'affine')

    a = next_level(out, 'affine', True)
    b = next_level(out, 'affine', False)

    print(a)
    print(b)
