import torch

torch.set_default_dtype(torch.float32)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


def param_update(warp_in: torch.Tensor, delta_p: torch.Tensor, transform: str):
    B, C, _, _ = warp_in.shape
    dev, dt = warp_in.device, warp_in.dtype

    if transform == 'affine':
        warp_out = torch.zeros((B, C, 3, 3), dtype=dt, device=dev)
        warp_out[:, :, :2, :] = warp_in[:, :, :2, :] + delta_p

        warp_out[:, :, 2, 2] = 1.

    return warp_out


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.float64

    w = torch.tensor([[[[1, 2, 3], [4, 5, 6], [0, 0, 1]]]],
                     device=dev, dtype=dt)
    delt = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], device=dev, dtype=dt)

    out = param_update(w, delt, 'affine')

    print(out)
