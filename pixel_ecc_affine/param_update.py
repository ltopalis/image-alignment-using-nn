import torch


def param_update(warp_in: torch.Tensor, delta_p: torch.Tensor, transform: str):
    B, C, _, _ = warp_in.shape
    dev, dt = warp_in.device, warp_in.dtype

    if transform == 'affine':
        warp_out = warp_in.clone()
        warp_out[:, :, :2, :] = warp_in[:, :, :2, :] + delta_p

        warp_out[:, :, 2, 2] = 1.
        warp_out[:, :, 2, 1] = 0.
        warp_out[:, :, 2, 0] = 0.

    return warp_out


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = torch.float64

    w = torch.tensor([[[[1, 2, 3], [4, 5, 6], [0, 0, 1]]]],
                     device=dev, dtype=dt)
    delt = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], device=dev, dtype=dt)

    out = param_update(w, delt, 'affine')

    print(out)
