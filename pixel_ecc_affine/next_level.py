import torch


def next_level(warp_in: torch.Tensor, transform: str, high_flag: bool):
    if transform != 'affine':
        raise NotImplementedError(f"Transform {transform} not implemented")

    warp = warp_in.clone()

    scale = 0.5 if high_flag else 2.
    warp[:, :, :2, -1] *= scale

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
