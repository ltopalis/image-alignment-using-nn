import torch

torch.set_default_dtype(torch.float64)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


def ComputePointError(test_pts: torch.Tensor,
                      template_affine: torch.Tensor,
                      warp_p: torch.Tensor,
                      m: torch.Tensor):
    dev, dt = test_pts.device, test_pts.dtype
    B = test_pts.shape[0]

    warp_p = warp_p - m
    M = torch.zeros((B, 3, 3), device=dev, dtype=dt)
    M[:, :2, :] = warp_p
    M[:,  2, 2] = 1
    M[:,  0, 0] += 1
    M[:,  1, 1] += 1

    tmplt_affine = torch.ones((B, 3, 3), device=dev, dtype=dt)
    tmplt_affine[:, :2, :] = template_affine
    iteration_pts = M @ tmplt_affine

    diff_pts = test_pts - iteration_pts[:, :2, :]
    diff_pts = diff_pts[:, :, :2]
    rms_pt_error = torch.sqrt(torch.mean(diff_pts.permute(
        0, 2, 1).reshape(B, -1) ** 2, dim=1))

    return rms_pt_error
