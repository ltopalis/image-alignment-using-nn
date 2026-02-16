import torch
from .matlab_functions import fspecial, filter2, grad
from .spatial_interp import spatial_interp
from .make_pyramid import make_pyramid
from .next_level import next_level
from .param_update import param_update


def check_finite(name, x):
    if not torch.isfinite(x).all():
        with torch.no_grad():
            fin = torch.isfinite(x)
            print(f"[NON-FINITE] {name}: finite={fin.float().mean().item():.4f} "
                  f"min={torch.nan_to_num(x).min().item():.4e} "
                  f"max={torch.nan_to_num(x).max().item():.4e}")
        raise RuntimeError(f"Non-finite detected at {name}")


def ECC_PIXEL_IA(wimage: torch.Tensor, template: torch.Tensor, init: torch.Tensor, in_levels: int = 3, DEBUG: bool = False):
    dt = wimage.dtype
    dev = wimage.device
    eps = 1e-6
    B, C, _, _ = wimage.shape

    hh = fspecial('gaussian', 21, sigma=2, dtype=dt, device=dev)
    levels = in_levels
    type_ = 'gaussian'

    template = template / (filter2(hh, template) + eps)
    template = filter2(hh, template)
    ttemplate = torch.zeros_like(template)
    ttemplate[:, :, ::2, ::2] = template[:, :, ::2, ::2]
    tttemplate = filter2(hh, ttemplate)
    ttemplate = torch.zeros_like(template)
    ttemplate[:, :, 1::2, 1::2] = template[:, :, 1::2, 1::2]
    template = (filter2(hh, ttemplate) + tttemplate) / 2

    wimage = wimage / (filter2(hh, wimage) + eps)
    wimage = filter2(hh, wimage)
    wwimage = torch.zeros_like(wimage)
    wwimage[:, :, ::2, ::2] = wimage[:, :, ::2, ::2]
    wwwimage = filter2(hh, wwimage)
    wwimage = torch.zeros_like(wimage)
    wwimage[:, :, 1::2, 1::2] = wimage[:, :, 1::2, 1::2]
    wimage = (filter2(hh, wwimage) + wwwimage) / 2

    template_pyr = make_pyramid(template, levels, type_)
    wimage_pyr = make_pyramid(wimage, levels, type_)

# Initialization
    warp = torch.zeros((B, C, 3, 3), dtype=dt, device=dev)
    warp[:, :, :2, :] = init

    I6 = torch.eye(6, device=dev, dtype=dt).view(1, 1, 6, 6)

    for l in range(levels - 1):
        warp = next_level(warp, 'affine', True)

    for l in range(levels, 0, -1):
        template = template_pyr[l - 1]
        wimage = wimage_pyr[l - 1]
        p = warp.clone()
        p = p + torch.eye(3, device=dev, dtype=dt).view(1, 1, 3, 3)

        noi = ((levels - l) ** 2 + 1) * 30 // 2 ** l

        tmplt = template.clone().contiguous()
        tH, tW = tmplt.shape[2:]

        xvector = torch.arange(tW, device=dev, dtype=dt)
        yvector = torch.arange(tH, device=dev, dtype=dt)

        wrpd = spatial_interp(wimage, p, 'linear', 'affine', xvector, yvector)
        wrpd = torch.nan_to_num(wrpd, nan=1.0, posinf=1.0, neginf=1.0)
        wrpd = torch.where(wrpd == 0.0, 1.0, wrpd)
        if DEBUG:
            check_finite("wrpd", wrpd)

        ROI_x = torch.arange(tW, device=dev, dtype=dt)
        ROI_y = torch.arange(tH, device=dev, dtype=dt)
        O2, O1 = torch.meshgrid(ROI_y, ROI_x, indexing='ij')
        O1 = O1.view(1, 1, tH, tW).expand(B, C, tH, tW)
        O2 = O2.view(1, 1, tH, tW).expand(B, C, tH, tW)

        # Template
        g_tmplt_x, g_tmplt_y = grad(tmplt)
        g_tmplt_x1 = g_tmplt_x / (g_tmplt_x ** 2 + g_tmplt_y ** 2 + eps).sqrt()
        g_tmplt_x1 = torch.nan_to_num(
            g_tmplt_x1, nan=0.0, posinf=0.0, neginf=0.0)
        g_tmplt_y1 = g_tmplt_y / (g_tmplt_x ** 2 + g_tmplt_y ** 2 + eps).sqrt()
        g_tmplt_y1 = torch.nan_to_num(
            g_tmplt_y1, nan=0.0, posinf=0.0, neginf=0.0)
        g_tmplt_y = g_tmplt_y1
        g_tmplt_x = g_tmplt_x1

        g_tmplt_xx, g_tmplt_xy = grad(g_tmplt_x)
        g_tmplt_yx, g_tmplt_yy = grad(g_tmplt_y)

        # Hessian matrix
        H_t = torch.zeros((B, C, 2, 2, tW * tH), device=dev, dtype=dt)
        H_t[:, :, 0, 0, :] = g_tmplt_xx.transpose(-1, -2).reshape(B, C, -1)
        H_t[:, :, 0, 1, :] = g_tmplt_xy.transpose(-1, -2).reshape(B, C, -1)
        H_t[:, :, 1, 0, :] = g_tmplt_yx.transpose(-1, -2).reshape(B, C, -1)
        H_t[:, :, 1, 1, :] = g_tmplt_yy.transpose(-1, -2).reshape(B, C, -1)

# Positiveness
        Determ_t = (H_t[:, :, 0, 0, :] * H_t[:, :, 1, 1, :] -
                    H_t[:, :, 0, 1, :] * H_t[:, :, 1, 0, :])
        Trace_t = H_t[:, :, 0, 0, :] + H_t[:, :, 1, 1, :]

        with torch.no_grad():
            o1 = O1.transpose(-1, -2).reshape(B, C, -1)
            o2 = O2.transpose(-1, -2).reshape(B, C, -1)
            Xp1 = torch.stack([o1 + 1, o2 + 1, torch.ones_like(o1)], dim=-1)

        fitt = None

        for iter in range(noi):
            g_wrpd_x, g_wrpd_y = grad(wrpd)
            g_wrpd_xx, g_wrpd_xy = grad(g_wrpd_x)
            g_wrpd_yx, g_wrpd_yy = grad(g_wrpd_y)

            # Gradient Matrix
            gx = g_wrpd_x.transpose(-1, -2).reshape(B, C, -1)
            gy = g_wrpd_y.transpose(-1, -2).reshape(B, C, -1)

            gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
            gy = torch.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0)

            if DEBUG:
                check_finite("gx", gx)
                check_finite("gy", gy)

            Ng2 = gx * gx + gy * gy
            Ng2 = torch.clamp(Ng2, min=1e-12)
            N_G = torch.sqrt(Ng2)

            # Hessian Matrix
            # H = torch.zeros((B, C, 2, 2, tH * tW), device=dev, dtype=dt)
            # H[:, :, 0, 0, :] = g_wrpd_xx.transpose(-1, -2).reshape(
            #     B, C, -1)
            # H[:, :, 0, 1, :] = g_wrpd_xy.transpose(-1, -2).reshape(
            #     B, C, -1)
            # H[:, :, 1, 0, :] = g_wrpd_yx.transpose(-1, -2).reshape(
            #     B, C, -1)
            # H[:, :, 1, 1, :] = g_wrpd_yy.transpose(-1, -2).reshape(
            #     B, C, -1)
            Tdim = tH * tW
            H00 = g_wrpd_xx.transpose(-1, -2).reshape(B, C, Tdim)
            H01 = g_wrpd_xy.transpose(-1, -2).reshape(B, C, Tdim)
            H10 = g_wrpd_yx.transpose(-1, -2).reshape(B, C, Tdim)
            H11 = g_wrpd_yy.transpose(-1, -2).reshape(B, C, Tdim)

            H00 = torch.nan_to_num(H00, nan=0.0, posinf=0.0, neginf=0.0)
            H01 = torch.nan_to_num(H01, nan=0.0, posinf=0.0, neginf=0.0)
            H10 = torch.nan_to_num(H10, nan=0.0, posinf=0.0, neginf=0.0)
            H11 = torch.nan_to_num(H11, nan=0.0, posinf=0.0, neginf=0.0)
            # H = torch.stack([
            #     torch.stack([H00, H01], dim=2),
            #     torch.stack([H10, H11], dim=2)
            # ], dim=2)
            # H = torch.nan_to_num(H, nan=1.0, posinf=0.0, neginf=0.0)
            if DEBUG:
                check_finite("H00", H00)
                check_finite("H11", H11)
                check_finite("H10", H10)
                check_finite("H01", H01)

# # Positiveness
#             Determ = H[:, :, 0, 0, :] * \
#                 H[:, :, 1, 1, :] - H[:, :, 0, 1, :] ** 2
            Determ = H00 * H11 - H01 * H01
            Determ = torch.clamp(Determ, min=1e-12)
            # Trace = H[:, :, 0, 0, :] + H[:, :, 1, 1, :]
            Trace = H00 + H11
            Phi_1 = (Determ_t / (Determ + eps)).view(B, C,
                                                     tW, tH).transpose(-1, -2)
            Phi_2 = (Trace_t / (N_G + eps) - (Determ_t / (Determ + eps))
                     * Trace / (N_G + eps)).view(B, C, tW, tH).transpose(3, 2)
            Detector = ((Phi_2.abs() * (Phi_1 > 0)) < 1e-4)

            # mesh (Detector)
            # G0 = torch.zeros((B, C, 2, 2, tW * tH), device=dev, dtype=dt)
            # G0[:, :, 0, 0, :] = g_tmplt_x.transpose(-1, -2).reshape(
            #     B, C, -1)
            # G0[:, :, 0, 1, :] = g_tmplt_x.transpose(-1, -2).reshape(
            #     B, C, -1)
            # G0[:, :, 1, 0, :] = g_tmplt_y.transpose(-1, -2).reshape(
            #     B, C, -1)
            # G0[:, :, 1, 1, :] = g_tmplt_y.transpose(-1, -2).reshape(
            #     B, C, -1)
            # G00 = g_tmplt_x.transpose(-1, -2).reshape(B, C, Tdim)
            # G01 = g_tmplt_x.transpose(-1, -2).reshape(B, C, Tdim)
            # G10 = g_tmplt_y.transpose(-1, -2).reshape(B, C, Tdim)
            # G11 = g_tmplt_y.transpose(-1, -2).reshape(B, C, Tdim)
            # G0 = torch.stack([
            #     torch.stack([G00, G01], dim=2),
            #     torch.stack([G10, G11], dim=2)
            # ], dim=2)
            G00 = g_tmplt_x.transpose(-1, -2).reshape(B, C, Tdim)
            G10 = g_tmplt_y.transpose(-1, -2).reshape(B, C, Tdim)

            # S = (G0 * H).sum(dim=2)
            S0 = G00 * H00 + G10 * H10
            S1 = G00 * H01 + G10 * H11

            # k1 = S[:, :, 0, :].unsqueeze(-1).repeat(1, 1, 1, 3)
            # k2 = S[:, :, 1, :].unsqueeze(-1).repeat(1, 1, 1, 3)
            # S = torch.cat([k1, k2], dim=-1)

            # T_left = S[..., :3] * Xp1
            # T_right = S[..., 3:] * Xp1
            # T = torch.cat([T_left, T_right], dim=-1)

            T_left = S0.unsqueeze(-1) * Xp1
            T_right = S1.unsqueeze(-1) * Xp1
            T = torch.cat([T_left, T_right], dim=-1)

            b1 = g_tmplt_x.transpose(3, 2).reshape(
                B, C, -1) * g_wrpd_x.transpose(-1, -2).reshape(B, C, -1)
            b2 = g_tmplt_y.transpose(3, 2).reshape(
                B, C, -1) * g_wrpd_y.transpose(-1, -2).reshape(B, C, -1)
            b = b1 + b2

            c2 = gx * gx + gy * gy
            c2 = torch.clamp(c2, min=1e-12)
            c = torch.sqrt(c2)

            M = (c - b).abs().amax(dim=2)
            temp = torch.logical_and(M < 1e-6, b.amin(dim=2) >= 0)

            cond = Detector.to(torch.bool).transpose(-2, -1).reshape(B, C, -1)

            lambda_ = torch.where(cond, c, torch.zeros_like(c))

            T1 = torch.where(cond.unsqueeze(-1), T, torch.zeros_like(T))

            b1 = torch.where(cond, b, torch.zeros_like(b))

            # A = T1^T T1, rhs = T1^T y
            A = T1.transpose(-1, -2) @ T1
            rhs = T1.transpose(-1, -2) @ (lambda_ - b1).unsqueeze(-1)

            # symmetrize + damping (scale-aware)
            A = 0.5 * (A + A.transpose(-1, -2))
            diag_mean = A.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
            A = A + (1e-3 * diag_mean).unsqueeze(-1) * I6
            if DEBUG:
                check_finite("A", A)
                check_finite("rhs", rhs)

            try:
                L = torch.linalg.cholesky(A)
                Dp = torch.cholesky_solve(rhs, L)
            except torch.linalg.LinAlgError:
                try:
                    Dp = torch.linalg.solve(A, rhs)
                except torch.linalg.LinAlgError:
                    Dp = torch.linalg.pinv(A, rcond=1e-3) @ rhs

            Dp = torch.where(temp.unsqueeze(-1).unsqueeze(-1),
                             torch.zeros_like(Dp), Dp)

            Dp = Dp.reshape(B, C, 2, 3)
            Dp = torch.nan_to_num(Dp, nan=0.0, posinf=0.0, neginf=0.0)

            warp = param_update(warp, Dp, 'affine')
            warp = torch.nan_to_num(warp, nan=0.0, posinf=0.0, neginf=0.0)
            fitt = warp[:, :, :2, :]
            Dps = torch.zeros(B, C, 3, 3, dtype=dt, device=dev)
            Dps[:, :, :2, :] = Dp
            p = (p + Dps)

            wrpd = spatial_interp(wimage, p, 'linear',
                                  'affine', xvector, yvector)
            wrpd = torch.nan_to_num(wrpd, nan=1.0, posinf=1.0, neginf=1.0)
            wrpd = torch.where(wrpd == 0.0, 1.0, wrpd)
            if DEBUG:
                check_finite("Dp", Dp)
                check_finite("warp", warp)
                check_finite("wrpd", wrpd)

        warp = next_level(warp, 'affine', False)
        warp = torch.nan_to_num(warp, nan=0.0, posinf=0.0, neginf=0.0)

        mask = torch.zeros_like(warp)
        mask[:, :, -1, -1] = 1.0
        warp = warp * (1 - mask)
        warp = torch.nan_to_num(warp, nan=0.0, posinf=0.0, neginf=0.0)
        if DEBUG:
            check_finite("warp", warp)

    return fitt


if __name__ == "__main__" and False:
    import os
    import h5py
    import time
    import matplotlib.pyplot as plt
    from ComputePointError import ComputePointError
    from scipy.io import savemat

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    batch_size = 800
    folder = "res/solve/"

    os.makedirs(folder, exist_ok=True)

    for levels in range(1, 6):
        print(f"Running with {levels} levels")

        error_list = []
        start_t = time.perf_counter()
        for start in range(0, 10_000, batch_size):

            with h5py.File('dataset_matlab.hdf5', 'r') as f:
                img = torch.from_numpy(
                    f['img'][start:start+batch_size]).to(dtype=dtype, device=device)
                tmplt = torch.from_numpy(
                    f['tmplt'][start:start+batch_size]).to(dtype=dtype, device=device)
                p_init = torch.from_numpy(f['p_init'][start:start+batch_size]).to(
                    dtype=dtype, device=device)
                test_pts = torch.from_numpy(f['test_pts'][start:start+batch_size]).to(
                    dtype=dtype, device=device)
                template_affine = torch.from_numpy(
                    f['template_affine'][start:start+batch_size]).to(dtype=dtype, device=device)
                m = torch.from_numpy(
                    f['m'][start:start+batch_size]).to(dtype=dtype, device=device)

            img_batch = img.unsqueeze(1)
            tmplt_batch = tmplt.unsqueeze(1)
            p_init_batch = p_init.unsqueeze(1)
            test_pts_batch = test_pts
            template_affine_batch = template_affine
            m_batch = m

            fitt = ECC_PIXEL_IA(img_batch, tmplt_batch, p_init_batch, levels)
            error = ComputePointError(
                test_pts_batch, template_affine_batch, fitt[-1]['warp_p'].squeeze(1), m_batch)

            error_list.extend(error.cpu().numpy())

        print(
            f"\tTime taken for {levels} levels: {time.perf_counter() - start_t:.2f} seconds")
        savemat(os.path.join(folder, f"levels_{levels}_results.mat"), {
                'errors': error_list})

if __name__ == '__main__' and True:
    import h5py
    import time
    import matplotlib.pyplot as plt
    from ComputePointError import ComputePointError
    from scipy.io import savemat, loadmat

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    batch_size = 100
    start = 0

    with h5py.File('dataset_matlab.hdf5', 'r') as f:
        img = torch.from_numpy(
            f['img'][start:start+batch_size]).to(dtype=dtype, device=device)
        tmplt = torch.from_numpy(
            f['tmplt'][start:start+batch_size]).to(dtype=dtype, device=device)
        p_init = torch.from_numpy(f['p_init'][start:start+batch_size]).to(
            dtype=dtype, device=device)
        test_pts = torch.from_numpy(f['test_pts'][start:start+batch_size]).to(
            dtype=dtype, device=device)
        template_affine = torch.from_numpy(
            f['template_affine'][start:start+batch_size]).to(dtype=dtype, device=device)
        m = torch.from_numpy(
            f['m'][start:start+batch_size]).to(dtype=dtype, device=device)

    img_batch = img.unsqueeze(1)
    tmplt_batch = tmplt.unsqueeze(1)
    p_init_batch = p_init.unsqueeze(1)
    test_pts_batch = test_pts
    template_affine_batch = template_affine
    m_batch = m

    fitt = ECC_PIXEL_IA(img_batch, tmplt_batch, p_init_batch, 4)
    error = ComputePointError(
        test_pts_batch, template_affine_batch, fitt.squeeze(), m_batch)

    print(error)

    # a = loadmat(
    #     '/home/ltopalis/Desktop/image-alignment-using-nn (Copy)/levels_4_results.mat')['errors']

    # print(a.squeeze()[:batch_size])
