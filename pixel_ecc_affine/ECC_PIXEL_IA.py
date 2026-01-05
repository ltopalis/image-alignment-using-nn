import torch
from .matlab_functions import fspecial, filter2, grad
from .spatial_interp import spatial_interp
from .make_pyramid import make_pyramid
from .next_level import next_level
from .param_update import param_update
from scipy.io import savemat, loadmat


torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=12)
# torch.use_deterministic_algorithms(True)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


def clamp_det(x, eps):
    return torch.sign(x) * torch.clamp(x.abs(), min=eps)


def ECC_PIXEL_IA(wimage: torch.Tensor, template: torch.Tensor, init: torch.Tensor, in_levels: int = 3):
    dt = wimage.dtype
    dev = wimage.device
    eps = torch.finfo(dt).eps
    eps_num = torch.tensor(1e-6, dtype=dt, device=dev)
    B, C, wH, wW = wimage.shape
    B, C, tH, tW = template.shape

    a = dict()

    h = fspecial('gaussian', 3, sigma=.5, dtype=dt, device=dev)
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
    qtemplate_pyr = make_pyramid(
        template / filter2(h, template), levels, type_)
    qwimage_pyr = make_pyramid(
        wimage / filter2(h, wimage), levels, type_)

# Initialization
    verbose = 0

    warp = torch.zeros((B, C, 3, 3), dtype=dt, device=dev)
    warp[:, :, :2, :] = init.clone()

    for l in range(levels - 1):
        warp = next_level(warp, 'affine', True)

    for l in range(levels, 0, -1):
        template = template_pyr[l - 1].clone()
        wimage = wimage_pyr[l - 1].clone()
        qtemplate = qtemplate_pyr[l - 1].clone()
        qwimage = qwimage_pyr[l - 1].clone()
        p = warp.clone()
        p += torch.eye(3, device=dev, dtype=dt).repeat(B, C, 1, 1)

        noi = ((levels - l) ** 2 + 1) * 30 // 2 ** l
        noi = 14

        tmplt = template.clone().contiguous()
        qtmplt = qtemplate.clone()
        # tmplt = tmplt[:, :, :-29, :-29]
        ker = fspecial('gaussian', 2 ** (levels - l + 5) +
                       1, sigma=1, dtype=dt, device=dev)
        ker1 = fspecial('gaussian', 2 ** 2 + 1, sigma=4, dtype=dt, device=dev)
        ker1 = fspecial('gaussian', 2 ** (levels - l + 4) +
                        1, sigma=1, dtype=dt, device=dev)
        tH, tW = tmplt.shape[2:]

        xvector = torch.arange(tW, device=dev, dtype=dt)
        yvector = torch.arange(tH, device=dev, dtype=dt)

        phi = torch.tensor(45 * torch.pi / 180, dtype=dt, device=dev)

        wrpd = spatial_interp(wimage, p, 'linear', 'affine', xvector, yvector)
        wrpd = torch.nan_to_num(wrpd, nan=1.0, posinf=0.0, neginf=0.0)
        wrpd[wrpd == 0.0] = 1.0

        qwrpd = spatial_interp(qwimage, p, 'linear',
                               'affine', xvector, yvector)
        qwrpd = torch.nan_to_num(qwrpd, nan=1.0, posinf=1.0, neginf=1.0)
        qwrpd[qwrpd == 0.0] = 1.0

        ROI_x = torch.arange(tW, device=dev, dtype=dt)
        ROI_y = torch.arange(tH, device=dev, dtype=dt)
        X = torch.ones((B, C, 3, tW * tH), device=dev, dtype=dt)
        O2, O1 = torch.meshgrid(ROI_y, ROI_x, indexing='ij')
        O1 = O1.repeat(B, C, 1, 1)
        O2 = O2.repeat(B, C, 1, 1)

        # Template

        g_tmplt_x, g_tmplt_y = grad(tmplt)
        g_t_x = g_tmplt_x.clone()
        g_t_y = g_tmplt_y.clone()
        g_tmplt_x1 = g_tmplt_x / (g_tmplt_x ** 2 + g_tmplt_y ** 2 + eps).sqrt()
        g_tmplt_x1 = g_tmplt_x1.nan_to_num(nan=1.0)
        g_tmplt_y1 = g_tmplt_y / (g_tmplt_x ** 2 + g_tmplt_y ** 2 + eps).sqrt()
        g_tmplt_y1 = g_tmplt_y1.nan_to_num(nan=1.0)
        g_tmplt_y = g_tmplt_y1.clone()
        g_tmplt_x = g_tmplt_x1.clone()

        g_tmplt_xx, g_tmplt_xy = grad(g_tmplt_x)
        g_tmplt_yx, g_tmplt_yy = grad(g_tmplt_y)

        # Gradient matrix
        G_t = torch.zeros((B, C, 2, 2, tH * tW), device=dev, dtype=dt)
        G_t[:, :, 0, 0, :] = g_tmplt_x.permute(0, 1, 3, 2).reshape(B, C, -1)
        G_t[:, :, 0, 1, :] = g_tmplt_y.permute(0, 1, 3, 2).reshape(B, C, -1)
        G_t[:, :, 1, 0, :] = g_tmplt_x.permute(0, 1, 3, 2).reshape(B, C, -1)
        G_t[:, :, 1, 1, :] = g_tmplt_y.permute(0, 1, 3, 2).reshape(B, C, -1)
        G = G_t.permute(0, 1, 3, 2, 4)
        G = torch.nan_to_num(G, nan=1.0, posinf=0.0, neginf=0.0)
        N_G_t = torch.sqrt(g_tmplt_x.transpose(3, 2).reshape(
            B, C, -1) ** 2 + g_tmplt_y.transpose(3, 2).reshape(B, C, -1) ** 2)

        # Hessian matrix
        H_t = torch.zeros((B, C, 2, 2, tW * tH), device=dev, dtype=dt)
        H_t[:, :, 0, 0, :] = g_tmplt_xx.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        H_t[:, :, 0, 1, :] = g_tmplt_xy.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        H_t[:, :, 1, 0, :] = g_tmplt_yx.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        H_t[:, :, 1, 1, :] = g_tmplt_yy.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        P_H_t = H_t.permute(0, 1, 3, 2, 4)

# Positiveness
        Determ_t = (H_t[:, :, 0, 0, :] * H_t[:, :, 1, 1, :] -
                    H_t[:, :, 0, 1, :] * H_t[:, :, 1, 0, :])
        Trace_t = H_t[:, :, 0, 0, :] + H_t[:, :, 1, 1, :]

        I_G_tx = filter2(ker1, g_tmplt_x ** 2)
        I_G_ty = filter2(ker1, g_tmplt_y ** 2)
        I_G_txy = filter2(ker1, g_tmplt_y * g_tmplt_x)

        I_H_t_11 = filter2(ker, H_t[:, :,  0, 0, :].view(
            B, C, tW, tH).transpose(3, 2))
        I_H_t_12 = filter2(ker, H_t[:, :,  0, 1, :].view(
            B, C, tW, tH).transpose(3, 2))
        I_H_t_21 = filter2(ker, H_t[:, :,  1, 0, :].view(
            B, C, tW, tH).transpose(3, 2))
        I_H_t_22 = filter2(ker, H_t[:, :,  1, 1, :].view(
            B, C, tW, tH).transpose(3, 2))

        Determ_mgrt = I_G_tx * I_G_ty - I_G_txy ** 2
        Determ_mt = I_H_t_11 * I_H_t_22 - I_H_t_12 * I_H_t_21
        Determ_mt = Determ_mt.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        Trace_mt = I_H_t_11 + I_H_t_22
        Trace_mt = Trace_mt.permute(0, 1, 3, 2).reshape(B, C, -1)

        H_t_in = torch.zeros((B, C, 2, 2, tH * tW), device=dev, dtype=dt)
        H_t_in[:, :, 0, 0, :] = I_H_t_22.transpose(
            2, 3).reshape(B, C, -1) / Determ_mt
        H_t_in[:, :, 1, 1, :] = I_H_t_11.transpose(
            2, 3).reshape(B, C, -1) / Determ_mt
        H_t_in[:, :, 0, 1, :] = -I_H_t_12.transpose(
            2, 3).reshape(B, C, -1) / Determ_mt
        H_t_in[:, :, 1, 0, :] = -I_H_t_21.transpose(
            2, 3).reshape(B, C, -1) / Determ_mt

        GG_t = torch.stack([I_G_tx.permute(0, 1, 3, 2).reshape(B, C, -1),
                            I_G_ty.permute(0, 1, 3, 2).reshape(B, C, -1)], dim=2)
        GG_t = GG_t / (torch.sqrt((GG_t**2).sum(dim=2, keepdim=True)) + eps)

        cond_a = (((Trace_mt / 2)-(Determ_mt).abs())
                  > 0.).view(B, C, tW, tH).transpose(3, 2)

        X[:, :, 0, :] = O1.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        X[:, :, 1, :] = O2.permute(0, 1, 3, 2).reshape(
            B, C, -1)
        X[:, :, 2, :] = 0.0
        X_t = X.transpose(3, 2)
        XX = torch.cat([X_t, X_t], dim=3)

        fitt = []

        for iter in range(noi):
            mb = filter2(h, wrpd)
            g_wrpd_x, g_wrpd_y = grad(wrpd)
            g_wrpd_xx, g_wrpd_xy = grad(g_wrpd_x)
            g_wrpd_yx, g_wrpd_yy = grad(g_wrpd_y)

            # Gradient Matrix
            G = torch.zeros((B, C, 2, 2, tH * tW), device=dev, dtype=dt)
            G[:, :, 0, 0, :] = g_wrpd_x.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            G[:, :, 0, 1, :] = g_wrpd_y.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            G[:, :, 1, 0, :] = g_wrpd_x.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            G[:, :, 1, 1, :] = g_wrpd_y.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            P_G = G.permute(0, 1, 3, 2, 4)
            N_G = torch.sqrt(g_wrpd_x.transpose(3, 2).reshape(
                B, C, -1) ** 2 + g_wrpd_y.permute(0, 1, 3, 2).reshape(B, C, -1) ** 2)

            # Hessian Matrix
            H = torch.zeros((B, C, 2, 2, tH * tW), device=dev, dtype=dt)
            H[:, :, 0, 0, :] = g_wrpd_xx.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            H[:, :, 0, 1, :] = g_wrpd_xy.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            H[:, :, 1, 0, :] = g_wrpd_yx.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            H[:, :, 1, 1, :] = g_wrpd_yy.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            H = torch.nan_to_num(H, nan=1.0, posinf=0.0, neginf=0.0)
            P_H = H.permute(0, 1, 3, 2, 4)

            S_N = (P_G * P_H_t * H * P_G).sum(dim=(2, 3))
            S_D = (P_G * P_H * H * P_G).sum(dim=(2, 3))
            Phi = (S_N / S_D).view(B, C, tW, tH).transpose(3, 2)

# Positiveness
            Determ = H[:, :, 0, 0, :] * \
                H[:, :, 1, 1, :] - H[:, :, 0, 1, :] ** 2
            Trace = H[:, :, 0, 0, :] + H[:, :, 1, 1, :]
            Phi_1 = (Determ_t / Determ).view(B, C,
                                             tW, tH)  .permute(0, 1, 3, 2)
            Phi_2 = (Trace_t / (N_G) - (Determ_t / (Determ))
                     * Trace / (N_G)).view(B, C, tW, tH)   .transpose(3, 2)
            Detector = ((Phi_2.abs() * (Phi_1 > 0)) < 1e-4)

            # mesh (Detector)
            I_G_x = filter2(ker1, g_wrpd_x ** 2)
            I_G_y = filter2(ker1, g_wrpd_y ** 2)
            I_G_xy = filter2(ker, g_wrpd_x * g_wrpd_y)

            I_H_11 = filter2(ker, H[:, :, 0, 0, :].view(
                B, C, tW, tH).transpose(3, 2))
            I_H_22 = filter2(ker, H[:, :, 1, 1, :].view(
                B, C, tW, tH).transpose(3, 2))
            I_H_12 = filter2(ker, H[:, :, 0, 1, :].view(
                B, C, tW, tH).transpose(3, 2))
            I_H_21 = filter2(ker, H[:, :, 1, 0, :].view(
                B, C, tW, tH).transpose(3, 2))

            Determ_mgr = I_G_x * I_G_y - I_G_xy ** 2
            Determ_m = I_H_11 * I_H_22 - I_H_12 * I_H_21
            Determ_m = Determ_m.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            Determ_m = clamp_det(Determ_m, eps_num)
            Trace_m = I_H_11 + I_H_22
            Trace_m = Trace_m.permute(0, 1, 3, 2).reshape(
                B, C, -1)

            H_in = torch.zeros((B, C, 2, 2, tH * tW), device=dev, dtype=dt)
            H_in[:, :, 0, 0, :] = I_H_22.transpose(
                2, 3).reshape(B, C, -1) / (Determ_m + eps)
            H_in[:, :, 1, 1, :] = I_H_11.transpose(
                2, 3).reshape(B, C, -1) / (Determ_m + eps)
            H_in[:, :, 0, 1, :] = -I_H_12.transpose(
                2, 3).reshape(B, C, -1) / (Determ_m + eps)
            H_in[:, :, 1, 0, :] = -I_H_21.transpose(
                2, 3).reshape(B, C, -1) / (Determ_m + eps)

            GG = torch.stack([I_G_x.permute(0, 1, 3, 2).reshape(B, C, -1),
                              I_G_y.permute(0, 1, 3, 2).reshape(B, C, -1)], dim=2)
            GG = GG / (torch.sqrt((GG ** 2).sum(dim=2, keepdim=True)) + eps)

            SS1 = (torch.sign(GG_t * GG).sum(dim=2).abs()
                   ).view(B, C, tW, tH).transpose(3, 2)

            A = torch.sqrt(1 - 4 * Determ_m / ((Trace_m ** 2) + eps))
            C_1 = torch.tan(phi) > A

            cond_b = (((Trace_m / 2) - Determ_m.abs()) >
                      0).view(B, C, tW, tH).transpose(3, 2)
            cond_c = cond_b * cond_a
            cond_cc = cond_a * cond_b * (SS1 == 1)

            G0 = torch.zeros((B, C, 2, 2, tW * tH), device=dev, dtype=dt)
            G0[:, :, 0, 0, :] = g_tmplt_x.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            G0[:, :, 0, 1, :] = g_tmplt_x.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            G0[:, :, 1, 0, :] = g_tmplt_y.permute(0, 1, 3, 2).reshape(
                B, C, -1)
            G0[:, :, 1, 1, :] = g_tmplt_y.permute(0, 1, 3, 2).reshape(
                B, C, -1)

            S = (G0 * H).sum(dim=2)

            k1 = S[:, :, 0, :].unsqueeze(-1).repeat(1, 1, 1, 3)
            k2 = S[:, :, 1, :].unsqueeze(-1).repeat(1, 1, 1, 3)
            S = torch.cat([k1, k2], dim=-1)

            T = S * (XX + 1)

            b1 = g_tmplt_x.transpose(3, 2).reshape(
                B, C, -1) * g_wrpd_x.permute(0, 1, 3, 2).reshape(B, C, -1)
            b2 = g_tmplt_y.transpose(3, 2).reshape(
                B, C, -1) * g_wrpd_y.permute(0, 1, 3, 2).reshape(B, C, -1)
            b = b1 + b2
            c = torch.sqrt(g_wrpd_x.transpose(3, 2).reshape(
                B, C, -1) ** 2 + g_wrpd_y.permute(0, 1, 3, 2).reshape(B, C, -1) ** 2)
            d = torch.sqrt(g_t_x.transpose(3, 2).reshape(
                B, C, -1) ** 2 + g_t_y.permute(0, 1, 3, 2).reshape(B, C, -1) ** 2)

            V, S1, U = torch.linalg.svd(T, full_matrices=False)

            r = (V.permute(0, 1, 3, 2) @ b.unsqueeze(-1))
            M = (c - b).abs().amax(dim=2)

            cond = Detector.clone().to(torch.bool)
            lambda_ = c[cond.permute(0, 1, 3, 2).reshape(
                B, C, -1)].unsqueeze(-1)
            T1 = T[cond.permute(0, 1, 3, 2).reshape(B, C, -1), :]
            b1 = b[cond.permute(0, 1, 3, 2).reshape(B, C, -1)].unsqueeze(-1)

            if B == 1 and C == 1:
                lambda_ = lambda_.unsqueeze(0).unsqueeze(0)
                T1 = T1.unsqueeze(0).unsqueeze(0)
                b1 = b1.unsqueeze(0).unsqueeze(0)
            elif B == 1:
                lambda_ = lambda_.unsqueeze(0)
                T1 = T1.unsqueeze(0).unsqueeze(0)
                b1 = b1.unsqueeze(0)

            V, S, U = torch.linalg.svd(T1, full_matrices=False)

            # S_inv = torch.diagflat(S).inverse()
            S_inv = torch.linalg.pinv(torch.diagflat(S))
            V_T = V.permute(0, 1, 3, 2)
            U = U.permute(0, 1, 3, 2)
            if B == 1 and C == 1:
                S_inv = S_inv.unsqueeze(0).unsqueeze(0)
            elif B == 1:
                S_inv = S_inv.unsqueeze(0)

            Dp = U @ S_inv @ (V_T @ (lambda_ - b1))
            temp = torch.logical_and(M < 1e-6, b.amin(dim=2) >= 0)
            Dp[temp, ...] = 0

            Dp = Dp.reshape(B, C, 2, 3)

            warp = param_update(warp, Dp, 'affine')
            fitt.append({'warp_p': warp[:, :, :2, :]})
            Dps = torch.zeros(B, C, 3, 3, dtype=dt, device=dev)
            Dps[:, :, :2, :] = Dp
            p = p + Dps

            if torch.norm(Dp) < 1e-9:
                return fitt

            wrpd = spatial_interp(wimage, p, 'linear',
                                  'affine', xvector, yvector)
            wrpd = torch.nan_to_num(wrpd, nan=1.0, posinf=0.0, neginf=0.0)
            wrpd[wrpd == 0.0] = 1.0

            qwrpd = spatial_interp(qwimage, p, 'linear',
                                   'affine', xvector, yvector)
            qwrpd = torch.nan_to_num(qwrpd, nan=1.0, posinf=1.0, neginf=1.0)
            qwrpd[qwrpd == 0.0] = 1.0

        warp = next_level(warp, 'affine', False)
        warp[:, :, -1, -1] = 0.

    return fitt


if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt
    from ComputePointError import ComputePointError

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # res = torch.zeros(10_000, dtype=dtype, device=device)
    res = loadmat(
        "/home/ltopalis/Desktop/image-alignment-using-nn/pixel_ecc_affine/levels_3_results.mat")['results']
    res = torch.from_numpy(res).to(dtype=dtype, device=device).squeeze()

    for sel in [7685, 7686, 7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694, 7695, 7696,
                7697, 7698, 7699, 7700, 7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708,
                7709, 7710, 7711, 7712, 7713, 7714, 7715, 7716, 7717, 7718, 7719, 7720,
                7721, 7722, 7723, 7724, 7725, 7726, 7727, 7728, 7729, 7730, 7731, 7732,
                7733, 7734, 7735, 7736, 7737, 7738, 7739, 7740, 7741, 7742, 7743, 7744,
                7745, 7746, 7747, 7748, 7749]:

        with h5py.File("dataset_matlab.hdf5", "r") as f:
            test_pts = f['test_pts'][sel]
            template_affine = f['template_affine'][sel]
            m = f['m'][sel]
            img = f['img'][sel]
            tmplt = f['tmplt'][sel]
            p_init = f['p_init'][sel]

        img = torch.from_numpy(
            img).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        tmplt = torch.from_numpy(
            tmplt).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        p_init = torch.from_numpy(
            p_init).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        test_pts = torch.from_numpy(test_pts).to(
            dtype=dtype, device=device).unsqueeze(0)
        template_affine = torch.from_numpy(template_affine).to(
            dtype=dtype, device=device).unsqueeze(0)
        m = torch.from_numpy(m).to(dtype=dtype, device=device).unsqueeze(0)

        out = ECC_PIXEL_IA(img, tmplt, p_init)

        rms = ComputePointError(
            test_pts, template_affine, out[-1]['warp_p'], m)

        res[sel] = rms

        if (sel + 1) % 25 == 0:
            print(f'{sel + 1} samples have been computed')
            savemat("/home/ltopalis/Desktop/image-alignment-using-nn/pixel_ecc_affine/levels_3_results.mat", mdict={
                    "results": res.detach().cpu()})

    savemat("/home/ltopalis/Desktop/image-alignment-using-nn/pixel_ecc_affine/levels_3_results.mat",
            mdict={"results": res.detach().cpu()})
