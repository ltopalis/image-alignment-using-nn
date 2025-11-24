import torch
import torch.nn as nn
import torch.nn.functional as F


class pixel_ECC_layer(nn.Module):
    def __init__(self, max_iters=200):
        super().__init__()
        self.max_iters = max_iters

    def forward(self, F_T: torch, F_I: torch, p0: torch):
        B, _, H, W = F_T.shape
        F_T = F_T[:, 0, :, :]
        F_I = F_I[:, 0, :, :]

        ys = torch.arange(H, device=F_T.device)
        xs = torch.arange(W, device=F_T.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')      # [H, W]
        ones = torch.ones_like(grid_x)
        pixels = torch.stack([grid_x, grid_y, ones],
                             dim=0)            # [3, H, W]
        pixels = pixels.view(3, -1).unsqueeze(0).repeat(B,
                                                        # [B, 3, N]
                                                        1, 1).type(torch.float32)

        p = p0.clone()
        Hmat = torch.cat(
            [p, torch.ones(B, 1, device=p.device)], dim=1).view(B, 3, 3)

        for _ in range(self.max_iters):
            H_inv = torch.inverse(Hmat)
            src = H_inv.bmm(pixels)
            x_s = src[:, 0]/src[:, 2]
            y_s = src[:, 1]/src[:, 2]
            x_norm = (x_s/(W-1))*2 - 1
            y_norm = (y_s/(H-1))*2 - 1
            grid = torch.stack([x_norm, y_norm], dim=-1).view(B, H, W, 2)
            F_I_w = F.grid_sample(
                F_I.unsqueeze(1), grid, align_corners=False, mode='bilinear', padding_mode='border')[:, 0]

            gy_t, gx_t = torch.gradient(F_T, dim=(1, 2))
            gy_w, gx_w = torch.gradient(F_I_w, dim=(1, 2))
            grad_t = torch.stack([gy_t, gx_t], dim=1)  # [B,2,H,W]
            grad_w = torch.stack([gy_w, gx_w], dim=1)  # [B,2,H,W]

            def compute_hessian(g):
                pass

            Ht = compute_hessian(grad_t)
            Hw = compute_hessian(grad_w)

        return p


if __name__ == "__main__":
    ftr = torch.randint(0, 255, (8, 5, 24, 24)).float()
    inp = torch.randint(0, 255, (8, 1, 24, 24)).float()
    mot = torch.randint(0, 255, (8, 8)).float()

    pixelECC = pixel_ECC_layer(1)
    pixelECC(ftr, inp, mot)
