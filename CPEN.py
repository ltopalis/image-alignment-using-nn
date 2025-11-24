import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
from translation_cnn import compute_initial_motion
from pixel_ecc_affine.ECC_PIXEL_IA import ECC_PIXEL_IA

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=12)
torch.use_deterministic_algorithms(True)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


class CPEN(nn.Module):
    def __init__(self, levels=4, out_channels=32, device='cpu', dtype=torch.float32):
        super().__init__()
        self.levels = levels
        self.out_ch = out_channels
        self.models = nn.ModuleList(
            [CNN(in_ch=1, hidden_ch=32, out_ch=self.out_ch, downsampling=True, level=l)
             for l in range(levels)][::-1]
        )
        self.aggrigator = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)

        )
        self.dev = device
        self.dt = dtype
        self.to(device=device, dtype=dtype)

    def forward(self,
                warped: torch.Tensor,
                template: torch.Tensor,):

        with torch.no_grad():
            init_p = compute_initial_motion(
                warped, template, levels=self.levels)
            init_p = init_p.unsqueeze(1).repeat(1, 33, 1, 1)

        for i, model in enumerate(self.models):
            wimage = model(warped).to(dtype=self.dt, device=self.dev)
            tmplt = model(template).to(dtype=self.dt, device=self.dev)

            wimage = torch.cat(
                [warped[:, :, ::2**(self.levels-i-1), ::2**(self.levels-i-1)], wimage], dim=1)
            tmplt = torch.cat(
                [template[:, :, ::2**(self.levels-i-1), ::2**(self.levels-i-1)], tmplt], dim=1)

            B, C, H, W = wimage.shape

            if H < 30 or W < 30:
                continue

            out = ECC_PIXEL_IA(wimage, tmplt, init_p)
            init_p = out[-1]['warp_p']

            # for b in range(B):
            #     for c in range(C):

            #         init = init_p[b, c, :, :].unsqueeze(0).unsqueeze(0)
            #         w = wimage[b, c, :, :].unsqueeze(0).unsqueeze(0)
            #         t = tmplt[b, c, :, :].unsqueeze(0).unsqueeze(0)

            #         out = ECC_PIXEL_IA(w, t, init)

            # init_p[b, c, :, :] = out[-1]['warp_p'].squeeze()

        init = init_p.view(B, C, -1)
        x = init.view(B * C, 6)
        logits = self.aggrigator(x).view(B, C)
        w = F.softmax(logits, dim=1).unsqueeze(-1)
        weighted = (init * w).sum(dim=1)

        return weighted.view(B, 2, 3)


if __name__ == '__main__':
    from Dataset import Dataset

    dt = torch.float32
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CPEN(device=dev, dtype=dt, levels=3)

    d = Dataset('data/d.mat', dtype=dt, device=dev)

    j = 1
    for i in range(1):
        # print(i)
        a = d[i]

        wimage = a['wimage'].unsqueeze(0).to(dtype=dt, device=dev)
        template = a['template'].unsqueeze(0).to(dtype=dt, device=dev)
        pred = model(template, wimage)

        print(pred)
