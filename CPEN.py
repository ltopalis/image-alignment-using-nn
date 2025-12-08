import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
import matplotlib.pyplot as plt
from translation_cnn import compute_initial_motion
from pixel_ecc_affine.ECC_PIXEL_IA import ECC_PIXEL_IA
from pixel_ecc_affine.next_level import next_level

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=12)
torch.use_deterministic_algorithms(False)
torch.set_default_device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))


class WeightedChannelSum(nn.Module):
    def __init__(self, channels, length=None, per_position=False, normalize=True):
        super().__init__()
        self.channels = channels
        self.length = length  # αν None, δεν υποστηρίζουμε per_position=True με fixed L
        self.per_position = per_position
        self.normalize = normalize
        if per_position:
            assert length is not None, "για per_position πρέπει να ξέρεις L ή να χρησιμοποιήσεις άλλο μηχανισμό"
            param = torch.zeros(channels, length)
            param[0, :] = 1
            self.raw_w = nn.Parameter(param)
        else:
            self.raw_w = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: (B, C, L) όπου B μπορεί να είναι οτιδήποτε
        B, C, L = x.shape
        assert C == self.channels
        if self.per_position:
            assert L == self.length, "per_position απαιτεί γνωστό fixed L"
            w = self.raw_w
            if self.normalize:
                w = F.softmax(w, dim=0)
            w = w.unsqueeze(0)          # (1, C, L)
            out = (x * w).sum(dim=1)    # -> (B, L)
            return out
        else:
            w = self.raw_w
            if self.normalize:
                w = F.softmax(w, dim=0)
            w = w.view(1, C, 1)        # (1, C, 1) broadcast σε B,L
            out = (x * w).sum(dim=1)   # -> (B, L)
            return out


class CPEN(nn.Module):
    def __init__(self, levels=4, out_channels=32, device='cpu', dtype=torch.float32):
        super().__init__()
        self.levels = levels
        self.out_ch = out_channels
        self.models = nn.ModuleList(
            [CNN(in_ch=1, hidden_ch=32, out_ch=self.out_ch, downsampling=True, level=l)
             for l in range(levels)][::-1]
        )
        self.aggrigator = WeightedChannelSum(
            channels=33, per_position=True, length=6)  # nn.Sequential(
        #     nn.Linear(6, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 1)

        # )
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

            print('model ' + str(i))

            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(wimage[0, 1, :, :].squeeze(
            # ).detach().cpu(), cmap=plt.cm.gray)
            # plt.subplot(1, 2, 2)
            # plt.imshow(tmplt[0, 1, :, :].squeeze(
            # ).detach().cpu(), cmap=plt.cm.gray)
            # plt.savefig(f'a/model_{i}.png', dpi=200)

            B, C, H, W = wimage.shape

            if H < 30 or W < 30:
                continue

            out = ECC_PIXEL_IA(wimage, tmplt, init_p)
            init_p = out[-1]['warp_p']
            init_p = next_level(init_p, transform='affine', high_flag=False)
            # print(i, init_p[0, 0:3, :, :])

        init = init_p.view(B, C, -1)
        logits = self.aggrigator(init)
        print(logits, wimage[0, 1, :, :])

        # print(init)
        # print(logits)

        return logits.view(B, 2, 3)


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
