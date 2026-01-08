from Dataset import Dataset
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
        self.length = length
        self.per_position = per_position
        self.normalize = normalize
        if per_position:
            assert length is not None, "για per_position πρέπει να ξέρεις L ή να χρησιμοποιήσεις άλλο μηχανισμό"
            param = torch.zeros(channels, length)
            param[0, :] = 1
            self.raw_w = nn.Parameter(param, requires_grad=False)
        else:
            self.raw_w = nn.Parameter(
                torch.zeros(channels), requires_grad=False)

    def forward(self, x):
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
    def __init__(self, levels=4, out_channels=32, device='cpu', dtype=torch.float64):
        super().__init__()
        self.levels = levels
        self.out_ch = out_channels
        self.model = CNN(in_ch=1, hidden_ch=32, out_ch=self.out_ch)
        self.aggrigator = WeightedChannelSum(
            # nn.Sequential(
            channels=out_channels+1, per_position=True, length=6)
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
            init_p = init_p.unsqueeze(1).repeat(1, self.out_ch + 1, 1, 1)

        wimage = self.model(warped).to(dtype=self.dt, device=self.dev)
        tmplt = self.model(template).to(dtype=self.dt, device=self.dev)

        wimage = torch.cat([warped[:, :, :, :], wimage], dim=1)
        tmplt = torch.cat([tmplt[:, :, :, :], template], dim=1)

        with torch.no_grad():
            a = compute_initial_motion(
                wimage[:, 0, :, :], tmplt[:, 0, :, :], levels=0)
            a = a.unsqueeze(1).repeat(1, self.out_ch + 1, 1, 1)

        B, C, H, W = wimage.shape

        for batch in range(B):
            for chan in range(C):
                out = ECC_PIXEL_IA(wimage[batch:batch+1, chan:chan+1, :, :], tmplt[batch:batch+1,
                                   chan:chan+1, :, :], init_p[batch:batch+1, chan:chan+1, :, :])
                init_p[batch, chan, :, :] = out[-1]['warp_p'].squeeze()

        init = init_p.view(B, C, -1)
        logits = self.aggrigator(init)

        return logits.view(B, 2, 3)


if __name__ == '__main__' and False:
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


if __name__ == "__main__":
    dt = torch.float32
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CPEN(device=dev, dtype=dt, levels=1)

    d = Dataset('data/dataset/data_01.mat', dtype=dt, device=dev)
    data = d[0]

    tmplt = data['template'].unsqueeze(0)
    wimage = data['template'][:, 100:228, 100:228].unsqueeze(0)

    pre = model(tmplt, wimage)

    print(pre)
