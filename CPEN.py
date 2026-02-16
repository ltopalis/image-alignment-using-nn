import time
from Dataset import FirstDataset, collate_batch
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
from pixel_ecc_affine.ECC_PIXEL_IA import ECC_PIXEL_IA


def check_finite(name, x):
    if not torch.isfinite(x).all():
        with torch.no_grad():
            fin = torch.isfinite(x)
            print(f"[NON-FINITE] {name}: finite={fin.float().mean().item():.4f} "
                  f"min={torch.nan_to_num(x).min().item():.4e} "
                  f"max={torch.nan_to_num(x).max().item():.4e}")
        raise RuntimeError(f"Non-finite detected at {name}")


class ChannelAggregator(nn.Module):
    def __init__(self, num_channels, temperature=1.0):
        super().__init__()

        self.num_channels = num_channels
        self.temperature = temperature

        self.logits = nn.Parameter(torch.zeros(num_channels))

        with torch.no_grad():
            self.logits[0] = 2.0
            self.logits[1:] = 0.0

    def forward(self, x):

        temp = self.logits / torch.clamp(
            torch.tensor(self.temperature, device=self.logits.device,
                         dtype=self.logits.dtype),
            1e-3, 10.0
        )

        finite = torch.isfinite(temp)

        if not finite.all():
            temp = torch.zeros_like(temp)
            temp[0] = 2.0

        weights = F.softmax(temp, dim=0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights / (weights.sum() + 1e-12)

        check_finite("agg_weights", weights)
        out = (x * weights.view(1, -1, 1)).sum(dim=1)

        return out


class CPEN(nn.Module):
    def __init__(self, levels=4, out_channels=32, device='cpu', dtype=torch.float32, DEBUG=False):
        super().__init__()
        self.levels = levels
        self.out_ch = out_channels
        self.model = CNN(in_ch=1, hidden_ch=32, out_ch=self.out_ch)
        self.aggregator = ChannelAggregator(self.out_ch + 1, temperature=0.1)

        self.dev = device
        self.dt = dtype
        self.DEBUG = DEBUG
        self.to(device=device, dtype=dtype)

    def forward(self,
                warped: torch.Tensor,
                template: torch.Tensor,
                init_p: torch.Tensor = None) -> torch.Tensor:

        init_p = init_p.unsqueeze(1).expand(-1, self.out_ch + 1, -1, -1)

        wimage = self.model(warped).to(dtype=self.dt, device=self.dev)
        tmplt = self.model(template).to(dtype=self.dt, device=self.dev)

        wimage = torch.nan_to_num(wimage, nan=0.0, posinf=0.0, neginf=0.0)
        tmplt = torch.nan_to_num(tmplt, nan=0.0, posinf=0.0, neginf=0.0)

        wimage = torch.cat([warped[:, :, :, :], wimage], dim=1)
        tmplt = torch.cat([template[:, :, :, :], tmplt], dim=1)

        B, C, H, W = wimage.shape

        out = ECC_PIXEL_IA(wimage, tmplt, init_p,
                           in_levels=self.levels, DEBUG=self.DEBUG)

        init = out.reshape(B, C, -1)
        init = torch.nan_to_num(init, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.aggregator(init)

        logits = logits.view(B, 2, 3)

        if self.DEBUG:
            check_finite("cpen_init_pre_fix", init)
            check_finite("cpen_init_post_fix", init)
            check_finite("cpen_logits", logits)
            check_finite("cpen_logits", logits)

        return logits


if __name__ == "__main__":
    torch.set_default_device("cpu")
    dt = torch.float32
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CPEN(device=dev, dtype=dt, levels=1)

    h5_path = '/home/ltopalis/Desktop/image-alignment-using-nn/dataset_matlab.hdf5'
    full_ds = FirstDataset(h5_path)
    N = len(full_ds)
    n_train = int(0.8 * N)
    n_test = N - n_train

    batch = 5

    train_ds, test_ds = random_split(full_ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available(),
                              persistent_workers=True, prefetch_factor=2,
                              collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             num_workers=2, pin_memory=torch.cuda.is_available(),
                             persistent_workers=True, collate_fn=collate_batch)

    model = CPEN(levels=3, out_channels=32, device=dev, dtype=dt)
    model = model.to(device=dev, dtype=dt)

    for data in train_loader:
        img = data['img'].unsqueeze(1).to(dtype=dt, device=dev)
        tmplt = data['tmplt'].unsqueeze(1).to(dtype=dt, device=dev)
        p_init = data['p_init'].to(dtype=dt, device=dev)

        start = time.perf_counter()
        out = model(img, tmplt, p_init)
        end = time.perf_counter()
        print(
            f"Output shape: {out.shape}, Time taken: {end - start:.4f} seconds")
