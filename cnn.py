import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 hidden_ch: int = 32,
                 out_ch: int = 32,
                 *,
                 downsampling: bool = True,
                 level: int = 0,
                 dt: torch.dtype = torch.float32,
                 dev: torch.device | str = torch.device('cpu')):
        super().__init__()
        stride0 = (2 ** level) if (downsampling and level > 0) else 1

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3,
                      stride=stride0, padding=1, bias=False, dtype=dt, device=dev),
            nn.BatchNorm2d(hidden_ch, dtype=dt, device=dev),
            # nn.GroupNorm(num_groups=8, num_channels=hidden_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3,
                      stride=1, padding=1, bias=False, dtype=dt, device=dev),
            nn.BatchNorm2d(hidden_ch, dtype=dt, device=dev),
            # nn.GroupNorm(num_groups=8, num_channels=hidden_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_ch, out_ch,    kernel_size=3,
                      stride=1, padding=1, bias=False, dtype=dt, device=dev),
            nn.BatchNorm2d(out_ch, dtype=dt, device=dev),
            # nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


def CoarseToFineFeatureExtractor(
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 32,
        device: torch.device | str = 'cpu',
        dtype: torch.dtype = torch.float32,
        downsampling: bool = True,
        levels: int = 4):

    models = []
    for level in range(levels):
        m = CNN(in_channels, hidden_channels, out_channels,
                downsampling=downsampling, level=level, dt=dtype, dev=device)

        models.append(m)

    return models


if __name__ == '__main__':
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    dev = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dt = torch.float64

    img = loadmat('data/myYaleCropped.mat')['tmplts']
    img = torch.from_numpy(img)
    img = img[:, :, 7].unsqueeze(0).unsqueeze(
        0).to(dtype=dt, device=dev)

    plt.imshow(img.squeeze().cpu(), cmap=plt.cm.gray)

    plt.savefig('plot_test.png', dpi=200)

    models = CoarseToFineFeatureExtractor(
        1, 32, 32, device=dev, dtype=dt, levels=4, downsampling=True)

    plt.figure()
    for i, model in enumerate(models[::-1], start=1):
        out = model(img)

        plt.subplot(2, 2, i)
        plt.imshow(out[0, 2, :, :].squeeze().detach().cpu(), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()

        a = out[0, 2, :, :].clone()

    print(torch.gradient(a.squeeze()))

    y, x = torch.gradient(a.squeeze())

    xy, xx = torch.gradient(x)

    plt.savefig('plot.png', dpi=200)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(a.squeeze().detach().cpu(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(xy.squeeze().detach().cpu(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(xx.squeeze().detach().cpu(), cmap=plt.cm.gray)
    plt.axis('off')

    plt.savefig('plot_grad.png', dpi=200)
