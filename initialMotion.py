import torch.nn as nn


class HomographyRegressionHead(nn.Module):
    def __init__(self, in_channel=128, out_channel=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channel, out_channel)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x -> [B, C, H, W]
        x = self.global_pool(x)    # x -> [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        x = self.fc(x)             # [B, 8]
        return x
