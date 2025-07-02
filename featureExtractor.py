import cv2
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Level_1(nn.Module):
    def __init__(self, out_channels=8, device='cpu'):
        super(Level_1, self).__init__()

        self.device = device
        self.out_channels = out_channels

        # Stride = 2 for Downsampling

        self.conv1a = nn.Conv2d(1, 32, kernel_size=3,
                                stride=2, padding=1, device=self.device)
        self.bn_1a = nn.BatchNorm2d(32)

        self.conv1b = nn.Conv2d(32, 64, kernel_size=3,
                                stride=2, padding=1, device=self.device)
        self.bn_1b = nn.BatchNorm2d(64)

        self.conv1c = nn.Conv2d(64, 128, kernel_size=3,
                                stride=2, padding=1, device=self.device)
        self.bn_1c = nn.BatchNorm2d(128)

        # Stride = 1 for feature extraction

        self.conv1d = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_1d = nn.BatchNorm2d(128)

        self.conv1e = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_1e = nn.BatchNorm2d(128)

        self.conv1f = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_1f = nn.BatchNorm2d(128)

        self.output = nn.Conv2d(128, self.out_channels,
                                kernel_size=1, device=self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, initial_motion=False):
        # input [batch, 1, H, W]
        x = F.leaky_relu(self.bn_1a(self.conv1a(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_1b(self.conv1b(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_1c(self.conv1c(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_1d(self.conv1d(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_1e(self.conv1e(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_1f(self.conv1f(x)), negative_slope=.1)

        if initial_motion:
            return x

        out = self.output(x)  # [batch, 8, H', W']

        return out


class Level_2(nn.Module):
    def __init__(self, out_channels=8, device="cpu"):
        super(Level_2, self).__init__()

        self.device = device
        self.out_channels = out_channels

        # Stride = 2 for Downsampling

        self.conv2a = nn.Conv2d(2, 32, kernel_size=3,
                                stride=2, padding=1, device=self.device)
        self.bn_2a = nn.BatchNorm2d(32)

        self.conv2b = nn.Conv2d(32, 64, kernel_size=3,
                                stride=2, padding=1, device=self.device)
        self.bn_2b = nn.BatchNorm2d(64)

        # Stride = 1 for feature extraction

        self.conv2c = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_2c = nn.BatchNorm2d(64)

        self.conv2d = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_2d = nn.BatchNorm2d(64)

        self.conv2e = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_2e = nn.BatchNorm2d(64)

        self.conv2f = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_2f = nn.BatchNorm2d(64)

        self.output = nn.Conv2d(64, self.out_channels,
                                kernel_size=1, device=self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # input [batch, 2, H, W]
        x = F.leaky_relu(self.bn_2a(self.conv2a(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_2b(self.conv2b(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_2c(self.conv2c(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_2d(self.conv2d(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_2e(self.conv2e(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_2f(self.conv2f(x)), negative_slope=.1)

        out = self.output(x)  # [batch, 8, H', W']

        return out


class Level_3(nn.Module):
    def __init__(self, out_channels=8, device="cpu"):
        super(Level_3, self).__init__()

        self.device = device
        self.out_channels = out_channels

        # Stride = 2 for Downsampling

        self.conv3a = nn.Conv2d(2, 32, kernel_size=3,
                                stride=2, padding=1, device=self.device)
        self.bn_3a = nn.BatchNorm2d(32)

        # Stride = 1 for feature extraction

        self.conv3b = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_3b = nn.BatchNorm2d(32)

        self.conv3c = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_3c = nn.BatchNorm2d(32)

        self.conv3d = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_3d = nn.BatchNorm2d(32)

        self.conv3e = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_3e = nn.BatchNorm2d(32)

        self.conv3f = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_3f = nn.BatchNorm2d(32)

        self.output = nn.Conv2d(32, self.out_channels,
                                kernel_size=1, device=self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # input [batch, 2, H, W]
        x = F.leaky_relu(self.bn_3a(self.conv3a(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_3b(self.conv3b(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_3c(self.conv3c(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_3d(self.conv3d(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_3e(self.conv3e(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_3f(self.conv3f(x)), negative_slope=.1)

        out = self.output(x)  # [batch, 8, H', W']

        return out


class Level_4(nn.Module):
    def __init__(self, out_channels=8, device="cpu"):
        super(Level_4, self).__init__()

        self.device = device
        self.out_channels = out_channels

        self.conv4a = nn.Conv2d(2, 16, kernel_size=3,
                                stride=1, padding=1, device=self.device)
        self.bn_4a = nn.BatchNorm2d(16)

        self.conv4b = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_4b = nn.BatchNorm2d(16)

        self.conv4c = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_4c = nn.BatchNorm2d(16)

        self.conv4d = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding=1, device=self.device)
        self.bn_4d = nn.BatchNorm2d(16)

        self.output = nn.Conv2d(16, 8, kernel_size=1, stride=1)

    def forward(self, x):
        # input [batch, 2, H, W]
        x = F.leaky_relu(self.bn_4a(self.conv4a(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_4b(self.conv4b(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_4c(self.conv4c(x)), negative_slope=.1)
        x = F.leaky_relu(self.bn_4d(self.conv4d(x)), negative_slope=.1)

        out = self.output(x)  # [batch, 8, H', W']

        return out
