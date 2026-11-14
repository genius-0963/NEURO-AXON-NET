"""
Candidate Variant V1: Residual U-Net.

Incorporates residual skip connections inside encoder/decoder blocks for improved gradient flow.
Guarded behind configuration flag: model.variant == 'residual'
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.shortcut = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = self.shortcut(x)
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.relu(out + res)
            return out

    class ResidualAxoNetUNet(nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 32):
            super().__init__()
            c = base_channels

            self.enc1 = ResidualBlock(in_channels, c)
            self.pool1 = nn.MaxPool2d(2, 2)

            self.enc2 = ResidualBlock(c, c * 2)
            self.pool2 = nn.MaxPool2d(2, 2)

            self.enc3 = ResidualBlock(c * 2, c * 4)
            self.pool3 = nn.MaxPool2d(2, 2)

            self.enc4 = ResidualBlock(c * 4, c * 8)
            self.pool4 = nn.MaxPool2d(2, 2)

            self.bottleneck = ResidualBlock(c * 8, c * 16)

            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec4 = ResidualBlock(c * 16 + c * 8, c * 8)

            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec3 = ResidualBlock(c * 8 + c * 4, c * 4)

            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec2 = ResidualBlock(c * 4 + c * 2, c * 2)

            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec1 = ResidualBlock(c * 2 + c, c)

            self.head = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            p1 = self.pool1(e1)
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)
            e3 = self.enc3(p2)
            p3 = self.pool3(e3)
            e4 = self.enc4(p3)
            p4 = self.pool4(e4)

            b = self.bottleneck(p4)

            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.head(d1)
