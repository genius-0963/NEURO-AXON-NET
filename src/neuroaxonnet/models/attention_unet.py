"""
Candidate Variant V2: Attention U-Net.

Incorporates additive attention gates on encoder-to-decoder skip connections.
Guarded behind configuration flag: model.variant == 'attention'
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class AttentionGate(nn.Module):
        """Additive Attention Gate (Oktay et al.)."""
        def __init__(self, f_g: int, f_l: int, f_int: int):
            super().__init__()
            self.w_g = nn.Sequential(
                nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            )
            self.w_x = nn.Sequential(
                nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            )
            self.psi = nn.Sequential(
                nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            g1 = self.w_g(g)
            x1 = self.w_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class AttentionAxoNetUNet(nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 32):
            super().__init__()
            c = base_channels

            # Encoder
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            )
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Sequential(
                nn.Conv2d(c, c * 2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(),
            )
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Sequential(
                nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.ReLU(),
            )
            self.pool3 = nn.MaxPool2d(2, 2)

            self.conv4 = nn.Sequential(
                nn.Conv2d(c * 4, c * 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 8, c * 8, 3, padding=1), nn.ReLU(),
            )
            self.pool4 = nn.MaxPool2d(2, 2)

            self.bottleneck = nn.Sequential(
                nn.Conv2d(c * 8, c * 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 16, c * 16, 3, padding=1), nn.ReLU(),
            )

            # Attention Gates
            self.ag4 = AttentionGate(f_g=c * 16, f_l=c * 8, f_int=c * 4)
            self.ag3 = AttentionGate(f_g=c * 8, f_l=c * 4, f_int=c * 2)
            self.ag2 = AttentionGate(f_g=c * 4, f_l=c * 2, f_int=c)
            self.ag1 = AttentionGate(f_g=c * 2, f_l=c, f_int=c // 2)

            # Decoder Convs
            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec4 = nn.Sequential(
                nn.Conv2d(c * 16 + c * 8, c * 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 8, c * 8, 3, padding=1), nn.ReLU(),
            )

            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec3 = nn.Sequential(
                nn.Conv2d(c * 8 + c * 4, c * 4, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.ReLU(),
            )

            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec2 = nn.Sequential(
                nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(),
            )

            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec1 = nn.Sequential(
                nn.Conv2d(c * 2 + c, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            )

            self.head = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.conv1(x)
            p1 = self.pool1(e1)
            e2 = self.conv2(p1)
            p2 = self.pool2(e2)
            e3 = self.conv3(p2)
            p3 = self.pool3(e3)
            e4 = self.conv4(p3)
            p4 = self.pool4(e4)

            b = self.bottleneck(p4)

            # Upsample and apply attention gate on skip
            u4 = self.up4(b)
            a4 = self.ag4(g=u4, x=e4)
            d4 = self.dec4(torch.cat([u4, a4], dim=1))

            u3 = self.up3(d4)
            a3 = self.ag3(g=u3, x=e3)
            d3 = self.dec3(torch.cat([u3, a3], dim=1))

            u2 = self.up2(d3)
            a2 = self.ag2(g=u2, x=e2)
            d2 = self.dec2(torch.cat([u2, a2], dim=1))

            u1 = self.up1(d2)
            a1 = self.ag1(g=u1, x=e1)
            d1 = self.dec1(torch.cat([u1, a1], dim=1))

            return self.head(d1)
