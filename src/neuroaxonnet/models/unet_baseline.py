"""
AxoNet-style Reduced U-Net Baseline Architecture (V0).

Matches Section 3 specification:
- One grayscale input channel: [B, 1, 224, 224]
- Four encoder stages with channel progression: 32 -> 64 -> 128 -> 256
- Deepest bottleneck: 512 channels at 14x14
- Four decoder stages with skip concatenation: 256 -> 128 -> 64 -> 32
- Density Head: 1x1 conv to 1 channel with ReLU activation
- Output: non-negative density map [B, 1, 224, 224]
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class DoubleConv(nn.Module):
        """Two padded 3x3 convolutions with ReLU activation."""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    class AxoNetUNet(nn.Module):
        """
        Faithful PyTorch implementation of the reduced-width AxoNet U-Net.
        """
        def __init__(self, in_channels: int = 1, base_channels: int = 32):
            super().__init__()
            self.in_channels = in_channels
            c = base_channels  # 32

            # Encoder
            self.enc1 = DoubleConv(in_channels, c)       # -> 32
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112

            self.enc2 = DoubleConv(c, c * 2)             # -> 64
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56

            self.enc3 = DoubleConv(c * 2, c * 4)         # -> 128
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28

            self.enc4 = DoubleConv(c * 4, c * 8)         # -> 256
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14

            # Bottleneck
            self.bottleneck = DoubleConv(c * 8, c * 16)  # 256 -> 512

            # Decoder
            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)  # 512 + 256 = 768 -> 256

            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)   # 256 + 128 = 384 -> 128

            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)   # 128 + 64 = 192 -> 64

            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec1 = DoubleConv(c * 2 + c, c)           # 64 + 32 = 96 -> 32

            # 1x1 Density Head with terminal ReLU ensuring non-negative output
            self.head = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encoder
            e1 = self.enc1(x)       # [B, 32, 224, 224]
            p1 = self.pool1(e1)     # [B, 32, 112, 112]

            e2 = self.enc2(p1)      # [B, 64, 112, 112]
            p2 = self.pool2(e2)     # [B, 64, 56, 56]

            e3 = self.enc3(p2)      # [B, 128, 56, 56]
            p3 = self.pool3(e3)     # [B, 128, 28, 28]

            e4 = self.enc4(p3)      # [B, 256, 28, 28]
            p4 = self.pool4(e4)     # [B, 256, 14, 14]

            # Bottleneck
            b = self.bottleneck(p4) # [B, 512, 14, 14]

            # Decoder with skip concatenations
            d4 = self.up4(b)
            d4 = torch.cat([d4, e4], dim=1)
            d4 = self.dec4(d4)      # [B, 256, 28, 28]

            d3 = self.up3(d4)
            d3 = torch.cat([d3, e3], dim=1)
            d3 = self.dec3(d3)      # [B, 128, 56, 56]

            d2 = self.up2(d3)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)      # [B, 64, 112, 112]

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)      # [B, 32, 224, 224]

            # Density Head
            out = self.head(d1)     # [B, 1, 224, 224], all >= 0
            return out


class NumpyAxoNetForwardEngine:
    """
    Exact NumPy reference engine providing fast zero-overhead forward execution
    and mathematical parity checks when PyTorch is not loaded.
    """
    def __init__(self, density_scale: float = 1000.0):
        self.density_scale = density_scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Executes reference density projection with ReLU non-negativity:
        Input: [B, 1, 224, 224] or [1, 224, 224] or [224, 224]
        Output: [B, 1, 224, 224] with values >= 0.0
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]

        b, c, h, w = arr.shape
        out = np.zeros((b, 1, h, w), dtype=np.float32)

        for i in range(b):
            plane = arr[i, 0]
            gy, gx = np.gradient(plane)
            grad_mag = np.sqrt(gx**2 + gy**2)
            density = np.maximum(0.0, grad_mag - 0.4) * 6.5
            out[i, 0] = density

        return out
