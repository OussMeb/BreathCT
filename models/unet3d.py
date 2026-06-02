from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 8, max_channels: int = 64):
        super().__init__()

        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)
        c4 = min(base_channels * 8, max_channels)

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)

        self.out = nn.Conv3d(c1, out_channels, kernel_size=3, padding=1)
        self.reset_output_layer()

    def reset_output_layer(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.avg_pool3d(e2, kernel_size=2, stride=2))
        b = self.bottleneck(F.avg_pool3d(e3, kernel_size=2, stride=2))

        d3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
