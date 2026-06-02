#!/usr/bin/env python3
"""
File: models/residual_refiner.py

Small 3D residual registration network for conservative refinement after uniGradICON.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualRefiner3d(nn.Module):
    """
    Lightweight 3D U-Net that predicts a small residual displacement/SVF.

    Input channels by default:
      moving, fixed, warped_moving_by_initializer, absolute_difference, foreground_mask
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 4,
        max_channels: int = 32,
        out_channels: int = 3,
        max_residual_voxels: float = 1.5,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)
        c4 = min(base_channels * 8, max_channels)

        self.enc1 = ConvBlock3d(in_channels, c1)
        self.enc2 = ConvBlock3d(c1, c2)
        self.enc3 = ConvBlock3d(c2, c3)
        self.bottleneck = ConvBlock3d(c3, c4)

        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3d(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(c1 + c1, c1)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=3, padding=1)
        self.max_residual_voxels = float(max_residual_voxels)
        self._init_head()

    def _init_head(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.tanh(self.head(d1)) * self.max_residual_voxels
