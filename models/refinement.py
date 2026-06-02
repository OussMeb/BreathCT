from __future__ import annotations

import torch
from torch import nn

from utils.spatial import compose_pull_dvfs, integrate_svf, warp_pull
from models.icon_model import ICONInitializer, IdentityInitializer
from models.unet3d import UNet3D


class ResidualRegistrationNet(nn.Module):
    def __init__(
        self,
        *,
        initializer: ICONInitializer | None = None,
        in_channels: int = 5,
        base_channels: int = 8,
        max_channels: int = 64,
        predict: str = "svf",
        integration_steps: int = 7,
        max_residual_voxels: float = 8.0,
    ):
        super().__init__()
        if predict not in {"svf", "dvf"}:
            raise ValueError("predict must be 'svf' or 'dvf'.")

        self.initializer = initializer or IdentityInitializer()
        self.predict = predict
        self.integration_steps = integration_steps
        self.max_residual_voxels = float(max_residual_voxels)
        self.net = UNet3D(in_channels=in_channels, out_channels=3, base_channels=base_channels, max_channels=max_channels)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        foreground_mask: torch.Tensor | None = None,
        case_ids: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            init_dvf = self.initializer.predict(moving, fixed, case_ids=case_ids)

        init_warped = warp_pull(moving, init_dvf, mode="bilinear", padding_mode="border")
        diff = fixed - init_warped

        if foreground_mask is None:
            foreground_mask = torch.ones_like(fixed)

        features = torch.cat([fixed, moving, init_warped, foreground_mask, diff.abs()], dim=1)
        raw = self.net(features)
        residual_raw = torch.tanh(raw) * self.max_residual_voxels

        if self.predict == "svf":
            residual_dvf = integrate_svf(residual_raw, steps=self.integration_steps)
            residual_svf = residual_raw
        else:
            residual_dvf = residual_raw
            residual_svf = residual_raw.new_zeros(residual_raw.shape)

        final_dvf = compose_pull_dvfs(init_dvf, residual_dvf)
        warped = warp_pull(moving, final_dvf, mode="bilinear", padding_mode="border")

        return {
            "warped": warped,
            "final_dvf": final_dvf,
            "init_dvf": init_dvf,
            "init_warped": init_warped,
            "residual_dvf": residual_dvf,
            "residual_svf": residual_svf,
        }
