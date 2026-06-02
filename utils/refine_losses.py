#!/usr/bin/env python3
"""
File: utils/refine_losses.py

Losses for conservative residual refinement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.refine_spatial import jacobian_determinant, spatial_gradients


def masked_mse_local_ncc(fixed: torch.Tensor, moving: torch.Tensor, mask: torch.Tensor, window: int = 7) -> torch.Tensor:
    """
    Stable masked LNCC loss. Returns 1 - NCC.
    """
    fixed = fixed.float()
    moving = moving.float()
    mask = mask.float()
    pad = window // 2
    kernel = torch.ones((1, 1, window, window, window), device=fixed.device, dtype=fixed.dtype)

    mask_sum = F.conv3d(mask, kernel, padding=pad).clamp_min(1.0)
    f_mean = F.conv3d(fixed * mask, kernel, padding=pad) / mask_sum
    m_mean = F.conv3d(moving * mask, kernel, padding=pad) / mask_sum

    f_centered = (fixed - f_mean) * mask
    m_centered = (moving - m_mean) * mask

    cross = F.conv3d(f_centered * m_centered, kernel, padding=pad)
    f_var = F.conv3d(f_centered.square(), kernel, padding=pad).clamp_min(1e-5)
    m_var = F.conv3d(m_centered.square(), kernel, padding=pad).clamp_min(1e-5)

    ncc = cross.square() / (f_var * m_var + 1e-5)
    valid = (mask_sum > 8).float()
    denom = valid.sum().clamp_min(1.0)
    return 1.0 - (ncc * valid).sum() / denom


def bending_energy(dvf: torch.Tensor) -> torch.Tensor:
    dx, dy, dz = spatial_gradients(dvf)
    dxx, _, _ = spatial_gradients(dx)
    _, dyy, _ = spatial_gradients(dy)
    _, _, dzz = spatial_gradients(dz)
    return dxx.square().mean() + dyy.square().mean() + dzz.square().mean()


def jacobian_folding_loss(dvf: torch.Tensor, epsilon: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    det = jacobian_determinant(dvf)
    folding = (det <= 0).float().mean() * 100.0
    loss = F.relu(epsilon - det).square().mean()
    return loss, folding
