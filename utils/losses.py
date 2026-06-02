from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.spatial import gradient_3d, jacobian_determinant


def lncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    *,
    window: int = 9,
    mask: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if moving.shape != fixed.shape:
        raise ValueError(f"LNCC shape mismatch: {moving.shape} vs {fixed.shape}")
    if window % 2 == 0:
        raise ValueError("LNCC window must be odd.")

    original_dtype = moving.dtype
    moving_f = torch.nan_to_num(moving.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    fixed_f = torch.nan_to_num(fixed.float(), nan=0.0, posinf=1.0, neginf=-1.0)

    padding = window // 2
    channels = moving_f.shape[1]
    kernel = torch.ones(
        (channels, 1, window, window, window),
        device=moving_f.device,
        dtype=torch.float32,
    )

    m = moving_f.permute(0, 1, 4, 3, 2).contiguous()
    f = fixed_f.permute(0, 1, 4, 3, 2).contiguous()

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, kernel, padding=padding, groups=channels)

    win_size = float(window ** 3)
    m_sum = conv(m)
    f_sum = conv(f)
    m2_sum = conv(m * m)
    f2_sum = conv(f * f)
    mf_sum = conv(m * f)

    mean_m = m_sum / win_size
    mean_f = f_sum / win_size

    cross = mf_sum - mean_f * m_sum - mean_m * f_sum + mean_m * mean_f * win_size
    m_var = m2_sum - 2.0 * mean_m * m_sum + mean_m.square() * win_size
    f_var = f2_sum - 2.0 * mean_f * f_sum + mean_f.square() * win_size

    m_var = m_var.clamp_min(eps)
    f_var = f_var.clamp_min(eps)

    cc = cross.square() / (m_var * f_var + eps)
    cc = torch.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)
    cc = cc.permute(0, 1, 4, 3, 2).contiguous()

    if mask is not None:
        valid = (mask.float() > 0.5).to(cc.dtype)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        denom = valid.sum().clamp_min(1.0)
        value = 1.0 - (cc * valid).sum() / denom
    else:
        value = 1.0 - cc.mean()

    return value.to(original_dtype)


def multiscale_lncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    *,
    windows: tuple[int, ...] = (9, 15, 21),
    weights: tuple[float, ...] = (0.5, 0.3, 0.2),
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if len(windows) != len(weights):
        raise ValueError("windows and weights must have equal lengths.")

    total = moving.new_tensor(0.0)
    norm = float(sum(weights))
    for window, weight in zip(windows, weights):
        total = total + float(weight) * lncc_loss(moving, fixed, window=window, mask=mask)
    return total / max(1e-6, norm)


def gradient_smoothness_loss(dvf: torch.Tensor) -> torch.Tensor:
    dvf_f = torch.nan_to_num(dvf.float(), nan=0.0, posinf=0.0, neginf=0.0)
    dx, dy, dz = gradient_3d(dvf_f)
    return dx.square().mean() + dy.square().mean() + dz.square().mean()


def bending_energy_loss(dvf: torch.Tensor) -> torch.Tensor:
    dvf_f = torch.nan_to_num(dvf.float(), nan=0.0, posinf=0.0, neginf=0.0)
    dx, dy, dz = gradient_3d(dvf_f)
    dxx = dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :]
    dyy = dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]
    dzz = dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]
    return dxx.square().mean() + dyy.square().mean() + dzz.square().mean()


def jacobian_folding_loss(dvf: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    dvf_f = torch.nan_to_num(dvf.float(), nan=0.0, posinf=0.0, neginf=0.0)
    det = jacobian_determinant(dvf_f)
    det = torch.nan_to_num(det, nan=-1.0, posinf=1.0, neginf=-1.0)
    return F.relu(float(epsilon) - det).square().mean()


def folding_percentage(dvf: torch.Tensor) -> torch.Tensor:
    dvf_f = torch.nan_to_num(dvf.float(), nan=0.0, posinf=0.0, neginf=0.0)
    det = jacobian_determinant(dvf_f)
    det = torch.nan_to_num(det, nan=-1.0, posinf=1.0, neginf=-1.0)
    return (det <= 0).to(torch.float32).mean() * 100.0


def registration_loss(
    warped_moving: torch.Tensor,
    fixed: torch.Tensor,
    dvf: torch.Tensor,
    *,
    foreground_mask: torch.Tensor | None,
    image_weight: float,
    bending_weight: float,
    jacobian_weight: float,
    smooth_weight: float = 0.0,
    lncc_windows: tuple[int, ...] = (9, 15, 21),
    lncc_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image = multiscale_lncc_loss(
        warped_moving,
        fixed,
        windows=lncc_windows,
        weights=lncc_weights,
        mask=foreground_mask,
    )
    bending = bending_energy_loss(dvf)
    jac = jacobian_folding_loss(dvf)
    smooth = gradient_smoothness_loss(dvf) if smooth_weight > 0 else dvf.new_tensor(0.0).float()
    fold = folding_percentage(dvf)

    total = (
        float(image_weight) * image.float()
        + float(bending_weight) * bending.float()
        + float(jacobian_weight) * jac.float()
        + float(smooth_weight) * smooth.float()
    )
    total = torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)

    return total, {
        "total": total.detach(),
        "image": image.detach().float(),
        "bending": bending.detach().float(),
        "jacobian": jac.detach().float(),
        "smooth": smooth.detach().float(),
        "folding_pct": fold.detach().float(),
    }
