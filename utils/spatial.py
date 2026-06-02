from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_dvf(batch: int, shape_xyz: tuple[int, int, int], device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((batch, 3, *shape_xyz), device=device, dtype=dtype)


def _base_grid_zyx(
    batch: int,
    shape_xyz: tuple[int, int, int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    x, y, z = shape_xyz
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, device=device, dtype=dtype),
        torch.arange(y, device=device, dtype=dtype),
        torch.arange(x, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy, zz), dim=-1)
    return grid.unsqueeze(0).repeat(batch, 1, 1, 1, 1)


def _normalize_grid_zyx(grid: torch.Tensor, shape_xyz: tuple[int, int, int]) -> torch.Tensor:
    x, y, z = shape_xyz
    out = grid.clone()
    out[..., 0] = 2.0 * out[..., 0] / max(1, x - 1) - 1.0
    out[..., 1] = 2.0 * out[..., 1] / max(1, y - 1) - 1.0
    out[..., 2] = 2.0 * out[..., 2] / max(1, z - 1) - 1.0
    return out


def warp_pull(
    volume: torch.Tensor,
    dvf: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """Pull warp using CXYZ voxel DVF: output[x] = input[x + dvf[x]]."""
    if volume.ndim != 5:
        raise ValueError(f"Expected volume BCHW-like B,C,X,Y,Z, got {volume.shape}")
    if dvf.ndim != 5 or dvf.shape[1] != 3:
        raise ValueError(f"Expected DVF B,3,X,Y,Z, got {dvf.shape}")

    b, _, x, y, z = volume.shape
    if tuple(dvf.shape[2:]) != (x, y, z):
        raise ValueError(f"Volume/DVF grid mismatch: {volume.shape} vs {dvf.shape}")

    volume_zyx = volume.permute(0, 1, 4, 3, 2).contiguous()
    dvf_zyx = dvf.permute(0, 4, 3, 2, 1).contiguous()

    grid = _base_grid_zyx(b, (x, y, z), volume.device, volume.dtype)
    sample_grid = _normalize_grid_zyx(grid + dvf_zyx, (x, y, z))

    warped = F.grid_sample(
        volume_zyx,
        sample_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return warped.permute(0, 1, 4, 3, 2).contiguous()


def resize_dvf(dvf: torch.Tensor, shape_xyz: tuple[int, int, int]) -> torch.Tensor:
    if dvf.ndim != 5 or dvf.shape[1] != 3:
        raise ValueError(f"Expected DVF B,3,X,Y,Z, got {dvf.shape}")

    old_x, old_y, old_z = dvf.shape[2:]
    new_x, new_y, new_z = shape_xyz

    dvf_zyx = dvf.permute(0, 1, 4, 3, 2)
    resized = F.interpolate(dvf_zyx, size=(new_z, new_y, new_x), mode="trilinear", align_corners=True)
    resized = resized.permute(0, 1, 4, 3, 2).contiguous()

    scale = torch.tensor(
        [new_x / old_x, new_y / old_y, new_z / old_z],
        device=dvf.device,
        dtype=dvf.dtype,
    ).view(1, 3, 1, 1, 1)

    return resized * scale


def compose_pull_dvfs(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose pull fields so final sampling is x + second(x) + first(x + second(x))."""
    sampled_first = warp_pull(first, second, mode="bilinear", padding_mode="border")
    return second + sampled_first


def integrate_svf(svf: torch.Tensor, steps: int = 7) -> torch.Tensor:
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    dvf = svf / float(2 ** steps)
    for _ in range(steps):
        dvf = compose_pull_dvfs(dvf, dvf)
    return dvf


def gradient_3d(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = field[:, :, 1:, :, :] - field[:, :, :-1, :, :]
    dy = field[:, :, :, 1:, :] - field[:, :, :, :-1, :]
    dz = field[:, :, :, :, 1:] - field[:, :, :, :, :-1]
    return dx, dy, dz


def jacobian_determinant(dvf: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant of transform phi(x)=x+u(x), returned on inner grid."""
    if dvf.ndim != 5 or dvf.shape[1] != 3:
        raise ValueError(f"Expected DVF B,3,X,Y,Z, got {dvf.shape}")

    u = dvf
    ux = u[:, 0]
    uy = u[:, 1]
    uz = u[:, 2]

    dux_dx = (ux[:, 2:, 1:-1, 1:-1] - ux[:, :-2, 1:-1, 1:-1]) * 0.5
    dux_dy = (ux[:, 1:-1, 2:, 1:-1] - ux[:, 1:-1, :-2, 1:-1]) * 0.5
    dux_dz = (ux[:, 1:-1, 1:-1, 2:] - ux[:, 1:-1, 1:-1, :-2]) * 0.5

    duy_dx = (uy[:, 2:, 1:-1, 1:-1] - uy[:, :-2, 1:-1, 1:-1]) * 0.5
    duy_dy = (uy[:, 1:-1, 2:, 1:-1] - uy[:, 1:-1, :-2, 1:-1]) * 0.5
    duy_dz = (uy[:, 1:-1, 1:-1, 2:] - uy[:, 1:-1, 1:-1, :-2]) * 0.5

    duz_dx = (uz[:, 2:, 1:-1, 1:-1] - uz[:, :-2, 1:-1, 1:-1]) * 0.5
    duz_dy = (uz[:, 1:-1, 2:, 1:-1] - uz[:, 1:-1, :-2, 1:-1]) * 0.5
    duz_dz = (uz[:, 1:-1, 1:-1, 2:] - uz[:, 1:-1, 1:-1, :-2]) * 0.5

    j11 = 1.0 + dux_dx
    j12 = dux_dy
    j13 = dux_dz
    j21 = duy_dx
    j22 = 1.0 + duy_dy
    j23 = duy_dz
    j31 = duz_dx
    j32 = duz_dy
    j33 = 1.0 + duz_dz

    return (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
