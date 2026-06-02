#!/usr/bin/env python3
"""
File: utils/refine_spatial.py

Spatial ops for residual registration refinement.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def resize_volume_5d(volume: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
    if factor == 1:
        return volume
    size = tuple(max(1, dim // factor) for dim in volume.shape[-3:])
    if mode == "nearest":
        return F.interpolate(volume, size=size, mode=mode)
    return F.interpolate(volume, size=size, mode=mode, align_corners=True)


def downsample_dvf_5d(dvf: torch.Tensor, factor: int) -> torch.Tensor:
    """
    DVF tensor shape: [B, 3, X, Y, Z], voxel units on original grid.
    Downsampled DVF remains in voxel units of the downsampled grid.
    """
    if factor == 1:
        return dvf
    size = tuple(max(1, dim // factor) for dim in dvf.shape[-3:])
    return F.interpolate(dvf, size=size, mode="trilinear", align_corners=True) / float(factor)


def upsample_dvf_5d(dvf: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    source_shape = tuple(int(v) for v in dvf.shape[-3:])
    if source_shape == target_shape:
        return dvf
    scale = torch.tensor(
        [target_shape[0] / source_shape[0], target_shape[1] / source_shape[1], target_shape[2] / source_shape[2]],
        dtype=dvf.dtype,
        device=dvf.device,
    ).view(1, 3, 1, 1, 1)
    return F.interpolate(dvf, size=target_shape, mode="trilinear", align_corners=True) * scale


def make_identity_grid(shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, shape[0], device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, shape[1], device=device, dtype=dtype)
    z = torch.linspace(-1.0, 1.0, shape[2], device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1)


def voxel_dvf_to_normalized_grid(dvf: torch.Tensor) -> torch.Tensor:
    """
    DVF tensor shape: [B, 3, X, Y, Z], component order X,Y,Z.
    Returns grid_sample grid shape [B, X, Y, Z, 3], component order Z,Y,X.
    """
    b, _, x, y, z = dvf.shape
    base = make_identity_grid((x, y, z), dvf.device, dvf.dtype).unsqueeze(0).repeat(b, 1, 1, 1, 1)
    norm_x = 2.0 * dvf[:, 0] / max(1, x - 1)
    norm_y = 2.0 * dvf[:, 1] / max(1, y - 1)
    norm_z = 2.0 * dvf[:, 2] / max(1, z - 1)
    disp = torch.stack([norm_z, norm_y, norm_x], dim=-1)
    return base + disp


def warp_volume(volume: torch.Tensor, dvf: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    grid = voxel_dvf_to_normalized_grid(dvf)
    return F.grid_sample(volume, grid, mode=mode, padding_mode="border", align_corners=True)


def compose_additive(init_dvf: torch.Tensor, residual_dvf: torch.Tensor) -> torch.Tensor:
    return init_dvf + residual_dvf


def integrate_svf(svf: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Scaling-and-squaring approximation. Conservative use only.
    """
    if steps <= 0:
        return svf
    flow = svf / float(2**steps)
    for _ in range(steps):
        flow = flow + warp_volume(flow, flow, mode="bilinear")
    return flow


def spatial_gradients(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = field[..., 1:, :, :] - field[..., :-1, :, :]
    dy = field[..., :, 1:, :] - field[..., :, :-1, :]
    dz = field[..., :, :, 1:] - field[..., :, :, :-1]
    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = F.pad(dz, (0, 1, 0, 0, 0, 0))
    return dx, dy, dz


def jacobian_determinant(dvf: torch.Tensor) -> torch.Tensor:
    ux = dvf[:, 0]
    uy = dvf[:, 1]
    uz = dvf[:, 2]
    dux_dx, dux_dy, dux_dz = spatial_gradients(ux)
    duy_dx, duy_dy, duy_dz = spatial_gradients(uy)
    duz_dx, duz_dy, duz_dz = spatial_gradients(uz)

    j11 = 1.0 + dux_dx
    j12 = dux_dy
    j13 = dux_dz
    j21 = duy_dx
    j22 = 1.0 + duy_dy
    j23 = duy_dz
    j31 = duz_dx
    j32 = duz_dy
    j33 = 1.0 + duz_dz

    return j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)


def tensor_from_xyzc(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32, copy=False)).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor.to(device=device)


def tensor_from_volume(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    return tensor.to(device=device)


def xyzc_from_dvf_tensor(dvf: torch.Tensor) -> np.ndarray:
    return dvf.detach().cpu().squeeze(0).permute(1, 2, 3, 0).numpy().astype(np.float32, copy=False)


def ras_canonical_to_original_grid_xyzc(ras_dvf_xyzc: np.ndarray, reference_image: nib.Nifti1Image) -> np.ndarray:
    """
    Convert a RAS-canonical XYZC voxel DVF back to the original NIfTI grid.

    For this challenge training images are RAS and validation images are LPS.
    The current validation conversion therefore flips X/Y array axes and vector signs.
    This function is intentionally limited to axis flips, which matches inspected data.
    """
    axcodes = nib.aff2axcodes(reference_image.affine)
    output = ras_dvf_xyzc
    if axcodes == ("R", "A", "S"):
        return output.astype(np.float32, copy=False)
    if axcodes == ("L", "P", "S"):
        output = output[::-1, ::-1, :, :].copy()
        output[..., 0] *= -1.0
        output[..., 1] *= -1.0
        return output.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported output orientation {axcodes}; expected RAS or LPS.")
