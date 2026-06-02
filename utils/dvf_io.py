from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


def _as_axcodes(path: str | Path) -> tuple[str, str, str]:
    image = nib.load(str(path))
    return tuple(str(value) for value in aff2axcodes(image.affine))  # type: ignore[return-value]


def load_dvf_xyzc(path: str | Path) -> np.ndarray:
    data = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"Expected XYZC DVF with 3 components, got {data.shape}: {path}")
    return data


def save_dvf_xyzc(
    dvf_xyzc: np.ndarray,
    reference_image_path: str | Path,
    output_path: str | Path,
) -> None:
    reference = nib.load(str(reference_image_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if dvf_xyzc.ndim != 4 or dvf_xyzc.shape[-1] != 3:
        raise ValueError(f"Expected XYZC DVF with 3 components, got {dvf_xyzc.shape}")
    if tuple(dvf_xyzc.shape[:3]) != tuple(reference.shape[:3]):
        raise ValueError(f"DVF grid {dvf_xyzc.shape[:3]} does not match reference {reference.shape[:3]}")

    image = nib.Nifti1Image(np.asarray(dvf_xyzc, dtype=np.float32), reference.affine, header=reference.header)
    image.header.set_data_dtype(np.float32)
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(output))


def original_dvf_xyzc_to_canonical_ras_cxyz(
    dvf_xyzc: np.ndarray,
    reference_image_path: str | Path,
) -> np.ndarray:
    """Convert original-grid XYZC voxel DVF into canonical RAS CXYZ voxel DVF.

    This intentionally supports the orientations observed in this dataset:
    training RAS and validation LPS.
    """
    if dvf_xyzc.ndim != 4 or dvf_xyzc.shape[-1] != 3:
        raise ValueError(f"Expected XYZC DVF with 3 components, got {dvf_xyzc.shape}")

    axcodes = _as_axcodes(reference_image_path)
    cxyz = np.moveaxis(np.asarray(dvf_xyzc, dtype=np.float32), -1, 0)

    if axcodes == ("R", "A", "S"):
        return cxyz.astype(np.float32, copy=False)

    if axcodes == ("L", "P", "S"):
        out = cxyz.copy()
        out[0] *= -1.0
        out[1] *= -1.0
        out = out[:, ::-1, ::-1, :]
        return out.astype(np.float32, copy=False)

    raise NotImplementedError(f"Unsupported orientation {axcodes}; add explicit DVF reorientation.")


def canonical_ras_cxyz_to_original_dvf_xyzc(
    dvf_cxyz: np.ndarray,
    reference_image_path: str | Path,
) -> np.ndarray:
    """Inverse of original_dvf_xyzc_to_canonical_ras_cxyz."""
    if dvf_cxyz.ndim != 4 or dvf_cxyz.shape[0] != 3:
        raise ValueError(f"Expected CXYZ DVF with 3 components, got {dvf_cxyz.shape}")

    axcodes = _as_axcodes(reference_image_path)
    cxyz = np.asarray(dvf_cxyz, dtype=np.float32)

    if axcodes == ("R", "A", "S"):
        return np.moveaxis(cxyz, 0, -1).astype(np.float32)

    if axcodes == ("L", "P", "S"):
        out = cxyz[:, ::-1, ::-1, :].copy()
        out[0] *= -1.0
        out[1] *= -1.0
        return np.moveaxis(out, 0, -1).astype(np.float32)

    raise NotImplementedError(f"Unsupported orientation {axcodes}; add explicit DVF reorientation.")


def save_canonical_ras_dvf_xyzc(
    dvf_cxyz: np.ndarray,
    reference_image_path: str | Path,
    output_path: str | Path,
) -> None:
    """Save canonical RAS DVF as XYZC NIfTI for easier inspection.

    The saved data layout is XYZC even though the in-memory input is CXYZ.
    """
    reference = nib.as_closest_canonical(nib.load(str(reference_image_path)))
    dvf_xyzc = np.moveaxis(np.asarray(dvf_cxyz, dtype=np.float32), 0, -1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(dvf_xyzc, reference.affine, header=reference.header)
    image.header.set_data_dtype(np.float32)
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(output))


def dvf_stats(dvf_xyzc: np.ndarray) -> dict[str, object]:
    flat = np.asarray(dvf_xyzc, dtype=np.float32).reshape(-1, 3)
    mag = np.linalg.norm(flat, axis=1)
    return {
        "shape": [int(v) for v in dvf_xyzc.shape],
        "component_min": [float(v) for v in flat.min(axis=0)],
        "component_p1": [float(v) for v in np.percentile(flat, 1, axis=0)],
        "component_p50": [float(v) for v in np.percentile(flat, 50, axis=0)],
        "component_p99": [float(v) for v in np.percentile(flat, 99, axis=0)],
        "component_max": [float(v) for v in flat.max(axis=0)],
        "magnitude_p50": float(np.percentile(mag, 50)),
        "magnitude_p95": float(np.percentile(mag, 95)),
        "magnitude_p99": float(np.percentile(mag, 99)),
        "magnitude_max": float(np.max(mag)),
    }
