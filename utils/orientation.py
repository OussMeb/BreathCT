from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


def canonicalize_image_to_ras(data: np.ndarray, image: nib.Nifti1Image) -> tuple[np.ndarray, nib.Nifti1Image]:
    canonical_image = nib.as_closest_canonical(image)
    canonical_data = np.asanyarray(canonical_image.dataobj)

    if canonical_data.shape != data.shape and data.shape == image.shape:
        # Shape changes are still valid for axis permutations; this guard catches impossible loads only.
        pass

    return np.asarray(canonical_data), canonical_image


def canonical_dvf_to_original_grid(dvf_canonical: np.ndarray, reference_image_path: str | Path) -> np.ndarray:
    """Convert CXYZ RAS voxel DVF to original-grid XYZC DVF.

    Supported orientations are exactly what the dataset showed:
    - RAS training-like grid: identity
    - LPS validation grid: flip X/Y spatial axes and negate X/Y components
    """
    image = nib.load(str(reference_image_path))
    axcodes = tuple(str(v) for v in aff2axcodes(image.affine))

    if dvf_canonical.ndim != 4 or dvf_canonical.shape[0] != 3:
        raise ValueError(f"Expected canonical DVF CXYZ with 3 components, got {dvf_canonical.shape}")

    dvf = np.asarray(dvf_canonical, dtype=np.float32)

    if axcodes == ("R", "A", "S"):
        original_cxyz = dvf
    elif axcodes == ("L", "P", "S"):
        original_cxyz = dvf[:, ::-1, ::-1, :].copy()
        original_cxyz[0] *= -1.0
        original_cxyz[1] *= -1.0
    else:
        raise NotImplementedError(
            f"Unsupported original orientation {axcodes}. "
            "Add explicit DVF reorientation before using this case."
        )

    return np.moveaxis(original_cxyz, 0, -1).astype(np.float32)


def save_dvf_nifti_xyzc(dvf_xyzc: np.ndarray, reference_image_path: str | Path, output_path: str | Path) -> None:
    reference = nib.load(str(reference_image_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if dvf_xyzc.shape[:3] != reference.shape[:3] or dvf_xyzc.shape[-1] != 3:
        raise ValueError(
            f"DVF shape {dvf_xyzc.shape} does not match reference grid {reference.shape} + vector dim."
        )

    image = nib.Nifti1Image(np.asarray(dvf_xyzc, dtype=np.float32), reference.affine, header=reference.header)
    image.header.set_data_dtype(np.float32)
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(output))
