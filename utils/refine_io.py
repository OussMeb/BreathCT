#!/usr/bin/env python3
"""
File: utils/refine_io.py

I/O helpers for cached uniGradICON residual refinement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class PairCase:
    case_id: str
    fixed_path: Path
    moving_path: Path
    init_dvf_path: Path
    fixed_lobe_path: Path | None = None
    moving_lobe_path: Path | None = None


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_nifti_array(path: str | Path, dtype: np.dtype | type = np.float32) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    return data.astype(dtype, copy=False), image


def save_nifti_like(data: np.ndarray, reference_image: nib.Nifti1Image, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), reference_image.affine, reference_image.header)
    nib.save(image, str(output_path))


def normalize_ct_hu(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(volume.astype(np.float32, copy=False), hu_min, hu_max)
    return ((clipped - hu_min) / max(1e-6, hu_max - hu_min) * 2.0 - 1.0).astype(np.float32, copy=False)


def discover_training_cases(raw_root: Path, init_dvf_dir: Path) -> list[PairCase]:
    train_root = raw_root / "training"
    cases: list[PairCase] = []
    for fixed_path in sorted(train_root.glob("NLST_*_INSP.nii.gz")):
        case_id = fixed_path.name.replace("_INSP.nii.gz", "")
        moving_path = train_root / f"{case_id}_EXP.nii.gz"
        init_dvf_path = init_dvf_dir / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
        if not moving_path.exists() or not init_dvf_path.exists():
            continue
        cases.append(PairCase(case_id=case_id, fixed_path=fixed_path, moving_path=moving_path, init_dvf_path=init_dvf_path))
    return cases


def discover_validation_cases(raw_root: Path, init_dvf_dir: Path) -> list[PairCase]:
    ct_root = raw_root / "validation" / "ct_data"
    seg_root = raw_root / "validation" / "seg_net"
    cases: list[PairCase] = []
    for fixed_path in sorted(ct_root.glob("NLST_*_INSP.nii.gz")):
        case_id = fixed_path.name.replace("_INSP.nii.gz", "")
        moving_path = ct_root / f"{case_id}_EXP.nii.gz"
        init_dvf_path = init_dvf_dir / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
        if not moving_path.exists() or not init_dvf_path.exists():
            continue
        fixed_lobe = seg_root / f"{case_id}_INSP_lobe.nii.gz"
        moving_lobe = seg_root / f"{case_id}_EXP_lobe.nii.gz"
        cases.append(
            PairCase(
                case_id=case_id,
                fixed_path=fixed_path,
                moving_path=moving_path,
                init_dvf_path=init_dvf_path,
                fixed_lobe_path=fixed_lobe if fixed_lobe.exists() else None,
                moving_lobe_path=moving_lobe if moving_lobe.exists() else None,
            )
        )
    return cases
