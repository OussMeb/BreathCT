from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import aff2axcodes
from torch.utils.data import DataLoader, Dataset

from utils.config import DataConfig
from utils.orientation import canonicalize_image_to_ras


CASE_RE = re.compile(r"^(NLST_\d{4})_(EXP|INSP)(?:_(lobe|fissure))?\.nii\.gz$")


@dataclass(frozen=True)
class CaseFiles:
    split: str
    case_id: str
    exp_ct: Path
    insp_ct: Path
    source_root: Path
    exp_lobe: Path | None = None
    insp_lobe: Path | None = None
    exp_fissure: Path | None = None
    insp_fissure: Path | None = None
    foreground_mask: Path | None = None


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name))


def _collect_ct_pairs(split_root: Path, recursive: bool) -> dict[str, dict[str, Path]]:
    pattern = "**/NLST_*_*.nii.gz" if recursive else "NLST_*_*.nii.gz"
    ct_by_case: dict[str, dict[str, Path]] = {}

    for path in sorted(split_root.glob(pattern), key=natural_key):
        match = CASE_RE.match(path.name)
        if not match:
            continue

        case_id, phase, file_type = match.groups()
        if file_type is not None:
            continue

        ct_by_case.setdefault(case_id, {})[phase] = path

    return ct_by_case


def _collect_labels(split_root: Path, case_id: str) -> dict[str, Path]:
    labels: dict[str, Path] = {}

    for path in sorted(split_root.rglob(f"{case_id}_*.nii.gz"), key=natural_key):
        match = CASE_RE.match(path.name)
        if not match:
            continue

        _, phase, file_type = match.groups()
        if file_type is None:
            continue

        labels[f"{phase}_{file_type}"] = path

    return labels


def discover_training_cases(train_data_root: str | Path) -> list[CaseFiles]:
    root = Path(train_data_root).expanduser()
    split_root = root / "training"
    if not split_root.exists():
        raise FileNotFoundError(f"Training split not found: {split_root}")

    cases: list[CaseFiles] = []
    ct_by_case = _collect_ct_pairs(split_root, recursive=False)

    for case_id in sorted(ct_by_case):
        phases = ct_by_case[case_id]
        if "EXP" not in phases or "INSP" not in phases:
            continue

        mask = split_root / "masks" / f"{case_id}_foreground_mask.nii.gz"
        cases.append(
            CaseFiles(
                split="training",
                case_id=case_id,
                exp_ct=phases["EXP"],
                insp_ct=phases["INSP"],
                source_root=split_root,
                foreground_mask=mask if mask.exists() else None,
            )
        )

    return cases


def discover_validation_cases(raw_data_root: str | Path) -> list[CaseFiles]:
    root = Path(raw_data_root).expanduser()
    split_root = root / "validation"
    if not split_root.exists():
        raise FileNotFoundError(f"Validation split not found: {split_root}")

    cases: list[CaseFiles] = []
    ct_by_case = _collect_ct_pairs(split_root, recursive=True)

    for case_id in sorted(ct_by_case):
        phases = ct_by_case[case_id]
        if "EXP" not in phases or "INSP" not in phases:
            continue

        labels = _collect_labels(split_root, case_id)
        cases.append(
            CaseFiles(
                split="validation",
                case_id=case_id,
                exp_ct=phases["EXP"],
                insp_ct=phases["INSP"],
                source_root=split_root,
                exp_lobe=labels.get("EXP_lobe"),
                insp_lobe=labels.get("INSP_lobe"),
                exp_fissure=labels.get("EXP_fissure"),
                insp_fissure=labels.get("INSP_fissure"),
            )
        )

    return cases


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {data.shape}: {path}")
    return data, image


def normalize_ct(data: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(np.asarray(data, dtype=np.float32), hu_min, hu_max)
    return ((clipped - hu_min) / max(1e-6, hu_max - hu_min) * 2.0 - 1.0).astype(np.float32)


def foreground_from_normalized(
    moving: np.ndarray,
    fixed: np.ndarray,
    *,
    hu_min: float,
    hu_max: float,
    threshold_hu: float,
) -> np.ndarray:
    threshold = ((threshold_hu - hu_min) / max(1e-6, hu_max - hu_min) * 2.0 - 1.0)
    mask = np.logical_or(moving > threshold, fixed > threshold)
    return mask.astype(np.float32)


def to_cxyz_tensor(data: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(data), dtype=dtype).unsqueeze(0)


def _load_canonical_ct(path: Path, config: DataConfig) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, ...]]:
    raw, image = load_nifti(path)
    axcodes = tuple(str(v) for v in aff2axcodes(image.affine))
    canonical, canonical_image = canonicalize_image_to_ras(raw, image)
    normalized = normalize_ct(canonical, config.hu_min, config.hu_max)
    return normalized, canonical_image, axcodes


def _load_canonical_label(path: Path) -> np.ndarray:
    raw, image = load_nifti(path)
    canonical, _ = canonicalize_image_to_ras(raw, image)
    return canonical.astype(np.int16, copy=False)


class Learn2BreathDataset(Dataset[dict[str, Any]]):
    def __init__(self, cases: list[CaseFiles], config: DataConfig):
        if not cases:
            raise ValueError("Dataset received zero cases.")
        self.cases = cases
        self.config = config

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index]

        moving, moving_can_img, exp_axcodes = _load_canonical_ct(case.exp_ct, self.config)
        fixed, fixed_can_img, insp_axcodes = _load_canonical_ct(case.insp_ct, self.config)

        if moving.shape != fixed.shape:
            raise ValueError(f"{case.case_id}: moving/fixed shape mismatch {moving.shape} vs {fixed.shape}")

        if case.foreground_mask is not None:
            mask_raw, mask_img = load_nifti(case.foreground_mask)
            mask, _ = canonicalize_image_to_ras(mask_raw, mask_img)
            foreground = (mask > 0).astype(np.float32)
        else:
            foreground = foreground_from_normalized(
                moving,
                fixed,
                hu_min=self.config.hu_min,
                hu_max=self.config.hu_max,
                threshold_hu=self.config.foreground_threshold_hu,
            )

        sample: dict[str, Any] = {
            "case_id": case.case_id,
            "split": case.split,
            "moving": to_cxyz_tensor(moving),
            "fixed": to_cxyz_tensor(fixed),
            "foreground_mask": to_cxyz_tensor(foreground),
            "meta": {
                "tensor_axis_order": "CXYZ",
                "exp_original_axcodes": exp_axcodes,
                "insp_original_axcodes": insp_axcodes,
                "canonical_axcodes": tuple(self.config.canonical_axcodes),
                "canonical_shape": tuple(int(v) for v in fixed.shape),
                "original_insp_path": str(case.insp_ct),
                "original_exp_path": str(case.exp_ct),
                "dvf_required_grid": "original_INSP_grid",
            },
        }

        if case.exp_lobe and case.insp_lobe:
            sample["moving_lobe"] = to_cxyz_tensor(_load_canonical_label(case.exp_lobe), torch.long)
            sample["fixed_lobe"] = to_cxyz_tensor(_load_canonical_label(case.insp_lobe), torch.long)

        if case.exp_fissure and case.insp_fissure:
            sample["moving_fissure"] = to_cxyz_tensor(_load_canonical_label(case.exp_fissure), torch.long)
            sample["fixed_fissure"] = to_cxyz_tensor(_load_canonical_label(case.insp_fissure), torch.long)

        return sample


def build_dataset(split: str, config: DataConfig) -> Learn2BreathDataset:
    if split == "training":
        cases = discover_training_cases(config.train_data_root)
    elif split == "validation":
        cases = discover_validation_cases(config.raw_data_root)
    else:
        raise ValueError(f"Unsupported split: {split}")

    return Learn2BreathDataset(cases, config)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    tensor_keys = {
        "moving",
        "fixed",
        "foreground_mask",
        "moving_lobe",
        "fixed_lobe",
        "moving_fissure",
        "fixed_fissure",
    }

    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in tensor_keys and all(torch.is_tensor(value) for value in values):
            output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values

    return output


def build_dataloader(
    split: str,
    config: DataConfig,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = build_dataset(split, config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
