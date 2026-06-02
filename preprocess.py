#!/usr/bin/env python3
"""
File: preprocess_learn2breath_fullgrid.py

Safe Learn2Breath preprocessing.

Default pipeline:
- discover paired EXP/INSP cases under training/ and validation/
- keep the original grid by default: no trimming, no resizing
- clip CT HU to [-1000, 600]
- scale CT to [-1, 1]
- copy validation lobe/fissure labels unchanged by default
- optionally create foreground masks for masked image losses
- optionally enable safe pair-consistent trimming and/or resizing for experiments
- save JSON manifest + CSV summary

Run:
    PYTORCH_NVML_BASED_CUDA_CHECK=0 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    python preprocess_learn2breath_fullgrid.py \
      --config configs/preprocess_fullgrid.yaml \
      --data-root "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data" \
      --output-name Learn2Breath_preprocessed_fullgrid_v1 \
      --hu-min -1000 \
      --hu-max 600 \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


CASE_RE = re.compile(r"^(NLST_\d{4})_(EXP|INSP)(?:_(lobe|fissure))?\.nii\.gz$")


@dataclass(frozen=True)
class CaseFiles:
    split: str
    case_id: str
    exp_ct: Path
    insp_ct: Path
    exp_lobe: Path | None = None
    insp_lobe: Path | None = None
    exp_fissure: Path | None = None
    insp_fissure: Path | None = None


@dataclass(frozen=True)
class GridInfo:
    original_shape: tuple[int, int, int]
    processed_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    processed_spacing: tuple[float, float, float]
    exp_orientation: tuple[str, str, str]
    insp_orientation: tuple[str, str, str]
    exp_insp_affine_equal: bool


@dataclass(frozen=True)
class TrimInfo:
    enabled: bool
    axis: int
    edge_search: int
    exp_start_empty: int
    exp_end_empty: int
    insp_start_empty: int
    insp_end_empty: int
    trim_start: int
    trim_end: int


@dataclass(frozen=True)
class ProcessedCase:
    split: str
    case_id: str
    grid: GridInfo
    trim: TrimInfo
    resize_ratio: float
    outputs: dict[str, str]
    labels_present: dict[str, bool]
    ct_stats_before: dict[str, dict[str, float]]
    ct_stats_after: dict[str, dict[str, float]]
    warnings: list[str]


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml") from exc
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError("Config must be .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a key-value mapping.")

    if "ratio" in data and "resize_ratio" not in data:
        data["resize_ratio"] = data.pop("ratio")

    return data


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, remaining = pre.parse_known_args()

    config = load_config(known.config)

    parser = argparse.ArgumentParser(
        description="Safe full-grid preprocessing for Learn2Breath.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre],
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data"),
        help="Root folder containing training/ and validation/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Exact output folder. Overrides --output-name when set.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="Learn2Breath_preprocessed_fullgrid_v1",
        help="Output folder name created next to data-root when --output-root is not set.",
    )
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=600.0)

    parser.add_argument(
        "--resize-ratio",
        "--ratio",
        dest="resize_ratio",
        type=float,
        default=1.0,
        help="Spatial resize ratio. Keep 1.0 for final pipeline.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))

    parser.add_argument("--enable-trim", dest="enable_trim", action="store_true")
    parser.add_argument("--disable-trim", dest="enable_trim", action="store_false")
    parser.set_defaults(enable_trim=False)

    parser.add_argument("--edge-search", type=int, default=30)
    parser.add_argument("--axis", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--empty-range-threshold", type=float, default=70.0)
    parser.add_argument("--empty-std-threshold", type=float, default=25.0)
    parser.add_argument("--empty-nonbg-threshold", type=float, default=0.015)
    parser.add_argument("--center-fraction", type=float, default=0.70)
    parser.add_argument("--min-depth-after-trim", type=int, default=96)

    parser.add_argument("--make-foreground-masks", dest="make_foreground_masks", action="store_true")
    parser.add_argument("--no-foreground-masks", dest="make_foreground_masks", action="store_false")
    parser.set_defaults(make_foreground_masks=True)
    parser.add_argument("--foreground-threshold", type=float, default=-995.0)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-affine", action="store_true", help="Fail if EXP/INSP affines differ.")
    parser.add_argument("--dry-run", action="store_true")

    valid_defaults = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid_defaults)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining)
    args.config = known.config

    args.data_root = Path(args.data_root).expanduser()
    args.output_root = Path(args.output_root).expanduser() if args.output_root is not None else None
    args.config = known.config
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    if args.hu_min >= args.hu_max:
        raise ValueError("--hu-min must be smaller than --hu-max")

    if not 0.1 <= args.resize_ratio <= 1.0:
        raise ValueError("--resize-ratio must be in [0.1, 1.0]. Use 1.0 for no resize.")

    if args.edge_search < 0:
        raise ValueError("--edge-search must be non-negative")

    if not 0.25 <= args.center_fraction <= 1.0:
        raise ValueError("--center-fraction must be in [0.25, 1.0]")


def output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    return args.data_root.parent / args.output_name


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name))


def discover_cases(data_root: Path) -> list[CaseFiles]:
    cases: list[CaseFiles] = []

    for split in ("training", "validation"):
        split_root = data_root / split
        if not split_root.exists():
            continue

        ct_by_case: dict[str, dict[str, Path]] = {}

        for path in sorted(split_root.glob("NLST_*_*.nii.gz"), key=natural_key):
            match = CASE_RE.match(path.name)
            if not match:
                continue

            case_id, phase, file_type = match.groups()
            if file_type is not None:
                continue

            ct_by_case.setdefault(case_id, {})[phase] = path

        for case_id in sorted(ct_by_case):
            phases = ct_by_case[case_id]
            if "EXP" not in phases or "INSP" not in phases:
                continue

            labels = find_labels(split_root, case_id)
            cases.append(
                CaseFiles(
                    split=split,
                    case_id=case_id,
                    exp_ct=phases["EXP"],
                    insp_ct=phases["INSP"],
                    exp_lobe=labels.get("EXP_lobe"),
                    insp_lobe=labels.get("INSP_lobe"),
                    exp_fissure=labels.get("EXP_fissure"),
                    insp_fissure=labels.get("INSP_fissure"),
                )
            )

    return cases


def find_labels(split_root: Path, case_id: str) -> dict[str, Path]:
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


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape}: {path}")
    return data, image


def spacing(image: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = image.header.get_zooms()[:3]
    return tuple(float(v) for v in zooms)


def orientation(image: nib.Nifti1Image) -> tuple[str, str, str]:
    return tuple(str(v) for v in nib.aff2axcodes(image.affine))  # type: ignore[return-value]


def ct_stats(data: np.ndarray) -> dict[str, float]:
    finite = np.asarray(data[np.isfinite(data)], dtype=np.float32)
    if finite.size == 0:
        return {
            "min": math.nan,
            "p0_5": math.nan,
            "p1": math.nan,
            "p50": math.nan,
            "p99": math.nan,
            "p99_5": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
        }

    return {
        "min": float(np.min(finite)),
        "p0_5": float(np.percentile(finite, 0.5)),
        "p1": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "p99_5": float(np.percentile(finite, 99.5)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def normalize_ct(data: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(np.asarray(data, dtype=np.float32), hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min) * 2.0 - 1.0).astype(np.float32)


def take_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return data[index, :, :]
    if axis == 1:
        return data[:, index, :]
    return data[:, :, index]


def center_crop_2d(data: np.ndarray, fraction: float) -> np.ndarray:
    h, w = data.shape
    ch = max(1, int(round(h * fraction)))
    cw = max(1, int(round(w * fraction)))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return data[y0 : y0 + ch, x0 : x0 + cw]


def is_empty_slice(
    slice_2d: np.ndarray,
    *,
    center_fraction: float,
    range_threshold: float,
    std_threshold: float,
    nonbg_threshold: float,
) -> bool:
    center = center_crop_2d(np.asarray(slice_2d, dtype=np.float32), center_fraction)
    finite = center[np.isfinite(center)]
    if finite.size == 0:
        return True

    p1 = float(np.percentile(finite, 1))
    p50 = float(np.percentile(finite, 50))
    p99 = float(np.percentile(finite, 99))
    robust_range = p99 - p1
    std = float(np.std(finite))
    non_background = np.abs(finite - p50) > max(10.0, 0.25 * range_threshold)
    non_background_fraction = float(np.mean(non_background))

    return (
        robust_range <= range_threshold
        or std <= std_threshold
        or non_background_fraction <= nonbg_threshold
    )


def count_empty_from_start(data: np.ndarray, args: argparse.Namespace) -> int:
    depth = data.shape[args.axis]
    limit = min(args.edge_search, depth)
    count = 0

    for idx in range(limit):
        if is_empty_slice(
            take_slice(data, args.axis, idx),
            center_fraction=args.center_fraction,
            range_threshold=args.empty_range_threshold,
            std_threshold=args.empty_std_threshold,
            nonbg_threshold=args.empty_nonbg_threshold,
        ):
            count += 1
        else:
            break

    return count


def count_empty_from_end(data: np.ndarray, args: argparse.Namespace) -> int:
    depth = data.shape[args.axis]
    limit = min(args.edge_search, depth)
    count = 0

    for offset in range(limit):
        idx = depth - 1 - offset
        if is_empty_slice(
            take_slice(data, args.axis, idx),
            center_fraction=args.center_fraction,
            range_threshold=args.empty_range_threshold,
            std_threshold=args.empty_std_threshold,
            nonbg_threshold=args.empty_nonbg_threshold,
        ):
            count += 1
        else:
            break

    return count


def compute_safe_pair_trim(exp: np.ndarray, insp: np.ndarray, args: argparse.Namespace) -> TrimInfo:
    exp_start = count_empty_from_start(exp, args)
    exp_end = count_empty_from_end(exp, args)
    insp_start = count_empty_from_start(insp, args)
    insp_end = count_empty_from_end(insp, args)

    if not args.enable_trim:
        return TrimInfo(
            enabled=False,
            axis=args.axis,
            edge_search=args.edge_search,
            exp_start_empty=exp_start,
            exp_end_empty=exp_end,
            insp_start_empty=insp_start,
            insp_end_empty=insp_end,
            trim_start=0,
            trim_end=0,
        )

    # Keep anatomy from both phases. Only remove slices empty in both EXP and INSP.
    trim_start = min(exp_start, insp_start)
    trim_end = min(exp_end, insp_end)

    depth = exp.shape[args.axis]
    if depth - trim_start - trim_end < args.min_depth_after_trim:
        allowed = max(0, depth - args.min_depth_after_trim)
        total = trim_start + trim_end
        if total > 0:
            trim_start = min(trim_start, int(round(allowed * trim_start / total)))
            trim_end = min(trim_end, allowed - trim_start)

    return TrimInfo(
        enabled=True,
        axis=args.axis,
        edge_search=args.edge_search,
        exp_start_empty=exp_start,
        exp_end_empty=exp_end,
        insp_start_empty=insp_start,
        insp_end_empty=insp_end,
        trim_start=trim_start,
        trim_end=trim_end,
    )


def crop_along_axis(data: np.ndarray, axis: int, start: int, end: int) -> np.ndarray:
    if start == 0 and end == 0:
        return data

    stop = data.shape[axis] - end if end > 0 else data.shape[axis]
    if stop <= start:
        raise ValueError(f"Invalid crop: axis={axis}, start={start}, end={end}, shape={data.shape}")

    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = slice(start, stop)
    return data[tuple(slices)]


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_array(data: np.ndarray, ratio: float, *, is_label: bool, device: torch.device) -> np.ndarray:
    if abs(ratio - 1.0) < 1e-8:
        return data

    original_shape = tuple(int(v) for v in data.shape)
    new_shape = tuple(max(1, int(round(v * ratio))) for v in original_shape)

    tensor = torch.from_numpy(np.asarray(data)).to(device=device)
    tensor = tensor[None, None]

    if is_label:
        resized = F.interpolate(tensor.float(), size=new_shape, mode="nearest")
        return resized[0, 0].cpu().numpy().astype(data.dtype, copy=False)

    resized = F.interpolate(tensor.float(), size=new_shape, mode="trilinear", align_corners=False)
    return resized[0, 0].cpu().numpy().astype(np.float32, copy=False)


def updated_affine_for_resize(affine: np.ndarray, original_shape: tuple[int, int, int], new_shape: tuple[int, int, int]) -> np.ndarray:
    if original_shape == new_shape:
        return affine.copy()

    updated = affine.copy()
    for axis in range(3):
        scale = original_shape[axis] / new_shape[axis]
        updated[:3, axis] *= scale
    return updated


def updated_affine_for_crop(affine: np.ndarray, axis: int, start: int) -> np.ndarray:
    if start == 0:
        return affine.copy()

    updated = affine.copy()
    updated[:3, 3] = affine[:3, 3] + affine[:3, axis] * float(start)
    return updated


def processed_affine(
    source_affine: np.ndarray,
    original_shape: tuple[int, int, int],
    cropped_shape: tuple[int, int, int],
    processed_shape: tuple[int, int, int],
    trim: TrimInfo,
) -> np.ndarray:
    affine = updated_affine_for_crop(source_affine, trim.axis, trim.trim_start)
    return updated_affine_for_resize(affine, cropped_shape, processed_shape)


def save_nifti(
    data: np.ndarray,
    *,
    source_image: nib.Nifti1Image,
    affine: np.ndarray,
    output_path: Path,
    dtype: np.dtype | type,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.asarray(data, dtype=dtype)

    header = source_image.header.copy()
    header.set_data_dtype(dtype)

    image = nib.Nifti1Image(clean, affine, header=header)
    image.set_qform(affine, code=int(source_image.header["qform_code"]) or 1)
    image.set_sform(affine, code=int(source_image.header["sform_code"]) or 1)
    nib.save(image, str(output_path))


def build_foreground_mask(exp_raw: np.ndarray, insp_raw: np.ndarray, threshold: float) -> np.ndarray:
    mask = (np.asarray(exp_raw) > threshold) | (np.asarray(insp_raw) > threshold)
    return mask.astype(np.uint8)


def output_paths(case: CaseFiles, root: Path) -> dict[str, Path]:
    split_root = root / case.split
    seg_root = split_root / "seg_net"
    mask_root = split_root / "masks"

    paths = {
        "exp_ct": split_root / f"{case.case_id}_EXP.nii.gz",
        "insp_ct": split_root / f"{case.case_id}_INSP.nii.gz",
        "foreground_mask": mask_root / f"{case.case_id}_foreground_mask.nii.gz",
    }

    if case.exp_lobe:
        paths["exp_lobe"] = seg_root / f"{case.case_id}_EXP_lobe.nii.gz"
    if case.insp_lobe:
        paths["insp_lobe"] = seg_root / f"{case.case_id}_INSP_lobe.nii.gz"
    if case.exp_fissure:
        paths["exp_fissure"] = seg_root / f"{case.case_id}_EXP_fissure.nii.gz"
    if case.insp_fissure:
        paths["insp_fissure"] = seg_root / f"{case.case_id}_INSP_fissure.nii.gz"

    return paths


def ensure_can_write(paths: dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return

    existing = [path for path in paths.values() if path.exists()]
    if existing:
        first = existing[0]
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {first}")


def process_label(
    label_path: Path,
    *,
    source_image: nib.Nifti1Image,
    output_path: Path,
    trim: TrimInfo,
    resize_ratio: float,
    device: torch.device,
    affine: np.ndarray,
    dry_run: bool,
) -> None:
    label, label_image = load_nifti(label_path)
    if tuple(label.shape) != tuple(source_image.shape):
        raise ValueError(f"Label shape mismatch: {label_path}, label={label.shape}, source={source_image.shape}")

    label = crop_along_axis(label, trim.axis, trim.trim_start, trim.trim_end)
    label = resize_array(label, resize_ratio, is_label=True, device=device)

    if not dry_run:
        save_nifti(label, source_image=label_image, affine=affine, output_path=output_path, dtype=np.uint16)


def process_case(case: CaseFiles, args: argparse.Namespace, root: Path, device: torch.device) -> ProcessedCase:
    paths = output_paths(case, root)
    if not args.make_foreground_masks:
        paths.pop("foreground_mask", None)
    ensure_can_write(paths, args.overwrite)

    exp_raw, exp_image = load_nifti(case.exp_ct)
    insp_raw, insp_image = load_nifti(case.insp_ct)

    if exp_raw.shape != insp_raw.shape:
        raise ValueError(f"EXP/INSP shape mismatch for {case.case_id}: {exp_raw.shape} vs {insp_raw.shape}")

    warnings: list[str] = []
    affines_equal = bool(np.allclose(exp_image.affine, insp_image.affine, atol=1e-5))
    if not affines_equal:
        message = f"{case.case_id}: EXP/INSP affines differ."
        if args.strict_affine:
            raise ValueError(message)
        warnings.append(message)

    before_stats = {
        "exp": ct_stats(exp_raw),
        "insp": ct_stats(insp_raw),
    }

    trim = compute_safe_pair_trim(exp_raw, insp_raw, args)
    exp_cropped_raw = crop_along_axis(exp_raw, trim.axis, trim.trim_start, trim.trim_end)
    insp_cropped_raw = crop_along_axis(insp_raw, trim.axis, trim.trim_start, trim.trim_end)

    exp_norm = normalize_ct(exp_cropped_raw, args.hu_min, args.hu_max)
    insp_norm = normalize_ct(insp_cropped_raw, args.hu_min, args.hu_max)

    cropped_shape = tuple(int(v) for v in exp_norm.shape)
    exp_norm = resize_array(exp_norm, args.resize_ratio, is_label=False, device=device)
    insp_norm = resize_array(insp_norm, args.resize_ratio, is_label=False, device=device)

    processed_shape = tuple(int(v) for v in exp_norm.shape)
    exp_affine = processed_affine(exp_image.affine, tuple(exp_raw.shape), cropped_shape, processed_shape, trim)
    insp_affine = processed_affine(insp_image.affine, tuple(insp_raw.shape), cropped_shape, processed_shape, trim)

    if not args.dry_run:
        save_nifti(exp_norm, source_image=exp_image, affine=exp_affine, output_path=paths["exp_ct"], dtype=np.float32)
        save_nifti(insp_norm, source_image=insp_image, affine=insp_affine, output_path=paths["insp_ct"], dtype=np.float32)

    if args.make_foreground_masks:
        mask = build_foreground_mask(exp_cropped_raw, insp_cropped_raw, args.foreground_threshold)
        mask = resize_array(mask, args.resize_ratio, is_label=True, device=device)
        if not args.dry_run:
            save_nifti(mask, source_image=insp_image, affine=insp_affine, output_path=paths["foreground_mask"], dtype=np.uint8)

    label_specs = [
        ("exp_lobe", case.exp_lobe, exp_image, exp_affine),
        ("insp_lobe", case.insp_lobe, insp_image, insp_affine),
        ("exp_fissure", case.exp_fissure, exp_image, exp_affine),
        ("insp_fissure", case.insp_fissure, insp_image, insp_affine),
    ]
    for key, label_path, source_image, affine in label_specs:
        if label_path is None or key not in paths:
            continue
        process_label(
            label_path,
            source_image=source_image,
            output_path=paths[key],
            trim=trim,
            resize_ratio=args.resize_ratio,
            device=device,
            affine=affine,
            dry_run=args.dry_run,
        )

    after_stats = {
        "exp": ct_stats(exp_norm),
        "insp": ct_stats(insp_norm),
    }

    processed_spacing = spacing(nib.Nifti1Image(np.zeros(processed_shape, dtype=np.float32), insp_affine))

    return ProcessedCase(
        split=case.split,
        case_id=case.case_id,
        grid=GridInfo(
            original_shape=tuple(int(v) for v in exp_raw.shape),
            processed_shape=processed_shape,
            original_spacing=spacing(insp_image),
            processed_spacing=processed_spacing,
            exp_orientation=orientation(exp_image),
            insp_orientation=orientation(insp_image),
            exp_insp_affine_equal=affines_equal,
        ),
        trim=trim,
        resize_ratio=float(args.resize_ratio),
        outputs={key: str(path) for key, path in paths.items()},
        labels_present={
            "exp_lobe": case.exp_lobe is not None,
            "insp_lobe": case.insp_lobe is not None,
            "exp_fissure": case.exp_fissure is not None,
            "insp_fissure": case.insp_fissure is not None,
        },
        ct_stats_before=before_stats,
        ct_stats_after=after_stats,
        warnings=warnings,
    )


def split_counts(cases: list[CaseFiles]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        item = counts.setdefault(
            case.split,
            {
                "cases": 0,
                "cases_with_lobes": 0,
                "cases_with_fissures": 0,
            },
        )
        item["cases"] += 1
        if case.exp_lobe and case.insp_lobe:
            item["cases_with_lobes"] += 1
        if case.exp_fissure and case.insp_fissure:
            item["cases_with_fissures"] += 1
    return counts


def write_manifest(root: Path, args: argparse.Namespace, cases: list[CaseFiles], processed: list[ProcessedCase]) -> None:
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "data_root": str(args.data_root),
        "output_root": str(root),
        "config": {
            "hu_min": args.hu_min,
            "hu_max": args.hu_max,
            "resize_ratio": args.resize_ratio,
            "enable_trim": args.enable_trim,
            "axis": args.axis,
            "edge_search": args.edge_search,
            "make_foreground_masks": args.make_foreground_masks,
            "foreground_threshold": args.foreground_threshold,
            "strict_affine": args.strict_affine,
            "dry_run": args.dry_run,
        },
        "discovered_case_counts": split_counts(cases),
        "processed_case_count": len(processed),
        "processed_split_counts": split_counts_from_processed(processed),
        "warnings": [warning for item in processed for warning in item.warnings],
        "cases": [asdict(item) for item in processed],
    }

    (root / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2))


def split_counts_from_processed(processed: list[ProcessedCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in processed:
        counts[item.split] = counts.get(item.split, 0) + 1
    return counts


def write_summary_csv(root: Path, processed: list[ProcessedCase]) -> None:
    rows: list[dict[str, Any]] = []
    for item in processed:
        rows.append(
            {
                "split": item.split,
                "case_id": item.case_id,
                "original_shape": "x".join(map(str, item.grid.original_shape)),
                "processed_shape": "x".join(map(str, item.grid.processed_shape)),
                "original_spacing": "x".join(f"{v:g}" for v in item.grid.original_spacing),
                "processed_spacing": "x".join(f"{v:g}" for v in item.grid.processed_spacing),
                "exp_orientation": "".join(item.grid.exp_orientation),
                "insp_orientation": "".join(item.grid.insp_orientation),
                "affine_equal": item.grid.exp_insp_affine_equal,
                "trim_enabled": item.trim.enabled,
                "trim_start": item.trim.trim_start,
                "trim_end": item.trim.trim_end,
                "resize_ratio": item.resize_ratio,
                "exp_lobe": item.labels_present["exp_lobe"],
                "insp_lobe": item.labels_present["insp_lobe"],
                "exp_fissure": item.labels_present["exp_fissure"],
                "insp_fissure": item.labels_present["insp_fissure"],
                "warnings": " | ".join(item.warnings),
            }
        )

    output_path = root / "preprocess_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["split", "case_id"])
        writer.writeheader()
        writer.writerows(rows)


def print_report(root: Path, processed: list[ProcessedCase]) -> None:
    by_split = split_counts_from_processed(processed)
    shapes = sorted({item.grid.processed_shape for item in processed})
    warnings = [warning for item in processed for warning in item.warnings]

    print("Preprocessing complete.")
    print(f"Output root: {root}")
    print(f"Processed cases: {len(processed)}")
    print(f"Split counts: {by_split}")
    print(f"Processed shapes: {shapes}")
    print(f"Warnings: {len(warnings)}")
    if warnings:
        print("First warnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")


def main() -> None:
    args = parse_args()
    root = output_root(args)
    device = resolve_device(args.device)

    cases = discover_cases(args.data_root)
    if not cases:
        raise RuntimeError(f"No complete paired EXP/INSP cases found under: {args.data_root}")

    processed: list[ProcessedCase] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:03d}/{len(cases):03d}] {case.split}/{case.case_id}")
        processed.append(process_case(case, args, root, device))

    if not args.dry_run:
        write_manifest(root, args, cases, processed)
        write_summary_csv(root, processed)

    print_report(root, processed)


if __name__ == "__main__":
    main()
