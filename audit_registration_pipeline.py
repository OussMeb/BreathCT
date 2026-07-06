#!/usr/bin/env python3
"""
Learn2Breath registration pipeline audit
========================================

Purpose
-------
Audit the current BreathCT / Learn2Breath pipeline before changing the
registration method. The script is designed around the uploaded project code
and checks only observable implementation behavior.

Main checks
-----------
1. Dataset pairing, shapes, spacing, affine/orientation consistency, labels.
2. Synthetic warp sanity checks for pull DVFs and component order.
3. Synthetic Jacobian checks (identity/translation/scaling/folding).
4. Baseline challenge-DVF grid/header/finite-value checks.
5. Project evaluator vs an independent SciPy pull warp.
6. Project Jacobian vs an independent NumPy finite-difference Jacobian.
7. Saved challenge DVF vs saved canonical-RAS DVF consistency.
8. Optional uniGradICON HDF5 transform -> DVF cross-check with SimpleITK.
9. Optional official warped-moving image vs saved-DVF warp comparison.
10. Refiner array-frame audit: raw CT orientation vs canonical-RAS DVFs.
11. Optional checkpoint diagnostic that reproduces the current refiner path
    and compares it with a frame-aligned diagnostic path.

Outputs
-------
<output-dir>/audit_report.json
<output-dir>/audit_checks.csv
<output-dir>/audit_summary.txt

Typical usage
-------------
Run from the project root:

    PYTHONPATH=. python audit_registration_pipeline.py \
      --project-root "/home/oussama/Desktop/MICCAI FRANCE" \
      --raw-data-root "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data" \
      --baseline-output-dir "/home/oussama/Desktop/MICCAI FRANCE/outputs/unigradicon_raw_validation" \
      --refiner-config "/home/oussama/Desktop/MICCAI FRANCE/configs/refiner_cached_lowspace.yaml" \
      --output-dir "/home/oussama/Desktop/MICCAI FRANCE/outputs/pipeline_audit"

With a trained refiner checkpoint:

    PYTHONPATH=. python audit_registration_pipeline.py \
      --project-root "/home/oussama/Desktop/MICCAI FRANCE" \
      --raw-data-root "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data" \
      --baseline-output-dir "/home/oussama/Desktop/MICCAI FRANCE/outputs/unigradicon_raw_validation" \
      --refiner-config "/home/oussama/Desktop/MICCAI FRANCE/configs/refiner_cached_lowspace.yaml" \
      --checkpoint "/home/oussama/Desktop/MICCAI FRANCE/outputs/refiner_cached_100/checkpoints/latest.pt" \
      --output-dir "/home/oussama/Desktop/MICCAI FRANCE/outputs/pipeline_audit_with_checkpoint"

Notes
-----
- The script does not modify any dataset, DVF, model, or checkpoint.
- SimpleITK checks are skipped when SimpleITK is unavailable.
- Independent label warping uses scipy.ndimage.map_coordinates.
- The checkpoint diagnostic can be expensive; it is only run when
  --checkpoint is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("nibabel is required. Install it with: pip install nibabel") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("PyTorch is required. Install it before running this audit.") from exc

try:
    from scipy.ndimage import map_coordinates
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("SciPy is required. Install it with: pip install scipy") from exc


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

VALID_STATUSES = {"PASS", "WARN", "FAIL", "INFO", "SKIP"}


@dataclass
class CheckRow:
    section: str
    check: str
    status: str
    case_id: str = ""
    message: str = ""
    values: dict[str, Any] | None = None


class AuditRecorder:
    def __init__(self) -> None:
        self.rows: list[CheckRow] = []

    def add(
        self,
        section: str,
        check: str,
        status: str,
        *,
        case_id: str = "",
        message: str = "",
        values: dict[str, Any] | None = None,
    ) -> None:
        status = status.upper()
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        self.rows.append(CheckRow(section, check, status, case_id, message, values or {}))

    def guarded(self, section: str, check: str, fn) -> Any:
        try:
            return fn()
        except Exception as exc:  # audit must continue after isolated failures
            self.add(
                section,
                check,
                "FAIL",
                message=f"Exception: {type(exc).__name__}: {exc}",
                values={"traceback": traceback.format_exc(limit=8)},
            )
            return None

    def counts(self) -> dict[str, int]:
        c = Counter(row.status for row in self.rows)
        return {status: int(c.get(status, 0)) for status in ("PASS", "WARN", "FAIL", "INFO", "SKIP")}

    def write(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": metadata,
            "summary": self.counts(),
            "checks": [asdict(row) for row in self.rows],
        }
        (output_dir / "audit_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with (output_dir / "audit_checks.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["section", "check", "status", "case_id", "message", "values_json"],
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "section": row.section,
                        "check": row.check,
                        "status": row.status,
                        "case_id": row.case_id,
                        "message": row.message,
                        "values_json": json.dumps(row.values or {}, sort_keys=True),
                    }
                )

        counts = self.counts()
        lines = [
            "Learn2Breath registration pipeline audit",
            "========================================",
            "",
            "Summary:",
            *(f"  {status}: {counts[status]}" for status in ("PASS", "WARN", "FAIL", "INFO", "SKIP")),
            "",
            "FAIL checks:",
        ]
        fails = [row for row in self.rows if row.status == "FAIL"]
        if fails:
            for row in fails:
                suffix = f" [{row.case_id}]" if row.case_id else ""
                lines.append(f"  - {row.section} / {row.check}{suffix}: {row.message}")
        else:
            lines.append("  none")

        lines += ["", "WARN checks:"]
        warns = [row for row in self.rows if row.status == "WARN"]
        if warns:
            for row in warns:
                suffix = f" [{row.case_id}]" if row.case_id else ""
                lines.append(f"  - {row.section} / {row.check}{suffix}: {row.message}")
        else:
            lines.append("  none")

        (output_dir / "audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Learn2Breath registration pipeline implementation.")
    p.add_argument("--project-root", default=".", help="BreathCT project root containing utils/, models/, etc.")
    p.add_argument("--raw-data-root", required=True, help="Learn2Breath_train_val_data root.")
    p.add_argument(
        "--baseline-output-dir",
        default="outputs/unigradicon_raw_validation",
        help="Raw uniGradICON validation output directory.",
    )
    p.add_argument(
        "--refiner-config",
        default="configs/refiner_cached_lowspace.yaml",
        help="Residual refiner YAML/JSON config.",
    )
    p.add_argument("--checkpoint", default=None, help="Optional residual-refiner checkpoint for inference diagnostics.")
    p.add_argument("--output-dir", default="outputs/pipeline_audit", help="Audit output directory.")
    p.add_argument("--device", default="cuda", help="cuda or cpu for project-warp/checkpoint checks.")
    p.add_argument("--max-validation-cases", type=int, default=0, help="0 means all validation cases.")
    p.add_argument("--max-training-cases", type=int, default=0, help="0 means all training cases for metadata checks.")
    p.add_argument("--chunk-x", type=int, default=16, help="Chunk size for independent SciPy warping.")
    p.add_argument("--expected-training-cases", type=int, default=200)
    p.add_argument("--expected-validation-cases", type=int, default=10)
    p.add_argument("--skip-official-warp-check", action="store_true")
    p.add_argument("--skip-transform-check", action="store_true")
    p.add_argument("--strict", action="store_true", help="Return exit code 2 if any FAIL check is recorded.")
    return p.parse_args()


def resolve_path(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (project_root / p)


def load_array(path: Path, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    return arr.astype(dtype, copy=False) if dtype is not None else np.asarray(arr)


def spacing_xyz(image: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    return tuple(float(v) for v in image.header.get_zooms()[:3])


def axcodes(image: nib.spatialimages.SpatialImage) -> tuple[str, str, str]:
    return tuple(str(v) for v in nib.aff2axcodes(image.affine))  # type: ignore[return-value]


def finite_stats(array: np.ndarray) -> dict[str, float]:
    a = np.asarray(array)
    finite = np.isfinite(a)
    return {
        "finite_fraction": float(finite.mean()),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    denom = int(aa.sum() + bb.sum())
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(aa, bb).sum() / denom)


def lobe_dice_dict(pred: np.ndarray, target: np.ndarray, labels: tuple[int, ...]) -> dict[int, float]:
    return {label: dice_binary(pred == label, target == label) for label in labels}


def mean_lobe_dice_independent(pred: np.ndarray, target: np.ndarray, labels: tuple[int, ...]) -> float:
    return float(np.mean(list(lobe_dice_dict(pred, target, labels).values())))


def global_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    aa = aa[valid]
    bb = bb[valid]
    if aa.size < 2:
        return float("nan")
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 0 else float("nan")


def independent_warp_pull_chunked(
    volume_xyz: np.ndarray,
    dvf_xyzc: np.ndarray,
    *,
    order: int,
    chunk_x: int = 16,
) -> np.ndarray:
    """Independent pull warp: output[x] = input[x + dvf[x]]."""
    volume = np.asarray(volume_xyz)
    dvf = np.asarray(dvf_xyzc, dtype=np.float64)
    if dvf.shape != (*volume.shape, 3):
        raise ValueError(f"Volume/DVF mismatch: volume={volume.shape}, dvf={dvf.shape}")

    out_dtype = volume.dtype if order == 0 else np.float32
    out = np.empty(volume.shape, dtype=out_dtype)
    nx, ny, nz = volume.shape
    yy, zz = np.meshgrid(np.arange(ny, dtype=np.float64), np.arange(nz, dtype=np.float64), indexing="ij")

    for x0 in range(0, nx, chunk_x):
        x1 = min(nx, x0 + chunk_x)
        xx = np.arange(x0, x1, dtype=np.float64)[:, None, None]
        xx = np.broadcast_to(xx, (x1 - x0, ny, nz))
        yb = np.broadcast_to(yy[None, ...], (x1 - x0, ny, nz))
        zb = np.broadcast_to(zz[None, ...], (x1 - x0, ny, nz))
        d = dvf[x0:x1]
        coords = [xx + d[..., 0], yb + d[..., 1], zb + d[..., 2]]
        warped = map_coordinates(volume, coords, order=order, mode="nearest", prefilter=(order > 1))
        out[x0:x1] = warped.astype(out_dtype, copy=False)
    return out


def independent_jacobian_det_inner(dvf_xyzc: np.ndarray) -> np.ndarray:
    """NumPy central-difference Jacobian of phi(x)=x+u(x), inner grid only."""
    u = np.asarray(dvf_xyzc, dtype=np.float64)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"Expected XYZC field, got {u.shape}")

    # np.gradient uses centered finite differences on interior voxels.
    grads = [[np.gradient(u[..., comp], axis=axis, edge_order=1) for axis in range(3)] for comp in range(3)]
    g = [[grads[c][a][1:-1, 1:-1, 1:-1] for a in range(3)] for c in range(3)]

    j11 = 1.0 + g[0][0]
    j12 = g[0][1]
    j13 = g[0][2]
    j21 = g[1][0]
    j22 = 1.0 + g[1][1]
    j23 = g[1][2]
    j31 = g[2][0]
    j32 = g[2][1]
    j33 = 1.0 + g[2][2]

    return (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )


def voxel_dvf_to_canonical_ras_independent(
    dvf_xyzc: np.ndarray,
    reference_image: nib.Nifti1Image,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """
    Independent orientation conversion.

    1. Convert voxel-vector components to physical/world vectors.
    2. Reorient each physical scalar component with nib.as_closest_canonical.
    3. Convert physical vectors to canonical voxel-vector components.
    """
    dvf = np.asarray(dvf_xyzc, dtype=np.float32)
    if dvf.ndim != 4 or dvf.shape[-1] != 3:
        raise ValueError(f"Expected XYZC DVF, got {dvf.shape}")

    a_orig = np.asarray(reference_image.affine[:3, :3], dtype=np.float64)
    d_phys = np.einsum("...j,ij->...i", dvf.astype(np.float64), a_orig)

    canonical_ref = nib.as_closest_canonical(reference_image)
    phys_components = []
    for c in range(3):
        scalar = nib.Nifti1Image(d_phys[..., c].astype(np.float32), reference_image.affine)
        scalar_can = nib.as_closest_canonical(scalar)
        phys_components.append(np.asanyarray(scalar_can.dataobj).astype(np.float64))
    d_phys_can = np.stack(phys_components, axis=-1)

    a_can = np.asarray(canonical_ref.affine[:3, :3], dtype=np.float64)
    inv_a_can = np.linalg.inv(a_can)
    d_can = np.einsum("...j,ij->...i", d_phys_can, inv_a_can)
    return d_can.astype(np.float32), canonical_ref


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config files") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def limited(items: list[Any], n: int) -> list[Any]:
    return items if n <= 0 else items[:n]


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------


def import_project_modules(project_root: Path) -> dict[str, Any]:
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from utils.metrics import LOBE_LABELS, jacobian_stats as project_jacobian_stats
    from utils.spatial import (
        compose_pull_dvfs,
        jacobian_determinant as project_jacobian_det,
        warp_pull,
    )

    modules: dict[str, Any] = {
        "LOBE_LABELS": tuple(int(v) for v in LOBE_LABELS),
        "project_jacobian_stats": project_jacobian_stats,
        "project_jacobian_det": project_jacobian_det,
        "warp_pull": warp_pull,
        "compose_pull_dvfs": compose_pull_dvfs,
    }

    try:
        from utils.dvf_io import original_dvf_xyzc_to_canonical_ras_cxyz

        modules["project_to_canonical_cxyz"] = original_dvf_xyzc_to_canonical_ras_cxyz
    except Exception:
        pass

    return modules


# -----------------------------------------------------------------------------
# Dataset audit
# -----------------------------------------------------------------------------


def discover_pairs(raw_root: Path, split: str) -> list[tuple[str, Path, Path]]:
    if split == "training":
        ct_root = raw_root / "training"
    elif split == "validation":
        ct_root = raw_root / "validation" / "ct_data"
    else:
        raise ValueError(split)

    pairs: list[tuple[str, Path, Path]] = []
    for fixed in sorted(ct_root.glob("NLST_*_INSP.nii.gz")):
        case_id = fixed.name.replace("_INSP.nii.gz", "")
        moving = ct_root / f"{case_id}_EXP.nii.gz"
        if moving.exists():
            pairs.append((case_id, moving, fixed))
    return pairs


def audit_dataset(
    rec: AuditRecorder,
    raw_root: Path,
    *,
    expected_train: int,
    expected_val: int,
    max_train: int,
    max_val: int,
) -> dict[str, Any]:
    section = "dataset"
    train_pairs = discover_pairs(raw_root, "training")
    val_pairs = discover_pairs(raw_root, "validation")

    rec.add(
        section,
        "training_pair_count",
        "PASS" if len(train_pairs) == expected_train else "WARN",
        message=f"Found {len(train_pairs)} complete training pairs; expected {expected_train}.",
        values={"found": len(train_pairs), "expected": expected_train},
    )
    rec.add(
        section,
        "validation_pair_count",
        "PASS" if len(val_pairs) == expected_val else "WARN",
        message=f"Found {len(val_pairs)} complete validation pairs; expected {expected_val}.",
        values={"found": len(val_pairs), "expected": expected_val},
    )

    orientation_counts: dict[str, Counter[str]] = {"training": Counter(), "validation": Counter()}
    shape_counts: Counter[str] = Counter()
    spacing_counts: Counter[str] = Counter()
    pair_affine_mismatch = 0

    for split, pairs, max_n in (
        ("training", train_pairs, max_train),
        ("validation", val_pairs, max_val),
    ):
        for case_id, moving, fixed in limited(pairs, max_n):
            m_img = nib.load(str(moving))
            f_img = nib.load(str(fixed))
            m_shape = tuple(int(v) for v in m_img.shape[:3])
            f_shape = tuple(int(v) for v in f_img.shape[:3])
            m_sp = spacing_xyz(m_img)
            f_sp = spacing_xyz(f_img)
            m_ax = axcodes(m_img)
            f_ax = axcodes(f_img)

            orientation_counts[split]["".join(f_ax)] += 1
            shape_counts[str(f_shape)] += 1
            spacing_counts[str(tuple(round(v, 6) for v in f_sp))] += 1

            if m_shape != f_shape:
                rec.add(section, "pair_shape_match", "FAIL", case_id=case_id, message=f"EXP {m_shape} vs INSP {f_shape}")
            if not np.allclose(m_sp, f_sp, atol=1e-6):
                rec.add(section, "pair_spacing_match", "FAIL", case_id=case_id, message=f"EXP {m_sp} vs INSP {f_sp}")
            if m_ax != f_ax:
                rec.add(section, "pair_orientation_match", "FAIL", case_id=case_id, message=f"EXP {m_ax} vs INSP {f_ax}")
            if not np.allclose(m_img.affine, f_img.affine, atol=1e-5, rtol=1e-6):
                pair_affine_mismatch += 1
                rec.add(
                    section,
                    "pair_affine_match",
                    "WARN",
                    case_id=case_id,
                    message="EXP and INSP affines differ; array-only warping requires explicit physical-grid handling.",
                    values={"max_abs_affine_diff": float(np.max(np.abs(m_img.affine - f_img.affine)))},
                )

    rec.add(
        section,
        "orientation_summary",
        "INFO",
        message="Observed INSP orientations by split.",
        values={split: dict(counter) for split, counter in orientation_counts.items()},
    )
    rec.add(section, "shape_summary", "INFO", values=dict(shape_counts), message="Observed INSP shape counts.")
    rec.add(section, "spacing_summary", "INFO", values=dict(spacing_counts), message="Observed INSP spacing counts.")
    rec.add(
        section,
        "pair_affine_mismatch_summary",
        "PASS" if pair_affine_mismatch == 0 else "WARN",
        message=f"Pairs with EXP/INSP affine mismatch: {pair_affine_mismatch}.",
        values={"mismatch_count": pair_affine_mismatch},
    )

    # Validation label inventory.
    seg_root = raw_root / "validation" / "seg_net"
    expected_lobes = {0, 8, 16, 32, 64, 128}
    for case_id, _, _ in limited(val_pairs, max_val):
        for phase in ("EXP", "INSP"):
            path = seg_root / f"{case_id}_{phase}_lobe.nii.gz"
            if not path.exists():
                rec.add(section, "validation_lobe_exists", "FAIL", case_id=case_id, message=f"Missing {path.name}")
                continue
            labels = set(int(v) for v in np.unique(load_array(path)))
            status = "PASS" if labels.issubset(expected_lobes) and expected_lobes - {0} <= labels else "WARN"
            rec.add(
                section,
                "validation_lobe_labels",
                status,
                case_id=case_id,
                message=f"{phase} labels: {sorted(labels)}",
                values={"phase": phase, "labels": sorted(labels)},
            )

    return {"training_pairs": train_pairs, "validation_pairs": val_pairs, "orientation_counts": orientation_counts}


# -----------------------------------------------------------------------------
# Synthetic audits
# -----------------------------------------------------------------------------


def audit_synthetic_ops(rec: AuditRecorder, modules: dict[str, Any], device: torch.device) -> None:
    section = "synthetic_ops"
    warp_pull = modules["warp_pull"]
    project_jac = modules["project_jacobian_det"]
    compose = modules["compose_pull_dvfs"]

    # Pull translation: +1 X samples input at x+1.
    shape = (17, 19, 21)
    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    volume_np = (10000 * x + 100 * y + z).astype(np.float32)
    volume_t = torch.from_numpy(volume_np).view(1, 1, *shape).to(device)
    dvf_t = torch.zeros((1, 3, *shape), dtype=torch.float32, device=device)
    dvf_t[:, 0] = 1.0
    warped = warp_pull(volume_t, dvf_t, mode="nearest", padding_mode="border")[0, 0].detach().cpu().numpy()
    expected = np.concatenate([volume_np[1:], volume_np[-1:]], axis=0)
    err = float(np.max(np.abs(warped - expected)))
    rec.add(
        section,
        "pull_translation_component_order",
        "PASS" if err == 0.0 else "FAIL",
        message=f"Synthetic +1 X pull translation max error={err:.6g}.",
        values={"max_abs_error": err},
    )

    # Identity and constant translation Jacobian det = 1.
    for name, field in (
        ("identity", torch.zeros((1, 3, *shape), device=device)),
        ("constant_translation", torch.ones((1, 3, *shape), device=device) * 2.5),
    ):
        det = project_jac(field).detach().cpu().numpy()
        max_err = float(np.max(np.abs(det - 1.0)))
        rec.add(
            section,
            f"jacobian_{name}",
            "PASS" if max_err < 1e-6 else "FAIL",
            message=f"Expected determinant 1; max error={max_err:.6g}.",
            values={"max_abs_error": max_err},
        )

    # Diagonal scaling phi=(1.1x, 0.95y, 1.2z), determinant known exactly.
    scale = (1.1, 0.95, 1.2)
    field = torch.zeros((1, 3, *shape), dtype=torch.float32, device=device)
    field[:, 0] = (scale[0] - 1.0) * torch.arange(shape[0], device=device).view(1, -1, 1, 1)
    field[:, 1] = (scale[1] - 1.0) * torch.arange(shape[1], device=device).view(1, 1, -1, 1)
    field[:, 2] = (scale[2] - 1.0) * torch.arange(shape[2], device=device).view(1, 1, 1, -1)
    det = project_jac(field).detach().cpu().numpy()
    expected_det = float(np.prod(scale))
    max_err = float(np.max(np.abs(det - expected_det)))
    rec.add(
        section,
        "jacobian_known_scaling",
        "PASS" if max_err < 2e-5 else "FAIL",
        message=f"Expected determinant {expected_det:.6f}; max error={max_err:.6g}.",
        values={"expected": expected_det, "max_abs_error": max_err},
    )

    # Reflection phi_x=-x gives determinant -1.
    fold = torch.zeros((1, 3, *shape), dtype=torch.float32, device=device)
    fold[:, 0] = -2.0 * torch.arange(shape[0], device=device).view(1, -1, 1, 1)
    det_fold = project_jac(fold).detach().cpu().numpy()
    neg_fraction = float(np.mean(det_fold <= 0.0))
    median = float(np.median(det_fold))
    rec.add(
        section,
        "jacobian_known_fold",
        "PASS" if neg_fraction > 0.999 and abs(median + 1.0) < 2e-5 else "FAIL",
        message=f"Reflection expected det=-1; median={median:.6f}, non-positive fraction={neg_fraction:.6f}.",
        values={"median_det": median, "nonpositive_fraction": neg_fraction},
    )

    # Pull composition with integer translations: sequential warp == composed warp.
    first = torch.zeros((1, 3, *shape), dtype=torch.float32, device=device)
    second = torch.zeros_like(first)
    first[:, 0] = 1.0
    second[:, 1] = 1.0
    composed = compose(first, second)
    sequential = warp_pull(warp_pull(volume_t, first, mode="nearest"), second, mode="nearest")
    one_shot = warp_pull(volume_t, composed, mode="nearest")
    comp_err = float(torch.max(torch.abs(sequential - one_shot)).detach().cpu())
    rec.add(
        section,
        "pull_composition",
        "PASS" if comp_err == 0.0 else "FAIL",
        message=f"Sequential-vs-composed integer pull warp max error={comp_err:.6g}.",
        values={"max_abs_error": comp_err},
    )

    # Refiner spatial implementation cross-check when available.
    try:
        from utils.refine_spatial import (
            downsample_dvf_5d,
            jacobian_determinant as refine_jac,
            upsample_dvf_5d,
            warp_volume,
        )

        ref_warp = warp_volume(volume_t, dvf_t, mode="nearest")[0, 0].detach().cpu().numpy()
        ref_err = float(np.max(np.abs(ref_warp - expected)))
        rec.add(
            section,
            "refiner_pull_translation_component_order",
            "PASS" if ref_err == 0.0 else "FAIL",
            message=f"Refiner synthetic +1 X pull translation max error={ref_err:.6g}.",
            values={"max_abs_error": ref_err},
        )

        ref_det = refine_jac(field).detach().cpu().numpy()
        # The refiner Jacobian uses forward differences with padded boundary; compare interior.
        interior = ref_det[:, 1:-1, 1:-1, 1:-1]
        ref_j_err = float(np.max(np.abs(interior - expected_det)))
        rec.add(
            section,
            "refiner_jacobian_known_scaling",
            "PASS" if ref_j_err < 2e-5 else "FAIL",
            message=f"Expected determinant {expected_det:.6f}; interior max error={ref_j_err:.6g}.",
            values={"expected": expected_det, "max_abs_error": ref_j_err},
        )

        # Measure low-res round-trip behavior; record, do not assume it is a bug.
        linear = field.clone()
        low = downsample_dvf_5d(linear, 2)
        up = upsample_dvf_5d(low, shape)
        rt_mae = float(torch.mean(torch.abs(up - linear)).detach().cpu())
        rt_max = float(torch.max(torch.abs(up - linear)).detach().cpu())
        rec.add(
            section,
            "refiner_dvf_down_up_roundtrip",
            "INFO" if rt_max < 0.1 else "WARN",
            message="Measured downsample->upsample error on a linear synthetic DVF.",
            values={"mae_vox": rt_mae, "max_abs_vox": rt_max},
        )
    except Exception as exc:
        rec.add(section, "refiner_synthetic_ops", "SKIP", message=f"Could not import/run refiner spatial ops: {exc}")


# -----------------------------------------------------------------------------
# Baseline DVF audit
# -----------------------------------------------------------------------------


def project_warp_label(
    label_xyz: np.ndarray,
    dvf_xyzc: np.ndarray,
    warp_pull,
    device: torch.device,
) -> np.ndarray:
    vol_t = torch.as_tensor(label_xyz.astype(np.float32), device=device).view(1, 1, *label_xyz.shape)
    dvf_t = torch.as_tensor(np.moveaxis(dvf_xyzc.astype(np.float32), -1, 0), device=device).unsqueeze(0)
    with torch.no_grad():
        out = warp_pull(vol_t, dvf_t, mode="nearest", padding_mode="border")
    return out[0, 0].detach().cpu().numpy().astype(np.int16)


def project_jacobian_numpy(dvf_xyzc: np.ndarray, project_jac, device: torch.device) -> np.ndarray:
    t = torch.as_tensor(np.moveaxis(dvf_xyzc.astype(np.float32), -1, 0), device=device).unsqueeze(0)
    with torch.no_grad():
        det = project_jac(t)
    return det[0].detach().cpu().numpy().astype(np.float64)


def independent_transform_to_voxel_dvf_xyzc(transform_path: Path, fixed_path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK unavailable") from exc

    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    transform = sitk.ReadTransform(str(transform_path))
    field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection(),
    )
    physical_zyxc = sitk.GetArrayFromImage(field).astype(np.float64)
    physical_xyzc = np.transpose(physical_zyxc, (2, 1, 0, 3))
    direction = np.asarray(fixed.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(fixed.GetSpacing(), dtype=np.float64)
    physical_to_index = np.linalg.inv(direction @ np.diag(spacing))
    voxel = np.einsum("...j,ij->...i", physical_xyzc, physical_to_index)
    return voxel.astype(np.float32)


def audit_baseline(
    rec: AuditRecorder,
    *,
    raw_root: Path,
    baseline_dir: Path,
    val_pairs: list[tuple[str, Path, Path]],
    modules: dict[str, Any],
    device: torch.device,
    max_val: int,
    chunk_x: int,
    skip_official_warp: bool,
    skip_transform: bool,
) -> dict[str, Any]:
    section = "baseline"
    dvf_dir = baseline_dir / "dvfs"
    canonical_dir = baseline_dir / "dvfs_canonical_ras"
    transform_dir = baseline_dir / "transforms"
    warped_dir = baseline_dir / "warped"
    seg_root = raw_root / "validation" / "seg_net"
    labels = modules["LOBE_LABELS"]
    warp_pull = modules["warp_pull"]
    project_jac = modules["project_jacobian_det"]

    if not dvf_dir.exists():
        rec.add(section, "dvf_directory", "FAIL", message=f"Missing baseline DVF directory: {dvf_dir}")
        return {}
    rec.add(section, "dvf_directory", "PASS", message=str(dvf_dir))

    reported = (
        load_json_if_exists(baseline_dir / "eval" / "validation_metrics.json")
        or load_json_if_exists(baseline_dir / "validation_metrics.json")
    )
    reported_by_case: dict[str, dict[str, Any]] = {}
    if reported:
        for row in reported.get("case_metrics", []):
            if isinstance(row, dict) and "case_id" in row:
                reported_by_case[str(row["case_id"])] = row
        rec.add(section, "reported_metrics_json", "PASS", message="Found validation_metrics.json for reproducibility checks.")
    else:
        rec.add(section, "reported_metrics_json", "WARN", message="No validation_metrics.json found; metric reproduction comparison skipped.")

    case_rows: list[dict[str, Any]] = []
    for case_id, moving_ct_path, fixed_ct_path in limited(val_pairs, max_val):
        dvf_path = dvf_dir / f"{case_id}_DVF.nii.gz"
        if not dvf_path.exists():
            rec.add(section, "dvf_exists", "FAIL", case_id=case_id, message=f"Missing {dvf_path}")
            continue

        fixed_img = nib.load(str(fixed_ct_path))
        moving_img = nib.load(str(moving_ct_path))
        dvf_img = nib.load(str(dvf_path))
        dvf = np.asanyarray(dvf_img.dataobj).astype(np.float32)

        # Grid/header.
        shape_ok = dvf.shape == (*fixed_img.shape[:3], 3)
        affine_diff = float(np.max(np.abs(dvf_img.affine - fixed_img.affine)))
        affine_ok = np.allclose(dvf_img.affine, fixed_img.affine, atol=1e-5, rtol=1e-6)
        rec.add(
            section,
            "dvf_fixed_grid",
            "PASS" if shape_ok and affine_ok else "FAIL",
            case_id=case_id,
            message=f"shape_ok={shape_ok}, max_abs_affine_diff={affine_diff:.6g}",
            values={
                "dvf_shape": list(dvf.shape),
                "fixed_shape": list(fixed_img.shape[:3]),
                "max_abs_affine_diff": affine_diff,
                "dvf_axcodes": list(axcodes(dvf_img)),
                "fixed_axcodes": list(axcodes(fixed_img)),
            },
        )
        fstats = finite_stats(dvf)
        rec.add(
            section,
            "dvf_finite",
            "PASS" if fstats["finite_fraction"] == 1.0 else "FAIL",
            case_id=case_id,
            message=f"finite_fraction={fstats['finite_fraction']:.9f}",
            values=fstats,
        )

        qcode = int(dvf_img.header["qform_code"])
        scode = int(dvf_img.header["sform_code"])
        rec.add(
            section,
            "dvf_qform_sform",
            "PASS" if qcode > 0 and scode > 0 else "WARN",
            case_id=case_id,
            message=f"qform_code={qcode}, sform_code={scode}",
            values={"qform_code": qcode, "sform_code": scode},
        )

        moving_lobe_path = seg_root / f"{case_id}_EXP_lobe.nii.gz"
        fixed_lobe_path = seg_root / f"{case_id}_INSP_lobe.nii.gz"
        if not moving_lobe_path.exists() or not fixed_lobe_path.exists():
            rec.add(section, "lobe_paths", "FAIL", case_id=case_id, message="Missing validation lobe segmentation.")
            continue

        moving_lobe = load_array(moving_lobe_path, np.int16)
        fixed_lobe = load_array(fixed_lobe_path, np.int16)

        # Project warp vs independent SciPy warp.
        pred_project = project_warp_label(moving_lobe, dvf, warp_pull, device)
        pred_ind = independent_warp_pull_chunked(moving_lobe, dvf, order=0, chunk_x=chunk_x).astype(np.int16)
        disagree = float(np.mean(pred_project != pred_ind))
        dice_project = mean_lobe_dice_independent(pred_project, fixed_lobe, labels)
        dice_ind = mean_lobe_dice_independent(pred_ind, fixed_lobe, labels)
        rec.add(
            section,
            "project_vs_independent_label_warp",
            "PASS" if disagree < 1e-6 else ("WARN" if disagree < 1e-4 else "FAIL"),
            case_id=case_id,
            message=f"voxel disagreement={disagree:.9g}; Dice(project)={dice_project:.6f}; Dice(independent)={dice_ind:.6f}",
            values={
                "voxel_disagreement_fraction": disagree,
                "dice_project": dice_project,
                "dice_independent": dice_ind,
                "abs_dice_diff": abs(dice_project - dice_ind),
            },
        )

        # Project Jacobian vs independent NumPy Jacobian.
        det_project = project_jacobian_numpy(dvf, project_jac, device)
        det_ind = independent_jacobian_det_inner(dvf)
        jac_max = float(np.max(np.abs(det_project - det_ind)))
        jac_mae = float(np.mean(np.abs(det_project - det_ind)))
        fold_project = float(np.mean(det_project <= 0.0) * 100.0)
        fold_ind = float(np.mean(det_ind <= 0.0) * 100.0)
        rec.add(
            section,
            "project_vs_independent_jacobian",
            "PASS" if jac_max < 5e-5 else ("WARN" if jac_max < 1e-3 else "FAIL"),
            case_id=case_id,
            message=f"max|det diff|={jac_max:.6g}; fold project={fold_project:.6f}%; independent={fold_ind:.6f}%",
            values={
                "max_abs_det_diff": jac_max,
                "mae_det_diff": jac_mae,
                "fold_project_pct": fold_project,
                "fold_independent_pct": fold_ind,
                "abs_fold_diff_pct": abs(fold_project - fold_ind),
            },
        )

        # Reproduce saved metrics.
        if case_id in reported_by_case:
            rr = reported_by_case[case_id]
            reported_dice = float(rr.get("mean_lobe_dice", float("nan")))
            reported_fold = float(rr.get("folding_percentage", float("nan")))
            dd = abs(reported_dice - dice_project)
            df = abs(reported_fold - fold_project)
            rec.add(
                section,
                "reported_metric_reproduction",
                "PASS" if dd < 1e-7 and df < 1e-6 else "FAIL",
                case_id=case_id,
                message=f"|Dice diff|={dd:.3g}; |fold diff|={df:.3g}%",
                values={
                    "reported_dice": reported_dice,
                    "recomputed_dice": dice_project,
                    "reported_fold_pct": reported_fold,
                    "recomputed_fold_pct": fold_project,
                },
            )

        # Challenge DVF -> canonical RAS cross-check.
        canonical_path = canonical_dir / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
        if canonical_path.exists():
            saved_can_img = nib.load(str(canonical_path))
            saved_can = np.asanyarray(saved_can_img.dataobj).astype(np.float32)
            independent_can, expected_can_ref = voxel_dvf_to_canonical_ras_independent(dvf, fixed_img)
            can_shape_ok = saved_can.shape == independent_can.shape
            if can_shape_ok:
                can_diff = np.abs(saved_can - independent_can)
                can_max = float(can_diff.max())
                can_mae = float(can_diff.mean())
            else:
                can_max = float("inf")
                can_mae = float("inf")
            can_affine_diff = float(np.max(np.abs(saved_can_img.affine - expected_can_ref.affine)))
            status = "PASS" if can_shape_ok and can_max < 2e-5 and can_affine_diff < 1e-5 else "FAIL"
            rec.add(
                section,
                "canonical_ras_dvf_consistency",
                status,
                case_id=case_id,
                message=f"shape_ok={can_shape_ok}; max vector diff={can_max:.6g}; affine diff={can_affine_diff:.6g}",
                values={
                    "max_abs_vector_diff_vox": can_max,
                    "mae_vector_diff_vox": can_mae,
                    "max_abs_affine_diff": can_affine_diff,
                    "saved_axcodes": list(axcodes(saved_can_img)),
                    "expected_axcodes": list(axcodes(expected_can_ref)),
                },
            )
        else:
            rec.add(section, "canonical_ras_dvf_consistency", "SKIP", case_id=case_id, message=f"Missing {canonical_path.name}")

        # Optional HDF5 transform conversion check.
        transform_path = transform_dir / f"{case_id}_unigradicon.hdf5"
        if skip_transform:
            pass
        elif transform_path.exists():
            try:
                from_transform = independent_transform_to_voxel_dvf_xyzc(transform_path, fixed_ct_path)
                diff = np.abs(from_transform - dvf)
                max_diff = float(diff.max())
                mae_diff = float(diff.mean())
                rec.add(
                    section,
                    "hdf5_transform_to_saved_dvf",
                    "PASS" if max_diff < 2e-5 else "FAIL",
                    case_id=case_id,
                    message=f"max vector diff={max_diff:.6g} vox; MAE={mae_diff:.6g} vox",
                    values={"max_abs_vox": max_diff, "mae_vox": mae_diff},
                )
            except RuntimeError as exc:
                rec.add(section, "hdf5_transform_to_saved_dvf", "SKIP", case_id=case_id, message=str(exc))
            except Exception as exc:
                rec.add(section, "hdf5_transform_to_saved_dvf", "FAIL", case_id=case_id, message=f"{type(exc).__name__}: {exc}")
        else:
            rec.add(section, "hdf5_transform_to_saved_dvf", "SKIP", case_id=case_id, message="Transform file not found.")

        # Optional official warped-moving check.
        if not skip_official_warp:
            official_path = warped_dir / f"{case_id}_warped_EXP.nii.gz"
            pair_affine_equal = np.allclose(moving_img.affine, fixed_img.affine, atol=1e-5, rtol=1e-6)
            if official_path.exists() and pair_affine_equal:
                moving_ct = np.asanyarray(moving_img.dataobj).astype(np.float32)
                official = load_array(official_path, np.float32)
                independent_ct = independent_warp_pull_chunked(moving_ct, dvf, order=1, chunk_x=chunk_x)
                if official.shape == independent_ct.shape:
                    # Ignore a small outer rim where padding conventions may differ.
                    rim = 2
                    sl = tuple(slice(rim, -rim) for _ in range(3))
                    a = official[sl]
                    b = independent_ct[sl]
                    mae = float(np.mean(np.abs(a - b)))
                    p1, p99 = np.percentile(official[np.isfinite(official)], [1, 99])
                    dynamic = max(1e-6, float(p99 - p1))
                    norm_mae = mae / dynamic
                    ncc = global_ncc(a, b)
                    status = "PASS" if ncc > 0.999 and norm_mae < 0.01 else ("WARN" if ncc > 0.99 else "FAIL")
                    rec.add(
                        section,
                        "official_warp_vs_saved_dvf_warp",
                        status,
                        case_id=case_id,
                        message=f"NCC={ncc:.6f}; normalized MAE={norm_mae:.6g}",
                        values={"ncc": ncc, "mae_hu": mae, "normalized_mae": norm_mae},
                    )
                else:
                    rec.add(section, "official_warp_vs_saved_dvf_warp", "FAIL", case_id=case_id, message=f"Shape mismatch official={official.shape}, independent={independent_ct.shape}")
            elif official_path.exists() and not pair_affine_equal:
                rec.add(section, "official_warp_vs_saved_dvf_warp", "SKIP", case_id=case_id, message="EXP/INSP affines differ; voxel-only independent comparison is not valid.")
            else:
                rec.add(section, "official_warp_vs_saved_dvf_warp", "SKIP", case_id=case_id, message="Official warped-moving output not found.")

        case_rows.append(
            {
                "case_id": case_id,
                "dice_project": dice_project,
                "dice_independent": dice_ind,
                "fold_project_pct": fold_project,
                "fold_independent_pct": fold_ind,
            }
        )

    if case_rows:
        mean_dice = float(np.mean([r["dice_project"] for r in case_rows]))
        mean_fold = float(np.mean([r["fold_project_pct"] for r in case_rows]))
        worst_dice = min(case_rows, key=lambda r: r["dice_project"])
        worst_fold = max(case_rows, key=lambda r: r["fold_project_pct"])
        rec.add(
            section,
            "recomputed_summary",
            "INFO",
            message=f"Mean Dice={mean_dice:.6f}; mean folding={mean_fold:.6f}%.",
            values={
                "mean_dice": mean_dice,
                "mean_folding_pct": mean_fold,
                "lowest_dice_case": worst_dice,
                "highest_folding_case": worst_fold,
            },
        )

    return {"case_rows": case_rows}


# -----------------------------------------------------------------------------
# Refiner frame audit and optional checkpoint diagnostic
# -----------------------------------------------------------------------------


def audit_refiner_frame_consistency(
    rec: AuditRecorder,
    *,
    project_root: Path,
    raw_root: Path,
    refiner_config_path: Path,
    max_val: int,
    max_train: int,
) -> dict[str, Any]:
    section = "refiner_frame"
    if not refiner_config_path.exists():
        rec.add(section, "config_exists", "SKIP", message=f"Missing refiner config: {refiner_config_path}")
        return {}

    config = load_yaml_or_json(refiner_config_path)
    rec.add(section, "config_exists", "PASS", message=str(refiner_config_path))

    infer_init = Path(config.get("infer", {}).get("init_dvf_dir", ""))
    train_init = Path(config.get("data", {}).get("init_dvf_dir", ""))
    if not infer_init.is_absolute():
        infer_init = project_root / infer_init
    if not train_init.is_absolute():
        train_init = project_root / train_init

    source_path = project_root / "infer_refiner_cached.py"
    source = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    source_loads_raw_arrays = (
        "load_nifti_array(case.moving_path)" in source and "load_nifti_array(case.fixed_path)" in source
    )
    source_has_explicit_canonicalization = any(
        token in source
        for token in (
            "as_closest_canonical",
            "canonicalize_image_to_ras",
            "canonicalize",
            "reorient",
        )
    )
    rec.add(
        section,
        "inference_source_orientation_handling",
        "INFO",
        message="Static source inspection of infer_refiner_cached.py.",
        values={
            "loads_raw_arrays": source_loads_raw_arrays,
            "explicit_canonicalization_token_found": source_has_explicit_canonicalization,
            "source_path": str(source_path),
        },
    )

    val_pairs = limited(discover_pairs(raw_root, "validation"), max_val)
    mismatch_cases = []
    for case_id, _, fixed_path in val_pairs:
        init_path = infer_init / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
        if not init_path.exists():
            rec.add(section, "validation_init_exists", "WARN", case_id=case_id, message=f"Missing {init_path}")
            continue
        fixed_img = nib.load(str(fixed_path))
        init_img = nib.load(str(init_path))
        fixed_ax = axcodes(fixed_img)
        init_ax = axcodes(init_img)
        same_shape = tuple(fixed_img.shape[:3]) == tuple(init_img.shape[:3])
        same_ax = fixed_ax == init_ax
        if not same_ax:
            mismatch_cases.append(case_id)
        status = "PASS" if same_ax and same_shape else "WARN"
        message = f"raw fixed axcodes={fixed_ax}, cached init axcodes={init_ax}, shape_match={same_shape}"
        if (
            not same_ax
            and source_loads_raw_arrays
            and not source_has_explicit_canonicalization
        ):
            status = "FAIL"
            message += "; inference source loads raw CT arrays and no explicit canonicalization was detected before combining with the cached DVF"
        rec.add(
            section,
            "validation_ct_vs_cached_dvf_array_frame",
            status,
            case_id=case_id,
            message=message,
            values={
                "fixed_axcodes": list(fixed_ax),
                "init_dvf_axcodes": list(init_ax),
                "shape_match": same_shape,
            },
        )

    train_pairs = limited(discover_pairs(raw_root, "training"), max_train)
    train_mismatch = 0
    for case_id, _, fixed_path in train_pairs:
        init_path = train_init / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
        if not init_path.exists():
            continue
        fixed_ax = axcodes(nib.load(str(fixed_path)))
        init_ax = axcodes(nib.load(str(init_path)))
        if fixed_ax != init_ax:
            train_mismatch += 1
    rec.add(
        section,
        "training_ct_vs_cached_dvf_frame_summary",
        "PASS" if train_mismatch == 0 else "WARN",
        message=f"Training cases with CT/DVF axcode mismatch among inspected cases: {train_mismatch}.",
        values={"mismatch_count": train_mismatch, "inspected": len(train_pairs)},
    )

    rec.add(
        section,
        "validation_frame_mismatch_summary",
        "PASS" if not mismatch_cases else "FAIL",
        message=f"Validation CT/canonical-DVF orientation mismatch cases: {len(mismatch_cases)}.",
        values={"case_ids": mismatch_cases},
    )

    return {"config": config, "infer_init_dir": infer_init, "train_init_dir": train_init, "mismatch_cases": mismatch_cases}


def canonicalize_ct_array(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    can = nib.as_closest_canonical(image)
    return np.asanyarray(can.dataobj).astype(np.float32), can


def checkpoint_diagnostic(
    rec: AuditRecorder,
    *,
    project_root: Path,
    raw_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    baseline_dir: Path,
    device: torch.device,
    max_val: int,
    chunk_x: int,
    modules: dict[str, Any],
) -> dict[str, Any]:
    section = "checkpoint_diagnostic"
    if not checkpoint_path.exists():
        rec.add(section, "checkpoint_exists", "SKIP", message=f"Missing checkpoint: {checkpoint_path}")
        return {}

    try:
        from models.residual_refiner import ResidualRefiner3d
        from utils.refine_io import normalize_ct_hu
        from utils.refine_spatial import (
            compose_additive,
            integrate_svf,
            ras_canonical_to_original_grid_xyzc,
            resize_volume_5d,
            tensor_from_volume,
            tensor_from_xyzc,
            upsample_dvf_5d,
            downsample_dvf_5d,
            warp_volume,
            xyzc_from_dvf_tensor,
        )
        from utils.spatial import compose_pull_dvfs
    except Exception as exc:
        rec.add(section, "project_imports", "FAIL", message=f"Could not import refiner modules: {exc}")
        return {}

    config = load_yaml_or_json(config_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    train_config = checkpoint.get("config", config)
    model_cfg = train_config["model"]
    model = ResidualRefiner3d(
        in_channels=int(model_cfg["input_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        max_channels=int(model_cfg["max_channels"]),
        max_residual_voxels=float(model_cfg["max_residual_voxels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rec.add(
        section,
        "checkpoint_exists",
        "PASS",
        message=str(checkpoint_path),
        values={"epoch": checkpoint.get("epoch")},
    )

    # Zero-head initialization is not expected after training; just record checkpoint metadata.
    hu_min = float(config["data"]["hu_min"])
    hu_max = float(config["data"]["hu_max"])
    threshold_hu = float(config["data"]["foreground_threshold_hu"])
    threshold_norm = (np.clip(threshold_hu, hu_min, hu_max) - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    reduce_factor = int(train_config["train"]["reduce_factor"])
    predict = str(train_config["model"]["predict"])
    integration_steps = int(train_config["model"]["integration_steps"])

    infer_init = Path(config["infer"]["init_dvf_dir"])
    if not infer_init.is_absolute():
        infer_init = project_root / infer_init

    val_pairs = limited(discover_pairs(raw_root, "validation"), max_val)
    seg_root = raw_root / "validation" / "seg_net"
    labels = modules["LOBE_LABELS"]
    baseline_dvf_dir = baseline_dir / "dvfs"

    rows: list[dict[str, Any]] = []

    def run_model(moving_np: np.ndarray, fixed_np: np.ndarray, init_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        moving = tensor_from_volume(normalize_ct_hu(moving_np, hu_min, hu_max), device)
        fixed = tensor_from_volume(normalize_ct_hu(fixed_np, hu_min, hu_max), device)
        init = tensor_from_xyzc(init_np, device)
        mask = ((moving > threshold_norm) | (fixed > threshold_norm)).float()
        moving_low = resize_volume_5d(moving, reduce_factor)
        fixed_low = resize_volume_5d(fixed, reduce_factor)
        mask_low = resize_volume_5d(mask, reduce_factor, mode="nearest")
        init_low = downsample_dvf_5d(init, reduce_factor)
        warped0 = warp_volume(moving_low, init_low)
        abs_diff = (fixed_low - warped0).abs()
        inputs = torch.cat([moving_low, fixed_low, warped0, abs_diff, mask_low], dim=1)
        residual_low_raw = model(inputs)
        residual_low = integrate_svf(residual_low_raw, integration_steps) if predict == "svf" else residual_low_raw
        residual_full = upsample_dvf_5d(residual_low, tuple(int(v) for v in init.shape[-3:]))
        return init, residual_full

    with torch.no_grad():
        for case_id, moving_path, fixed_path in val_pairs:
            init_path = infer_init / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
            if not init_path.exists():
                rec.add(section, "init_exists", "WARN", case_id=case_id, message=f"Missing {init_path}")
                continue

            moving_raw = load_array(moving_path, np.float32)
            fixed_raw = load_array(fixed_path, np.float32)
            fixed_img = nib.load(str(fixed_path))
            init_can = load_array(init_path, np.float32)
            moving_can, moving_can_img = canonicalize_ct_array(moving_path)
            fixed_can, fixed_can_img = canonicalize_ct_array(fixed_path)

            if init_can.shape[:3] != fixed_can.shape[:3]:
                rec.add(
                    section,
                    "canonical_shape_match",
                    "FAIL",
                    case_id=case_id,
                    message=f"init={init_can.shape[:3]}, fixed canonical={fixed_can.shape[:3]}",
                )
                continue

            # A. Reproduce current inference array path: raw CT arrays + canonical DVF array.
            init_t_raw, residual_raw = run_model(moving_raw, fixed_raw, init_can)
            final_current = compose_additive(init_t_raw, residual_raw)

            # B. Frame-aligned diagnostic: canonical CT arrays + canonical DVF array.
            init_t_can, residual_can = run_model(moving_can, fixed_can, init_can)
            final_aligned = compose_additive(init_t_can, residual_can)

            # C/D. Composition diagnostics using the same frame-aligned residual.
            composed_after = compose_pull_dvfs(init_t_can, residual_can)
            composed_before = compose_pull_dvfs(residual_can, init_t_can)

            def to_original(field_t: torch.Tensor) -> np.ndarray:
                can_xyzc = xyzc_from_dvf_tensor(field_t)
                return ras_canonical_to_original_grid_xyzc(can_xyzc, fixed_img)

            fields_original = {
                "current_additive": to_original(final_current),
                "aligned_additive": to_original(final_aligned),
                "aligned_compose_after": to_original(composed_after),
                "aligned_compose_before": to_original(composed_before),
            }

            moving_lobe = load_array(seg_root / f"{case_id}_EXP_lobe.nii.gz", np.int16)
            fixed_lobe = load_array(seg_root / f"{case_id}_INSP_lobe.nii.gz", np.int16)

            metrics: dict[str, Any] = {}
            baseline_path = baseline_dvf_dir / f"{case_id}_DVF.nii.gz"
            if baseline_path.exists():
                base = load_array(baseline_path, np.float32)
                base_pred = independent_warp_pull_chunked(moving_lobe, base, order=0, chunk_x=chunk_x)
                metrics["baseline_dice"] = mean_lobe_dice_independent(base_pred, fixed_lobe, labels)
                metrics["baseline_fold_pct"] = float(np.mean(independent_jacobian_det_inner(base) <= 0) * 100)

            residual_stats = {
                "raw_residual_abs_mean_vox": float(torch.mean(torch.abs(residual_raw)).cpu()),
                "raw_residual_abs_max_vox": float(torch.max(torch.abs(residual_raw)).cpu()),
                "aligned_residual_abs_mean_vox": float(torch.mean(torch.abs(residual_can)).cpu()),
                "aligned_residual_abs_max_vox": float(torch.max(torch.abs(residual_can)).cpu()),
            }

            for name, field in fields_original.items():
                pred = independent_warp_pull_chunked(moving_lobe, field, order=0, chunk_x=chunk_x)
                metrics[f"{name}_dice"] = mean_lobe_dice_independent(pred, fixed_lobe, labels)
                metrics[f"{name}_fold_pct"] = float(np.mean(independent_jacobian_det_inner(field) <= 0) * 100)

            row = {"case_id": case_id, **metrics, **residual_stats}
            rows.append(row)
            rec.add(
                section,
                "per_case_refiner_diagnostic",
                "INFO",
                case_id=case_id,
                message="Reproduced current path and frame-aligned/composition diagnostics.",
                values=row,
            )

    if rows:
        numeric_keys = sorted({k for row in rows for k, v in row.items() if k != "case_id" and isinstance(v, (int, float))})
        means = {k: float(np.mean([float(row[k]) for row in rows if k in row])) for k in numeric_keys}
        rec.add(
            section,
            "mean_refiner_diagnostic",
            "INFO",
            message="Mean metrics over checkpoint-diagnostic cases.",
            values=means,
        )
        return {"rows": rows, "means": means}
    return {}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    raw_root = Path(args.raw_data_root).expanduser().resolve()
    baseline_dir = resolve_path(project_root, args.baseline_output_dir)
    refiner_config = resolve_path(project_root, args.refiner_config)
    checkpoint = resolve_path(project_root, args.checkpoint)
    output_dir = resolve_path(project_root, args.output_dir)
    assert baseline_dir is not None and refiner_config is not None and output_dir is not None

    rec = AuditRecorder()
    metadata = {
        "project_root": str(project_root),
        "raw_data_root": str(raw_root),
        "baseline_output_dir": str(baseline_dir),
        "refiner_config": str(refiner_config),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "output_dir": str(output_dir),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "argv": sys.argv,
    }

    # Core paths.
    rec.add("setup", "project_root", "PASS" if project_root.exists() else "FAIL", message=str(project_root))
    rec.add("setup", "raw_data_root", "PASS" if raw_root.exists() else "FAIL", message=str(raw_root))
    rec.add("setup", "baseline_output_dir", "PASS" if baseline_dir.exists() else "WARN", message=str(baseline_dir))

    if not project_root.exists() or not raw_root.exists():
        rec.write(output_dir, metadata)
        print((output_dir / "audit_summary.txt").read_text(encoding="utf-8"))
        return 2

    modules = rec.guarded("setup", "project_imports", lambda: import_project_modules(project_root))
    if modules is None:
        rec.write(output_dir, metadata)
        print((output_dir / "audit_summary.txt").read_text(encoding="utf-8"))
        return 2
    rec.add("setup", "project_imports", "PASS", message="Imported project spatial/metric modules.")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        rec.add("setup", "device", "WARN", message=f"Requested {requested_device}, CUDA unavailable; using CPU.")
    else:
        device = torch.device(requested_device)
        rec.add("setup", "device", "PASS", message=str(device))

    dataset_info = rec.guarded(
        "dataset",
        "dataset_audit_execution",
        lambda: audit_dataset(
            rec,
            raw_root,
            expected_train=args.expected_training_cases,
            expected_val=args.expected_validation_cases,
            max_train=args.max_training_cases,
            max_val=args.max_validation_cases,
        ),
    ) or {}

    rec.guarded("synthetic_ops", "synthetic_audit_execution", lambda: audit_synthetic_ops(rec, modules, device))

    val_pairs = dataset_info.get("validation_pairs") or discover_pairs(raw_root, "validation")
    rec.guarded(
        "baseline",
        "baseline_audit_execution",
        lambda: audit_baseline(
            rec,
            raw_root=raw_root,
            baseline_dir=baseline_dir,
            val_pairs=val_pairs,
            modules=modules,
            device=device,
            max_val=args.max_validation_cases,
            chunk_x=args.chunk_x,
            skip_official_warp=args.skip_official_warp_check,
            skip_transform=args.skip_transform_check,
        ),
    )

    rec.guarded(
        "refiner_frame",
        "refiner_frame_audit_execution",
        lambda: audit_refiner_frame_consistency(
            rec,
            project_root=project_root,
            raw_root=raw_root,
            refiner_config_path=refiner_config,
            max_val=args.max_validation_cases,
            max_train=args.max_training_cases,
        ),
    )

    if checkpoint is not None:
        rec.guarded(
            "checkpoint_diagnostic",
            "checkpoint_diagnostic_execution",
            lambda: checkpoint_diagnostic(
                rec,
                project_root=project_root,
                raw_root=raw_root,
                config_path=refiner_config,
                checkpoint_path=checkpoint,
                baseline_dir=baseline_dir,
                device=device,
                max_val=args.max_validation_cases,
                chunk_x=args.chunk_x,
                modules=modules,
            ),
        )
    else:
        rec.add(
            "checkpoint_diagnostic",
            "checkpoint_supplied",
            "SKIP",
            message="No --checkpoint supplied; checkpoint-specific refiner diagnostics were not run.",
        )

    rec.write(output_dir, metadata)
    summary_path = output_dir / "audit_summary.txt"
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Wrote: {output_dir / 'audit_report.json'}")
    print(f"Wrote: {output_dir / 'audit_checks.csv'}")
    print(f"Wrote: {summary_path}")

    if args.strict and rec.counts()["FAIL"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
