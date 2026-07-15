#!/usr/bin/env python3
"""
Audit, freeze, package, and verify the C8A GT-assisted validation submission.

Default project root:
    /home/oussama/Desktop/MICCAI FRANCE

Source DVFs:
    outputs/sgr_fm/c8a_gt_oracle/dvfs

Final ZIP:
    SGR_FM_C8A_GT_ORACLE_validation.zip

The script:
  1. verifies exactly ten expected DVFs;
  2. checks NIfTI vector layout, finite floating values, fixed-INSP grid and affine;
  3. copies the exact DVFs into a frozen staging directory;
  4. writes SHA-256 manifests and preserves compact C8A result files;
  5. creates a ZIP with exactly ten top-level DVFs;
  6. verifies ZIP structure, CRC, and source-to-ZIP byte identity.

This packages a GT-assisted validation result. It does not establish that hidden-test
labels will be available to the submitted algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


EXPECTED_NAMES = [f"NLST_{i:04d}_DVF.nii.gz" for i in range(1, 11)]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def vector_spatial_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D vector field, got shape={shape}")
    if shape[-1] == 3:
        return tuple(int(v) for v in shape[:3])
    if shape[0] == 3:
        return tuple(int(v) for v in shape[1:])
    raise ValueError(
        f"Expected vector layout [X,Y,Z,3] or [3,X,Y,Z], got shape={shape}"
    )


def audit_dvfs(dvf_dir: Path, fixed_dir: Path, affine_atol: float) -> list[dict]:
    actual = sorted(p.name for p in dvf_dir.glob("*.nii.gz"))
    if actual != EXPECTED_NAMES:
        missing = sorted(set(EXPECTED_NAMES) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_NAMES))
        raise RuntimeError(
            "DVF inventory failed.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    records: list[dict] = []
    print("[1/5] DVF INVENTORY: PASS — exactly 10 expected files")

    for name in EXPECTED_NAMES:
        case_id = name.removesuffix("_DVF.nii.gz")
        dvf_path = dvf_dir / name
        fixed_path = fixed_dir / f"{case_id}_INSP.nii.gz"

        if not fixed_path.is_file():
            raise FileNotFoundError(f"Missing fixed INSP image: {fixed_path}")

        dvf_img = nib.load(str(dvf_path))
        fixed_img = nib.load(str(fixed_path))
        dvf_shape = tuple(int(v) for v in dvf_img.shape)
        fixed_shape = tuple(int(v) for v in fixed_img.shape[:3])
        spatial_shape = vector_spatial_shape(dvf_shape)

        if spatial_shape != fixed_shape:
            raise RuntimeError(
                f"{case_id}: shape mismatch, DVF={spatial_shape}, fixed={fixed_shape}"
            )

        affine_diff = float(
            np.max(np.abs(np.asarray(dvf_img.affine) - np.asarray(fixed_img.affine)))
        )
        if affine_diff > affine_atol:
            raise RuntimeError(
                f"{case_id}: affine mismatch, max_abs_diff={affine_diff:.9g}"
            )

        data = np.asanyarray(dvf_img.dataobj)
        if not np.issubdtype(data.dtype, np.floating):
            raise RuntimeError(f"{case_id}: DVF dtype is not floating: {data.dtype}")
        if not np.isfinite(data).all():
            bad = int(data.size - np.count_nonzero(np.isfinite(data)))
            raise RuntimeError(f"{case_id}: non-finite values found: {bad}")

        records.append(
            {
                "case_id": case_id,
                "filename": name,
                "source_path": str(dvf_path),
                "fixed_path": str(fixed_path),
                "dvf_shape": list(dvf_shape),
                "fixed_shape": list(fixed_shape),
                "dtype": str(data.dtype),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "affine_max_abs_diff": affine_diff,
                "sha256": sha256_file(dvf_path),
                "size_bytes": dvf_path.stat().st_size,
            }
        )
        print(
            f"  {case_id}: shape={dvf_shape}, dtype={data.dtype}, "
            f"range=({float(np.min(data)):.4f}, {float(np.max(data)):.4f})"
        )

    print("[2/5] NIFTI + FIXED-GRID AUDIT: PASS")
    return records


def prepare_frozen_staging(
    source_root: Path,
    source_dvf_dir: Path,
    staging_root: Path,
    records: list[dict],
) -> tuple[Path, Path]:
    if staging_root.exists():
        shutil.rmtree(staging_root)

    frozen_dvf_dir = staging_root / "dvfs"
    frozen_dvf_dir.mkdir(parents=True, exist_ok=True)

    frozen_records = []
    for rec in records:
        src = source_dvf_dir / rec["filename"]
        dst = frozen_dvf_dir / rec["filename"]
        shutil.copy2(src, dst)

        src_hash = rec["sha256"]
        dst_hash = sha256_file(dst)
        if src_hash != dst_hash:
            raise RuntimeError(f"Copy hash mismatch: {rec['filename']}")

        frozen = dict(rec)
        frozen["frozen_path"] = str(dst)
        frozen_records.append(frozen)

    # Preserve compact evidence, not large images or DVFs beyond the ten submitted fields.
    evidence_names = [
        source_root / "c8a_gt_oracle_summary.json",
        source_root / "eval" / "validation_metrics.json",
        source_root / "eval" / "validation_metrics.csv",
        source_root / "gap_audit" / "c8a_gt_oracle_gap_per_case.csv",
        source_root / "gap_audit" / "c8a_gt_oracle_gap_per_lobe.csv",
    ]
    evidence_dir = staging_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied_evidence = []
    for src in evidence_names:
        if src.is_file():
            dst = evidence_dir / src.name
            shutil.copy2(src, dst)
            copied_evidence.append(
                {
                    "filename": dst.name,
                    "sha256": sha256_file(dst),
                    "size_bytes": dst.stat().st_size,
                }
            )

    manifest = {
        "schema_version": 1,
        "method": "SGR-FM C8A GT-oracle validation submission",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "uses_validation_gt_in_optimizer": True,
        "warning": (
            "This is a GT-assisted validation submission. Hidden-test label "
            "availability and rule compliance must be confirmed separately."
        ),
        "source_root": str(source_root),
        "source_dvf_dir": str(source_dvf_dir),
        "staging_root": str(staging_root),
        "expected_archive_members": EXPECTED_NAMES,
        "dvfs": frozen_records,
        "evidence": copied_evidence,
    }

    manifest_path = staging_root / "c8a_submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    marker = staging_root / "C8A_GT_ASSISTED_SUBMISSION.txt"
    marker.write_text(
        "SGR-FM C8A GT-assisted validation submission staging.\n"
        f"created_utc={manifest['created_utc']}\n"
        "uses_validation_gt_in_optimizer=true\n"
        "Do not describe this branch as hidden-test-safe unless organizers confirm "
        "that test labels are exposed to the submitted method.\n"
    )

    # Make staging files read-only while leaving the original C8A outputs untouched.
    for path in staging_root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    print(f"[3/5] FROZEN STAGING: PASS — {staging_root}")
    return frozen_dvf_dir, manifest_path


def build_and_verify_zip(frozen_dvf_dir: Path, zip_path: Path) -> str:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    # Files are already .nii.gz; storing avoids unnecessary recompression.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in EXPECTED_NAMES:
            zf.write(frozen_dvf_dir / name, arcname=name)

    print(f"[4/5] ZIP CREATED: PASS — {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != EXPECTED_NAMES:
            raise RuntimeError(
                "ZIP member list failed.\n"
                f"Expected: {EXPECTED_NAMES}\n"
                f"Actual: {names}"
            )
        bad_crc = zf.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"ZIP CRC failed for: {bad_crc}")

        for name in EXPECTED_NAMES:
            if zf.read(name) != (frozen_dvf_dir / name).read_bytes():
                raise RuntimeError(f"ZIP byte-identity failed: {name}")

    digest = sha256_file(zip_path)
    print("[5/5] ZIP STRUCTURE + CRC + BYTE IDENTITY: PASS")
    print(f"ZIP SHA256: {digest}")
    return digest


def parse_args() -> argparse.Namespace:
    default_root = Path("/home/oussama/Desktop/MICCAI FRANCE")
    parser = argparse.ArgumentParser(
        description="Freeze, audit, package, and verify the C8A validation submission."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_root / "outputs/sgr_fm/c8a_gt_oracle",
    )
    parser.add_argument(
        "--dvf-dir",
        type=Path,
        default=default_root / "outputs/sgr_fm/c8a_gt_oracle/dvfs",
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=default_root / "Learn2Breath_train_val_data/validation/ct_data",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=default_root
        / "outputs/sgr_fm/frozen_assets/c8a_gt_oracle_submission_20260715",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=default_root / "SGR_FM_C8A_GT_ORACLE_validation.zip",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for required in [args.source_root, args.dvf_dir, args.fixed_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    records = audit_dvfs(args.dvf_dir, args.fixed_dir, args.affine_atol)
    frozen_dvf_dir, manifest_path = prepare_frozen_staging(
        args.source_root,
        args.dvf_dir,
        args.staging_root,
        records,
    )
    zip_sha = build_and_verify_zip(frozen_dvf_dir, args.zip_path)

    report = {
        "status": "PASS",
        "uses_validation_gt_in_optimizer": True,
        "zip_path": str(args.zip_path),
        "zip_sha256": zip_sha,
        "manifest_path": str(manifest_path),
        "archive_members": EXPECTED_NAMES,
    }
    report_path = args.staging_root / "submission_report.json"
    # Staging was made read-only; temporarily create the report in project root, then copy.
    temp_report = args.project_root / ".c8a_submission_report.tmp.json"
    temp_report.write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(temp_report, report_path)
    temp_report.unlink(missing_ok=True)
    report_path.chmod(report_path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    print("\nFINAL C8A PRE-SUBMISSION STATUS: PASS")
    print(f"ZIP: {args.zip_path}")
    print(f"ZIP SHA256: {zip_sha}")
    print(f"MANIFEST: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nFINAL C8A PRE-SUBMISSION STATUS: FAIL\n{exc}", file=sys.stderr)
        raise
