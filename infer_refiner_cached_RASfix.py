#!/usr/bin/env python3
"""
File: infer_refiner_cached.py

Full-grid validation inference for cached uniGradICON + residual refiner.
Writes only final DVFs and optional submission ZIP.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from models.residual_refiner import ResidualRefiner3d
from utils.refine_io import (
    discover_validation_cases,
    load_config,
    load_nifti_array,
    load_nifti_array_ras,
    normalize_ct_hu,
    save_nifti_like,
)
from utils.refine_spatial import (
    compose_additive,
    downsample_dvf_5d,
    integrate_svf,
    ras_canonical_to_original_grid_xyzc,
    resize_volume_5d,
    tensor_from_volume,
    tensor_from_xyzc,
    upsample_dvf_5d,
    warp_volume,
    xyzc_from_dvf_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/refiner_cached_lowspace.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-data-root")
    parser.add_argument("--init-dvf-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--make-zip", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.raw_data_root:
        config["data"]["raw_data_root"] = args.raw_data_root
    if args.init_dvf_dir:
        config["infer"]["init_dvf_dir"] = args.init_dvf_dir
    if args.output_dir:
        config["infer"]["output_dir"] = args.output_dir
    if args.device:
        config["infer"]["device"] = args.device
    config["infer"]["make_zip"] = bool(args.make_zip or config["infer"].get("make_zip", False))
    config["infer"]["overwrite"] = bool(args.overwrite or config["infer"].get("overwrite", False))
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_config = checkpoint.get("config", config)

    device = torch.device(config["infer"]["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["infer"]["output_dir"])
    dvf_dir = output_dir / "dvfs"
    dvf_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = train_config["model"]
    model = ResidualRefiner3d(
        in_channels=int(model_cfg["input_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        max_channels=int(model_cfg["max_channels"]),
        max_residual_voxels=float(model_cfg["max_residual_voxels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    init_dvf_dir = Path(config["infer"]["init_dvf_dir"])
    cases = discover_validation_cases(Path(config["data"]["raw_data_root"]), init_dvf_dir)
    if not cases:
        raise RuntimeError("No validation cases discovered with cached initializer DVFs.")

    hu_min = float(config["data"]["hu_min"])
    hu_max = float(config["data"]["hu_max"])
    threshold_hu = float(config["data"]["foreground_threshold_hu"])
    threshold_norm = (np.clip(threshold_hu, hu_min, hu_max) - hu_min) / (hu_max - hu_min) * 2.0 - 1.0

    reduce_factor = int(train_config["train"]["reduce_factor"])
    predict = str(train_config["model"]["predict"])
    integration_steps = int(train_config["model"]["integration_steps"])

    summary = {"cases": [], "output_dir": str(output_dir), "dvf_dir": str(dvf_dir)}
    with torch.no_grad():
        for case in cases:
            output_path = dvf_dir / f"{case.case_id}_DVF.nii.gz"
            if output_path.exists() and not bool(config["infer"]["overwrite"]):
                summary["cases"].append({"case_id": case.case_id, "status": "exists", "dvf": str(output_path)})
                continue

            # Validation CTs are stored as LPS, whereas cached initializer DVFs
            # are saved on a canonical RAS grid. Run the complete refiner path
            # in RAS and convert the final DVF back to the original INSP grid
            # only once, immediately before saving.
            moving_np, _, moving_ras_img = load_nifti_array_ras(case.moving_path)
            fixed_np, fixed_img, fixed_ras_img = load_nifti_array_ras(case.fixed_path)
            init_np, init_img = load_nifti_array(case.init_dvf_path)

            moving_axcodes = tuple(nib.aff2axcodes(moving_ras_img.affine))
            fixed_axcodes = tuple(nib.aff2axcodes(fixed_ras_img.affine))
            init_axcodes = tuple(nib.aff2axcodes(init_img.affine))

            if moving_axcodes != ("R", "A", "S"):
                raise RuntimeError(
                    f"{case.case_id}: canonical moving CT is not RAS: {moving_axcodes}"
                )
            if fixed_axcodes != ("R", "A", "S"):
                raise RuntimeError(
                    f"{case.case_id}: canonical fixed CT is not RAS: {fixed_axcodes}"
                )
            if init_axcodes != ("R", "A", "S"):
                raise RuntimeError(
                    f"{case.case_id}: cached initializer DVF is not RAS: {init_axcodes}"
                )

            if moving_np.shape != fixed_np.shape:
                raise RuntimeError(
                    f"{case.case_id}: canonical moving/fixed shape mismatch: "
                    f"{moving_np.shape} vs {fixed_np.shape}"
                )
            if fixed_np.shape != init_np.shape[:3]:
                raise RuntimeError(
                    f"{case.case_id}: canonical fixed/DVF shape mismatch: "
                    f"{fixed_np.shape} vs {init_np.shape[:3]}"
                )
            if init_np.ndim != 4 or init_np.shape[-1] != 3:
                raise RuntimeError(
                    f"{case.case_id}: expected XYZC initializer DVF, got {init_np.shape}"
                )

            if not np.allclose(
                moving_ras_img.affine,
                fixed_ras_img.affine,
                atol=1e-5,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{case.case_id}: canonical moving/fixed CT grids differ."
                )
            if not np.allclose(
                fixed_ras_img.affine,
                init_img.affine,
                atol=1e-5,
                rtol=0.0,
            ):
                max_diff = float(np.max(np.abs(fixed_ras_img.affine - init_img.affine)))
                raise RuntimeError(
                    f"{case.case_id}: canonical fixed CT and initializer DVF grids differ; "
                    f"max affine difference={max_diff:.6g}"
                )

            moving = tensor_from_volume(normalize_ct_hu(moving_np, hu_min, hu_max), device)
            fixed = tensor_from_volume(normalize_ct_hu(fixed_np, hu_min, hu_max), device)
            init_dvf = tensor_from_xyzc(init_np, device)
            mask = ((moving > threshold_norm) | (fixed > threshold_norm)).float()

            moving_low = resize_volume_5d(moving, reduce_factor)
            fixed_low = resize_volume_5d(fixed, reduce_factor)
            mask_low = resize_volume_5d(mask, reduce_factor, mode="nearest")
            init_low = downsample_dvf_5d(init_dvf, reduce_factor)
            warped0 = warp_volume(moving_low, init_low)
            abs_diff = (fixed_low - warped0).abs()
            inputs = torch.cat([moving_low, fixed_low, warped0, abs_diff, mask_low], dim=1)

            residual_low_raw = model(inputs)
            residual_low = integrate_svf(residual_low_raw, integration_steps) if predict == "svf" else residual_low_raw
            residual_full = upsample_dvf_5d(residual_low, tuple(int(v) for v in init_dvf.shape[-3:]))
            final_ras = compose_additive(init_dvf, residual_full)

            final_ras_xyzc = xyzc_from_dvf_tensor(final_ras)
            final_original_xyzc = ras_canonical_to_original_grid_xyzc(final_ras_xyzc, fixed_img)
            save_nifti_like(final_original_xyzc, fixed_img, output_path)

            summary["cases"].append(
                {
                    "case_id": case.case_id,
                    "status": "ok",
                    "dvf": str(output_path),
                    "refiner_frame": "canonical_RAS",
                    "output_frame": "original_INSP_grid",
                }
            )
            print(f"wrote {output_path}")

    if bool(config["infer"]["make_zip"]):
        zip_path = output_dir / "submission_refined.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dvf_dir.glob("NLST_*_DVF.nii.gz")):
                archive.write(path, arcname=path.name)
        summary["zip_path"] = str(zip_path)

    with (output_dir / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
