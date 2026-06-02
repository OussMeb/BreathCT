#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from utils.config import add_common_args, apply_cli_overrides, build_experiment_config, load_config
from utils.data import build_dataloader
from models.refinement import ResidualRegistrationNet
from utils.orientation import canonical_dvf_to_original_grid, save_dvf_nifti_xyzc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict validation DVFs on original INSP grid.")
    add_common_args(parser)
    parser.add_argument("--make-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(build_experiment_config(load_config(args.config)), args)
    if args.make_zip:
        config.infer.make_zip = True

    output_dir = Path(config.infer.output_dir)
    dvf_dir = output_dir / "dvfs"
    if dvf_dir.exists() and any(dvf_dir.glob("*.nii.gz")) and not config.infer.overwrite:
        raise FileExistsError(f"DVF output dir is not empty. Use --overwrite: {dvf_dir}")
    dvf_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")
    loader = build_dataloader(
        "validation",
        config.data,
        batch_size=1,
        num_workers=config.train.num_workers,
        shuffle=False,
    )

    model = ResidualRegistrationNet(
        in_channels=config.model.input_channels,
        base_channels=config.model.base_channels,
        max_channels=config.model.max_channels,
        predict=config.model.predict,
        integration_steps=config.model.integration_steps,
        max_residual_voxels=config.model.max_residual_voxels,
    ).to(device)

    if not config.infer.checkpoint:
        raise ValueError("Set infer.checkpoint or pass --checkpoint.")
    checkpoint = torch.load(config.infer.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    outputs: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation inference", dynamic_ncols=True):
            moving = batch["moving"].to(device, non_blocking=True)
            fixed = batch["fixed"].to(device, non_blocking=True)
            mask = batch["foreground_mask"].to(device, non_blocking=True)
            case_id = batch["case_id"][0]
            meta = batch["meta"][0]

            pred = model(moving, fixed, foreground_mask=mask, case_ids=[case_id])
            dvf_cxyz = pred["final_dvf"][0].detach().cpu().numpy().astype(np.float32)
            dvf_xyzc = canonical_dvf_to_original_grid(dvf_cxyz, meta["original_insp_path"])

            out_path = dvf_dir / f"{case_id}_DVF.nii.gz"
            save_dvf_nifti_xyzc(dvf_xyzc, meta["original_insp_path"], out_path)
            outputs.append(str(out_path))

    zip_path = None
    if config.infer.make_zip:
        zip_path = output_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path_text in outputs:
                path = Path(path_text)
                archive.write(path, arcname=path.name)

    summary = {
        "config": asdict(config),
        "checkpoint": config.infer.checkpoint,
        "dvf_dir": str(dvf_dir),
        "zip_path": str(zip_path) if zip_path else None,
        "dvfs": outputs,
        "dvf_convention": {
            "grid": "original_INSP_grid",
            "layout": "X,Y,Z,3",
            "units": "voxel",
            "warp": "pull: warped_EXP[x] = EXP[x + DVF[x]]",
        },
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
