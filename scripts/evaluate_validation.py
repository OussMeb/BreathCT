#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from learn2breath.config import DataConfig
from learn2breath.data import discover_validation_cases
from learn2breath.metrics import LOBE_LABELS, lobe_dice, mean_lobe_dice, jacobian_stats
from learn2breath.spatial import warp_pull


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation DVFs against lobe segmentations.")
    parser.add_argument("--raw-data-root", required=True)
    parser.add_argument("--dvf-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cases = discover_validation_cases(args.raw_data_root)
    rows = []

    for case in tqdm(cases, desc="evaluate", dynamic_ncols=True):
        if case.exp_lobe is None or case.insp_lobe is None:
            raise FileNotFoundError(f"Missing lobe labels for {case.case_id}")

        dvf_path = Path(args.dvf_dir) / f"{case.case_id}_DVF.nii.gz"
        if not dvf_path.exists():
            raise FileNotFoundError(f"Missing DVF: {dvf_path}")

        moving_lobe = load_array(case.exp_lobe).astype(np.float32)
        fixed_lobe = load_array(case.insp_lobe).astype(np.int16)
        dvf = load_array(dvf_path).astype(np.float32)

        if dvf.shape[-1] != 3:
            raise ValueError(f"Expected XYZC DVF, got {dvf.shape}: {dvf_path}")

        moving_t = torch.as_tensor(moving_lobe, device=device).view(1, 1, *moving_lobe.shape)
        dvf_t = torch.as_tensor(np.moveaxis(dvf, -1, 0), device=device).view(1, 3, *moving_lobe.shape)

        warped = warp_pull(moving_t, dvf_t, mode="nearest", padding_mode="border")[0, 0].cpu().numpy().astype(np.int16)
        dice_by_lobe = lobe_dice(warped, fixed_lobe, LOBE_LABELS)
        jac = jacobian_stats(dvf_t.cpu())

        row = {
            "case_id": case.case_id,
            "mean_lobe_dice": mean_lobe_dice(warped, fixed_lobe, LOBE_LABELS),
            "folding_percentage": jac["folding_percentage"],
            **{f"dice_{label}": value for label, value in dice_by_lobe.items()},
            **{f"jac_{key}": value for key, value in jac.items()},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "case_count": int(len(df)),
        "mean_lobe_dice": float(df["mean_lobe_dice"].mean()),
        "min_case_mean_lobe_dice": float(df["mean_lobe_dice"].min()),
        "max_case_mean_lobe_dice": float(df["mean_lobe_dice"].max()),
        "mean_folding_percentage": float(df["folding_percentage"].mean()),
        "max_folding_percentage": float(df["folding_percentage"].max()),
        "case_metrics": rows,
    }

    df.to_csv(output_dir / "validation_metrics.csv", index=False)
    (output_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
