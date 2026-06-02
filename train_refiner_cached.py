#!/usr/bin/env python3
"""
File: train_refiner_cached.py

Train a low-space residual refiner from cached raw uniGradICON DVFs.

Run:
    PYTHONPATH=. python train_refiner_cached.py --config configs/refiner_cached_lowspace.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from models.residual_refiner import ResidualRefiner3d
from utils.refine_io import discover_training_cases, load_config, load_nifti_array, normalize_ct_hu
from utils.refine_losses import bending_energy, jacobian_folding_loss, masked_mse_local_ncc
from utils.refine_spatial import (
    compose_additive,
    downsample_dvf_5d,
    integrate_svf,
    resize_volume_5d,
    tensor_from_volume,
    tensor_from_xyzc,
    warp_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/refiner_cached_lowspace.yaml")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--init-dvf-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--reduce-factor", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.raw_data_root:
        config["data"]["raw_data_root"] = args.raw_data_root
    if args.init_dvf_dir:
        config["data"]["init_dvf_dir"] = args.init_dvf_dir
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.reduce_factor is not None:
        config["train"]["reduce_factor"] = args.reduce_factor
    if args.base_channels is not None:
        config["model"]["base_channels"] = args.base_channels
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.device:
        config["train"]["device"] = args.device
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_batch(case, config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    hu_min = float(config["data"]["hu_min"])
    hu_max = float(config["data"]["hu_max"])
    threshold_hu = float(config["data"]["foreground_threshold_hu"])
    threshold_norm = (np.clip(threshold_hu, hu_min, hu_max) - hu_min) / (hu_max - hu_min) * 2.0 - 1.0

    moving_np, _ = load_nifti_array(case.moving_path)
    fixed_np, _ = load_nifti_array(case.fixed_path)
    init_dvf_np, _ = load_nifti_array(case.init_dvf_path)

    moving = tensor_from_volume(normalize_ct_hu(moving_np, hu_min, hu_max), device)
    fixed = tensor_from_volume(normalize_ct_hu(fixed_np, hu_min, hu_max), device)
    init_dvf = tensor_from_xyzc(init_dvf_np, device)

    mask = ((moving > threshold_norm) | (fixed > threshold_norm)).float()

    factor = int(config["train"]["reduce_factor"])
    moving_low = resize_volume_5d(moving, factor)
    fixed_low = resize_volume_5d(fixed, factor)
    mask_low = resize_volume_5d(mask, factor, mode="nearest")
    init_low = downsample_dvf_5d(init_dvf, factor)

    warped0 = warp_volume(moving_low, init_low)
    abs_diff = (fixed_low - warped0).abs()
    inputs = torch.cat([moving_low, fixed_low, warped0, abs_diff, mask_low], dim=1)

    return {
        "moving": moving_low,
        "fixed": fixed_low,
        "mask": mask_low,
        "init_dvf": init_low,
        "inputs": inputs,
    }


def train_one_epoch(
    model: nn.Module,
    cases,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    random.shuffle(cases)

    image_weight = float(config["loss"]["image"])
    bending_weight = float(config["loss"]["bending"])
    jacobian_weight = float(config["loss"]["jacobian"])
    lncc_window = int(config["loss"]["lncc_window"])
    integration_steps = int(config["model"]["integration_steps"])
    predict = str(config["model"]["predict"])
    amp = bool(config["train"]["amp"]) and device.type == "cuda"

    totals: list[dict[str, float]] = []

    for index, case in enumerate(cases, start=1):
        batch = build_batch(case, config, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            residual_raw = model(batch["inputs"])
            residual = integrate_svf(residual_raw, integration_steps) if predict == "svf" else residual_raw
            final_dvf = compose_additive(batch["init_dvf"], residual)
            warped = warp_volume(batch["moving"], final_dvf)

            loss_image = masked_mse_local_ncc(batch["fixed"], warped, batch["mask"], window=lncc_window)
            loss_bending = bending_energy(residual)
            loss_jac, folding_pct = jacobian_folding_loss(final_dvf)

            total = image_weight * loss_image + bending_weight * loss_bending + jacobian_weight * loss_jac

        if not torch.isfinite(total):
            print(f"[WARN] skipping non-finite batch {case.case_id}: total={float(total.detach().cpu())}")
            continue

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()

        totals.append(
            {
                "total": float(total.detach().cpu()),
                "image": float(loss_image.detach().cpu()),
                "bending": float(loss_bending.detach().cpu()),
                "jacobian": float(loss_jac.detach().cpu()),
                "folding_pct": float(folding_pct.detach().cpu()),
            }
        )

        if index % int(config["train"]["log_every"]) == 0:
            print(
                f"epoch={epoch:03d} case={index:03d}/{len(cases):03d} id={case.case_id} "
                f"total={totals[-1]['total']:.5f} image={totals[-1]['image']:.5f} "
                f"jac={totals[-1]['jacobian']:.6f} folding={totals[-1]['folding_pct']:.5f}%"
            )

    if not totals:
        raise RuntimeError("No finite batches were processed in this epoch.")

    return {key: float(np.mean([row[key] for row in totals])) for key in totals[0]}


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    seed_everything(int(config["train"]["seed"]))

    output_dir = Path(config["train"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    cases = discover_training_cases(Path(config["data"]["raw_data_root"]), Path(config["data"]["init_dvf_dir"]))
    if not cases:
        raise RuntimeError("No training cases discovered with cached initializer DVFs.")

    print(f"training_cases={len(cases)}")
    print(f"device={device}")
    print(f"reduce_factor={config['train']['reduce_factor']}")
    print(f"init_dvf_dir={config['data']['init_dvf_dir']}")

    model = ResidualRefiner3d(
        in_channels=int(config["model"]["input_channels"]),
        base_channels=int(config["model"]["base_channels"]),
        max_channels=int(config["model"]["max_channels"]),
        max_residual_voxels=float(config["model"]["max_residual_voxels"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    history = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        metrics = train_one_epoch(model, cases, optimizer, scaler, config, device, epoch)
        row = {"epoch": epoch, **metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            ckpt_dir / "latest.pt",
        )
        if epoch % int(config["train"]["save_every"]) == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": config}, ckpt_dir / f"epoch_{epoch:03d}.pt")

        with (output_dir / "train_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    print(f"done: {output_dir}")


if __name__ == "__main__":
    main()
