#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import random
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.config import add_common_args, apply_cli_overrides, build_experiment_config, load_config
from utils.data import build_dataloader
from utils.losses import registration_loss
from models.refinement import ResidualRegistrationNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Learn2Breath residual deformable registration refiner.")
    add_common_args(parser)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(build_experiment_config(load_config(args.config)), args)
    set_seed(config.train.seed)

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")
    loader = build_dataloader(
        "training",
        config.data,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
    )

    model = ResidualRegistrationNet(
        in_channels=config.model.input_channels,
        base_channels=config.model.base_channels,
        max_channels=config.model.max_channels,
        predict=config.model.predict,
        integration_steps=config.model.integration_steps,
        max_residual_voxels=config.model.max_residual_voxels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scaler = GradScaler(enabled=config.train.amp and device.type == "cuda")

    history: list[dict[str, float]] = []

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        epoch_sums: dict[str, float] = {}
        steps = 0

        progress = tqdm(loader, desc=f"epoch {epoch}/{config.train.epochs}", dynamic_ncols=True)
        for batch in progress:
            moving = batch["moving"].to(device, non_blocking=True)
            fixed = batch["fixed"].to(device, non_blocking=True)
            mask = batch["foreground_mask"].to(device, non_blocking=True)
            case_ids = batch["case_id"]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.train.amp and device.type == "cuda"):
                outputs = model(moving, fixed, foreground_mask=mask, case_ids=case_ids)
                loss, terms = registration_loss(
                    outputs["warped"],
                    fixed,
                    outputs["final_dvf"],
                    foreground_mask=mask,
                    image_weight=config.loss.image,
                    bending_weight=config.loss.bending,
                    jacobian_weight=config.loss.jacobian,
                    smooth_weight=config.loss.smooth,
                    lncc_windows=tuple(config.loss.lncc_windows),
                    lncc_weights=tuple(config.loss.lncc_weights),
                )

            if not torch.isfinite(loss):
                epoch_sums["nonfinite_batches"] = epoch_sums.get("nonfinite_batches", 0.0) + 1.0
                print(f"[WARN] Skipping non-finite loss for cases={list(case_ids)} loss={float(loss.detach().cpu())}")
                continue

            scaler.scale(loss).backward()
            if config.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            for key, value in terms.items():
                value_float = float(torch.nan_to_num(value.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0))
                epoch_sums[key] = epoch_sums.get(key, 0.0) + value_float

            progress.set_postfix({key: epoch_sums.get(key, 0.0) / max(1, steps) for key in ("total", "image", "folding_pct")})

        if steps == 0:
            raise RuntimeError("All batches produced non-finite losses. Stop and inspect LNCC/data/model outputs.")

        epoch_log = {"epoch": float(epoch), **{key: value / max(1, steps) for key, value in epoch_sums.items()}}
        history.append(epoch_log)

        if epoch % config.train.log_every == 0:
            print(json.dumps(epoch_log, indent=2))

        if epoch % config.train.save_every == 0 or epoch == config.train.epochs:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                asdict(config),
            )
            save_checkpoint(output_dir / "checkpoints" / "latest.pt", model, optimizer, epoch, asdict(config))

        (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
