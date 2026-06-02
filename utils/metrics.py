from __future__ import annotations

import numpy as np
import torch

from utils.spatial import jacobian_determinant


LOBE_LABELS = (8, 16, 32, 64, 128)


def dice_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    denom = int(pred_b.sum() + target_b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_b, target_b).sum() / denom)


def lobe_dice(pred: np.ndarray, target: np.ndarray, labels: tuple[int, ...] = LOBE_LABELS) -> dict[int, float]:
    return {label: dice_binary(pred == label, target == label) for label in labels}


def mean_lobe_dice(pred: np.ndarray, target: np.ndarray, labels: tuple[int, ...] = LOBE_LABELS) -> float:
    values = list(lobe_dice(pred, target, labels).values())
    return float(np.mean(values))


def jacobian_stats(dvf: torch.Tensor) -> dict[str, float]:
    det = jacobian_determinant(dvf).detach().float().cpu().numpy().ravel()
    return {
        "min": float(np.min(det)),
        "p0_1": float(np.percentile(det, 0.1)),
        "p1": float(np.percentile(det, 1.0)),
        "p50": float(np.percentile(det, 50.0)),
        "p99": float(np.percentile(det, 99.0)),
        "max": float(np.max(det)),
        "folding_percentage": float(np.mean(det <= 0.0) * 100.0),
    }
