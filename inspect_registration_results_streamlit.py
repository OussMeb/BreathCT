#!/usr/bin/env python3
"""
File: inspect_registration_results_streamlit.py

Streamlit viewer for Learn2Breath registration results.

Run:
    streamlit run inspect_registration_results_streamlit.py

Typical usage:
    - raw_data_root: /home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data
    - experiment_dir: /home/oussama/Desktop/MICCAI FRANCE/outputs/unigradicon_raw_validation
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import map_coordinates


DEFAULT_RAW_ROOT = "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data"
DEFAULT_EXPERIMENT_DIR = "/home/oussama/Desktop/MICCAI FRANCE/outputs/unigradicon_raw_validation"


@dataclass(frozen=True)
class CaseArtifacts:
    case_id: str
    exp_ct: Path
    insp_ct: Path
    exp_lobe: Path | None
    insp_lobe: Path | None
    exp_fissure: Path | None
    insp_fissure: Path | None
    warped_exp: Path | None
    dvf: Path | None


def ensure_streamlit_runtime() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return
    if get_script_run_ctx() is None:
        print("Run this app with:")
        print("  streamlit run inspect_registration_results_streamlit.py")
        sys.exit(1)


@st.cache_data(show_spinner=False)
def load_nifti(path_text: str) -> tuple[np.ndarray, list[str]]:
    image = nib.load(path_text)
    data = np.asarray(image.dataobj)
    axcodes = list(nib.aff2axcodes(image.affine))
    return data, axcodes


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict[str, Any] | list[Any]:
    with open(path_text, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def warp_volume_pull(moving_path: str, dvf_path: str, order: int) -> np.ndarray:
    moving, _ = load_nifti(moving_path)
    dvf, _ = load_nifti(dvf_path)
    moving = np.asarray(moving, dtype=np.float32)
    dvf = np.asarray(dvf, dtype=np.float32)

    if dvf.ndim != 4 or dvf.shape[-1] != 3:
        raise ValueError(f"Expected DVF shape [X,Y,Z,3], got {dvf.shape}")
    if moving.shape != dvf.shape[:3]:
        raise ValueError(f"Moving volume shape {moving.shape} and DVF shape {dvf.shape[:3]} differ")

    x = np.arange(moving.shape[0], dtype=np.float32)
    y = np.arange(moving.shape[1], dtype=np.float32)
    z = np.arange(moving.shape[2], dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    coords = [xx + dvf[..., 0], yy + dvf[..., 1], zz + dvf[..., 2]]
    warped = map_coordinates(moving, coords, order=order, mode="nearest")
    if order == 0:
        warped = np.rint(warped)
    return warped.astype(np.float32, copy=False)


@st.cache_data(show_spinner=False)
def jacobian_determinant_from_dvf(dvf_path: str) -> np.ndarray:
    dvf, _ = load_nifti(dvf_path)
    dvf = np.asarray(dvf, dtype=np.float32)
    ux = dvf[..., 0]
    uy = dvf[..., 1]
    uz = dvf[..., 2]

    dux_dx, dux_dy, dux_dz = np.gradient(ux, edge_order=1)
    duy_dx, duy_dy, duy_dz = np.gradient(uy, edge_order=1)
    duz_dx, duz_dy, duz_dz = np.gradient(uz, edge_order=1)

    j11 = 1.0 + dux_dx
    j12 = dux_dy
    j13 = dux_dz
    j21 = duy_dx
    j22 = 1.0 + duy_dy
    j23 = duy_dz
    j31 = duz_dx
    j32 = duz_dy
    j33 = 1.0 + duz_dz

    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return det.astype(np.float32, copy=False)


@st.cache_data(show_spinner=False)
def discover_cases(raw_root_text: str, experiment_dir_text: str) -> list[CaseArtifacts]:
    raw_root = Path(raw_root_text)
    experiment_dir = Path(experiment_dir_text)
    ct_root = raw_root / "validation" / "ct_data"
    seg_root = raw_root / "validation" / "seg_net"
    warped_root = experiment_dir / "warped"
    dvf_root = experiment_dir / "dvfs"

    cases: list[CaseArtifacts] = []
    for insp_path in sorted(ct_root.glob("NLST_*_INSP.nii.gz")):
        case_id = insp_path.name.replace("_INSP.nii.gz", "")
        exp_path = ct_root / f"{case_id}_EXP.nii.gz"
        if not exp_path.exists():
            continue
        cases.append(
            CaseArtifacts(
                case_id=case_id,
                exp_ct=exp_path,
                insp_ct=insp_path,
                exp_lobe=(seg_root / f"{case_id}_EXP_lobe.nii.gz") if (seg_root / f"{case_id}_EXP_lobe.nii.gz").exists() else None,
                insp_lobe=(seg_root / f"{case_id}_INSP_lobe.nii.gz") if (seg_root / f"{case_id}_INSP_lobe.nii.gz").exists() else None,
                exp_fissure=(seg_root / f"{case_id}_EXP_fissure.nii.gz") if (seg_root / f"{case_id}_EXP_fissure.nii.gz").exists() else None,
                insp_fissure=(seg_root / f"{case_id}_INSP_fissure.nii.gz") if (seg_root / f"{case_id}_INSP_fissure.nii.gz").exists() else None,
                warped_exp=(warped_root / f"{case_id}_warped_EXP.nii.gz") if (warped_root / f"{case_id}_warped_EXP.nii.gz").exists() else None,
                dvf=(dvf_root / f"{case_id}_DVF.nii.gz") if (dvf_root / f"{case_id}_DVF.nii.gz").exists() else None,
            )
        )
    return cases


@st.cache_data(show_spinner=False)
def load_metrics_table(metrics_json_text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = load_json(metrics_json_text)
    if not isinstance(payload, dict):
        return pd.DataFrame(), {}
    case_metrics = payload.get("case_metrics", [])
    df = pd.DataFrame(case_metrics)
    return df, payload


@st.cache_data(show_spinner=False)
def load_train_history_table(history_path_text: str) -> pd.DataFrame:
    payload = load_json(history_path_text)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_unigradicon_case_summary(summary_path_text: str) -> pd.DataFrame:
    payload = load_json(summary_path_text)
    if not isinstance(payload, dict):
        return pd.DataFrame()
    return pd.DataFrame(payload.get("cases", []))


LOBE_COLORS = {
    8: (1.0, 0.1, 0.1, 0.55),
    16: (0.1, 0.8, 0.1, 0.55),
    32: (0.1, 0.5, 1.0, 0.55),
    64: (1.0, 0.8, 0.1, 0.55),
    128: (0.9, 0.2, 0.9, 0.55),
}
FISSURE_COLORS = {
    1: (1.0, 0.2, 0.2, 0.8),
    2: (0.1, 1.0, 1.0, 0.8),
}


def plane_size(volume: np.ndarray, plane: str) -> int:
    return {"Axial": volume.shape[2], "Coronal": volume.shape[1], "Sagittal": volume.shape[0]}[plane]


def get_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "Axial":
        return volume[:, :, index]
    if plane == "Coronal":
        return volume[:, index, :]
    return volume[index, :, :]


def rotate_for_display(image_2d: np.ndarray) -> np.ndarray:
    return np.rot90(image_2d)


def make_overlay_rgba(label_slice: np.ndarray, palette: dict[int, tuple[float, float, float, float]]) -> np.ndarray:
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for value, color in palette.items():
        mask = label_slice == value
        if np.any(mask):
            rgba[mask] = color
    return rgba


def display_panel(
    volume: np.ndarray,
    *,
    title: str,
    plane: str,
    index: int,
    vmin: float,
    vmax: float,
    overlay: np.ndarray | None = None,
    overlay_palette: dict[int, tuple[float, float, float, float]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    sl = rotate_for_display(get_slice(volume, plane, index))
    ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)

    if overlay is not None and overlay_palette is not None:
        overlay_sl = rotate_for_display(get_slice(overlay, plane, index))
        rgba = make_overlay_rgba(overlay_sl, overlay_palette)
        if np.any(rgba[..., 3] > 0):
            ax.imshow(rgba, interpolation="nearest")

    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


@st.cache_data(show_spinner=False)
def diff_volume(a_path: str, b_path: str) -> np.ndarray:
    a, _ = load_nifti(a_path)
    b, _ = load_nifti(b_path)
    return (np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).astype(np.float32, copy=False)


def pick_metrics_path(experiment_dir: Path) -> Path | None:
    candidates = [
        experiment_dir / "eval" / "validation_metrics.json",
        experiment_dir / "validation_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def pick_summary_path(experiment_dir: Path) -> Path | None:
    candidates = [
        experiment_dir / "unigradicon_summary.json",
        experiment_dir / "validation_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_metrics_tab(metrics_df: pd.DataFrame, metrics_payload: dict[str, Any], history_df: pd.DataFrame, case_df: pd.DataFrame) -> None:
    st.subheader("Validation summary")
    if metrics_payload:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Dice", f"{metrics_payload.get('mean_lobe_dice', float('nan')):.5f}")
        c2.metric("Min case Dice", f"{metrics_payload.get('min_case_mean_lobe_dice', float('nan')):.5f}")
        c3.metric("Max case Dice", f"{metrics_payload.get('max_case_mean_lobe_dice', float('nan')):.5f}")
        c4.metric("Mean folding %", f"{metrics_payload.get('mean_folding_percentage', float('nan')):.6f}")

    if not metrics_df.empty:
        bar_df = metrics_df[["case_id", "mean_lobe_dice", "folding_percentage"]].copy()
        bar_df = bar_df.sort_values("mean_lobe_dice")
        st.write("Per-case Dice")
        st.bar_chart(bar_df.set_index("case_id")["mean_lobe_dice"])
        st.write("Per-case folding percentage")
        st.bar_chart(bar_df.set_index("case_id")["folding_percentage"])
        st.dataframe(bar_df, use_container_width=True)

    if not case_df.empty:
        st.subheader("uniGradICON DVF stats")
        if "dvf_stats_voxel" in case_df.columns:
            expanded = pd.json_normalize(case_df["dvf_stats_voxel"])
            display = pd.concat([case_df[["case_id"]].reset_index(drop=True), expanded], axis=1)
            st.dataframe(display, use_container_width=True)

    st.subheader("Training history")
    if history_df.empty:
        st.info("No train_history.json loaded.")
    else:
        numeric_cols = [col for col in ["total", "image", "bending", "jacobian", "folding_pct"] if col in history_df.columns]
        hist = history_df.copy()
        hist = hist.replace([np.inf, -np.inf], np.nan)
        st.line_chart(hist.set_index("epoch")[numeric_cols])
        st.dataframe(hist, use_container_width=True)


def render_viewer_tab(case: CaseArtifacts, metrics_df: pd.DataFrame) -> None:
    exp_ct, exp_axes = load_nifti(str(case.exp_ct))
    insp_ct, insp_axes = load_nifti(str(case.insp_ct))
    warped_exp = None
    if case.warped_exp and case.warped_exp.exists():
        warped_exp, _ = load_nifti(str(case.warped_exp))

    exp_lobe = insp_lobe = exp_fissure = insp_fissure = None
    if case.exp_lobe and case.exp_lobe.exists():
        exp_lobe, _ = load_nifti(str(case.exp_lobe))
    if case.insp_lobe and case.insp_lobe.exists():
        insp_lobe, _ = load_nifti(str(case.insp_lobe))
    if case.exp_fissure and case.exp_fissure.exists():
        exp_fissure, _ = load_nifti(str(case.exp_fissure))
    if case.insp_fissure and case.insp_fissure.exists():
        insp_fissure, _ = load_nifti(str(case.insp_fissure))

    plane = st.selectbox("Plane", ["Axial", "Coronal", "Sagittal"], index=0)
    size = plane_size(insp_ct, plane)
    default_index = size // 2
    index = st.slider("Slice index", 0, size - 1, default_index, 1)
    overlay_type = st.selectbox(
        "Overlay",
        [
            "None",
            "Lobes",
            "Fissures",
            "Warped moving lobes",
            "Folding mask",
        ],
    )

    vmin = float(np.percentile(np.asarray(insp_ct, dtype=np.float32), 1))
    vmax = float(np.percentile(np.asarray(insp_ct, dtype=np.float32), 99))

    overlay_exp = overlay_insp = overlay_warped = None
    palette = None

    if overlay_type == "Lobes":
        overlay_exp = exp_lobe
        overlay_insp = insp_lobe
        overlay_warped = insp_lobe
        palette = LOBE_COLORS
    elif overlay_type == "Fissures":
        overlay_exp = exp_fissure
        overlay_insp = insp_fissure
        overlay_warped = insp_fissure
        palette = FISSURE_COLORS
    elif overlay_type == "Warped moving lobes" and case.dvf and case.exp_lobe:
        overlay_exp = exp_lobe
        overlay_insp = insp_lobe
        overlay_warped = warp_volume_pull(str(case.exp_lobe), str(case.dvf), order=0)
        palette = LOBE_COLORS
    elif overlay_type == "Folding mask" and case.dvf:
        jac = jacobian_determinant_from_dvf(str(case.dvf))
        folding = (jac <= 0).astype(np.int16)
        overlay_exp = None
        overlay_insp = folding
        overlay_warped = folding
        palette = {1: (1.0, 0.0, 0.0, 0.9)}

    before_diff = exp_ct - insp_ct
    after_diff = warped_exp - insp_ct if warped_exp is not None else None

    case_row = metrics_df[metrics_df["case_id"] == case.case_id] if not metrics_df.empty else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Case", case.case_id)
    c2.metric("EXP axes", "".join(exp_axes))
    c3.metric("INSP axes", "".join(insp_axes))
    if not case_row.empty:
        c4.metric("Mean Dice", f"{float(case_row.iloc[0]['mean_lobe_dice']):.5f}")

    row1 = st.columns(3)
    with row1[0]:
        display_panel(exp_ct, title="Moving EXP", plane=plane, index=index, vmin=vmin, vmax=vmax, overlay=overlay_exp, overlay_palette=palette)
    with row1[1]:
        display_panel(insp_ct, title="Fixed INSP", plane=plane, index=index, vmin=vmin, vmax=vmax, overlay=overlay_insp, overlay_palette=palette)
    with row1[2]:
        if warped_exp is not None:
            display_panel(warped_exp, title="Warped EXP", plane=plane, index=index, vmin=vmin, vmax=vmax, overlay=overlay_warped, overlay_palette=palette)
        else:
            st.info("No warped EXP file found.")

    row2 = st.columns(3)
    with row2[0]:
        display_panel(before_diff, title="EXP - INSP", plane=plane, index=index, vmin=-400, vmax=400)
    with row2[1]:
        if after_diff is not None:
            display_panel(after_diff, title="Warped EXP - INSP", plane=plane, index=index, vmin=-400, vmax=400)
        else:
            st.info("No warped EXP diff available.")
    with row2[2]:
        if case.dvf and case.dvf.exists():
            jac = jacobian_determinant_from_dvf(str(case.dvf))
            display_panel(jac, title="Jacobian determinant", plane=plane, index=index, vmin=0.0, vmax=2.0)
        else:
            st.info("No DVF found.")

    if not case_row.empty:
        st.subheader("Per-case metrics")
        metric_cols = [col for col in case_row.columns if col not in {"case_id"}]
        st.dataframe(case_row[["case_id", *metric_cols]], use_container_width=True)


def main() -> None:
    ensure_streamlit_runtime()
    st.set_page_config(page_title="Learn2Breath registration viewer", layout="wide")
    st.title("Learn2Breath registration results viewer")

    with st.sidebar:
        st.header("Paths")
        raw_root = Path(st.text_input("Raw data root", DEFAULT_RAW_ROOT)).expanduser()
        experiment_dir = Path(st.text_input("Experiment output dir", DEFAULT_EXPERIMENT_DIR)).expanduser()
        metrics_path_default = pick_metrics_path(experiment_dir)
        summary_path_default = pick_summary_path(experiment_dir)
        metrics_path_text = st.text_input("Metrics JSON", str(metrics_path_default) if metrics_path_default else "")
        summary_path_text = st.text_input("Summary JSON", str(summary_path_default) if summary_path_default else "")
        history_path_text = st.text_input("Train history JSON (optional)", "")
        refresh = st.button("Reload")
        if refresh:
            st.cache_data.clear()

    if not raw_root.exists():
        st.error(f"Raw root not found: {raw_root}")
        return
    if not experiment_dir.exists():
        st.error(f"Experiment dir not found: {experiment_dir}")
        return

    cases = discover_cases(str(raw_root), str(experiment_dir))
    if not cases:
        st.error("No validation cases discovered.")
        return

    metrics_df = pd.DataFrame()
    metrics_payload: dict[str, Any] = {}
    if metrics_path_text and Path(metrics_path_text).exists():
        metrics_df, metrics_payload = load_metrics_table(metrics_path_text)

    case_df = pd.DataFrame()
    if summary_path_text and Path(summary_path_text).exists():
        case_df = load_unigradicon_case_summary(summary_path_text)

    history_df = pd.DataFrame()
    if history_path_text and Path(history_path_text).exists():
        history_df = load_train_history_table(history_path_text)

    case_ids = [case.case_id for case in cases]
    case_id = st.selectbox("Case", case_ids, index=0)
    case = next(item for item in cases if item.case_id == case_id)

    tab_metrics, tab_viewer = st.tabs(["Metrics", "Viewer"])
    with tab_metrics:
        render_metrics_tab(metrics_df, metrics_payload, history_df, case_df)
    with tab_viewer:
        render_viewer_tab(case, metrics_df)


if __name__ == "__main__":
    main()
