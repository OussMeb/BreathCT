#!/usr/bin/env python3
"""
File: inspect_learn2breath_streamlit_fixed.py

Streamlit viewer for raw or preprocessed Learn2Breath data.

Run:
    streamlit run inspect_learn2breath_streamlit_fixed.py

Alternative:
    python -m streamlit run inspect_learn2breath_streamlit_fixed.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st


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


def ensure_streamlit_runtime() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return

    if get_script_run_ctx() is None:
        print("Run this app with:")
        print("  streamlit run inspect_learn2breath_streamlit_fixed.py")
        print("or:")
        print("  python -m streamlit run inspect_learn2breath_streamlit_fixed.py")
        sys.exit(1)


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name))


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


def discover_cases(root: Path) -> list[CaseFiles]:
    cases: list[CaseFiles] = []

    for split in ("training", "validation"):
        split_root = root / split
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

        for case_id in sorted(ct_by_case, key=lambda value: natural_key(Path(value))):
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


@st.cache_data(show_spinner=False)
def load_volume(path_text: str) -> np.ndarray:
    image = nib.load(path_text)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {data.shape}: {path_text}")
    return data.astype(np.float32, copy=False)


def volume_stats(volume: np.ndarray) -> dict[str, float | tuple[int, ...]]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return {"shape": tuple(int(v) for v in volume.shape), "min": np.nan, "p1": np.nan, "p50": np.nan, "p99": np.nan, "max": np.nan}

    return {
        "shape": tuple(int(v) for v in volume.shape),
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def get_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "Axial":
        return volume[:, :, index]
    if plane == "Sagittal":
        return volume[index, :, :]
    if plane == "Coronal":
        return volume[:, index, :]
    raise ValueError(f"Unsupported plane: {plane}")


def plane_size(volume: np.ndarray, plane: str) -> int:
    if plane == "Axial":
        return volume.shape[2]
    if plane == "Sagittal":
        return volume.shape[0]
    if plane == "Coronal":
        return volume.shape[1]
    raise ValueError(f"Unsupported plane: {plane}")


def display_slice(
    volume: np.ndarray,
    *,
    title: str,
    plane: str,
    index: int,
    vmin: float,
    vmax: float,
    label: np.ndarray | None,
    alpha: float,
) -> None:
    image_slice = np.rot90(get_slice(volume, plane, index))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)

    if label is not None:
        label_slice = np.rot90(get_slice(label, plane, index))
        masked = np.ma.masked_where(label_slice <= 0, label_slice)
        ax.imshow(masked, alpha=alpha, interpolation="nearest")

    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def normalize_for_plot(volume: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    clipped = np.clip(volume, vmin, vmax)
    return (clipped - vmin) / max(1e-6, vmax - vmin)


def downsample_for_3d(volume: np.ndarray, max_dim: int) -> np.ndarray:
    stride = max(1, int(np.ceil(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride]


def display_3d_volume(volume: np.ndarray, *, vmin: float, vmax: float, max_dim: int, title: str) -> None:
    small = downsample_for_3d(volume, max_dim=max_dim)
    values = normalize_for_plot(small, vmin, vmax)

    x = np.arange(values.shape[0])
    y = np.arange(values.shape[1])
    z = np.arange(values.shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    fig = go.Figure(
        data=go.Volume(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=values.ravel(),
            isomin=0.05,
            isomax=0.95,
            opacity=0.08,
            surface_count=12,
        )
    )
    fig.update_layout(
        title=title,
        width=900,
        height=700,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def default_root() -> str:
    raw = Path("/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data")
    preprocessed = raw.parent / "Learn2Breath_preprocessed_v1"

    if raw.exists():
        return str(raw)
    if preprocessed.exists():
        return str(preprocessed)
    return str(raw)


def main() -> None:
    st.set_page_config(page_title="Learn2Breath Inspector", layout="wide")
    st.title("Learn2Breath 2D/3D Inspector")

    root_text = st.sidebar.text_input("Dataset root", value=default_root())
    root = Path(root_text).expanduser().resolve()

    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

    if not root.exists():
        st.error(f"Root does not exist: {root}")
        st.stop()

    cases = discover_cases(root)
    if not cases:
        st.error(f"No paired cases found under: {root}")
        st.info("Expected structure: root/training/NLST_XXXX_EXP.nii.gz and root/training/NLST_XXXX_INSP.nii.gz")
        st.stop()

    splits = sorted(set(case.split for case in cases))
    split = st.sidebar.selectbox("Split", splits)
    split_cases = [case for case in cases if case.split == split]

    case_ids = [case.case_id for case in split_cases]
    case_id = st.sidebar.selectbox("Case", case_ids)
    case = next(item for item in split_cases if item.case_id == case_id)

    exp = load_volume(str(case.exp_ct))
    insp = load_volume(str(case.insp_ct))

    exp_lobe = load_volume(str(case.exp_lobe)) if case.exp_lobe else None
    insp_lobe = load_volume(str(case.insp_lobe)) if case.insp_lobe else None
    exp_fissure = load_volume(str(case.exp_fissure)) if case.exp_fissure else None
    insp_fissure = load_volume(str(case.insp_fissure)) if case.insp_fissure else None

    st.sidebar.subheader("2D display")
    plane = st.sidebar.selectbox("Plane", ["Axial", "Sagittal", "Coronal"])
    max_index = min(plane_size(exp, plane), plane_size(insp, plane)) - 1
    index = st.sidebar.slider("Slice", 0, max_index, max_index // 2)

    exp_sample = exp.ravel()[:: max(1, exp.size // 500_000)]
    insp_sample = insp.ravel()[:: max(1, insp.size // 500_000)]
    combined = np.concatenate([exp_sample, insp_sample])
    auto_min = float(np.percentile(combined, 1))
    auto_max = float(np.percentile(combined, 99))

    if auto_min >= -1.5 and auto_max <= 1.5:
        default_vmin, default_vmax = -1.0, 1.0
    else:
        default_vmin, default_vmax = -1000.0, 500.0

    vmin = st.sidebar.number_input("Window min", value=float(default_vmin))
    vmax = st.sidebar.number_input("Window max", value=float(default_vmax))

    overlay_type = st.sidebar.selectbox("Overlay", ["None", "Lobe", "Fissure"])
    overlay_alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.35)

    if overlay_type == "Lobe":
        exp_overlay = exp_lobe
        insp_overlay = insp_lobe
    elif overlay_type == "Fissure":
        exp_overlay = exp_fissure
        insp_overlay = insp_fissure
    else:
        exp_overlay = None
        insp_overlay = None

    st.subheader(f"{split}/{case_id}")
    st.write(
        {
            "EXP": str(case.exp_ct),
            "INSP": str(case.insp_ct),
            "EXP stats": volume_stats(exp),
            "INSP stats": volume_stats(insp),
            "Has EXP lobe": exp_lobe is not None,
            "Has INSP lobe": insp_lobe is not None,
            "Has EXP fissure": exp_fissure is not None,
            "Has INSP fissure": insp_fissure is not None,
        }
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        display_slice(exp, title=f"EXP {plane} {index}", plane=plane, index=index, vmin=vmin, vmax=vmax, label=exp_overlay, alpha=overlay_alpha)

    with col2:
        display_slice(insp, title=f"INSP {plane} {index}", plane=plane, index=index, vmin=vmin, vmax=vmax, label=insp_overlay, alpha=overlay_alpha)

    with col3:
        if exp.shape == insp.shape:
            diff = insp - exp
            diff_max = float(np.percentile(np.abs(diff), 99))
            display_slice(diff, title=f"INSP - EXP {plane} {index}", plane=plane, index=index, vmin=-diff_max, vmax=diff_max, label=None, alpha=0.0)
        else:
            st.warning(f"Shape mismatch: EXP {exp.shape}, INSP {insp.shape}")

    st.subheader("3D volume rendering")
    enable_3d = st.checkbox("Enable 3D rendering", value=False)
    if enable_3d:
        phase = st.selectbox("3D phase", ["EXP", "INSP"], horizontal=True)
        max_dim = st.slider("3D max dimension", 32, 128, 64, step=8)
        volume = exp if phase == "EXP" else insp
        display_3d_volume(volume, vmin=vmin, vmax=vmax, max_dim=max_dim, title=f"{case_id} {phase}")
    else:
        st.info("3D rendering is disabled by default to avoid slow startup.")


if __name__ == "__main__":
    ensure_streamlit_runtime()
    main()
