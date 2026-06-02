from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes

from utils.data import CaseFiles, discover_training_cases, discover_validation_cases
from utils.dvf_io import (
    dvf_stats,
    original_dvf_xyzc_to_canonical_ras_cxyz,
    save_canonical_ras_dvf_xyzc,
    save_dvf_xyzc,
)


@dataclass
class UniGradICONConfig:
    raw_data_root: str = "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data"
    train_data_root: str = "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_preprocessed"
    output_dir: str = "outputs/unigradicon_raw_validation"
    split: str = "validation"
    command: str = "unigradicon-register"
    fixed_modality: str = "ct"
    moving_modality: str = "ct"
    io_iterations: str = "None"
    io_sim: str = "lncc2"
    device: str = ""  # Official uniGradICON CLI does not expose --device.
    make_zip: bool = True
    evaluate: bool = True
    overwrite: bool = False
    save_warped: bool = True
    save_canonical: bool = True
    stop_on_error: bool = True
    case_ids: list[str] = field(default_factory=list)


def load_unigradicon_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML configs: pip install pyyaml") from exc
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    return data.get("unigradicon", data)


def build_unigradicon_config(config_dict: dict[str, Any], cli: Any | None = None) -> UniGradICONConfig:
    base = asdict(UniGradICONConfig())
    for key, value in config_dict.items():
        if key in base:
            base[key] = value

    config = UniGradICONConfig(**base)

    if cli is not None:
        for key in (
            "raw_data_root",
            "train_data_root",
            "output_dir",
            "split",
            "command",
            "fixed_modality",
            "moving_modality",
            "io_iterations",
            "io_sim",
            "device",
        ):
            value = getattr(cli, key, None)
            if value is not None:
                setattr(config, key, value)

        for key in ("make_zip", "evaluate", "overwrite", "save_warped", "save_canonical", "stop_on_error"):
            value = getattr(cli, key, None)
            if value is not None:
                setattr(config, key, bool(value))

        case_ids = getattr(cli, "case_ids", None)
        if case_ids:
            config.case_ids = [str(v) for v in case_ids]

    return config


def _discover_cases(config: UniGradICONConfig) -> list[CaseFiles]:
    if config.split == "validation":
        cases = discover_validation_cases(config.raw_data_root)
    elif config.split == "training":
        # Use raw CTs for uniGradICON, not normalized CTs. Training CTs are direct children.
        raw_training_root = Path(config.raw_data_root).expanduser() / "training"
        if raw_training_root.exists():
            from utils.data import _collect_ct_pairs  # local import keeps public surface small

            ct_by_case = _collect_ct_pairs(raw_training_root, recursive=False)
            cases = []
            for case_id in sorted(ct_by_case):
                phases = ct_by_case[case_id]
                if "EXP" in phases and "INSP" in phases:
                    cases.append(
                        CaseFiles(
                            split="training",
                            case_id=case_id,
                            exp_ct=phases["EXP"],
                            insp_ct=phases["INSP"],
                            source_root=raw_training_root,
                        )
                    )
        else:
            cases = discover_training_cases(config.train_data_root)
    else:
        raise ValueError(f"Unsupported split: {config.split}")

    if config.case_ids:
        requested = set(config.case_ids)
        cases = [case for case in cases if case.case_id in requested]
        missing = sorted(requested - {case.case_id for case in cases})
        if missing:
            raise FileNotFoundError(f"Requested case IDs were not found: {missing}")

    if not cases:
        raise ValueError(f"No cases found for split={config.split}")

    return cases


def _check_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find '{command}' on PATH.\n"
            "Install uniGradICON first, for example:\n"
            "  pip install unigradicon SimpleITK\n"
            "Then verify:\n"
            "  unigradicon-register --help"
        )
    return resolved


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\n[exit_code] {completed.returncode}\n")
        log.write(f"[elapsed_sec] {time.time() - started:.3f}\n")
    return int(completed.returncode)


def _transform_to_voxel_dvf_xyzc(transform_path: Path, fixed_image_path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to convert uniGradICON HDF5 transforms: pip install SimpleITK") from exc

    fixed = sitk.ReadImage(str(fixed_image_path), sitk.sitkFloat32)
    transform = sitk.ReadTransform(str(transform_path))

    field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection(),
    )

    displacement_physical_zyx = sitk.GetArrayFromImage(field).astype(np.float64)
    displacement_physical_xyx = np.transpose(displacement_physical_zyx, (2, 1, 0, 3))

    direction = np.asarray(fixed.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(fixed.GetSpacing(), dtype=np.float64)
    index_to_physical_delta = direction @ np.diag(spacing)
    physical_to_index_delta = np.linalg.inv(index_to_physical_delta)

    flat = displacement_physical_xyx.reshape(-1, 3)
    displacement_voxel = flat @ physical_to_index_delta.T
    return displacement_voxel.reshape(displacement_physical_xyx.shape).astype(np.float32)


def _build_cli_command(
    config: UniGradICONConfig,
    *,
    fixed_path: Path,
    moving_path: Path,
    transform_out: Path,
    warped_out: Path | None,
) -> list[str]:
    command = [
        config.command,
        f"--fixed={fixed_path}",
        f"--fixed_modality={config.fixed_modality}",
        f"--moving={moving_path}",
        f"--moving_modality={config.moving_modality}",
        f"--transform_out={transform_out}",
    ]

    if warped_out is not None:
        command.append(f"--warped_moving_out={warped_out}")

    if config.io_iterations != "":
        command.append(f"--io_iterations={config.io_iterations}")

    if config.io_sim:
        command.append(f"--io_sim={config.io_sim}")

    # Do not pass --device: official uniGradICON CLI currently does not support it.
    return command


def _write_submission_zip(dvf_paths: list[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in dvf_paths:
            archive.write(path, arcname=path.name)


def run_unigradicon_baseline(config: UniGradICONConfig) -> dict[str, Any]:
    _check_command(config.command)

    output_dir = Path(config.output_dir).expanduser()
    transform_dir = output_dir / "transforms"
    warped_dir = output_dir / "warped"
    dvf_dir = output_dir / "dvfs"
    canonical_dir = output_dir / "dvfs_canonical_ras"
    log_dir = output_dir / "logs"

    if output_dir.exists() and any(output_dir.rglob("*")) and not config.overwrite:
        raise FileExistsError(f"Output directory is not empty. Use --overwrite: {output_dir}")

    for directory in (transform_dir, dvf_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if config.save_warped:
        warped_dir.mkdir(parents=True, exist_ok=True)
    if config.save_canonical:
        canonical_dir.mkdir(parents=True, exist_ok=True)

    cases = _discover_cases(config)
    rows: list[dict[str, Any]] = []
    dvf_paths: list[Path] = []

    for case in cases:
        case_id = case.case_id
        transform_path = transform_dir / f"{case_id}_unigradicon.hdf5"
        warped_path = warped_dir / f"{case_id}_warped_EXP.nii.gz" if config.save_warped else None
        challenge_dvf_path = dvf_dir / f"{case_id}_DVF.nii.gz"
        canonical_dvf_path = canonical_dir / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz"
        log_path = log_dir / f"{case_id}.log"

        cli_command = _build_cli_command(
            config,
            fixed_path=case.insp_ct,
            moving_path=case.exp_ct,
            transform_out=transform_path,
            warped_out=warped_path,
        )

        exit_code = _run_command(cli_command, log_path)
        if exit_code != 0:
            row = {
                "case_id": case_id,
                "status": "failed",
                "exit_code": exit_code,
                "log_path": str(log_path),
                "fixed": str(case.insp_ct),
                "moving": str(case.exp_ct),
            }
            rows.append(row)
            if config.stop_on_error:
                raise RuntimeError(f"uniGradICON failed for {case_id}. See log: {log_path}")
            continue

        dvf_xyzc = _transform_to_voxel_dvf_xyzc(transform_path, case.insp_ct)
        save_dvf_xyzc(dvf_xyzc, case.insp_ct, challenge_dvf_path)
        dvf_paths.append(challenge_dvf_path)

        canonical_path_text: str | None = None
        if config.save_canonical:
            dvf_cxyz = original_dvf_xyzc_to_canonical_ras_cxyz(dvf_xyzc, case.insp_ct)
            save_canonical_ras_dvf_xyzc(dvf_cxyz, case.insp_ct, canonical_dvf_path)
            canonical_path_text = str(canonical_dvf_path)

        fixed_img = nib.load(str(case.insp_ct))
        row = {
            "case_id": case_id,
            "status": "ok",
            "exit_code": exit_code,
            "fixed": str(case.insp_ct),
            "moving": str(case.exp_ct),
            "fixed_axcodes": list(aff2axcodes(fixed_img.affine)),
            "transform": str(transform_path),
            "warped_moving": str(warped_path) if warped_path else None,
            "challenge_dvf": str(challenge_dvf_path),
            "canonical_ras_dvf": canonical_path_text,
            "log_path": str(log_path),
            "dvf_stats_voxel": dvf_stats(dvf_xyzc),
        }
        rows.append(row)

    zip_path = output_dir / "submission_unigradicon_raw.zip"
    if config.make_zip and config.split == "validation":
        _write_submission_zip(dvf_paths, zip_path)

    summary = {
        "config": asdict(config),
        "case_count": len(cases),
        "ok_count": sum(row["status"] == "ok" for row in rows),
        "failed_count": sum(row["status"] != "ok" for row in rows),
        "output_dir": str(output_dir),
        "dvf_dir": str(dvf_dir),
        "canonical_dvf_dir": str(canonical_dir) if config.save_canonical else None,
        "zip_path": str(zip_path) if config.make_zip and config.split == "validation" else None,
        "dvf_convention": {
            "grid": "original_INSP_grid",
            "layout": "X,Y,Z,3",
            "units": "voxel",
            "warp": "pull: warped_EXP[x] = EXP[x + DVF[x]]",
            "source_transform": "uniGradICON HDF5 converted from physical displacement to voxel displacement",
        },
        "cases": rows,
    }

    summary_path = output_dir / "unigradicon_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if config.evaluate and config.split == "validation" and summary["failed_count"] == 0:
        eval_dir = output_dir / "eval"
        project_root = Path(__file__).resolve().parents[1]
        eval_script = project_root / "evaluate_validation.py"
        if not eval_script.exists():
            eval_script = project_root / "scripts" / "evaluate_validation.py"

        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--raw-data-root",
            config.raw_data_root,
            "--dvf-dir",
            str(dvf_dir),
            "--output-dir",
            str(eval_dir),
        ]
        eval_code = _run_command(eval_cmd, log_dir / "evaluate_validation.log")
        summary["evaluation"] = {
            "exit_code": eval_code,
            "output_dir": str(eval_dir),
            "log_path": str(log_dir / "evaluate_validation.log"),
        }
        if (eval_dir / "validation_metrics.json").exists():
            summary["evaluation"]["metrics_json"] = str(eval_dir / "validation_metrics.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
