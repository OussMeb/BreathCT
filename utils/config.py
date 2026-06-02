from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    raw_data_root: str = "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data"
    train_data_root: str = "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_preprocessed"
    hu_min: float = -1000.0
    hu_max: float = 600.0
    foreground_threshold_hu: float = -995.0
    canonical_axcodes: tuple[str, str, str] = ("R", "A", "S")


@dataclass
class ModelConfig:
    base_channels: int = 8
    max_channels: int = 64
    predict: str = "svf"
    integration_steps: int = 7
    max_residual_voxels: float = 8.0
    input_channels: int = 5


@dataclass
class LossConfig:
    image: float = 1.0
    bending: float = 0.05
    jacobian: float = 1.0
    smooth: float = 0.0
    lncc_windows: tuple[int, ...] = (9, 15, 21)
    lncc_weights: tuple[float, ...] = (0.5, 0.3, 0.2)


@dataclass
class TrainConfig:
    output_dir: str = "outputs/refiner"
    epochs: int = 100
    batch_size: int = 1
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-6
    seed: int = 2026
    amp: bool = True
    grad_clip: float = 1.0
    save_every: int = 10
    log_every: int = 1
    device: str = "cuda"


@dataclass
class InferConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/validation_prediction"
    make_zip: bool = True
    overwrite: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(config_path)

    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension: {config_path.suffix}")


def _recursive_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in values.items() if key in valid}
    return cls(**kwargs)


def build_experiment_config(config_dict: dict[str, Any]) -> ExperimentConfig:
    default = asdict(ExperimentConfig())
    merged = _recursive_update(default, config_dict)

    return ExperimentConfig(
        data=_dataclass_from_dict(DataConfig, merged.get("data", {})),
        model=_dataclass_from_dict(ModelConfig, merged.get("model", {})),
        loss=_dataclass_from_dict(LossConfig, merged.get("loss", {})),
        train=_dataclass_from_dict(TrainConfig, merged.get("train", {})),
        infer=_dataclass_from_dict(InferConfig, merged.get("infer", {})),
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--raw-data-root", type=str, default=None)
    parser.add_argument("--train-data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--max-residual-voxels", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.raw_data_root is not None:
        config.data.raw_data_root = args.raw_data_root
    if args.train_data_root is not None:
        config.data.train_data_root = args.train_data_root
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
        config.infer.output_dir = args.output_dir
    if args.checkpoint is not None:
        config.infer.checkpoint = args.checkpoint
    if args.device is not None:
        config.train.device = args.device
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.lr is not None:
        config.train.lr = args.lr
    if args.base_channels is not None:
        config.model.base_channels = args.base_channels
    if args.max_residual_voxels is not None:
        config.model.max_residual_voxels = args.max_residual_voxels
    if args.overwrite:
        config.infer.overwrite = True
    if args.no_amp:
        config.train.amp = False

    return config
