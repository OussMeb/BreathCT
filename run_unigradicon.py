#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.unigradicon_runner import (
    build_unigradicon_config,
    load_unigradicon_config,
    run_unigradicon_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw uniGradICON baseline and export challenge DVFs.")
    parser.add_argument("--config", default="configs/unigradicon_baseline.yaml")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--train-data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", choices=("training", "validation"))
    parser.add_argument("--command")
    parser.add_argument("--fixed-modality")
    parser.add_argument("--moving-modality")
    parser.add_argument("--io-iterations")
    parser.add_argument("--io-sim")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--make-zip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-warped", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-canonical", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_unigradicon_config(load_unigradicon_config(args.config), args)
    summary = run_unigradicon_baseline(config)
    print(json.dumps({
        "case_count": summary["case_count"],
        "ok_count": summary["ok_count"],
        "failed_count": summary["failed_count"],
        "output_dir": summary["output_dir"],
        "dvf_dir": summary["dvf_dir"],
        "zip_path": summary["zip_path"],
        "evaluation": summary.get("evaluation"),
    }, indent=2))


if __name__ == "__main__":
    main()
