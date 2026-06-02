#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn2Breath MICCAI runner.")
    parser.add_argument("command", choices=("train", "infer", "evaluate", "unigradicon"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()

    if parsed.command == "train":
        import train_refiner as module
        sys.argv = ["train_refiner.py", *parsed.args]
    elif parsed.command == "infer":
        import infer_validation as module
        sys.argv = ["infer_validation.py", *parsed.args]
    elif parsed.command == "evaluate":
        import evaluate_validation as module
        sys.argv = ["evaluate_validation.py", *parsed.args]
    else:
        import run_unigradicon as module
        sys.argv = ["run_unigradicon.py", *parsed.args]

    module.main()


if __name__ == "__main__":
    main()
