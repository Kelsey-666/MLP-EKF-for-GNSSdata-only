#!/usr/bin/env python3
"""Train MLP-Kalmannet model on CSV-preprocessed GNSS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent
    gnssfilter = root / "GNSSfilter"
    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(gnssfilter) not in sys.path:
        sys.path.append(str(gnssfilter))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP-Kalmannet model on preprocessed CSV data")
    parser.add_argument(
        "--config",
        default="config/MLP-train-seed44.json",
        help="Training config JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    _bootstrap_paths()
    from mlp_train import load_train_config, train_model

    args = parse_args()
    cfg = load_train_config(args.config)
    summary = train_model(cfg)
    print("training_done=1")
    print(f"best_checkpoint={summary['best_checkpoint']}")
    print(f"latest_checkpoint={summary['latest_checkpoint']}")
    print(f"best_score_3d_rmse_m={summary['best_score_3d_rmse_m']}")


if __name__ == "__main__":
    main()
