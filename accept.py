#!/usr/bin/env python3
"""Acceptance/evaluation entry for MLP-Kalmannet checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent
    gnssfilter = root / "GNSSfilter"
    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(gnssfilter) not in sys.path:
        sys.path.append(str(gnssfilter))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MLP-Kalmannet checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth path.")
    parser.add_argument("--observations", default="Data/mlp_train2000_test2000/mlp_observations.csv", help="Observation CSV.")
    parser.add_argument("--ekf-config", default="GNSSfilter/mlp_ekf/default_config.json", help="EKF config JSON.")
    parser.add_argument("--split", default="all", help="Split to evaluate: train|test|all.")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size.")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--init-mode", default="truth", help="Override EKF init mode.")
    parser.add_argument("--output-dir", default="mlp_results", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    _bootstrap_paths()
    from mlp_train import evaluate_checkpoint

    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    splits: List[str]
    if args.split == "all":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    for split in splits:
        out = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            observations_csv=args.observations,
            ekf_config=args.ekf_config,
            split=split,
            batch_size_eval=args.batch_size,
            device=device,
            init_mode_eval=args.init_mode,
            output_dir=args.output_dir,
        )
        m = out["metrics"]
        print(
            f"[{split}] E={m['east_rmse_m']:.3f} "
            f"N={m['north_rmse_m']:.3f} U={m['up_rmse_m']:.3f} "
            f"2D={m['2d_rmse_m']:.3f} 3D={m['3d_rmse_m']:.3f}"
        )


if __name__ == "__main__":
    main()
