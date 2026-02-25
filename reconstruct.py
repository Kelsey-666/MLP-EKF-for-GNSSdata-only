#!/usr/bin/env python3
"""Rebuild EKF trajectory from a trained checkpoint's R/bias outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent
    gnssfilter = root / "GNSSfilter"
    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(gnssfilter) not in sys.path:
        sys.path.append(str(gnssfilter))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct trajectory using checkpoint + adapted EKF")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth path.")
    parser.add_argument("--observations", default="Data/mlp_train2000_test2000/mlp_observations.csv", help="Observation CSV.")
    parser.add_argument("--ekf-config", default="GNSSfilter/mlp_ekf/default_config.json", help="EKF config JSON.")
    parser.add_argument("--split", default="test", help="Split: train|test")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for model forward.")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--init-mode", default="truth", help="Override EKF init mode.")
    parser.add_argument("--output", default="mlp_results/reconstruct_records.csv", help="Output reconstruction CSV.")
    return parser.parse_args()


def main() -> None:
    _bootstrap_paths()
    from mlp_train import reconstruct_from_checkpoint

    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = reconstruct_from_checkpoint(
        checkpoint_path=args.checkpoint,
        observations_csv=args.observations,
        ekf_config=args.ekf_config,
        split=args.split,
        batch_size_eval=args.batch_size,
        device=device,
        init_mode_eval=args.init_mode,
        output_path=args.output,
    )
    m = out["metrics"]
    print("reconstruct_done=1")
    print(f"records={out['records_count']}")
    print(f"records_csv={out['records_path']}")
    print(f"3d_rmse_m={m['3d_rmse_m']}")


if __name__ == "__main__":
    main()
