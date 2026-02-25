#!/usr/bin/env python3
"""Export trained MLP-Kalmannet model (TorchScript + metadata)."""

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
    parser = argparse.ArgumentParser(description="Export MLP-Kalmannet checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth path.")
    parser.add_argument("--output", default="trained_model/mlp/exported_model.ts", help="Output TorchScript path.")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    return parser.parse_args()


def main() -> None:
    _bootstrap_paths()
    from mlp_train import export_torchscript

    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    meta = export_torchscript(args.checkpoint, args.output, device)
    print("export_done=1")
    print(f"torchscript={meta['torchscript']}")
    print(f"meta_json={Path(meta['torchscript']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
