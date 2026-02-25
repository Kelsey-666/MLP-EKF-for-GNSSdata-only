#!/usr/bin/env python3
"""Run MLP-Kalmannet EKF on preprocessed CSV observations.

Supports two modes:
1) Baseline EKF: no checkpoint, pure filter run.
2) EKF + NN: load checkpoint, infer per-observation learned R/bias, then run EKF.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))  # .../GNSSfilter
    sys.path.append(str(Path(__file__).resolve().parents[2]))  # project root
    from mlp_ekf import compute_metrics, load_config, load_epoch_batches, run_filter
else:
    from . import compute_metrics, load_config, load_epoch_batches, run_filter


def _clamp_nn_outputs(
    r_diag,
    bias,
    mask,
    r_min_scale: float,
    r_max_scale: float,
    bias_abs_max_m: float,
):
    import torch

    r = torch.clamp(r_diag, min=float(r_min_scale), max=float(r_max_scale))
    b = torch.clamp(bias, min=-float(bias_abs_max_m), max=float(bias_abs_max_m))
    m = mask.to(r.dtype)
    return r * m, b * m


def _build_learned_by_epoch(
    checkpoint: str,
    observations_csv: str,
    split: str,
    device_text: str,
) -> Dict[int, Tuple[Sequence[float], Sequence[float]]]:
    import sys
    import torch
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parents[2]
    gnssfilter = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(gnssfilter) not in sys.path:
        sys.path.append(str(gnssfilter))

    from mlp_train import load_checkpoint_model
    from mlp_train.dataset import MLPDataset, collate_fn

    device = torch.device("cuda" if device_text == "cuda" and torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint_model(checkpoint, device)
    model = model.to(device).double().eval()
    norm_params = ckpt.get("norm_params")
    train_cfg = ckpt.get("train_config", {})
    r_min = float(train_cfg.get("learned_r_min_scale", 0.5))
    r_max = float(train_cfg.get("learned_r_max_scale", 20.0))
    b_abs = float(train_cfg.get("learned_bias_abs_max_m", 30.0))

    ds_split: Optional[str] = None if split == "all" else split
    ds = MLPDataset(observations_csv, split=ds_split, is_train=False, norm_params=norm_params)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    learned_by_epoch: Dict[int, Tuple[Sequence[float], Sequence[float]]] = {}
    with torch.no_grad():
        for inputs, _, masks, batches in loader:
            inputs = inputs.to(device=device, dtype=torch.double)
            masks = masks.to(device=device)
            r_diag, bias = model(inputs, masks)
            r_diag, bias = _clamp_nn_outputs(
                r_diag,
                bias,
                masks,
                r_min_scale=r_min,
                r_max_scale=r_max,
                bias_abs_max_m=b_abs,
            )
            for i, batch in enumerate(batches):
                valid = masks[i]
                rr = r_diag[i][valid].detach().cpu().tolist()
                bb = bias[i][valid].detach().cpu().tolist()
                learned_by_epoch[int(batch.epoch)] = (rr, bb)

    return learned_by_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLP-Kalmannet EKF")
    parser.add_argument(
        "--observations",
        default="Data/mlp_train2000_test2000/mlp_observations.csv",
        help="Preprocessed observation csv.",
    )
    parser.add_argument(
        "--config",
        default="GNSSfilter/mlp_ekf/default_config.json",
        help="EKF config JSON. Missing file falls back to defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default="GNSSfilter/mlp_ekf/output",
        help="Output directory.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "test"],
        help="Subset split for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path for learned R/bias inference.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for NN inference when checkpoint is provided.",
    )
    parser.add_argument(
        "--init-mode",
        default="",
        help="Optional override for EKF init_mode (truth|coarse).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = load_epoch_batches(args.observations)
    if args.split != "all":
        batches = [b for b in batches if b.split == args.split]
    cfg = load_config(args.config)
    if str(args.init_mode).strip():
        cfg.init_mode = str(args.init_mode).strip()

    learned_by_epoch: Optional[Dict[int, Tuple[Sequence[float], Sequence[float]]]] = None
    mode = "baseline"
    if str(args.checkpoint).strip():
        learned_by_epoch = _build_learned_by_epoch(
            checkpoint=args.checkpoint,
            observations_csv=args.observations,
            split=args.split,
            device_text=args.device,
        )
        mode = "nn"

    rows, debug = run_filter(batches, cfg, learned_by_epoch=learned_by_epoch)
    metrics = compute_metrics(rows)
    metrics["mode"] = mode
    metrics["epochs_input"] = len(batches)
    metrics["epochs_output"] = len(rows)
    metrics["split"] = args.split
    if mode == "nn":
        metrics["checkpoint"] = str(args.checkpoint)
        metrics["learned_epochs"] = len(learned_by_epoch or {})

    suffix = f"_{args.split}_{mode}" if args.split != "all" or mode != "baseline" else ""
    state_path = out_dir / f"mlp_filter_states{suffix}.csv"
    with state_path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    debug_path = out_dir / f"mlp_filter_debug{suffix}.json"
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in debug], f, indent=2)

    metric_path = out_dir / f"mlp_filter_metrics{suffix}.json"
    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"mode={mode}")
    print(f"split={args.split}")
    print(f"epochs_processed={len(rows)}")
    if mode == "nn":
        print(f"learned_epochs={len(learned_by_epoch or {})}")
    print(f"wrote: {state_path}")
    print(f"wrote: {debug_path}")
    print(f"wrote: {metric_path}")


if __name__ == "__main__":
    main()
