#!/usr/bin/env python3
"""Create fixed epoch-window splits for MLP-Kalmannet observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed train/test epoch windows.")
    parser.add_argument(
        "--input-observations",
        default="Data/mlp_compat/mlp_observations.csv",
        help="Input observation CSV.",
    )
    parser.add_argument(
        "--input-epochs",
        default="Data/mlp_compat/mlp_epochs.csv",
        help="Input epoch summary CSV.",
    )
    parser.add_argument(
        "--input-gt",
        default="Data/mlp_compat/mlp_gt.csv",
        help="Input GT CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/mlp_train2000_test2000",
        help="Output directory.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=2000,
        help="Number of leading epochs assigned to train.",
    )
    parser.add_argument(
        "--test-epochs",
        type=int,
        default=2000,
        help="Number of epochs after train window assigned to test.",
    )
    return parser.parse_args()


def _to_int(value: object, default: int = 0) -> int:
    text = "" if value is None else str(value).strip()
    if text == "":
        return default
    return int(float(text))


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pick_epoch_sets(all_epochs_sorted: Sequence[int], n_train: int, n_test: int) -> Dict[str, Set[int]]:
    train = set(all_epochs_sorted[:n_train])
    test = set(all_epochs_sorted[n_train : n_train + n_test])
    keep = train | test
    return {"train": train, "test": test, "keep": keep}


def _retag_rows_by_epoch(rows: Iterable[Dict[str, str]], train_set: Set[int], test_set: Set[int]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        epoch = _to_int(row.get("epoch"))
        if epoch in train_set:
            split = "train"
        elif epoch in test_set:
            split = "test"
        else:
            continue
        item = dict(row)
        item["split"] = split
        out.append(item)
    return out


def _filter_gt_rows(gt_rows: Iterable[Dict[str, str]], keep_set: Set[int]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in gt_rows:
        epoch = _to_int(row.get("epoch"))
        if epoch in keep_set:
            out.append(dict(row))
    return out


def _epoch_summary_from_obs(obs_rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    train_epochs: Set[int] = set()
    test_epochs: Set[int] = set()
    for row in obs_rows:
        epoch = _to_int(row.get("epoch"))
        split = str(row.get("split", ""))
        if split == "train":
            train_epochs.add(epoch)
        elif split == "test":
            test_epochs.add(epoch)
    return {
        "train_epochs": len(train_epochs),
        "test_epochs": len(test_epochs),
        "observations": len(obs_rows),
    }


def main() -> None:
    args = parse_args()
    obs_in = Path(args.input_observations)
    epochs_in = Path(args.input_epochs)
    gt_in = Path(args.input_gt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_src = _load_csv(obs_in)
    if not obs_src:
        raise RuntimeError(f"Input observations is empty: {obs_in}")

    all_epochs_sorted = sorted({_to_int(r.get("epoch")) for r in obs_src})
    wanted = int(args.train_epochs) + int(args.test_epochs)
    if len(all_epochs_sorted) < wanted:
        raise RuntimeError(
            f"Not enough epochs in source ({len(all_epochs_sorted)}) for requested "
            f"train+test window ({wanted})."
        )

    sets = _pick_epoch_sets(all_epochs_sorted, int(args.train_epochs), int(args.test_epochs))
    train_set = sets["train"]
    test_set = sets["test"]
    keep_set = sets["keep"]

    obs_out_rows = _retag_rows_by_epoch(obs_src, train_set, test_set)
    epochs_out_rows = _retag_rows_by_epoch(_load_csv(epochs_in), train_set, test_set) if epochs_in.exists() else []
    gt_out_rows = _filter_gt_rows(_load_csv(gt_in), keep_set) if gt_in.exists() else []

    obs_out = out_dir / "mlp_observations.csv"
    epochs_out = out_dir / "mlp_epochs.csv"
    gt_out = out_dir / "mlp_gt.csv"
    manifest_out = out_dir / "mlp_manifest.json"

    _write_csv(obs_out, obs_out_rows)
    _write_csv(epochs_out, epochs_out_rows)
    _write_csv(gt_out, gt_out_rows)

    split_counts = _epoch_summary_from_obs(obs_out_rows)
    dropped_epochs = len(all_epochs_sorted) - len(keep_set)

    manifest = {
        "source": {
            "input_observations": str(obs_in),
            "input_epochs": str(epochs_in),
            "input_gt": str(gt_in),
        },
        "window": {
            "train_epochs": int(args.train_epochs),
            "test_epochs": int(args.test_epochs),
            "train_epoch_min": min(train_set),
            "train_epoch_max": max(train_set),
            "test_epoch_min": min(test_set),
            "test_epoch_max": max(test_set),
            "dropped_tail_epochs": dropped_epochs,
        },
        "outputs": {
            "observations_csv": str(obs_out),
            "epochs_csv": str(epochs_out),
            "gt_csv": str(gt_out),
        },
        "stats": split_counts,
        "notes": [
            "Split policy: first train window + next test window, tail dropped.",
            "All rows preserve source features/residuals.",
        ],
    }

    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote: {obs_out}")
    print(f"wrote: {epochs_out}")
    print(f"wrote: {gt_out}")
    print(f"wrote: {manifest_out}")
    print(
        f"epochs_total_in={len(all_epochs_sorted)} train={len(train_set)} "
        f"test={len(test_set)} dropped={dropped_epochs} observations_out={len(obs_out_rows)}"
    )


if __name__ == "__main__":
    main()
