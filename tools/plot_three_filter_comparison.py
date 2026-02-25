#!/usr/bin/env python3
"""Plot org/baseline/+mlp comparison figures on a configurable epoch range.

Outputs all requested figures into D:\\MLPkalmannet\\mlp-gnss_results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys


def _add_import_paths(root: Path) -> None:
    root_str = str(root)
    gnssfilter_str = str(root / "GNSSfilter")
    if root_str not in sys.path:
        sys.path.append(root_str)
    if gnssfilter_str not in sys.path:
        sys.path.append(gnssfilter_str)


def _load_observations_by_epoch(observations_csv: Path) -> Dict[int, pd.DataFrame]:
    obs = pd.read_csv(observations_csv)
    obs = obs[obs["split"] == "test"].copy()
    obs = obs.sort_values(["epoch", "obs_index_in_epoch"]).reset_index(drop=True)
    return {int(ep): g for ep, g in obs.groupby("epoch", sort=True)}


def _load_debug(debug_json: Path) -> list:
    with debug_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0.0)
    out[mask] = a[mask] / b[mask]
    return out


def _series_from_mlp_filter(
    states_csv: Path,
    debug_json: Path,
    observations_by_epoch: Dict[int, pd.DataFrame],
    learned_r_by_epoch: Dict[int, Tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    states = pd.read_csv(states_csv).sort_values("epoch").reset_index(drop=True)
    debug = _load_debug(debug_json)
    if len(states) != len(debug):
        raise RuntimeError(f"Length mismatch: {states_csv} ({len(states)}) vs {debug_json} ({len(debug)})")

    epochs = states["epoch"].astype(int).to_numpy()
    used = states["used_obs"].to_numpy(dtype=float)
    rejected = states["rejected_obs"].to_numpy(dtype=float)
    total = used + rejected
    util = _safe_div(used, total)

    err_e = states["enu_e_m"].to_numpy(dtype=float)
    err_n = states["enu_n_m"].to_numpy(dtype=float)
    err_u = states["enu_u_m"].to_numpy(dtype=float)
    err_2d = np.sqrt(err_e * err_e + err_n * err_n)
    err_3d = np.sqrt(err_e * err_e + err_n * err_n + err_u * err_u)

    noise_med = np.full(epochs.shape, np.nan, dtype=float)
    norm_innov = np.full(epochs.shape, np.nan, dtype=float)
    for i, ep in enumerate(epochs):
        obs_ep = observations_by_epoch.get(int(ep))
        if obs_ep is None or obs_ep.empty:
            continue
        pr_noise = obs_ep["pr_noise_m"].to_numpy(dtype=float)
        dbg = debug[i]
        innov = np.asarray(dbg.get("pr_innovations", []), dtype=float)
        if innov.size == 0:
            continue
        infl = float(dbg.get("r_inflation", 1.0))

        if learned_r_by_epoch is None:
            scales = np.ones_like(pr_noise, dtype=float)
        else:
            rr = learned_r_by_epoch.get(int(ep))
            if rr is None or len(rr[0]) == 0:
                scales = np.ones_like(pr_noise, dtype=float)
            else:
                scales = np.asarray(rr[0], dtype=float)

        m = min(len(pr_noise), len(innov), len(scales))
        if m <= 0:
            continue
        sigma = pr_noise[:m] * np.sqrt(np.maximum(scales[:m], 1e-6) * max(infl, 1e-6))
        noise_med[i] = _safe_median(sigma)
        norm_innov[i] = _safe_mean(np.abs(innov[:m]) / np.maximum(sigma, 1e-6))

    pred_ecef = states[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    return {
        "epochs": epochs,
        "used": used,
        "rejected": rejected,
        "util": util,
        "noise": noise_med,
        "norm": norm_innov,
        "e": err_e,
        "n": err_n,
        "u": err_u,
        "err2d": err_2d,
        "err3d": err_3d,
        "pred_ecef": pred_ecef,
    }


def _series_from_org_filter(
    root: Path,
    raw_csv: Path,
    gt_csv: Path,
    epoch_start: int,
    epoch_end: int,
) -> tuple[dict, np.ndarray]:
    from org_ekf.io_csv import load_gnss_from_csv
    from org_ekf.core import pedestrian_kalman_filter
    from org_ekf.utils import ecef2ned_error

    gnss = load_gnss_from_csv(str(raw_csv), str(gt_csv), epoch_start=epoch_start, epoch_end=epoch_end)
    out = pedestrian_kalman_filter(gnss)

    epochs = np.arange(int(epoch_start), int(epoch_start) + len(out["num_obs"]), dtype=int)
    num_obs = np.asarray(out["num_obs"], dtype=float)
    pr_res = np.asarray(out["prRes"], dtype=float)
    pr_innov = np.asarray(out["prInnov"], dtype=float)
    used = np.sum(np.isfinite(pr_res), axis=1).astype(float)
    rejected = np.maximum(num_obs - used, 0.0)
    util = _safe_div(used, used + rejected)

    obs_id_xt = np.asarray(out["obsId"].xtId, dtype=np.int64)
    xt_to_col = {int(v): i for i, v in enumerate(obs_id_xt)}
    sigma_mat = np.full(pr_innov.shape, np.nan, dtype=float)
    for i in range(len(gnss.epoch)):
        row = int(gnss.epoch[i]) - 1
        col = xt_to_col.get(int(gnss.xtId[i]))
        if col is None or row < 0 or row >= sigma_mat.shape[0]:
            continue
        sigma_mat[row, col] = float(gnss.pr_noise[i])

    noise = np.full((len(epochs),), np.nan, dtype=float)
    norm = np.full((len(epochs),), np.nan, dtype=float)
    for i in range(len(epochs)):
        used_mask = np.isfinite(pr_res[i, :]) & np.isfinite(sigma_mat[i, :]) & (sigma_mat[i, :] > 0.0)
        innov_mask = np.isfinite(pr_innov[i, :]) & np.isfinite(sigma_mat[i, :]) & (sigma_mat[i, :] > 0.0)
        if np.any(used_mask):
            noise[i] = _safe_median(sigma_mat[i, used_mask])
        if np.any(innov_mask):
            ratio = np.abs(pr_innov[i, innov_mask]) / np.maximum(sigma_mat[i, innov_mask], 1e-6)
            norm[i] = _safe_mean(ratio)

    pred_ecef = np.asarray(out["x"], dtype=float)[:, :3]
    truth_ecef = np.asarray(gnss.pos_E, dtype=float)[: pred_ecef.shape[0], :]
    dpos_ned, _, _, _ = ecef2ned_error(pred_ecef, truth_ecef)
    n = dpos_ned[:, 0]
    e = dpos_ned[:, 1]
    u = -dpos_ned[:, 2]
    err2d = np.sqrt(e * e + n * n)
    err3d = np.sqrt(e * e + n * n + u * u)

    series = {
        "epochs": epochs,
        "used": used,
        "rejected": rejected,
        "util": util,
        "noise": noise,
        "norm": norm,
        "e": e,
        "n": n,
        "u": u,
        "err2d": err2d,
        "err3d": err3d,
        "pred_ecef": pred_ecef,
    }
    return series, truth_ecef


def _to_enu_track(ecef_xyz: np.ndarray, ref_lat: float, ref_lon: float, ref_h: float) -> np.ndarray:
    from mlp_ekf.geo import ecef_to_enu

    out = np.zeros((ecef_xyz.shape[0], 3), dtype=float)
    for i, (x, y, z) in enumerate(ecef_xyz):
        e, n, u = ecef_to_enu(float(x), float(y), float(z), ref_lat, ref_lon, ref_h)
        out[i, 0] = e
        out[i, 1] = n
        out[i, 2] = u
    return out


def _plot_overlay(
    out_path: Path,
    x: np.ndarray,
    y_org: np.ndarray,
    y_baseline: np.ndarray,
    y_mlp: np.ndarray,
    y_label: str,
    title: str,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(x, y_org, color="#d62728", linewidth=1.4, label="org-ekf")
    ax.plot(x, y_baseline, color="#1f77b4", linewidth=1.4, label="baseline-ekf")
    ax.plot(x, y_mlp, color="#2ca02c", linewidth=1.4, label="+mlp-ekf")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_utilization(
    out_path: Path,
    x: np.ndarray,
    org: dict,
    baseline: dict,
    mlp: dict,
    x_label: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, series, title, color in [
        (axes[0], org, "org-ekf", "#d62728"),
        (axes[1], baseline, "baseline-ekf", "#1f77b4"),
        (axes[2], mlp, "+mlp-ekf", "#2ca02c"),
    ]:
        l1 = ax.plot(x, series["used"], color=color, linewidth=1.0, label="used obs")
        l2 = ax.plot(x, series["rejected"], color="#ff7f0e", linewidth=1.0, label="rejected obs")
        ax.set_ylabel("obs count")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax2 = ax.twinx()
        l3 = ax2.plot(x, series["util"], color="#000000", linestyle="--", linewidth=1.0, label="utilization")
        ax2.set_ylabel("utilization")
        lines = l1 + l2 + l3
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc="lower right")
    axes[-1].set_xlabel(x_label)
    fig.suptitle("Observation Utilization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _empirical_cdf(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    vv = np.sort(vv)
    if vv.size == 0:
        return vv, vv
    p = np.arange(1, vv.size + 1, dtype=float) / float(vv.size)
    return vv, p


def _plot_cdf(out_path: Path, org: dict, baseline: dict, mlp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(axes[0], "err2d", "2D Error CDF"), (axes[1], "err3d", "3D Error CDF")]:
        for lab, color, series in [
            ("org-ekf", "#d62728", org),
            ("baseline-ekf", "#1f77b4", baseline),
            ("+mlp-ekf", "#2ca02c", mlp),
        ]:
            x, y = _empirical_cdf(series[key])
            ax.plot(x, y, label=lab, color=color, linewidth=1.5)
        ax.set_xlabel("error [m]")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_box(out_path: Path, org: dict, baseline: dict, mlp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["org-ekf", "baseline-ekf", "+mlp-ekf"]
    data_2d = [np.asarray(org["err2d"]), np.asarray(baseline["err2d"]), np.asarray(mlp["err2d"])]
    data_3d = [np.asarray(org["err3d"]), np.asarray(baseline["err3d"]), np.asarray(mlp["err3d"])]
    axes[0].boxplot(data_2d, tick_labels=labels, showfliers=False)
    axes[0].set_title("2D Error Boxplot")
    axes[0].set_ylabel("error [m]")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].boxplot(data_3d, tick_labels=labels, showfliers=False)
    axes[1].set_title("3D Error Boxplot")
    axes[1].set_ylabel("error [m]")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_trajectory_2d(
    out_path: Path,
    truth_en: np.ndarray,
    org_en: np.ndarray,
    baseline_en: np.ndarray,
    mlp_en: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(truth_en[:, 0], truth_en[:, 1], color="black", linewidth=2.0, label="truth")
    ax.plot(org_en[:, 0], org_en[:, 1], color="#d62728", linewidth=1.2, label="org-ekf")
    ax.plot(baseline_en[:, 0], baseline_en[:, 1], color="#1f77b4", linewidth=1.2, label="baseline-ekf")
    ax.plot(mlp_en[:, 0], mlp_en[:, 1], color="#2ca02c", linewidth=1.2, label="+mlp-ekf")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _rmse(v: np.ndarray) -> float:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vv * vv)))


def _save_summary(out_path: Path, org: dict, baseline: dict, mlp: dict) -> None:
    summary = {
        "org": {
            "2d_rmse_m": _rmse(org["err2d"]),
            "3d_rmse_m": _rmse(org["err3d"]),
        },
        "baseline": {
            "2d_rmse_m": _rmse(baseline["err2d"]),
            "3d_rmse_m": _rmse(baseline["err3d"]),
        },
        "mlp": {
            "2d_rmse_m": _rmse(mlp["err2d"]),
            "3d_rmse_m": _rmse(mlp["err3d"]),
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot org/baseline/+mlp comparison figures.")
    parser.add_argument("--root", default=r"D:\MLPkalmannet")
    parser.add_argument("--raw-csv", default=r"D:\MLPkalmannet\Data\raw_observation.csv")
    parser.add_argument("--gt-csv-org", default=r"D:\MLPkalmannet\Data\gt_processed.csv")
    parser.add_argument("--obs-csv", default=r"D:\MLPkalmannet\Data\mlp_train2000_test2000\mlp_observations.csv")
    parser.add_argument("--gt-csv-mlp", default=r"D:\MLPkalmannet\Data\mlp_train2000_test2000\mlp_gt.csv")
    parser.add_argument(
        "--baseline-states",
        default=r"D:\MLPkalmannet\GNSSfilter\mlp_ekf\output\baseline_recheck_seed44_test2000\mlp_filter_states_test_baseline.csv",
    )
    parser.add_argument(
        "--baseline-debug",
        default=r"D:\MLPkalmannet\GNSSfilter\mlp_ekf\output\baseline_recheck_seed44_test2000\mlp_filter_debug_test_baseline.json",
    )
    parser.add_argument(
        "--mlp-states",
        default=r"D:\MLPkalmannet\GNSSfilter\mlp_ekf\output\mlp_recheck_seed44_test2000\mlp_filter_states_test_nn.csv",
    )
    parser.add_argument(
        "--mlp-debug",
        default=r"D:\MLPkalmannet\GNSSfilter\mlp_ekf\output\mlp_recheck_seed44_test2000\mlp_filter_debug_test_nn.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"D:\MLPkalmannet\trained_model\mlp_seed44_2000train_2000test_40ep\mlp_seed44_2000train_2000test_40ep_s44_20260225_055926_best.pth",
    )
    parser.add_argument("--out-dir", default=r"D:\MLPkalmannet\mlp-gnss_results")
    parser.add_argument("--epoch-start", type=int, default=2001)
    parser.add_argument("--epoch-end", type=int, default=4000)
    return parser.parse_args()


def _slice_series(series: dict, epochs_keep: np.ndarray) -> dict:
    idx_map = {int(ep): i for i, ep in enumerate(series["epochs"].tolist())}
    idx = [idx_map[int(ep)] for ep in epochs_keep.tolist() if int(ep) in idx_map]
    out = {}
    for k, v in series.items():
        if isinstance(v, np.ndarray) and len(v) == len(series["epochs"]):
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _add_import_paths(root)

    from mlp_ekf.run_mlp_filter import _build_learned_by_epoch

    observations_by_epoch = _load_observations_by_epoch(Path(args.obs_csv))

    org, org_truth_ecef = _series_from_org_filter(
        root,
        Path(args.raw_csv),
        Path(args.gt_csv_org),
        epoch_start=int(args.epoch_start),
        epoch_end=int(args.epoch_end),
    )
    baseline = _series_from_mlp_filter(
        Path(args.baseline_states),
        Path(args.baseline_debug),
        observations_by_epoch=observations_by_epoch,
        learned_r_by_epoch=None,
    )

    learned_r_by_epoch = _build_learned_by_epoch(
        checkpoint=str(Path(args.checkpoint)),
        observations_csv=str(Path(args.obs_csv)),
        split="test",
        device_text="cpu",
    )
    mlp = _series_from_mlp_filter(
        Path(args.mlp_states),
        Path(args.mlp_debug),
        observations_by_epoch=observations_by_epoch,
        learned_r_by_epoch=learned_r_by_epoch,
    )

    epochs = np.asarray(org["epochs"], dtype=int)
    epochs = np.intersect1d(epochs, np.asarray(baseline["epochs"], dtype=int))
    epochs = np.intersect1d(epochs, np.asarray(mlp["epochs"], dtype=int))
    if epochs.size == 0:
        raise RuntimeError("No common epochs among org/baseline/+mlp series.")
    org = _slice_series(org, epochs)
    baseline = _slice_series(baseline, epochs)
    mlp = _slice_series(mlp, epochs)
    epoch_label = f"Epoch ({int(epochs[0])}-{int(epochs[-1])})"

    _plot_overlay(
        out_dir / "noise_epoch_org_baseline_mlp.png",
        epochs,
        org["noise"],
        baseline["noise"],
        mlp["noise"],
        y_label="median sigma [m]",
        title="Per-epoch Robust Noise (median sigma, org / baseline / +mlp)",
        x_label=epoch_label,
    )
    _plot_overlay(
        out_dir / "err2d_epoch_org_baseline_mlp.png",
        epochs,
        org["err2d"],
        baseline["err2d"],
        mlp["err2d"],
        y_label="2D error [m]",
        title="Per-epoch 2D Error (org / baseline / +mlp)",
        x_label=epoch_label,
    )
    _plot_overlay(
        out_dir / "err3d_epoch_org_baseline_mlp.png",
        epochs,
        org["err3d"],
        baseline["err3d"],
        mlp["err3d"],
        y_label="3D error [m]",
        title="Per-epoch 3D Error (org / baseline / +mlp)",
        x_label=epoch_label,
    )
    _plot_utilization(
        out_dir / "utilization_org_baseline_mlp.png",
        epochs,
        org=org,
        baseline=baseline,
        mlp=mlp,
        x_label=epoch_label,
    )
    _plot_overlay(
        out_dir / "norm_innov_org_baseline_mlp.png",
        epochs,
        org["norm"],
        baseline["norm"],
        mlp["norm"],
        y_label="|v|/sigma",
        title="Per-epoch Normalized Innovation |v|/sigma (org / baseline / +mlp)",
        x_label=epoch_label,
    )
    _plot_cdf(out_dir / "cdf_org_baseline_mlp_2d3d.png", org, baseline, mlp)
    _plot_box(out_dir / "boxplot_org_baseline_mlp_2d3d.png", org, baseline, mlp)

    gt = pd.read_csv(Path(args.gt_csv_mlp))
    gt_test = gt[(gt["epoch"] >= int(epochs[0])) & (gt["epoch"] <= int(epochs[-1]))].sort_values("epoch").reset_index(drop=True)
    if len(gt_test) != len(epochs):
        raise RuntimeError(f"Expected {len(epochs)} GT epochs, got {len(gt_test)}")
    ref_lat = float(gt_test.loc[0, "gt_lat_deg"])
    ref_lon = float(gt_test.loc[0, "gt_lon_deg"])
    ref_h = float(gt_test.loc[0, "gt_h_m"])
    truth_ecef = gt_test[["gt_ecef_x_m", "gt_ecef_y_m", "gt_ecef_z_m"]].to_numpy(dtype=float)

    # Use GT from mlp split for all trajectory plotting.
    truth_en = _to_enu_track(truth_ecef, ref_lat, ref_lon, ref_h)
    org_en = _to_enu_track(org["pred_ecef"], ref_lat, ref_lon, ref_h)
    baseline_en = _to_enu_track(baseline["pred_ecef"], ref_lat, ref_lon, ref_h)
    mlp_en = _to_enu_track(mlp["pred_ecef"], ref_lat, ref_lon, ref_h)

    _plot_trajectory_2d(
        out_dir / "trajectory2d_org_baseline_mlp_vs_gt.png",
        truth_en=truth_en,
        org_en=org_en,
        baseline_en=baseline_en,
        mlp_en=mlp_en,
        title=f"2D Trajectory Comparison (Test {epoch_label})",
    )

    _save_summary(out_dir / "compare_org_baseline_mlp_summary.json", org, baseline, mlp)

    print("figure_generation_done=1")
    print(f"out_dir={out_dir}")
    for name in [
        "noise_epoch_org_baseline_mlp.png",
        "err2d_epoch_org_baseline_mlp.png",
        "err3d_epoch_org_baseline_mlp.png",
        "utilization_org_baseline_mlp.png",
        "norm_innov_org_baseline_mlp.png",
        "cdf_org_baseline_mlp_2d3d.png",
        "boxplot_org_baseline_mlp_2d3d.png",
        "trajectory2d_org_baseline_mlp_vs_gt.png",
        "compare_org_baseline_mlp_summary.json",
    ]:
        print(f"wrote={out_dir / name}")


if __name__ == "__main__":
    main()
