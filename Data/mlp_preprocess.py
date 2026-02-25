#!/usr/bin/env python3
"""Build MLP-Kalmannet compatible intermediate data from raw_observation.csv + gt_processed.csv.

This script provides the first-stage bridge for LF-style training/inference pipelines:
1) Convert flat per-observation CSV to epoch-grouped canonical records.
2) Build PR residual features from coarse LS state (no GT leakage).
3) Emit split-ready observations/epochs/gt tables in one consistent schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
CLIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5


GNSS_ID_MAP = {
    0: "GPS",
    2: "GAL",
    3: "BDS",
    6: "GLO",
}


RAW_ALIASES: Dict[str, Sequence[str]] = {
    "epoch": ("epoch",),
    "week": ("week",),
    "tow": ("tow",),
    "time_gps_s": ("time_gps_s",),
    "gnss_id": ("gnss_id",),
    "sv_id": ("sv_id",),
    "sig_id": ("sig_id",),
    "pr_m": ("pr_m", "pr"),
    "dr_mps": ("dr_mps", "dr"),
    "pr_noise_m": ("pr_noise_m", "pr_noise"),
    "dr_noise_mps": ("dr_noise_mps", "dr_noise"),
    "sat_pos_x_m": ("sat_pos_E_x_m", "sat_pos_E_1"),
    "sat_pos_y_m": ("sat_pos_E_y_m", "sat_pos_E_2"),
    "sat_pos_z_m": ("sat_pos_E_z_m", "sat_pos_E_3"),
    "sat_vel_x_mps": ("sat_vel_E_x_mps", "sat_vel_E_1"),
    "sat_vel_y_mps": ("sat_vel_E_y_mps", "sat_vel_E_2"),
    "sat_vel_z_mps": ("sat_vel_E_z_mps", "sat_vel_E_3"),
    "sat_bias_m": ("sat_bias_m", "sat_bias"),
    "sat_drift_mps": ("sat_drift_mps", "sat_drift"),
    "elev_rad": ("elev_rad", "elev"),
    "azim_rad": ("azim_rad", "azim"),
    "elev_deg": ("elev_deg",),
    "azim_deg": ("azim_deg",),
    "cno_dbhz": ("cno", "cno_dbhz"),
}


GT_ALIASES: Dict[str, Sequence[str]] = {
    "epoch": ("epoch",),
    "week": ("week",),
    "tow": ("tow",),
    "time_gps_s": ("time_gps_s",),
    "ecef_x": ("gt_pos_E_x_m", "pos_E_X"),
    "ecef_y": ("gt_pos_E_y_m", "pos_E_Y"),
    "ecef_z": ("gt_pos_E_z_m", "pos_E_Z"),
    "vel_x": ("gt_vel_E_x_mps", "vel_E_X"),
    "vel_y": ("gt_vel_E_y_mps", "vel_E_Y"),
    "vel_z": ("gt_vel_E_z_mps", "vel_E_Z"),
    "lat_deg": ("gt_lat_deg", "lat_lon_alt_X"),
    "lon_deg": ("gt_lon_deg", "lat_lon_alt_Y"),
    "h_m": ("gt_h_m", "altitude", "lat_lon_alt_Z"),
    "speed_mps": ("speed_mps",),
    "is_static": ("is_static",),
}


@dataclass
class CoarseState:
    x: float
    y: float
    z: float
    cb: float
    ok: bool
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP-Kalmannet data preprocessor")
    parser.add_argument(
        "--raw",
        default="Data/raw_observation.csv",
        help="Input raw observation CSV.",
    )
    parser.add_argument(
        "--gt",
        default="Data/gt_processed.csv",
        help="Input GT CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/mlp_train2000_test2000",
        help="Output directory for bridge files.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["window", "ratio"],
        default="window",
        help="Split policy: fixed epoch window or continuous ratio.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=2000,
        help="Leading epochs assigned to train when split-mode=window.",
    )
    parser.add_argument(
        "--test-epochs",
        type=int,
        default=2000,
        help="Epochs after train window assigned to test when split-mode=window.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Continuous train ratio by epoch order.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Continuous val ratio by epoch order.",
    )
    parser.add_argument(
        "--cno-min",
        type=float,
        default=24.0,
        help="C/N0 minimum (dB-Hz) used for coarse LS and gate stats.",
    )
    parser.add_argument(
        "--elev-min-deg",
        type=float,
        default=12.0,
        help="Elevation minimum (deg) used for coarse LS and gate stats.",
    )
    parser.add_argument(
        "--static-speed-threshold",
        type=float,
        default=0.1,
        help="Speed threshold (m/s) to relabel static flag if GT has no is_static.",
    )
    return parser.parse_args()


def _pick_value(row: Dict[str, str], names: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in row:
            value = row.get(name)
            if value is not None and str(value).strip() != "":
                return str(value)
    return default


def get_raw(row: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    return _pick_value(row, RAW_ALIASES[key], default=default)


def get_gt(row: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    return _pick_value(row, GT_ALIASES[key], default=default)


def to_float(value: object, default: float = float("nan")) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        return default
    return float(text)


def to_int(value: object, default: int = 0) -> int:
    text = "" if value is None else str(value).strip()
    if text == "":
        return default
    return int(float(text))


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n
        lat_next = math.atan2(z, p * (1.0 - WGS84_E2 * (n / max(n + h, 1.0))))
        if abs(lat_next - lat) < 1e-13:
            lat = lat_next
            break
        lat = lat_next
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / max(math.cos(lat), 1e-12) - n
    return math.degrees(lat), math.degrees(lon), h


def solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def geometric_range(
    sat_pos: Tuple[float, float, float], rx_pos: Tuple[float, float, float]
) -> Tuple[float, Tuple[float, float, float]]:
    dx = sat_pos[0] - rx_pos[0]
    dy = sat_pos[1] - rx_pos[1]
    dz = sat_pos[2] - rx_pos[2]
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r < 1e-9:
        return 0.0, (0.0, 0.0, 0.0)
    sagnac = OMEGA_E * (sat_pos[0] * rx_pos[1] - sat_pos[1] * rx_pos[0]) / CLIGHT
    los = (dx / r, dy / r, dz / r)
    return r + sagnac, los


def least_squares_coarse_state(
    rows: List[Dict[str, str]],
    init_state: Optional[CoarseState],
    cno_min: float,
    elev_min_deg: float,
    max_iter: int = 8,
) -> CoarseState:
    if init_state is None:
        x, y, z, cb = 0.0, 0.0, 0.0, 0.0
    else:
        x, y, z, cb = init_state.x, init_state.y, init_state.z, init_state.cb

    good_rows: List[Dict[str, str]] = []
    elev_min_rad = math.radians(elev_min_deg)
    for row in rows:
        pr = to_float(get_raw(row, "pr_m"), float("nan"))
        cno = to_float(get_raw(row, "cno_dbhz"), float("nan"))
        elev = to_float(get_raw(row, "elev_rad"), float("nan"))
        sat_x = to_float(get_raw(row, "sat_pos_x_m"), float("nan"))
        sat_y = to_float(get_raw(row, "sat_pos_y_m"), float("nan"))
        sat_z = to_float(get_raw(row, "sat_pos_z_m"), float("nan"))
        if not math.isfinite(pr) or pr <= 0:
            continue
        if not math.isfinite(cno) or cno < cno_min:
            continue
        if not math.isfinite(elev) or elev < elev_min_rad:
            continue
        if not math.isfinite(sat_x + sat_y + sat_z):
            continue
        good_rows.append(row)

    if len(good_rows) < 4:
        return CoarseState(x=x, y=y, z=z, cb=cb, ok=False, iterations=0)

    for it in range(1, max_iter + 1):
        h_rows: List[List[float]] = []
        v_rows: List[float] = []
        for row in good_rows:
            sat_pos = (
                to_float(get_raw(row, "sat_pos_x_m")),
                to_float(get_raw(row, "sat_pos_y_m")),
                to_float(get_raw(row, "sat_pos_z_m")),
            )
            pr_meas = to_float(get_raw(row, "pr_m"))
            sat_bias = to_float(get_raw(row, "sat_bias_m"), 0.0)
            rho, los = geometric_range(sat_pos, (x, y, z))
            pred = rho + cb - sat_bias
            res = pr_meas - pred
            h_rows.append([-los[0], -los[1], -los[2], 1.0])
            v_rows.append(res)

        nmat = [[0.0] * 4 for _ in range(4)]
        uvec = [0.0] * 4
        for h, v in zip(h_rows, v_rows):
            for r in range(4):
                uvec[r] += h[r] * v
                for c in range(4):
                    nmat[r][c] += h[r] * h[c]

        dx = solve_linear_system(nmat, uvec)
        if dx is None:
            return CoarseState(x=x, y=y, z=z, cb=cb, ok=False, iterations=it)

        x += dx[0]
        y += dx[1]
        z += dx[2]
        cb += dx[3]
        if math.sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) < 1e-4:
            return CoarseState(x=x, y=y, z=z, cb=cb, ok=True, iterations=it)

    return CoarseState(x=x, y=y, z=z, cb=cb, ok=True, iterations=max_iter)


def assign_split(epoch_idx: int, n_epochs: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(n_epochs * train_ratio)
    val_end = int(n_epochs * (train_ratio + val_ratio))
    if epoch_idx < train_end:
        return "train"
    if epoch_idx < val_end:
        return "val"
    return "test"


def assign_split_window(epoch_idx: int, train_epochs: int, test_epochs: int) -> Optional[str]:
    if epoch_idx < train_epochs:
        return "train"
    if epoch_idx < (train_epochs + test_epochs):
        return "test"
    return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_gt_lookup(
    gt_rows: List[Dict[str, str]],
    raw_rows: List[Dict[str, str]],
    static_speed_threshold: float,
) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, float]]]:
    epoch_time: Dict[int, Tuple[int, float, float]] = {}
    for row in raw_rows:
        ep = to_int(get_raw(row, "epoch"))
        if ep in epoch_time:
            continue
        week = to_int(get_raw(row, "week"))
        tow = to_float(get_raw(row, "tow"), float("nan"))
        tgps = to_float(get_raw(row, "time_gps_s"), float("nan"))
        if not math.isfinite(tgps):
            tgps = float(week) * 604800.0 + tow if math.isfinite(tow) else float("nan")
        epoch_time[ep] = (week, tow, tgps)

    gt_lookup: Dict[int, Dict[str, float]] = {}
    gt_table: List[Dict[str, float]] = []

    for row in gt_rows:
        epoch = to_int(get_gt(row, "epoch"))
        if epoch <= 0:
            continue

        week = to_int(get_gt(row, "week"), epoch_time.get(epoch, (0, float("nan"), float("nan"))[0]))
        tow = to_float(get_gt(row, "tow"), epoch_time.get(epoch, (0, float("nan"), float("nan"))[1]))
        time_gps_s = to_float(get_gt(row, "time_gps_s"), epoch_time.get(epoch, (0, float("nan"), float("nan"))[2]))
        if not math.isfinite(time_gps_s):
            time_gps_s = float(week) * 604800.0 + tow if math.isfinite(tow) else float("nan")

        ecef_x = to_float(get_gt(row, "ecef_x"), float("nan"))
        ecef_y = to_float(get_gt(row, "ecef_y"), float("nan"))
        ecef_z = to_float(get_gt(row, "ecef_z"), float("nan"))

        lat_deg = to_float(get_gt(row, "lat_deg"), float("nan"))
        lon_deg = to_float(get_gt(row, "lon_deg"), float("nan"))
        h_m = to_float(get_gt(row, "h_m"), float("nan"))

        if not (math.isfinite(lat_deg) and math.isfinite(lon_deg) and math.isfinite(h_m)):
            if math.isfinite(ecef_x) and math.isfinite(ecef_y) and math.isfinite(ecef_z):
                lat_deg, lon_deg, h_m = ecef_to_geodetic(ecef_x, ecef_y, ecef_z)

        vel_x = to_float(get_gt(row, "vel_x"), 0.0)
        vel_y = to_float(get_gt(row, "vel_y"), 0.0)
        vel_z = to_float(get_gt(row, "vel_z"), 0.0)
        speed_mps = to_float(get_gt(row, "speed_mps"), float("nan"))
        if not math.isfinite(speed_mps):
            speed_mps = math.sqrt(vel_x * vel_x + vel_y * vel_y + vel_z * vel_z)

        raw_static = get_gt(row, "is_static")
        if raw_static is not None and str(raw_static).strip() != "":
            is_static = to_int(raw_static)
        else:
            is_static = 1 if speed_mps < static_speed_threshold else 0

        item = {
            "epoch": epoch,
            "week": week,
            "tow": tow,
            "time_gps_s": time_gps_s,
            "gt_ecef_x_m": ecef_x,
            "gt_ecef_y_m": ecef_y,
            "gt_ecef_z_m": ecef_z,
            "gt_lat_deg": lat_deg,
            "gt_lon_deg": lon_deg,
            "gt_h_m": h_m,
            "gt_vel_x_mps": vel_x,
            "gt_vel_y_mps": vel_y,
            "gt_vel_z_mps": vel_z,
            "speed_mps": speed_mps,
            "is_static": is_static,
        }
        gt_lookup[epoch] = item
        gt_table.append(item)

    gt_table.sort(key=lambda r: int(r["epoch"]))
    return gt_lookup, gt_table


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    gt_path = Path(args.gt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_csv_rows(raw_path)
    gt_rows = load_csv_rows(gt_path)
    gt_lookup, gt_table = build_gt_lookup(
        gt_rows=gt_rows,
        raw_rows=raw_rows,
        static_speed_threshold=float(args.static_speed_threshold),
    )

    by_epoch: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        ep = to_int(get_raw(row, "epoch"))
        if ep > 0:
            by_epoch[ep].append(row)

    epochs = sorted(by_epoch.keys())
    epoch_summary_rows: List[Dict[str, object]] = []
    obs_rows: List[Dict[str, object]] = []

    prev_state: Optional[CoarseState] = None
    coarse_ok_count = 0

    for idx, epoch in enumerate(epochs):
        rows = by_epoch[epoch]
        if args.split_mode == "window":
            split = assign_split_window(idx, int(args.train_epochs), int(args.test_epochs))
            if split is None:
                continue
        else:
            split = assign_split(idx, len(epochs), args.train_ratio, args.val_ratio)

        week = to_int(get_raw(rows[0], "week"))
        tow = to_float(get_raw(rows[0], "tow"), float("nan"))
        time_gps_s = to_float(get_raw(rows[0], "time_gps_s"), float("nan"))
        if not math.isfinite(time_gps_s):
            time_gps_s = float(week) * 604800.0 + tow if math.isfinite(tow) else float("nan")

        coarse = least_squares_coarse_state(
            rows=rows,
            init_state=prev_state,
            cno_min=float(args.cno_min),
            elev_min_deg=float(args.elev_min_deg),
        )
        if coarse.ok:
            prev_state = coarse
            coarse_ok_count += 1

        gt = gt_lookup.get(epoch, {})

        used_for_ls = 0
        elev_min_rad = math.radians(float(args.elev_min_deg))
        for r in rows:
            pr = to_float(get_raw(r, "pr_m"), float("nan"))
            cno = to_float(get_raw(r, "cno_dbhz"), float("nan"))
            elev = to_float(get_raw(r, "elev_rad"), float("nan"))
            if pr > 0.0 and cno >= float(args.cno_min) and elev >= elev_min_rad:
                used_for_ls += 1

        epoch_summary_rows.append(
            {
                "epoch": epoch,
                "split": split,
                "week": week,
                "tow": tow,
                "time_gps_s": time_gps_s,
                "obs_count": len(rows),
                "obs_count_ls_gate": used_for_ls,
                "coarse_ok": int(coarse.ok),
                "coarse_iter": coarse.iterations,
                "coarse_rx_x_m": coarse.x,
                "coarse_rx_y_m": coarse.y,
                "coarse_rx_z_m": coarse.z,
                "coarse_cb_m": coarse.cb,
                "gt_lat_deg": gt.get("gt_lat_deg", float("nan")),
                "gt_lon_deg": gt.get("gt_lon_deg", float("nan")),
                "gt_h_m": gt.get("gt_h_m", float("nan")),
                "speed_mps": gt.get("speed_mps", float("nan")),
                "is_static": gt.get("is_static", -1),
            }
        )

        for ridx, row in enumerate(rows):
            sat_pos = (
                to_float(get_raw(row, "sat_pos_x_m"), float("nan")),
                to_float(get_raw(row, "sat_pos_y_m"), float("nan")),
                to_float(get_raw(row, "sat_pos_z_m"), float("nan")),
            )
            sat_vel = (
                to_float(get_raw(row, "sat_vel_x_mps"), float("nan")),
                to_float(get_raw(row, "sat_vel_y_mps"), float("nan")),
                to_float(get_raw(row, "sat_vel_z_mps"), float("nan")),
            )
            pr = to_float(get_raw(row, "pr_m"), float("nan"))
            dr = to_float(get_raw(row, "dr_mps"), float("nan"))
            sat_bias = to_float(get_raw(row, "sat_bias_m"), 0.0)
            sat_drift = to_float(get_raw(row, "sat_drift_mps"), 0.0)

            elev_rad = to_float(get_raw(row, "elev_rad"), float("nan"))
            azim_rad = to_float(get_raw(row, "azim_rad"), float("nan"))
            elev_deg = to_float(get_raw(row, "elev_deg"), float("nan"))
            azim_deg = to_float(get_raw(row, "azim_deg"), float("nan"))
            if not math.isfinite(elev_deg) and math.isfinite(elev_rad):
                elev_deg = math.degrees(elev_rad)
            if not math.isfinite(azim_deg) and math.isfinite(azim_rad):
                azim_deg = math.degrees(azim_rad)

            rho, _ = geometric_range(sat_pos, (coarse.x, coarse.y, coarse.z))
            pred_pr = rho + coarse.cb - sat_bias
            residual = pr - pred_pr if math.isfinite(pr) else float("nan")

            gnss_id = to_int(get_raw(row, "gnss_id"))
            obs_rows.append(
                {
                    "epoch": epoch,
                    "obs_index_in_epoch": ridx,
                    "split": split,
                    "week": week,
                    "tow": tow,
                    "time_gps_s": time_gps_s,
                    "gnss_id_raw": gnss_id,
                    "gnss_system": GNSS_ID_MAP.get(gnss_id, "UNKNOWN"),
                    "sv_id": to_int(get_raw(row, "sv_id")),
                    "sig_id": to_int(get_raw(row, "sig_id")),
                    "pr_m": pr,
                    "dr_mps": dr,
                    "pr_noise_m": to_float(get_raw(row, "pr_noise_m"), float("nan")),
                    "dr_noise_mps": to_float(get_raw(row, "dr_noise_mps"), float("nan")),
                    "sat_pos_x_m": sat_pos[0],
                    "sat_pos_y_m": sat_pos[1],
                    "sat_pos_z_m": sat_pos[2],
                    "sat_vel_x_mps": sat_vel[0],
                    "sat_vel_y_mps": sat_vel[1],
                    "sat_vel_z_mps": sat_vel[2],
                    "sat_bias_m": sat_bias,
                    "sat_drift_mps": sat_drift,
                    "elev_rad": elev_rad,
                    "azim_rad": azim_rad,
                    "elev_deg": elev_deg,
                    "azim_deg": azim_deg,
                    "cno_dbhz": to_float(get_raw(row, "cno_dbhz"), float("nan")),
                    "coarse_ok": int(coarse.ok),
                    "coarse_rx_x_m": coarse.x,
                    "coarse_rx_y_m": coarse.y,
                    "coarse_rx_z_m": coarse.z,
                    "coarse_cb_m": coarse.cb,
                    "pr_pred_m": pred_pr,
                    "residual_m": residual,
                    "gt_ecef_x_m": gt.get("gt_ecef_x_m", float("nan")),
                    "gt_ecef_y_m": gt.get("gt_ecef_y_m", float("nan")),
                    "gt_ecef_z_m": gt.get("gt_ecef_z_m", float("nan")),
                    "gt_lat_deg": gt.get("gt_lat_deg", float("nan")),
                    "gt_lon_deg": gt.get("gt_lon_deg", float("nan")),
                    "gt_h_m": gt.get("gt_h_m", float("nan")),
                    "speed_mps": gt.get("speed_mps", float("nan")),
                    "is_static": gt.get("is_static", -1),
                }
            )

    obs_out = out_dir / "mlp_observations.csv"
    epoch_out = out_dir / "mlp_epochs.csv"
    gt_out = out_dir / "mlp_gt.csv"
    manifest_out = out_dir / "mlp_manifest.json"

    if obs_rows:
        with obs_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(obs_rows[0].keys()))
            writer.writeheader()
            writer.writerows(obs_rows)

    if epoch_summary_rows:
        with epoch_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_summary_rows)

    if gt_table:
        with gt_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gt_table[0].keys()))
            writer.writeheader()
            writer.writerows(gt_table)

    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in epoch_summary_rows:
        sp = str(row["split"])
        if sp not in split_counts:
            split_counts[sp] = 0
        split_counts[sp] += 1

    manifest = {
        "inputs": {
            "raw_csv": str(raw_path),
            "gt_csv": str(gt_path),
        },
        "outputs": {
            "observations_csv": str(obs_out),
            "epochs_csv": str(epoch_out),
            "gt_csv": str(gt_out),
        },
        "id_mapping": GNSS_ID_MAP,
        "quality_for_coarse_ls": {
            "cno_min_dbhz": float(args.cno_min),
            "elev_min_deg": float(args.elev_min_deg),
        },
        "split_policy": {
            "mode": str(args.split_mode),
            "train_epochs": int(args.train_epochs),
            "test_epochs": int(args.test_epochs),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
        },
        "feature_columns_for_model": [
            "cno_dbhz",
            "azim_deg",
            "elev_deg",
            "residual_m",
        ],
        "label_columns": [
            "gt_lat_deg",
            "gt_lon_deg",
            "gt_h_m",
        ],
        "leakage_policy": {
            "forbidden_in_model_input": [
                "gt_ecef_x_m",
                "gt_ecef_y_m",
                "gt_ecef_z_m",
                "gt_lat_deg",
                "gt_lon_deg",
                "gt_h_m",
            ]
        },
        "stats": {
            "epochs_total": len(epoch_summary_rows),
            "observations_total": len(obs_rows),
            "coarse_ok_epochs": coarse_ok_count,
            "split_counts": split_counts,
        },
        "notes": [
            "MLP-Kalmannet bridge approximates RINEX+EPH by using precomputed satellite states in raw observations.",
            "Residuals are generated from coarse LS, not from GT receiver positions.",
        ],
    }

    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote: {obs_out}")
    print(f"wrote: {epoch_out}")
    print(f"wrote: {gt_out}")
    print(f"wrote: {manifest_out}")
    print(f"epochs={len(epoch_summary_rows)} observations={len(obs_rows)} coarse_ok={coarse_ok_count}")


if __name__ == "__main__":
    main()
