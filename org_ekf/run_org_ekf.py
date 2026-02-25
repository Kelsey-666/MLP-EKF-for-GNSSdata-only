#!/usr/bin/env python3
"""Run migrated org EKF on CSV data and compute RMSE on selected epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from org_ekf.io_csv import load_gnss_from_csv
from org_ekf.core import pedestrian_kalman_filter
from org_ekf.utils import ecef2ned_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run migrated original MATLAB EKF.")
    parser.add_argument("--raw-csv", default="Data/raw_observation.csv")
    parser.add_argument("--gt-csv", default="Data/gt_processed.csv")
    parser.add_argument("--epoch-start", type=int, default=2001)
    parser.add_argument("--epoch-end", type=int, default=4000)
    parser.add_argument("--out-dir", default="org_results")
    return parser.parse_args()


def _state_index_to_names(state_idx) -> list[str]:
    names = ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "cb_m", "cd_mps"]
    for i in range(len(state_idx.ibb)):
        names.append(f"ibb_{i+1}_m")
    for i in range(len(state_idx.inter_gnss)):
        names.append(f"inter_gnss_{i+1}_m")
    return names


def _rmse(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def main() -> None:
    args = parse_args()
    raw_csv = Path(args.raw_csv).resolve()
    gt_csv = Path(args.gt_csv).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(args.out_dir).resolve() / f"org_ekf_test2000_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    gnss = load_gnss_from_csv(
        raw_csv=str(raw_csv),
        gt_csv=str(gt_csv),
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
    )
    out = pedestrian_kalman_filter(gnss)

    state_idx = out["state_indx"]
    x = out["x"]
    pos_est = x[:, state_idx.pos]
    vel_est = x[:, state_idx.vel]

    dpos_ned, dpos_ecef, dvel_ned, dvel_ecef = ecef2ned_error(
        pos_est,
        gnss.pos_E[: pos_est.shape[0], :],
        vel_est,
        gnss.vel_E[: vel_est.shape[0], :],
    )
    n = dpos_ned[:, 0]
    e = dpos_ned[:, 1]
    d = dpos_ned[:, 2]

    metrics = {
        "epoch_start_original": int(args.epoch_start),
        "epoch_end_original": int(args.epoch_end),
        "epoch_count": int(pos_est.shape[0]),
        "north_rmse_m": _rmse(n),
        "east_rmse_m": _rmse(e),
        "down_rmse_m": _rmse(d),
        "up_rmse_m": _rmse(-d),
        "2d_rmse_m": float(np.sqrt(np.mean(n * n + e * e))),
        "3d_rmse_m": float(np.sqrt(np.mean(n * n + e * e + d * d))),
    }

    state_cols = _state_index_to_names(state_idx)
    states_df = pd.DataFrame(x, columns=state_cols)
    states_df.insert(0, "epoch", np.arange(1, x.shape[0] + 1, dtype=int))
    states_df["week"] = out["week"]
    states_df["tow"] = out["tow"]
    states_df["num_obs"] = out["num_obs"]
    states_df["num_sat"] = out["numSat"]
    states_df["pdop"] = out["pdop"]
    states_csv = run_dir / "org_ekf_states.csv"
    states_df.to_csv(states_csv, index=False)

    ned_df = pd.DataFrame(
        {
            "epoch": np.arange(1, dpos_ned.shape[0] + 1, dtype=int),
            "north_m": n,
            "east_m": e,
            "down_m": d,
            "up_m": -d,
            "err_2d_m": np.sqrt(n * n + e * e),
            "err_3d_m": np.sqrt(n * n + e * e + d * d),
        }
    )
    ned_csv = run_dir / "org_ekf_ned_errors.csv"
    ned_df.to_csv(ned_csv, index=False)

    metrics_json = run_dir / "org_ekf_metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_md = run_dir / "org_ekf_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# org_ekf test2000 summary",
                "",
                f"- Raw CSV: `{raw_csv}`",
                f"- GT CSV: `{gt_csv}`",
                f"- Epoch range (original): `{args.epoch_start}-{args.epoch_end}`",
                f"- Epoch count: `{metrics['epoch_count']}`",
                f"- East RMSE: `{metrics['east_rmse_m']:.3f} m`",
                f"- North RMSE: `{metrics['north_rmse_m']:.3f} m`",
                f"- Up RMSE: `{metrics['up_rmse_m']:.3f} m`",
                f"- 2D RMSE: `{metrics['2d_rmse_m']:.3f} m`",
                f"- 3D RMSE: `{metrics['3d_rmse_m']:.3f} m`",
                "",
                f"- States CSV: `{states_csv}`",
                f"- NED errors CSV: `{ned_csv}`",
                f"- Metrics JSON: `{metrics_json}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"run_dir={run_dir}")
    print(f"E={metrics['east_rmse_m']:.3f} N={metrics['north_rmse_m']:.3f} D={metrics['down_rmse_m']:.3f}")
    print(f"2D={metrics['2d_rmse_m']:.3f} 3D={metrics['3d_rmse_m']:.3f}")


if __name__ == "__main__":
    main()

