"""CSV adapter that turns preprocessed MLP records into epoch batches."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .models import EpochBatch, Observation


def _to_float(text: str) -> float:
    s = "" if text is None else str(text).strip()
    return float("nan") if s == "" else float(s)


def _to_int(text: str) -> int:
    s = "" if text is None else str(text).strip()
    return 0 if s == "" else int(float(s))


def load_epoch_batches(observation_csv: str) -> List[EpochBatch]:
    path = Path(observation_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: Dict[int, List[Observation]] = defaultdict(list)
    epoch_meta: Dict[int, Dict[str, object]] = {}

    for row in rows:
        obs = Observation(
            epoch=_to_int(row["epoch"]),
            split=row.get("split", "train"),
            week=_to_int(row["week"]),
            tow=_to_float(row["tow"]),
            time_gps_s=_to_float(row["time_gps_s"]),
            gnss_id_raw=_to_int(row["gnss_id_raw"]),
            gnss_system=row.get("gnss_system", "UNKNOWN"),
            sv_id=_to_int(row["sv_id"]),
            sig_id=_to_int(row["sig_id"]),
            pr_m=_to_float(row["pr_m"]),
            pr_noise_m=_to_float(row["pr_noise_m"]),
            sat_pos_x_m=_to_float(row["sat_pos_x_m"]),
            sat_pos_y_m=_to_float(row["sat_pos_y_m"]),
            sat_pos_z_m=_to_float(row["sat_pos_z_m"]),
            sat_vel_x_mps=_to_float(row["sat_vel_x_mps"]),
            sat_vel_y_mps=_to_float(row["sat_vel_y_mps"]),
            sat_vel_z_mps=_to_float(row["sat_vel_z_mps"]),
            sat_bias_m=_to_float(row["sat_bias_m"]),
            sat_drift_mps=_to_float(row["sat_drift_mps"]),
            elev_rad=_to_float(row["elev_rad"]),
            azim_rad=_to_float(row["azim_rad"]),
            elev_deg=_to_float(row["elev_deg"]),
            azim_deg=_to_float(row["azim_deg"]),
            cno_dbhz=_to_float(row["cno_dbhz"]),
            residual_m=_to_float(row["residual_m"]),
            gt_ecef_x_m=_to_float(row["gt_ecef_x_m"]),
            gt_ecef_y_m=_to_float(row["gt_ecef_y_m"]),
            gt_ecef_z_m=_to_float(row["gt_ecef_z_m"]),
            gt_lat_deg=_to_float(row["gt_lat_deg"]),
            gt_lon_deg=_to_float(row["gt_lon_deg"]),
            gt_h_m=_to_float(row["gt_h_m"]),
            speed_mps=_to_float(row["speed_mps"]),
            is_static=_to_int(row["is_static"]),
        )
        grouped[obs.epoch].append(obs)
        if obs.epoch not in epoch_meta:
            epoch_meta[obs.epoch] = {
                "split": obs.split,
                "week": obs.week,
                "tow": obs.tow,
                "time_gps_s": obs.time_gps_s,
            }

    batches: List[EpochBatch] = []
    for epoch in sorted(grouped.keys()):
        meta = epoch_meta[epoch]
        batches.append(
            EpochBatch(
                epoch=epoch,
                split=str(meta["split"]),
                week=int(meta["week"]),
                tow=float(meta["tow"]),
                time_gps_s=float(meta["time_gps_s"]),
                observations=grouped[epoch],
            )
        )
    return batches
