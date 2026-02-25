"""CSV loader that recreates the MATLAB GnssMeasurements object."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .models import GnssMeasurements
from .utils import code_svids


@dataclass
class EpochSlice:
    start_epoch: int
    end_epoch: int


def _ensure_epoch_range(raw_df: pd.DataFrame, gt_df: pd.DataFrame | None, start_epoch: int | None, end_epoch: int | None):
    if start_epoch is None and end_epoch is None:
        return raw_df, gt_df

    se = int(start_epoch) if start_epoch is not None else int(raw_df["epoch"].min())
    ee = int(end_epoch) if end_epoch is not None else int(raw_df["epoch"].max())

    raw_df = raw_df[(raw_df["epoch"] >= se) & (raw_df["epoch"] <= ee)].copy()
    if gt_df is not None:
        gt_df = gt_df[(gt_df["epoch"] >= se) & (gt_df["epoch"] <= ee)].copy()
    return raw_df, gt_df


def _renumber_epochs(raw_df: pd.DataFrame, gt_df: pd.DataFrame | None):
    epochs = np.sort(raw_df["epoch"].unique())
    epoch_map = {int(e): i + 1 for i, e in enumerate(epochs)}
    raw_df = raw_df.copy()
    raw_df["epoch_old"] = raw_df["epoch"].astype(int)
    raw_df["epoch"] = raw_df["epoch_old"].map(epoch_map).astype(int)
    if gt_df is not None:
        gt_df = gt_df.copy()
        gt_df["epoch_old"] = gt_df["epoch"].astype(int)
        gt_df = gt_df[gt_df["epoch_old"].isin(epoch_map.keys())]
        gt_df["epoch"] = gt_df["epoch_old"].map(epoch_map).astype(int)
    return raw_df, gt_df, epoch_map


def load_gnss_from_csv(
    raw_csv: str,
    gt_csv: str | None = None,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
) -> GnssMeasurements:
    raw_df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    raw_df = raw_df.sort_values(["epoch", "obs_index"]).reset_index(drop=True)
    gt_df = pd.read_csv(gt_csv, encoding="utf-8-sig") if gt_csv else None

    raw_df, gt_df = _ensure_epoch_range(raw_df, gt_df, epoch_start, epoch_end)
    if raw_df.empty:
        raise ValueError("No observations after epoch filtering.")
    raw_df, gt_df, _ = _renumber_epochs(raw_df, gt_df)

    epoch_meta = (
        raw_df.groupby("epoch", as_index=False)
        .agg(
            week=("week", "first"),
            tow=("tow", "first"),
            time_gps_s=("time_gps_s", "first"),
            gt_pos_E_x_m=("gt_pos_E_x_m", "first"),
            gt_pos_E_y_m=("gt_pos_E_y_m", "first"),
            gt_pos_E_z_m=("gt_pos_E_z_m", "first"),
            gt_vel_E_x_mps=("gt_vel_E_x_mps", "first"),
            gt_vel_E_y_mps=("gt_vel_E_y_mps", "first"),
            gt_vel_E_z_mps=("gt_vel_E_z_mps", "first"),
            gt_pos_std_N_n_m=("gt_pos_std_N_n_m", "first"),
            gt_pos_std_N_e_m=("gt_pos_std_N_e_m", "first"),
            gt_pos_std_N_d_m=("gt_pos_std_N_d_m", "first"),
        )
        .sort_values("epoch")
    )

    if gt_df is not None and not gt_df.empty:
        gt_epoch = gt_df.sort_values("epoch").set_index("epoch")
        for col in [
            "week",
            "tow",
            "time_gps_s",
            "gt_pos_E_x_m",
            "gt_pos_E_y_m",
            "gt_pos_E_z_m",
            "gt_vel_E_x_mps",
            "gt_vel_E_y_mps",
            "gt_vel_E_z_mps",
            "gt_pos_std_N_n_m",
            "gt_pos_std_N_e_m",
            "gt_pos_std_N_d_m",
        ]:
            if col in gt_epoch.columns:
                epoch_meta[col] = gt_epoch.reindex(epoch_meta["epoch"]).reset_index(drop=True)[col].to_numpy()

    gnss = GnssMeasurements()
    gnss.tow = epoch_meta["tow"].to_numpy(dtype=float)
    gnss.week = epoch_meta["week"].to_numpy(dtype=float)
    gnss.pos_E = epoch_meta[["gt_pos_E_x_m", "gt_pos_E_y_m", "gt_pos_E_z_m"]].to_numpy(dtype=float)
    gnss.vel_E = epoch_meta[["gt_vel_E_x_mps", "gt_vel_E_y_mps", "gt_vel_E_z_mps"]].to_numpy(dtype=float)
    gnss.pos_std_N = epoch_meta[["gt_pos_std_N_n_m", "gt_pos_std_N_e_m", "gt_pos_std_N_d_m"]].to_numpy(dtype=float)

    gnss.epoch = raw_df["epoch"].to_numpy(dtype=int)
    gnss.pr = raw_df["pr_m"].to_numpy(dtype=float)
    gnss.dr = raw_df["dr_mps"].to_numpy(dtype=float)
    gnss.gnss_id = raw_df["gnss_id"].to_numpy(dtype=int)
    gnss.sv_id = raw_df["sv_id"].to_numpy(dtype=int)
    gnss.sig_id = raw_df["sig_id"].to_numpy(dtype=int)
    gnss.accs_id = raw_df["accs_id"].to_numpy(dtype=int)
    gnss.pr_noise = raw_df["pr_noise_m"].to_numpy(dtype=float)
    gnss.dr_noise = raw_df["dr_noise_mps"].to_numpy(dtype=float)
    gnss.sat_pos_E = raw_df[["sat_pos_E_x_m", "sat_pos_E_y_m", "sat_pos_E_z_m"]].to_numpy(dtype=float)
    gnss.sat_vel_E = raw_df[["sat_vel_E_x_mps", "sat_vel_E_y_mps", "sat_vel_E_z_mps"]].to_numpy(dtype=float)
    gnss.sat_bias = raw_df["sat_bias_m"].to_numpy(dtype=float)
    gnss.sat_drift = raw_df["sat_drift_mps"].to_numpy(dtype=float)
    gnss.elev = raw_df["elev_rad"].to_numpy(dtype=float)
    gnss.azim = raw_df["azim_rad"].to_numpy(dtype=float)
    gnss.wavelength = raw_df["wavelength_m"].to_numpy(dtype=float)
    gnss.cno = raw_df["cno"].to_numpy(dtype=float)
    gnss.iono = raw_df["iono_m"].to_numpy(dtype=float)

    z = np.zeros_like(gnss.sv_id, dtype=np.int64)
    gnss.xtId = code_svids(gnss.sv_id, gnss.gnss_id, gnss.sig_id, z)
    gnss.xtIdSv = code_svids(gnss.sv_id, gnss.gnss_id, z, z)
    gnss.validate()
    return gnss

