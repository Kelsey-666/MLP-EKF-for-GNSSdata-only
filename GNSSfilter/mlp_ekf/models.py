"""Data models for MLP-Kalmannet EKF pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Observation:
    epoch: int
    split: str
    week: int
    tow: float
    time_gps_s: float
    gnss_id_raw: int
    gnss_system: str
    sv_id: int
    sig_id: int
    pr_m: float
    pr_noise_m: float
    sat_pos_x_m: float
    sat_pos_y_m: float
    sat_pos_z_m: float
    sat_vel_x_mps: float
    sat_vel_y_mps: float
    sat_vel_z_mps: float
    sat_bias_m: float
    sat_drift_mps: float
    elev_rad: float
    azim_rad: float
    elev_deg: float
    azim_deg: float
    cno_dbhz: float
    residual_m: float
    gt_ecef_x_m: float
    gt_ecef_y_m: float
    gt_ecef_z_m: float
    gt_lat_deg: float
    gt_lon_deg: float
    gt_h_m: float
    speed_mps: float
    is_static: int


@dataclass
class EpochBatch:
    epoch: int
    split: str
    week: int
    tow: float
    time_gps_s: float
    observations: List[Observation]


@dataclass
class FilterResultRow:
    epoch: int
    split: str
    week: int
    tow: float
    time_gps_s: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    ref_sys_id: int
    cb_m: float
    cd_mps: float
    inter_gnss_m: str
    ibb_m: str
    used_obs: int
    rejected_obs: int
    pdop: float
    r_inflation: float
    gt_lat_deg: float
    gt_lon_deg: float
    gt_h_m: float
    pred_lat_deg: float
    pred_lon_deg: float
    pred_h_m: float
    enu_e_m: float
    enu_n_m: float
    enu_u_m: float
    enu_3d_m: float
    init_ok: int


@dataclass
class UpdateDebug:
    pr_innovations: List[float]
    rejected_count: int
    used_count: int
    pdop: float
    r_inflation: float
    init_ok: bool
    coarse_state: Optional[List[float]]
