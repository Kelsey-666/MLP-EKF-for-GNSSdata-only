#!/usr/bin/env python3
"""Transform smartLoc RXM-RAWX/NAV-POSLLH to MLPKalmannet CSV schema.

Main behaviors:
1) Keep the previous schema mapping.
2) Add satellite-state enrichment from BRDC RINEX nav:
   - sat_pos_E_*
   - sat_vel_E_*
   - sat_bias_m
   - sat_drift_mps
   - elev/azim (rad/deg)
3) Enforce project Doppler sign convention:
   dr_mps = doMes * wavelength_m
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


CLIGHT = 299_792_458.0
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
SECS_IN_WEEK = 604_800.0


RAW_SCHEMA = [
    "obs_index",
    "epoch",
    "week",
    "tow",
    "time_gps_s",
    "gnss_id",
    "sv_id",
    "sig_id",
    "accs_id",
    "pr_m",
    "dr_mps",
    "pr_noise_m",
    "dr_noise_mps",
    "sat_pos_E_x_m",
    "sat_pos_E_y_m",
    "sat_pos_E_z_m",
    "sat_vel_E_x_mps",
    "sat_vel_E_y_mps",
    "sat_vel_E_z_mps",
    "sat_bias_m",
    "sat_drift_mps",
    "elev_rad",
    "azim_rad",
    "elev_deg",
    "azim_deg",
    "cno",
    "iono_m",
    "wavelength_m",
    "gt_pos_E_x_m",
    "gt_pos_E_y_m",
    "gt_pos_E_z_m",
    "gt_vel_E_x_mps",
    "gt_vel_E_y_mps",
    "gt_vel_E_z_mps",
    "gt_pos_std_N_n_m",
    "gt_pos_std_N_e_m",
    "gt_pos_std_N_d_m",
    "is_static",
]

GT_SCHEMA = [
    "epoch",
    "week",
    "tow",
    "time_gps_s",
    "gt_pos_E_x_m",
    "gt_pos_E_y_m",
    "gt_pos_E_z_m",
    "gt_vel_E_x_mps",
    "gt_vel_E_y_mps",
    "gt_vel_E_z_mps",
    "gt_lat_deg",
    "gt_lon_deg",
    "gt_h_m",
    "gt_pos_std_N_n_m",
    "gt_pos_std_N_e_m",
    "gt_pos_std_N_d_m",
    "speed_mps",
    "is_static",
]

SAT_COLUMNS = [
    "sat_pos_E_x_m",
    "sat_pos_E_y_m",
    "sat_pos_E_z_m",
    "sat_vel_E_x_mps",
    "sat_vel_E_y_mps",
    "sat_vel_E_z_mps",
    "sat_bias_m",
    "sat_drift_mps",
    "elev_rad",
    "azim_rad",
    "elev_deg",
    "azim_deg",
]


def _resolve_project_root() -> Path:
    # .../MLPkalmannet/Data/smartLoc.../transform_smartloc_schema.py
    return Path(__file__).resolve().parents[2]


def _ensure_lf_foundation_on_path() -> None:
    project_root = _resolve_project_root()
    lf_root = project_root / "ref" / "LF-GNSS"
    if not lf_root.exists():
        raise FileNotFoundError(f"LF-GNSS root not found: {lf_root}")
    lf_root_str = str(lf_root)
    if lf_root_str not in sys.path:
        sys.path.insert(0, lf_root_str)


_ensure_lf_foundation_on_path()
from gnss_foundation.ephemeris import satpos  # noqa: E402
from gnss_foundation.gnss_type import (  # noqa: E402
    Nav,
    ecef2pos,
    geodist,
    gpst2time,
    prn2sat,
    satazel,
    timeadd,
    uGNSS,
)
from gnss_foundation.rinex import rnxdec  # noqa: E402


PROJECT_TO_UGNSS = {
    0: uGNSS.GPS,
    1: uGNSS.SBS,
    2: uGNSS.GAL,
    3: uGNSS.BDS,
    5: uGNSS.QZS,
    6: uGNSS.GLO,
}


def _read_smartloc_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")


def _norm_text(s: str) -> str:
    return str(s).strip().lower().replace("褢褍", "掳")


def _find_col(cols: Iterable[str], *contains_all: str) -> str:
    needles = [_norm_text(k) for k in contains_all]
    for c in cols:
        cc = _norm_text(c)
        if all(n in cc for n in needles):
            return c
    raise KeyError(f"Cannot find column with tokens {contains_all}")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _map_gnss_id(s: pd.Series) -> pd.Series:
    def one(v) -> float:
        if pd.isna(v):
            return np.nan
        t = str(v).strip().upper()
        if t in {"0", "GPS"}:
            return 0
        if t in {"2", "GAL", "GALILEO"}:
            return 2
        if t in {"3", "BDS", "BEIDOU", "BEI"}:
            return 3
        if t in {"5", "QZS", "QZSS"}:
            return 5
        if t in {"6", "GLO", "GLONASS"}:
            return 6
        if t in {"1", "SBAS"}:
            return 1
        try:
            return int(float(t))
        except Exception:
            return np.nan

    return s.apply(one).astype(float)


def _wavelength_m(gnss_id: np.ndarray, freq_id: np.ndarray) -> np.ndarray:
    out = np.full(gnss_id.shape, np.nan, dtype=float)
    gps_l1 = CLIGHT / 1_575_420_000.0
    bds_b1i = CLIGHT / 1_561_098_000.0
    for i in range(len(out)):
        g = gnss_id[i]
        if not np.isfinite(g):
            continue
        gi = int(g)
        if gi in (0, 2, 5):  # GPS/GAL/QZSS -> L1/E1
            out[i] = gps_l1
        elif gi == 3:  # BeiDou -> B1I
            out[i] = bds_b1i
        elif gi == 6:  # GLONASS L1 + slot
            k = freq_id[i] if np.isfinite(freq_id[i]) else 0.0
            f_hz = 1_602_000_000.0 + float(k) * 562_500.0
            if f_hz > 0.0:
                out[i] = CLIGHT / f_hz
    return out


def _llh_to_ecef(
    lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return x, y, z


def _vel_enu_to_ecef(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    v_e_mps: np.ndarray,
    v_n_mps: np.ndarray,
    v_u_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    vx = -sin_lon * v_e_mps - sin_lat * cos_lon * v_n_mps + cos_lat * cos_lon * v_u_mps
    vy = cos_lon * v_e_mps - sin_lat * sin_lon * v_n_mps + cos_lat * sin_lon * v_u_mps
    vz = cos_lat * v_n_mps + sin_lat * v_u_mps
    return vx, vy, vz


def _pos_cov_deg_to_m(
    lat_deg: np.ndarray, lon_cov_deg: np.ndarray, lat_cov_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat_deg)
    sigma_n = np.abs(lat_cov_deg) * np.pi / 180.0 * WGS84_A
    sigma_e = np.abs(lon_cov_deg) * np.pi / 180.0 * WGS84_A * np.cos(lat_rad)
    return sigma_n, sigma_e


def _build_epoch(df: pd.DataFrame, week_col: str, tow_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([week_col, tow_col], kind="stable").reset_index(drop=True)
    keys = list(zip(out[week_col].to_numpy(), out[tow_col].to_numpy()))
    out["epoch"] = pd.factorize(pd.Series(keys), sort=False)[0] + 1
    return out


def _discover_nav_file(scenario_dir: Path, nav_file_arg: Optional[Path] = None) -> Path:
    if nav_file_arg is not None:
        if not nav_file_arg.exists():
            raise FileNotFoundError(f"Provided nav file does not exist: {nav_file_arg}")
        return nav_file_arg

    patterns = [
        "aux_nav/*.rnx",
        "aux_nav/*.rnx.gz",
        "*.rnx",
        "*.rnx.gz",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(sorted(scenario_dir.glob(pat)))
    if not candidates:
        raise FileNotFoundError(
            f"No BRDC RINEX nav file found under: {scenario_dir} (or aux_nav/)."
        )
    return candidates[0]


def _ensure_unzipped_nav(nav_path: Path) -> Path:
    if nav_path.suffix.lower() != ".gz":
        return nav_path
    out = nav_path.with_suffix("")
    if (not out.exists()) or (nav_path.stat().st_mtime > out.stat().st_mtime):
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(nav_path, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out


def _load_nav(nav_path: Path) -> Nav:
    nav = Nav()
    dec = rnxdec()
    ret = dec.decode_nav(str(nav_path), nav)
    if ret == -1:
        raise RuntimeError(f"Failed to decode nav file: {nav_path}")
    return nav


def _project_to_sat(gnss_id: float, sv_id: float) -> int:
    if not np.isfinite(gnss_id) or not np.isfinite(sv_id):
        return 0
    gi = int(gnss_id)
    if gi not in PROJECT_TO_UGNSS:
        return 0
    sys_ = PROJECT_TO_UGNSS[gi]
    prn = int(round(float(sv_id)))
    if prn <= 0:
        return 0
    if sys_ == uGNSS.QZS and prn < 193:
        prn += 192
    if sys_ == uGNSS.SBS and prn < 120:
        prn += 100
    return prn2sat(sys_, prn)


def _sat_state_at_time(sat: int, t, nav: Nav):
    rs, vs, dts, _ = satpos(sat, t, nav)
    if rs is None or vs is None or dts is None:
        return None
    if len(rs) < 1 or len(vs) < 1 or len(dts) < 1:
        return None
    r = np.asarray(rs[0], dtype=float)
    v = np.asarray(vs[0], dtype=float)
    dt = float(dts[0])
    if (not np.isfinite(r).all()) or (not np.isfinite(v).all()) or (not np.isfinite(dt)):
        return None
    return r, v, dt


def _solve_tx_state(week: float, tow: float, pr_m: float, sat: int, nav: Nav):
    if sat <= 0 or (not np.isfinite(week)) or (not np.isfinite(tow)):
        return None
    tau = float(pr_m) / CLIGHT if np.isfinite(pr_m) else 0.0
    t_rx = gpst2time(int(round(float(week))), float(tow))
    t_tx = timeadd(t_rx, -tau)

    # One correction iteration is a good speed/accuracy tradeoff for this conversion step.
    st = _sat_state_at_time(sat, t_tx, nav)
    if st is None:
        return None
    _, _, dt = st
    t_tx = timeadd(t_rx, -tau - dt)

    st = _sat_state_at_time(sat, t_tx, nav)
    if st is None:
        return None
    return t_tx, st


def _compute_sat_columns(df: pd.DataFrame, nav: Nav, progress_step: int = 5000) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame(index=df.index)
    for c in SAT_COLUMNS:
        out[c] = np.nan

    range_rate_geo = np.full(n, np.nan, dtype=float)

    week = df["week"].to_numpy(dtype=float)
    tow = df["tow"].to_numpy(dtype=float)
    pr = df["pr_m"].to_numpy(dtype=float)
    gnss_id = df["gnss_id"].to_numpy(dtype=float)
    sv_id = df["sv_id"].to_numpy(dtype=float)
    rr_x = df["gt_pos_E_x_m"].to_numpy(dtype=float)
    rr_y = df["gt_pos_E_y_m"].to_numpy(dtype=float)
    rr_z = df["gt_pos_E_z_m"].to_numpy(dtype=float)
    rv_x = df["gt_vel_E_x_mps"].to_numpy(dtype=float)
    rv_y = df["gt_vel_E_y_mps"].to_numpy(dtype=float)
    rv_z = df["gt_vel_E_z_mps"].to_numpy(dtype=float)

    for i in range(n):
        if progress_step > 0 and (i % progress_step == 0):
            print(f"[sat-state] {i}/{n}")

        sat = _project_to_sat(gnss_id[i], sv_id[i])
        if sat <= 0:
            continue

        solved = _solve_tx_state(week[i], tow[i], pr[i], sat, nav)
        if solved is None:
            continue
        t_tx, st = solved
        r_s, v_s, dt_s = st

        out.iat[i, out.columns.get_loc("sat_pos_E_x_m")] = r_s[0]
        out.iat[i, out.columns.get_loc("sat_pos_E_y_m")] = r_s[1]
        out.iat[i, out.columns.get_loc("sat_pos_E_z_m")] = r_s[2]
        out.iat[i, out.columns.get_loc("sat_vel_E_x_mps")] = v_s[0]
        out.iat[i, out.columns.get_loc("sat_vel_E_y_mps")] = v_s[1]
        out.iat[i, out.columns.get_loc("sat_vel_E_z_mps")] = v_s[2]
        out.iat[i, out.columns.get_loc("sat_bias_m")] = dt_s * CLIGHT

        rr = np.array([rr_x[i], rr_y[i], rr_z[i]], dtype=float)
        if np.isfinite(rr).all():
            _, e = geodist(r_s, rr)
            pos = ecef2pos(rr)
            az, el = satazel(pos, e)
            out.iat[i, out.columns.get_loc("elev_rad")] = el
            out.iat[i, out.columns.get_loc("azim_rad")] = az
            out.iat[i, out.columns.get_loc("elev_deg")] = np.rad2deg(el)
            out.iat[i, out.columns.get_loc("azim_deg")] = np.rad2deg(az)
            rr_v = np.array([rv_x[i], rv_y[i], rv_z[i]], dtype=float)
            if np.isfinite(rr_v).all():
                range_rate_geo[i] = float(e @ (v_s - rr_v))

    # Estimate satellite clock drift from bias derivative per satellite track.
    out["sat_drift_mps"] = np.nan
    key = (
        df["gnss_id"].astype("Int64").astype(str)
        + "_"
        + df["sv_id"].astype("Int64").astype(str)
    ).to_numpy()
    t_all = df["time_gps_s"].to_numpy(dtype=float)
    b_all = out["sat_bias_m"].to_numpy(dtype=float)
    for sat_key in pd.unique(key):
        idx = np.where(key == sat_key)[0]
        if len(idx) < 2:
            continue
        t = t_all[idx]
        b = b_all[idx]
        valid = np.isfinite(t) & np.isfinite(b)
        if valid.sum() < 2:
            continue
        idx_v = idx[valid]
        t_v = t[valid]
        b_v = b[valid]
        order = np.argsort(t_v)
        idx_s = idx_v[order]
        t_s = t_v[order]
        b_s = b_v[order]
        dt = np.diff(t_s)
        if np.any(dt <= 0):
            # Keep stable derivative only on strictly increasing segments.
            keep = np.concatenate(([True], dt > 0))
            idx_s = idx_s[keep]
            t_s = t_s[keep]
            b_s = b_s[keep]
        if len(idx_s) < 2:
            continue
        drift = np.gradient(b_s, t_s)
        out.iloc[idx_s, out.columns.get_loc("sat_drift_mps")] = drift

    out["smartloc_range_rate_geo_mps"] = range_rate_geo
    return out


def transform_rxm(
    rxm_path: Path,
    nav: Optional[Nav],
    static_speed_threshold: float,
    max_epoch: Optional[int] = None,
) -> pd.DataFrame:
    src = _read_smartloc_csv(rxm_path)

    col_week = _find_col(src.columns, "gps week number (week)")
    col_tow = _find_col(src.columns, "measurement time of week (rcvtow)")
    col_gt_lon = _find_col(src.columns, "longitude (gt lon)")
    col_gt_lat = _find_col(src.columns, "latitude (gt lat)")
    col_gt_h = _find_col(src.columns, "gt height")
    col_gt_lon_cov = _find_col(src.columns, "longitude cov (gt lon)")
    col_gt_lat_cov = _find_col(src.columns, "latitude cov (gt lat)")
    col_gt_h_cov = _find_col(src.columns, "height above ellipsoid cov (gt height)")
    col_gt_speed = _find_col(src.columns, "velocity (gt velocity)")
    col_gt_heading = _find_col(src.columns, "gt heading")
    col_pr = _find_col(src.columns, "pseudorange measurement (prmes)")
    col_dop_hz = _find_col(src.columns, "doppler measurement (domes)")
    col_gnss = _find_col(src.columns, "gnss identifier (gnssid)")
    col_sv = _find_col(src.columns, "satellite identifier (svid)")
    col_freq = _find_col(src.columns, "freqid")
    col_cno = _find_col(src.columns, "cno")
    col_pr_std = _find_col(src.columns, "prstdev")
    col_do_std = _find_col(src.columns, "dostdev")
    col_nlos = _find_col(src.columns, "nlos")

    df = pd.DataFrame()
    df["week"] = _to_num(src[col_week]).astype("Int64")
    df["tow"] = _to_num(src[col_tow]).astype(float)
    df["time_gps_s"] = df["week"].astype(float) * SECS_IN_WEEK + df["tow"]
    df["gt_lon_deg"] = _to_num(src[col_gt_lon]).astype(float)
    df["gt_lat_deg"] = _to_num(src[col_gt_lat]).astype(float)
    df["gt_h_m"] = _to_num(src[col_gt_h]).astype(float)
    df["gt_lon_cov_deg"] = _to_num(src[col_gt_lon_cov]).astype(float)
    df["gt_lat_cov_deg"] = _to_num(src[col_gt_lat_cov]).astype(float)
    df["gt_h_cov_m"] = _to_num(src[col_gt_h_cov]).astype(float)
    df["speed_mps"] = _to_num(src[col_gt_speed]).astype(float)
    df["heading_rad"] = _to_num(src[col_gt_heading]).astype(float)
    df["pr_m"] = _to_num(src[col_pr]).astype(float)
    df["do_hz"] = _to_num(src[col_dop_hz]).astype(float)
    df["gnss_id"] = _map_gnss_id(src[col_gnss])
    df["sv_id"] = _to_num(src[col_sv]).astype(float)
    df["freq_id"] = _to_num(src[col_freq]).astype(float)
    df["cno"] = _to_num(src[col_cno]).astype(float)
    df["pr_noise_m"] = _to_num(src[col_pr_std]).astype(float)
    df["do_std_hz"] = _to_num(src[col_do_std]).astype(float)
    df["nlos"] = pd.to_numeric(src[col_nlos].replace("#", np.nan), errors="coerce")

    df = _build_epoch(df, "week", "tow")
    if max_epoch is not None:
        df = df[df["epoch"] <= int(max_epoch)].reset_index(drop=True)
    df["obs_index"] = np.arange(1, len(df) + 1, dtype=int)

    gx, gy, gz = _llh_to_ecef(
        df["gt_lat_deg"].to_numpy(), df["gt_lon_deg"].to_numpy(), df["gt_h_m"].to_numpy()
    )
    v = df["speed_mps"].to_numpy(dtype=float)
    hdg = df["heading_rad"].to_numpy(dtype=float)
    v_e = v * np.cos(hdg)  # heading: 0 rad = East, CCW positive.
    v_n = v * np.sin(hdg)
    v_u = np.zeros_like(v_e)
    gvx, gvy, gvz = _vel_enu_to_ecef(
        df["gt_lat_deg"].to_numpy(), df["gt_lon_deg"].to_numpy(), v_e, v_n, v_u
    )

    sigma_n, sigma_e = _pos_cov_deg_to_m(
        df["gt_lat_deg"].to_numpy(),
        df["gt_lon_cov_deg"].to_numpy(),
        df["gt_lat_cov_deg"].to_numpy(),
    )
    sigma_d = np.abs(df["gt_h_cov_m"].to_numpy(dtype=float))

    wl = _wavelength_m(df["gnss_id"].to_numpy(dtype=float), df["freq_id"].to_numpy(dtype=float))
    # Project convention verified against raw_observation: dr ~= -range_rate (+ clock drift term)
    dr_mps = df["do_hz"].to_numpy(dtype=float) * wl
    dr_noise_mps = df["do_std_hz"].to_numpy(dtype=float) * wl

    base = pd.DataFrame(index=df.index)
    base["obs_index"] = df["obs_index"].astype(int)
    base["epoch"] = df["epoch"].astype(int)
    base["week"] = df["week"].astype("Int64")
    base["tow"] = df["tow"].astype(float)
    base["time_gps_s"] = df["time_gps_s"].astype(float)
    base["gnss_id"] = df["gnss_id"].astype("Int64")
    base["sv_id"] = df["sv_id"].astype("Int64")
    base["sig_id"] = 0
    base["accs_id"] = 0
    base["pr_m"] = df["pr_m"].astype(float)
    base["dr_mps"] = dr_mps
    base["pr_noise_m"] = df["pr_noise_m"].astype(float)
    base["dr_noise_mps"] = dr_noise_mps
    base["cno"] = df["cno"].astype(float)
    base["iono_m"] = 0.0
    base["wavelength_m"] = wl
    base["gt_pos_E_x_m"] = gx
    base["gt_pos_E_y_m"] = gy
    base["gt_pos_E_z_m"] = gz
    base["gt_vel_E_x_mps"] = gvx
    base["gt_vel_E_y_mps"] = gvy
    base["gt_vel_E_z_mps"] = gvz
    base["gt_pos_std_N_n_m"] = sigma_n
    base["gt_pos_std_N_e_m"] = sigma_e
    base["gt_pos_std_N_d_m"] = sigma_d
    base["is_static"] = (df["speed_mps"].astype(float) < static_speed_threshold).astype(int)

    if nav is not None:
        sat_df = _compute_sat_columns(base, nav)
        for c in SAT_COLUMNS:
            base[c] = sat_df[c].astype(float)
        base["smartloc_range_rate_geo_mps"] = sat_df["smartloc_range_rate_geo_mps"].astype(float)
    else:
        for c in SAT_COLUMNS:
            base[c] = np.nan
        base["smartloc_range_rate_geo_mps"] = np.nan

    base["smartloc_nlos"] = df["nlos"].astype(float)
    base["smartloc_do_hz"] = df["do_hz"].astype(float)
    base["smartloc_do_std_hz"] = df["do_std_hz"].astype(float)
    base["smartloc_freq_id"] = df["freq_id"].astype(float)
    base["smartloc_heading_rad"] = df["heading_rad"].astype(float)
    base["smartloc_speed_mps"] = df["speed_mps"].astype(float)

    return base


def transform_nav(
    nav_path: Path,
    static_speed_threshold: float,
    max_epoch: Optional[int] = None,
) -> pd.DataFrame:
    src = _read_smartloc_csv(nav_path)

    col_week = _find_col(src.columns, "gpsweek")
    col_tow = _find_col(src.columns, "gpssecondsofweek")
    col_gt_lon = _find_col(src.columns, "longitude (gt lon)")
    col_gt_lat = _find_col(src.columns, "latitude (gt lat)")
    col_gt_h = _find_col(src.columns, "gt height")
    col_gt_lon_cov = _find_col(src.columns, "longitude cov (gt lon")
    col_gt_lat_cov = _find_col(src.columns, "latitude cov (gt lat")
    col_gt_h_cov = _find_col(src.columns, "height above ellipsoid cov (gt height")
    col_gt_speed = _find_col(src.columns, "velocity (gt velocity)")
    col_gt_heading = _find_col(src.columns, "gt heading")

    df = pd.DataFrame()
    df["week"] = _to_num(src[col_week]).astype("Int64")
    df["tow"] = _to_num(src[col_tow]).astype(float)
    df["time_gps_s"] = df["week"].astype(float) * SECS_IN_WEEK + df["tow"]
    df["gt_lon_deg"] = _to_num(src[col_gt_lon]).astype(float)
    df["gt_lat_deg"] = _to_num(src[col_gt_lat]).astype(float)
    df["gt_h_m"] = _to_num(src[col_gt_h]).astype(float)
    df["gt_lon_cov_deg"] = _to_num(src[col_gt_lon_cov]).astype(float)
    df["gt_lat_cov_deg"] = _to_num(src[col_gt_lat_cov]).astype(float)
    df["gt_h_cov_m"] = _to_num(src[col_gt_h_cov]).astype(float)
    df["speed_mps"] = _to_num(src[col_gt_speed]).astype(float)
    df["heading_rad"] = _to_num(src[col_gt_heading]).astype(float)

    df = _build_epoch(df, "week", "tow")
    if max_epoch is not None:
        df = df[df["epoch"] <= int(max_epoch)].reset_index(drop=True)

    gx, gy, gz = _llh_to_ecef(
        df["gt_lat_deg"].to_numpy(), df["gt_lon_deg"].to_numpy(), df["gt_h_m"].to_numpy()
    )
    v = df["speed_mps"].to_numpy(dtype=float)
    hdg = df["heading_rad"].to_numpy(dtype=float)
    v_e = v * np.cos(hdg)
    v_n = v * np.sin(hdg)
    v_u = np.zeros_like(v_e)
    gvx, gvy, gvz = _vel_enu_to_ecef(
        df["gt_lat_deg"].to_numpy(), df["gt_lon_deg"].to_numpy(), v_e, v_n, v_u
    )
    sigma_n, sigma_e = _pos_cov_deg_to_m(
        df["gt_lat_deg"].to_numpy(),
        df["gt_lon_cov_deg"].to_numpy(),
        df["gt_lat_cov_deg"].to_numpy(),
    )
    sigma_d = np.abs(df["gt_h_cov_m"].to_numpy(dtype=float))

    out = pd.DataFrame(index=df.index)
    out["epoch"] = df["epoch"].astype(int)
    out["week"] = df["week"].astype("Int64")
    out["tow"] = df["tow"].astype(float)
    out["time_gps_s"] = df["time_gps_s"].astype(float)
    out["gt_pos_E_x_m"] = gx
    out["gt_pos_E_y_m"] = gy
    out["gt_pos_E_z_m"] = gz
    out["gt_vel_E_x_mps"] = gvx
    out["gt_vel_E_y_mps"] = gvy
    out["gt_vel_E_z_mps"] = gvz
    out["gt_lat_deg"] = df["gt_lat_deg"].astype(float)
    out["gt_lon_deg"] = df["gt_lon_deg"].astype(float)
    out["gt_h_m"] = df["gt_h_m"].astype(float)
    out["gt_pos_std_N_n_m"] = sigma_n
    out["gt_pos_std_N_e_m"] = sigma_e
    out["gt_pos_std_N_d_m"] = sigma_d
    out["speed_mps"] = df["speed_mps"].astype(float)
    out["is_static"] = (df["speed_mps"].astype(float) < static_speed_threshold).astype(int)
    return out


def _inject_epoch_sat_summary(nav_df: pd.DataFrame, rxm_df: pd.DataFrame) -> pd.DataFrame:
    # nav is epoch-level, so we store epoch-mean satellite fields for compatibility.
    agg = (
        rxm_df.groupby("epoch", as_index=False)[SAT_COLUMNS]
        .mean(numeric_only=True)
        .rename(columns={c: c for c in SAT_COLUMNS})
    )
    out = nav_df.merge(agg, on="epoch", how="left")
    return out


def write_report(path: Path, nav_source: str) -> None:
    txt = f"""# smartLoc -> MLPKalmannet Schema Report

## Key differences handled
1. Delimiter difference:
- smartLoc files use semicolon (`;`) separators.
- Project files use comma separators.

2. Field name normalization:
- RXM/NAV long descriptive headers are mapped to short project-style names.

3. Unit normalization:
- Doppler from Hz (`doMes`) is converted to m/s (`dr_mps`) via `wavelength_m`.
- Doppler std from Hz (`doStdev`) is converted to m/s (`dr_noise_mps`) via `wavelength_m`.
- GT lon/lat covariance in degree is approximated to EN meters (`gt_pos_std_N_e_m`, `gt_pos_std_N_n_m`).

4. dr sign convention:
- Enforced project convention: `dr_mps = doMes * wavelength_m`.
- With smartLoc sign definition, this aligns `dr_mps` opposite to geometric range-rate.

5. Satellite state enrichment:
- Source nav file: `{nav_source}`
- Added per-observation `sat_pos/vel/bias/drift + elev/azim` via BRDC ephemeris propagation.

6. Static relabel:
- `is_static` is recomputed by `speed_mps < 0.1`.

## Assumptions
1. Signal ID is unavailable in RXM-RAWX:
- `sig_id = 0`.

2. Access ID is unavailable:
- `accs_id = 0`.

3. Wavelength inference by constellation:
- GPS/GAL/QZSS: L1/E1 (1575.42 MHz)
- BeiDou: B1I (1561.098 MHz)
- GLONASS: L1 + `freqId` slot (1602 MHz + k*0.5625 MHz)

4. GT velocity vector:
- Only scalar speed + heading are given.
- E/N velocity is inferred from heading (0 rad = East, CCW positive), then rotated to ECEF.

## Notes
- `trans_NAV-POSLLH.csv` is epoch-level GT; it additionally contains epoch-mean sat-state summary columns.
"""
    path.write_text(txt, encoding="utf-8")


def run(
    scenario_dir: Path,
    static_speed_threshold: float,
    nav_file: Optional[Path] = None,
    max_epoch: Optional[int] = None,
) -> tuple[Path, Path]:
    rxm_in = scenario_dir / "RXM-RAWX.csv"
    nav_in = scenario_dir / "NAV-POSLLH.csv"
    if not rxm_in.exists():
        raise FileNotFoundError(f"Missing {rxm_in}")
    if not nav_in.exists():
        raise FileNotFoundError(f"Missing {nav_in}")

    nav_src = _discover_nav_file(scenario_dir, nav_file_arg=nav_file)
    nav_rnx = _ensure_unzipped_nav(nav_src)
    nav_obj = _load_nav(nav_rnx)

    rxm_out = scenario_dir / "trans_RXM-RAWX.csv"
    nav_out = scenario_dir / "trans_NAV-POSLLH.csv"
    report_out = scenario_dir / "trans_schema_report.md"

    print(f"[nav] source={nav_src}")
    print(f"[nav] decoded={nav_rnx}")
    print(f"[nav] eph={len(nav_obj.eph)} geph={len(nav_obj.geph)}")
    if max_epoch is not None:
        print(f"[limit] max_epoch={max_epoch}")

    rxm = transform_rxm(
        rxm_in,
        nav=nav_obj,
        static_speed_threshold=static_speed_threshold,
        max_epoch=max_epoch,
    )
    nav_df = transform_nav(
        nav_in,
        static_speed_threshold=static_speed_threshold,
        max_epoch=max_epoch,
    )
    nav_df = _inject_epoch_sat_summary(nav_df, rxm)

    for c in RAW_SCHEMA:
        if c not in rxm.columns:
            rxm[c] = np.nan
    ordered_rxm = RAW_SCHEMA + [c for c in rxm.columns if c not in RAW_SCHEMA]
    rxm = rxm[ordered_rxm]

    for c in GT_SCHEMA:
        if c not in nav_df.columns:
            nav_df[c] = np.nan
    ordered_nav = GT_SCHEMA + [c for c in nav_df.columns if c not in GT_SCHEMA]
    nav_df = nav_df[ordered_nav]

    rxm.to_csv(rxm_out, index=False, encoding="utf-8-sig")
    nav_df.to_csv(nav_out, index=False, encoding="utf-8-sig")
    write_report(report_out, nav_source=str(nav_src))

    sat_filled = int(np.isfinite(rxm["sat_pos_E_x_m"].to_numpy(dtype=float)).sum())
    elev_filled = int(np.isfinite(rxm["elev_rad"].to_numpy(dtype=float)).sum())
    print(f"wrote={rxm_out}")
    print(f"rows={len(rxm)} cols={len(rxm.columns)} sat_pos_filled={sat_filled} elev_filled={elev_filled}")
    print(f"wrote={nav_out}")
    print(f"rows={len(nav_df)} cols={len(nav_df.columns)}")
    print(f"wrote={report_out}")
    return rxm_out, nav_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform smartLoc CSV schema to MLPKalmannet-like schema.")
    parser.add_argument(
        "--scenario-dir",
        default=r"D:\MLPkalmannet\Data\smartLoc (TU Chemnitz) urban GNSS dataset\berlin2_gendarmenmarkt",
        help="Directory containing RXM-RAWX.csv and NAV-POSLLH.csv",
    )
    parser.add_argument("--nav-file", default=None, help="Optional explicit BRDC RINEX nav file path.")
    parser.add_argument("--static-threshold", type=float, default=0.1, help="speed threshold for is_static.")
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=3000,
        help="Only keep epochs <= this value. Set <=0 to disable truncation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nav_file = Path(args.nav_file) if args.nav_file else None
    max_epoch = int(args.max_epoch) if args.max_epoch and int(args.max_epoch) > 0 else None
    run(
        Path(args.scenario_dir),
        static_speed_threshold=float(args.static_threshold),
        nav_file=nav_file,
        max_epoch=max_epoch,
    )


if __name__ == "__main__":
    main()
