"""ConvertToGnss.m migration (for MATLAB-like source structures)."""

from __future__ import annotations

import numpy as np

from .constants import Constants
from .models import GnssMeasurements, Truth
from .calc_predicted_obs import calc_predicted_obs
from .utils import code_svids


def _g(obj, *keys):
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur[k]
        else:
            cur = getattr(cur, k)
    return cur


def convert_to_gnss(rov_proc, rov_solver_proc, rov_truth: Truth | dict) -> GnssMeasurements:
    nrow = int(_g(rov_proc, "signal", "internal", "pr").shape[0])
    ncol = int(_g(rov_proc, "signal", "internal", "pr").shape[1])

    rov_pred_pr, rov_pred_dr = calc_predicted_obs(_g(rov_proc, "satellite"), _g(rov_proc, "signal"), False)

    epoch = np.tile(np.arange(1, len(_g(rov_proc, "receiver", "rover_time_est")) + 1, dtype=int), (nrow, 1))
    sv_idx = _g(rov_proc, "signal", "internal", "sv_idx")

    pr = _g(rov_proc, "signal", "internal", "pr")
    dr = _g(rov_proc, "signal", "internal", "doppler") * _g(rov_proc, "signal", "internal", "lambda")

    ms_in_m = Constants.c * 0.001
    clock_bias_round_1ms = np.round(_g(rov_proc, "receiver", "clock_bias") / ms_in_m) * ms_in_m
    geom_range_rov = _g(rov_proc, "satellite", "internal", "linearised", "pr")[sv_idx, :] - clock_bias_round_1ms
    geom_range_rate_rov = _g(rov_proc, "satellite", "internal", "linearised", "doppler")[sv_idx, :]

    pr_corr = rov_pred_pr - geom_range_rov
    dr_corr = rov_pred_dr - geom_range_rate_rov

    gnss_id = np.tile(_g(rov_proc, "satellite", "internal", "gnss_id")[sv_idx].reshape(-1, 1), (1, ncol))
    sv_id = np.tile(_g(rov_proc, "satellite", "internal", "sv_id")[sv_idx].reshape(-1, 1), (1, ncol))

    sat_pos_x = _g(rov_proc, "satellite", "internal", "pos_E")[sv_idx, :, 0]
    sat_pos_y = _g(rov_proc, "satellite", "internal", "pos_E")[sv_idx, :, 1]
    sat_pos_z = _g(rov_proc, "satellite", "internal", "pos_E")[sv_idx, :, 2]
    sat_vel_x = _g(rov_proc, "satellite", "internal", "vel_E")[sv_idx, :, 0]
    sat_vel_y = _g(rov_proc, "satellite", "internal", "vel_E")[sv_idx, :, 1]
    sat_vel_z = _g(rov_proc, "satellite", "internal", "vel_E")[sv_idx, :, 2]
    sat_bias = _g(rov_proc, "satellite", "internal", "clk")[sv_idx, :]
    sat_drift = _g(rov_proc, "satellite", "internal", "clk_drift")[sv_idx, :]

    pr_noise = _g(rov_solver_proc, "signal", "pr_noise")
    dr_noise = _g(rov_solver_proc, "signal", "do_noise")

    wavelength = np.tile(_g(rov_proc, "signal", "internal", "lambda").reshape(-1, 1), (1, _g(rov_proc, "signal", "internal", "pr").shape[1]))
    cno = _g(rov_proc, "signal", "internal", "cno")

    elev_mask = np.deg2rad(10.0)
    sat_elev = _g(rov_proc, "satellite", "internal", "sat_elev_N")[sv_idx, :]
    ok = (~np.isnan(pr) & ~np.isnan(pr_corr) & ~np.isnan(sat_elev) & ~np.isnan(sat_bias) & (sat_elev > elev_mask))
    if np.sum(ok) == 0:
        raise ValueError("No valid measurements found")

    truth_pos = _g(rov_truth, "pos_E") if isinstance(rov_truth, dict) else rov_truth.pos_E
    truth_vel = _g(rov_truth, "vel_E") if isinstance(rov_truth, dict) else rov_truth.vel_E
    truth_std = _g(rov_truth, "pos_std_N") if isinstance(rov_truth, dict) else rov_truth.pos_std_N

    gnss = GnssMeasurements()
    gnss.tow = np.asarray(_g(rov_proc, "receiver", "rover_time_est"), dtype=float).reshape(-1)
    gnss.week = np.asarray(_g(rov_proc, "receiver", "week"), dtype=float).reshape(-1)
    gnss.pos_E = np.asarray(truth_pos, dtype=float).T
    gnss.vel_E = np.asarray(truth_vel, dtype=float).T
    gnss.pos_std_N = np.asarray(truth_std, dtype=float).T

    gnss.epoch = epoch[ok].astype(int)
    gnss.pr = (pr[ok] - pr_corr[ok]).astype(float)
    gnss.dr = (dr[ok] - dr_corr[ok]).astype(float)
    gnss.gnss_id = gnss_id[ok].astype(int)
    gnss.sv_id = sv_id[ok].astype(int)
    gnss.sig_id = _g(rov_proc, "signal", "internal", "sig_id")[ok].astype(int)
    gnss.accs_id = _g(rov_proc, "signal", "internal", "access_id")[ok].astype(int)
    gnss.pr_noise = pr_noise[ok].astype(float)
    gnss.dr_noise = dr_noise[ok].astype(float)
    gnss.sat_pos_E = np.column_stack([sat_pos_x[ok], sat_pos_y[ok], sat_pos_z[ok]]).astype(float)
    gnss.sat_vel_E = np.column_stack([sat_vel_x[ok], sat_vel_y[ok], sat_vel_z[ok]]).astype(float)
    gnss.sat_bias = sat_bias[ok].astype(float)
    gnss.sat_drift = sat_drift[ok].astype(float)
    gnss.elev = sat_elev[ok].astype(float)
    gnss.azim = _g(rov_proc, "satellite", "internal", "sat_az_N")[sv_idx, :][ok].astype(float)
    gnss.wavelength = wavelength[ok].astype(float)
    gnss.cno = cno[ok].astype(float)

    z = np.zeros_like(gnss.sv_id, dtype=np.int64)
    gnss.xtId = code_svids(gnss.sv_id, gnss.gnss_id, gnss.sig_id, z)
    gnss.xtIdSv = code_svids(gnss.sv_id, gnss.gnss_id, z, z)
    gnss.validate()
    return gnss

