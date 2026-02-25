"""CalcPredictedObs.m migration.

This function keeps the original equations and expects MATLAB-like nested dicts.
"""

from __future__ import annotations

import numpy as np


def _g(obj, *keys):
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur[k]
        else:
            cur = getattr(cur, k)
    return cur


def calc_predicted_obs(satellite, signal, phase_smooth: bool):
    trop = _g(satellite, "model", "tropo_delay", "map_dry") * _g(satellite, "model", "tropo_delay", "zenith_dry")
    trop += _g(satellite, "model", "tropo_delay", "map_wet") * _g(satellite, "model", "tropo_delay", "zenith_wet")

    gim = _g(satellite, "model", "gim")
    if gim is not None and np.size(gim) > 0:
        freq = _g(signal, "internal", "freq")
        iono_scale = 40.3e16 / np.square(freq)
        sv_idx = _g(signal, "internal", "sv_idx")
        iono_sig_m = iono_scale * gim[sv_idx, :]
    else:
        iono_sig_m = 0.0

    sv_idx = _g(signal, "internal", "sv_idx")
    pred_pr = _g(satellite, "internal", "linearised", "pr")[sv_idx, :] + trop[sv_idx, :] + iono_sig_m

    sat_pco = _g(signal, "model", "sat_pco")
    if sat_pco is not None and np.size(sat_pco) > 0:
        pred_pr = pred_pr - sat_pco

    pco = _g(signal, "model", "pco")
    if pco is not None and np.size(pco) > 0:
        pred_pr = pred_pr + pco

    sat_pr_bias = _g(signal, "model", "sat_pr_bias")
    if sat_pr_bias is not None and np.size(sat_pr_bias) > 0:
        pred_pr = pred_pr + sat_pr_bias
        sat_phase_bias = _g(signal, "model", "sat_phase_bias")
        if phase_smooth and sat_phase_bias is not None and np.size(sat_phase_bias) > 0:
            pred_pr = pred_pr + sat_phase_bias

    cbv = _g(signal, "model", "cbv")
    if cbv is not None and np.size(cbv) > 0:
        pred_pr = pred_pr + cbv

    if phase_smooth:
        # MATLAB source references solver_proc.signal.pr_multipath from outer scope.
        # Keep behavior by checking optional key here.
        try:
            pr_multipath = _g(signal, "solver_proc", "signal", "pr_multipath")
            pred_pr = pred_pr + pr_multipath
        except Exception:
            pass
    else:
        sat_nadir_bias = _g(signal, "model", "sat_nadir_bias")
        if sat_nadir_bias is not None and np.size(sat_nadir_bias) > 0:
            pred_pr = pred_pr - sat_nadir_bias

    pred_dr = _g(satellite, "internal", "linearised", "doppler")[sv_idx, :]
    return pred_pr, pred_dr

