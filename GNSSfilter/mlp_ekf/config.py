"""Configuration for MLP-Kalmannet EKF."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class MLPEKFConfig:
    cno_min_dbhz: float = 24.0
    elev_min_deg: float = 12.0
    init_mode: str = "truth"  # coarse | truth
    use_fde: bool = False
    fde_sigma: float = 3.0
    use_clock_jump_compensation: bool = True
    clock_jump_ms: float = 0.5
    sig_qp: float = 1.0845183309789115
    sig_qv: float = 2.877122406965902
    sig_qcb: float = 0.4863722501147573
    sig_qcd: float = 0.009683001950682902
    sig_q_inter: float = 0.01
    sig_q_ibb: float = 0.01
    init_pos_std_m: float = 1000.0
    init_vel_std_mps: float = 10.0
    init_cb_std_m: float = 1000.0
    init_cd_std_mps: float = 100.0
    pr_noise_floor_m: float = 1.0
    max_init_iter: int = 8
    innovation_gate_sigma: float = 6.0
    max_speed_mps: float = 30.0
    max_clock_bias_m: float = 1.0e7
    max_clock_drift_mps: float = 5000.0
    freeze_velocity_state: bool = True
    freeze_clock_drift_state: bool = False
    use_engineered_r_model: bool = True
    r_blend_alpha: float = 0.5
    r_cno_ref_dbhz: float = 45.0
    r_cno_power: float = 1.0
    r_elev_power: float = 1.0
    r_scale_gps: float = 1.0
    r_scale_gal: float = 1.15
    r_scale_bds: float = 1.10
    r_scale_glo: float = 1.20
    learned_r_min_scale: float = 0.5
    learned_r_max_scale: float = 20.0
    use_pdop_r_inflation: bool = True
    pdop_threshold: float = 8.0
    pdop_r_inflation_alpha: float = 0.3
    pdop_r_inflation_max: float = 8.0
    starvation_fallback_enabled: bool = True
    starvation_min_updates: int = 4
    starvation_max_norm_innov: float = 25.0
    starvation_r_scale: float = 10.0


def load_config(path: str | None) -> MLPEKFConfig:
    cfg = MLPEKFConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    with p.open("r", encoding="utf-8-sig") as f:
        data: Dict[str, Any] = json.load(f)
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
