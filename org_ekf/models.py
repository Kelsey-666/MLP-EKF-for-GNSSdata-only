"""Data models aligned with MATLAB classes/structs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np


def _empty(shape: tuple[int, ...], dtype=float) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


@dataclass
class GnssMeasurements:
    tow: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    week: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    pos_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    vel_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    pos_std_N: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    pos_ref_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    epoch: np.ndarray = field(default_factory=lambda: _empty((0,), int))
    pr: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    cr: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    dr: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    gnss_id: np.ndarray = field(default_factory=lambda: _empty((0,), int))
    sv_id: np.ndarray = field(default_factory=lambda: _empty((0,), int))
    sig_id: np.ndarray = field(default_factory=lambda: _empty((0,), int))
    accs_id: np.ndarray = field(default_factory=lambda: _empty((0,), int))
    iono: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    pr_noise: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    cr_noise: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    dr_noise: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    sat_pos_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    sat_vel_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    sat_bias: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    sat_drift: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    track_idx: np.ndarray = field(default_factory=lambda: _empty((0,), int))
    elev: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    azim: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    wavelength: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    cno: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    xtId: np.ndarray = field(default_factory=lambda: _empty((0,), np.int64))
    xtIdSv: np.ndarray = field(default_factory=lambda: _empty((0,), np.int64))

    def validate(self) -> None:
        if self.epoch.size == 0:
            return
        n = self.epoch.size
        fields_1d = [
            "pr",
            "dr",
            "gnss_id",
            "sv_id",
            "sig_id",
            "accs_id",
            "pr_noise",
            "dr_noise",
            "sat_bias",
            "sat_drift",
            "elev",
            "azim",
            "cno",
        ]
        for name in fields_1d:
            val = getattr(self, name)
            if val.size != n:
                raise ValueError(f"{name} size mismatch: {val.size} != {n}")


@dataclass
class Truth:
    pos_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    vel_E: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    acc_B: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    lat_lon_alt: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    euler_N_B: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    vel_std_N: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    pos_std_N: np.ndarray = field(default_factory=lambda: _empty((0, 3), float))
    distance: np.ndarray = field(default_factory=lambda: _empty((0,), float))
    is_static: np.ndarray = field(default_factory=lambda: _empty((0,), bool))
    extrapolated: np.ndarray = field(default_factory=lambda: _empty((0,), bool))


@dataclass
class StateIndex:
    pos: np.ndarray
    vel: np.ndarray
    posvel: np.ndarray
    clock: np.ndarray
    ibb: np.ndarray
    inter_gnss: np.ndarray


@dataclass
class KFState:
    x: np.ndarray
    P: np.ndarray
    state_indx: StateIndex
    num_sv: int
    num_obs: int
    inter_gnss: int
    ibb: int
    num_states: int


@dataclass
class ObsId:
    xtIdSv: np.ndarray
    xtId: np.ndarray
    interGnss: np.ndarray
    ibb: np.ndarray


@dataclass
class ObsMap:
    sv: np.ndarray
    interGnss: np.ndarray
    ibb: np.ndarray
    chn: np.ndarray


@dataclass
class ObservationBlock:
    H: np.ndarray
    z: np.ndarray
    R: np.ndarray


@dataclass
class FilterOutput:
    values: Dict[str, Any]
