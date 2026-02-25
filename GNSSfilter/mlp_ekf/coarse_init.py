"""Coarse least-squares position initializer."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .linalg import solve_linear_system
from .models import Observation

CLIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5


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
    return r + sagnac, (dx / r, dy / r, dz / r)


def coarse_position_ls(
    observations: List[Observation],
    cno_min_dbhz: float,
    elev_min_deg: float,
    max_iter: int = 8,
) -> Optional[Tuple[float, float, float, float]]:
    use = [
        o
        for o in observations
        if o.pr_m > 0.0 and o.cno_dbhz >= cno_min_dbhz and o.elev_deg >= elev_min_deg
    ]
    if len(use) < 4:
        return None

    x, y, z, cb = 0.0, 0.0, 0.0, 0.0
    for _ in range(max_iter):
        nmat = [[0.0] * 4 for _ in range(4)]
        uvec = [0.0] * 4
        for obs in use:
            sat_pos = (obs.sat_pos_x_m, obs.sat_pos_y_m, obs.sat_pos_z_m)
            rho, los = geometric_range(sat_pos, (x, y, z))
            pred = rho + cb - obs.sat_bias_m
            v = obs.pr_m - pred
            h = [-los[0], -los[1], -los[2], 1.0]
            for r in range(4):
                uvec[r] += h[r] * v
                for c in range(4):
                    nmat[r][c] += h[r] * h[c]
        dx = solve_linear_system(nmat, uvec)
        if dx is None:
            return None
        x += dx[0]
        y += dx[1]
        z += dx[2]
        cb += dx[3]
        if math.sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) < 1e-4:
            break
    return x, y, z, cb
