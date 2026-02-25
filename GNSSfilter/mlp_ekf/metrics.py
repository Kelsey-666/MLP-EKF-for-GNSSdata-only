"""Evaluation metrics for MLP-Kalmannet EKF outputs."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .models import FilterResultRow


def _rmse(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def compute_metrics(rows: Iterable[FilterResultRow]) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    by_split: Dict[str, List[FilterResultRow]] = {"all": rows}
    for split in ("train", "val", "test"):
        by_split[split] = [r for r in rows if r.split == split]

    metrics: Dict[str, Dict[str, float]] = {}
    for split, items in by_split.items():
        e = [r.enu_e_m for r in items]
        n = [r.enu_n_m for r in items]
        u = [r.enu_u_m for r in items]
        d2 = [
            math.sqrt(r.enu_e_m * r.enu_e_m + r.enu_n_m * r.enu_n_m)
            for r in items
            if math.isfinite(r.enu_e_m) and math.isfinite(r.enu_n_m)
        ]
        d3 = [r.enu_3d_m for r in items]
        metrics[split] = {
            "count": float(len(items)),
            "east_rmse_m": _rmse(e),
            "north_rmse_m": _rmse(n),
            "up_rmse_m": _rmse(u),
            "2d_rmse_m": _rmse(d2),
            "3d_rmse_m": _rmse(d3),
        }
    return metrics
