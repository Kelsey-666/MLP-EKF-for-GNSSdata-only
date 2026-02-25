"""Constants.m equivalent."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Constants:
    c: float = 299792458.0
    GM: float = 3986004.418e8
    earth_rot_rate: float = 7.2921151467e-5
    rad2deg: float = 180.0 / math.pi
    deg2rad: float = math.pi / 180.0
    secs_in_week: float = float(24 * 7 * 60 * 60)
    beidou_week_offset: int = 1356

