"""MATLAB utils migration: codeSVIDs/ecef2llh/ecef2nedRot/ecef2nedError/cleanStruct."""

from __future__ import annotations

from typing import Any
import numpy as np


def code_svids(sv_id: np.ndarray, gnss_id: np.ndarray, sig_id: np.ndarray, accs_id: np.ndarray) -> np.ndarray:
    sv = np.asarray(sv_id, dtype=np.int64)
    gn = np.asarray(gnss_id, dtype=np.int64)
    sg = np.asarray(sig_id, dtype=np.int64)
    ac = np.asarray(accs_id, dtype=np.int64)
    ext = sv + 256 * (gn + 256 * (sg + 256 * ac * (gn == 6)))
    ext = ext * (sg != 255)
    return ext.astype(np.int64)


def ecef2llh(xyz: np.ndarray) -> np.ndarray:
    # Vermeille direct transform (same as MATLAB file).
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError("ecef2llh expects 3-element vector.")
    x, y, z0 = arr
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    e4 = e2 * e2

    lon = np.degrees(np.arctan2(y, x))
    z = z0 / a
    p = (x * x + y * y) / (a * a)
    q = (1.0 - e2) * z * z
    r = (p + q - e4) / 6.0
    s = (e4 * p * q) / (4.0 * (r**3))
    t = (1.0 + s + np.sqrt(s * (2.0 + s))) ** (1.0 / 3.0)
    u = r * (1.0 + t + (1.0 / t))
    v = np.sqrt(u * u + q * e4)
    w = e2 * (u + v - q) / (2.0 * v)
    k = np.sqrt(u + v + w * w) - w
    d = k * np.sqrt(p) / (k + e2)
    lat = np.degrees(np.arctan2(z, d))
    h = a * (k + e2 - 1.0) / k * (np.sqrt(d * d + z * z))
    return np.array([lat, lon, h], dtype=float)


def ecef2ned_rot(latitude: float, longitude: float, already_rad: bool = False) -> np.ndarray:
    lat = float(latitude)
    lon = float(longitude)
    if not already_rad:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)
    ren = np.array(
        [
            [-sinlat * coslon, -sinlat * sinlon, coslat],
            [-sinlon, coslon, 0.0],
            [-coslat * coslon, -coslat * sinlon, -sinlat],
        ],
        dtype=float,
    )
    return ren


def ecef2ned_error(
    pos_E: np.ndarray,
    truth_pos_E: np.ndarray,
    vel_E: np.ndarray | None = None,
    truth_vel_E: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    dpos_ecef = np.asarray(pos_E, dtype=float) - np.asarray(truth_pos_E, dtype=float)
    dpos_ned = np.full_like(dpos_ecef, np.nan, dtype=float)
    dvel_ecef = None
    dvel_ned = None
    if vel_E is not None and truth_vel_E is not None:
        dvel_ecef = np.asarray(vel_E, dtype=float) - np.asarray(truth_vel_E, dtype=float)
        dvel_ned = np.full_like(dvel_ecef, np.nan, dtype=float)

    n2n2 = np.eye(3)
    for i in range(dpos_ecef.shape[0]):
        llh = ecef2llh(np.asarray(pos_E[i], dtype=float))
        ren = ecef2ned_rot(llh[0], llh[1], already_rad=False)
        dpos_ned[i, :] = (n2n2 @ ren @ dpos_ecef[i, :].reshape(3, 1)).reshape(3)
        if dvel_ecef is not None and dvel_ned is not None:
            dvel_ned[i, :] = (n2n2 @ ren @ dvel_ecef[i, :].reshape(3, 1)).reshape(3)

    return dpos_ned, dpos_ecef, dvel_ned, dvel_ecef


def clean_struct(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            cv = clean_struct(v)
            remove = False
            if cv is None:
                remove = True
            elif isinstance(cv, np.ndarray):
                if cv.size == 0:
                    remove = True
                elif np.issubdtype(cv.dtype, np.number) and np.all(np.isnan(cv)):
                    remove = True
            elif isinstance(cv, dict) and len(cv) == 0:
                remove = True
            if not remove:
                out[k] = cv
        return out
    return obj


def ismember_indices(values: np.ndarray, unique_values: np.ndarray) -> np.ndarray:
    mapping = {int(v): i for i, v in enumerate(np.asarray(unique_values, dtype=np.int64))}
    vals = np.asarray(values, dtype=np.int64)
    return np.array([mapping.get(int(v), -1) for v in vals], dtype=int)

