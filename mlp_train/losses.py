"""Losses aligned with LF training style (ENU-based focal-like loss)."""

from __future__ import annotations

import torch
from torch import nn


WGS84_A = 6378137.0
WGS84_B = 6356752.3142
WGS84_F = (WGS84_A - WGS84_B) / WGS84_A
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic2ecef_torch(lat_deg: torch.Tensor, lon_deg: torch.Tensor, h_m: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    sin_lat = torch.sin(lat)
    n = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (h_m + n) * torch.cos(lat) * torch.cos(lon)
    y = (h_m + n) * torch.cos(lat) * torch.sin(lon)
    z = (h_m + (1.0 - WGS84_E2) * n) * sin_lat
    return torch.stack([x, y, z], dim=-1)


def ecef2enu_torch(xyz: torch.Tensor, ref_llh: torch.Tensor) -> torch.Tensor:
    # xyz: [B,3], ref_llh: [B,3] in deg/deg/m
    ref_ecef = geodetic2ecef_torch(ref_llh[:, 0], ref_llh[:, 1], ref_llh[:, 2])
    dx = xyz - ref_ecef
    lat = torch.deg2rad(ref_llh[:, 0])
    lon = torch.deg2rad(ref_llh[:, 1])
    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)
    sin_lon = torch.sin(lon)
    cos_lon = torch.cos(lon)

    east = -sin_lon * dx[:, 0] + cos_lon * dx[:, 1]
    north = -sin_lat * cos_lon * dx[:, 0] - sin_lat * sin_lon * dx[:, 1] + cos_lat * dx[:, 2]
    up = cos_lat * cos_lon * dx[:, 0] + cos_lat * sin_lon * dx[:, 1] + sin_lat * dx[:, 2]
    return torch.stack([east, north, up], dim=-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 1.5, dynamic_gamma: bool = True, scale_factor: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dynamic_gamma = dynamic_gamma
        self.scale_factor = scale_factor

    def forward(self, pred_ecef: torch.Tensor, gt_llh: torch.Tensor) -> torch.Tensor:
        # pred_ecef: [B,3], gt_llh: [B,3]
        enu = ecef2enu_torch(pred_ecef, gt_llh)
        dist = torch.norm(enu, p=2, dim=1)
        base = dist
        if self.dynamic_gamma:
            dyn_gamma = self.gamma * torch.exp(-self.scale_factor * base.detach())
        else:
            dyn_gamma = torch.full_like(base, self.gamma)
        max_base = torch.clamp(base.max().detach(), min=1.0e-8)
        ratio = torch.clamp(base / max_base, min=1.0e-8, max=1.0 - 1.0e-8)
        inner = torch.clamp(1.0 - torch.sqrt(ratio), min=1.0e-6, max=1.0)
        focal_weight = torch.exp(dyn_gamma * torch.log(inner))
        loss = self.alpha * focal_weight * base
        return torch.sqrt(torch.mean(loss * loss) + 1.0e-12)


def rmse_loss(pred_ecef: torch.Tensor, gt_llh: torch.Tensor) -> torch.Tensor:
    enu = ecef2enu_torch(pred_ecef, gt_llh)
    dist = torch.norm(enu, p=2, dim=1)
    return torch.sqrt(torch.mean(dist * dist) + 1.0e-12)
