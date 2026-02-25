"""Torch-based MLP-Kalmannet EKF stepper for training/evaluation loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mlp_ekf.coarse_init import coarse_position_ls, geometric_range
from mlp_ekf.config import MLPEKFConfig
from mlp_ekf.geo import ecef_to_enu, ecef_to_geodetic
from mlp_ekf.models import EpochBatch

CLIGHT = 299792458.0


def _is_finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    if n % 2 == 1:
        return arr[n // 2]
    return 0.5 * (arr[n // 2 - 1] + arr[n // 2])


@dataclass
class StepResult:
    ok: bool
    pred_ecef: torch.Tensor
    gt_llh: torch.Tensor
    time_gps_s: float
    epoch: int
    split: str
    used_obs: int
    rejected_obs: int
    enu_e_m: float
    enu_n_m: float
    enu_u_m: float
    enu_3d_m: float


class TorchMLPEKF:
    def __init__(
        self,
        cfg: MLPEKFConfig,
        system_ids: Sequence[int],
        ibb_keys: Sequence[Tuple[int, int]],
        ref_sys_id: int,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = torch.double

        systems = sorted(set(int(s) for s in system_ids)) or [0]
        self.system_ids = systems
        self.ref_sys_id = int(ref_sys_id) if int(ref_sys_id) in systems else (0 if 0 in systems else systems[0])

        self.inter_ids = [sid for sid in systems if sid != self.ref_sys_id]
        self.inter_map: Dict[int, int] = {sid: i for i, sid in enumerate(self.inter_ids)}

        keys = sorted(set((int(k[0]), int(k[1])) for k in ibb_keys))
        self.ibb_keys = keys
        self.ibb_map: Dict[Tuple[int, int], int] = {k: i for i, k in enumerate(self.ibb_keys)}

        self.cb_idx = 6
        self.cd_idx = 7
        self.inter_start = 8
        self.ibb_start = self.inter_start + len(self.inter_ids)
        self.nx = 8 + len(self.inter_ids) + len(self.ibb_keys)

        self.x = torch.zeros((self.nx,), dtype=self.dtype, device=self.device)
        self.P = torch.zeros((self.nx, self.nx), dtype=self.dtype, device=self.device)
        self.last_time: Optional[float] = None
        self.initialized = False
        self._reset_covariance()

    def _inter_idx(self, sys_id: int) -> Optional[int]:
        slot = self.inter_map.get(int(sys_id))
        return None if slot is None else self.inter_start + slot

    def _ibb_idx(self, sys_id: int, sig_id: int) -> Optional[int]:
        slot = self.ibb_map.get((int(sys_id), int(sig_id)))
        return None if slot is None else self.ibb_start + slot

    def _reset_covariance(self) -> None:
        self.P.zero_()
        for i in range(3):
            self.P[i, i] = self.cfg.init_pos_std_m**2
        for i in range(3, 6):
            self.P[i, i] = self.cfg.init_vel_std_mps**2
        self.P[self.cb_idx, self.cb_idx] = self.cfg.init_cb_std_m**2
        self.P[self.cd_idx, self.cd_idx] = self.cfg.init_cd_std_mps**2
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                self.P[iidx, iidx] = self.cfg.init_cb_std_m**2
        for sid, sig in self.ibb_keys:
            bidx = self._ibb_idx(sid, sig)
            if bidx is not None:
                self.P[bidx, bidx] = self.cfg.init_cb_std_m**2

    def _gate_measurement(self, cno_dbhz: float, elev_deg: float) -> bool:
        if not _is_finite(cno_dbhz) or cno_dbhz < self.cfg.cno_min_dbhz:
            return False
        if not _is_finite(elev_deg) or elev_deg < self.cfg.elev_min_deg:
            return False
        return True

    def _measurement_variance_pr(self, obs_noise_m: float, learned_scale: Optional[torch.Tensor]) -> torch.Tensor:
        floor_var = torch.tensor(self.cfg.pr_noise_floor_m**2, dtype=self.dtype, device=self.device)
        if _is_finite(obs_noise_m):
            base_var = torch.maximum(torch.tensor(obs_noise_m * obs_noise_m, dtype=self.dtype, device=self.device), floor_var)
        else:
            base_var = floor_var
        if learned_scale is not None:
            min_s = float(self.cfg.learned_r_min_scale)
            max_s = float(self.cfg.learned_r_max_scale)
            scale = torch.clamp(learned_scale.to(self.dtype), min=min_s, max=max_s)
            return torch.maximum(base_var * scale, floor_var)
        return base_var

    def _apply_limits(self) -> None:
        x_new = self.x.clone()
        max_v = abs(self.cfg.max_speed_mps)
        x_new[3:6] = torch.clamp(x_new[3:6], min=-max_v, max=max_v)
        max_cb = abs(self.cfg.max_clock_bias_m)
        max_cd = abs(self.cfg.max_clock_drift_mps)
        x_new[self.cb_idx] = torch.clamp(x_new[self.cb_idx], min=-max_cb, max=max_cb)
        x_new[self.cd_idx] = torch.clamp(x_new[self.cd_idx], min=-max_cd, max=max_cd)
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                x_new[iidx] = torch.clamp(x_new[iidx], min=-max_cb, max=max_cb)
        for sid, sig in self.ibb_keys:
            bidx = self._ibb_idx(sid, sig)
            if bidx is not None:
                x_new[bidx] = torch.clamp(x_new[bidx], min=-max_cb, max=max_cb)
        self.x = x_new

    def _predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        freeze_vel = self.cfg.freeze_velocity_state
        freeze_cd = self.cfg.freeze_clock_drift_state

        if freeze_vel:
            self.x[3:6] = 0.0
        else:
            self.x[0:3] = self.x[0:3] + self.x[3:6] * dt

        if freeze_cd:
            self.x[self.cd_idx] = 0.0
        else:
            self.x[self.cb_idx] = self.x[self.cb_idx] + self.x[self.cd_idx] * dt

        F = torch.eye(self.nx, dtype=self.dtype, device=self.device)
        if not freeze_vel:
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
        if not freeze_cd:
            F[self.cb_idx, self.cd_idx] = dt

        Q = torch.zeros((self.nx, self.nx), dtype=self.dtype, device=self.device)
        for i in range(3):
            Q[i, i] = self.cfg.sig_qp**2 * dt
        for i in range(3, 6):
            Q[i, i] = self.cfg.sig_qv**2 * dt
        Q[self.cb_idx, self.cb_idx] = self.cfg.sig_qcb**2 * dt
        if not freeze_cd:
            Q[self.cd_idx, self.cd_idx] = self.cfg.sig_qcd**2 * dt
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                Q[iidx, iidx] = self.cfg.sig_q_inter**2 * dt
        for sid, sig in self.ibb_keys:
            bidx = self._ibb_idx(sid, sig)
            if bidx is not None:
                Q[bidx, bidx] = self.cfg.sig_q_ibb**2 * dt

        self.P = F @ self.P @ F.transpose(0, 1) + Q

    def _batch_update(self, H: torch.Tensor, v: torch.Tensor, r_diag: torch.Tensor) -> bool:
        if H.numel() == 0:
            return False
        try:
            PHt = self.P @ H.transpose(0, 1)
            S = H @ PHt + torch.diag(r_diag)
            Sinv = torch.linalg.inv(S)
            K = PHt @ Sinv
        except RuntimeError:
            return False
        x_new = self.x + K @ v
        I = torch.eye(self.nx, dtype=self.dtype, device=self.device)
        IKH = I - K @ H
        P_new = IKH @ self.P @ IKH.transpose(0, 1) + K @ torch.diag(r_diag) @ K.transpose(0, 1)
        P_new = 0.5 * (P_new + P_new.transpose(0, 1))
        self.x = x_new
        self.P = P_new
        return True

    def _bias_candidates(
        self,
        batch: EpochBatch,
        rx_xyz: Tuple[float, float, float],
        gated_only: bool = True,
    ) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float]]:
        all_vals: List[float] = []
        by_sys: Dict[int, List[float]] = {sid: [] for sid in self.system_ids}
        by_key: Dict[Tuple[int, int], List[float]] = {k: [] for k in self.ibb_keys}

        for obs in batch.observations:
            sid = int(obs.gnss_id_raw)
            if sid not in by_sys:
                continue
            if gated_only and not self._gate_measurement(obs.cno_dbhz, obs.elev_deg):
                continue
            rho, _ = geometric_range((obs.sat_pos_x_m, obs.sat_pos_y_m, obs.sat_pos_z_m), rx_xyz)
            cb_i = obs.pr_m - rho + obs.sat_bias_m
            if not _is_finite(cb_i):
                continue
            all_vals.append(float(cb_i))
            by_sys[sid].append(float(cb_i))
            key = (sid, int(obs.sig_id))
            if key in by_key:
                by_key[key].append(float(cb_i))

        cb = _median(all_vals)
        inter_map: Dict[int, float] = {}
        for sid in self.inter_ids:
            vals = by_sys.get(sid, [])
            if vals:
                inter_map[sid] = _median(vals) - cb
        ibb_map: Dict[Tuple[int, int], float] = {}
        for key in self.ibb_keys:
            vals = by_key.get(key, [])
            sys_vals = by_sys.get(key[0], [])
            if vals and sys_vals:
                ibb_map[key] = _median(vals) - _median(sys_vals)
        return cb, inter_map, ibb_map

    def _init_if_needed(self, batch: EpochBatch) -> bool:
        if self.initialized:
            return True

        truth_mode = self.cfg.init_mode.lower() == "truth"
        if truth_mode and batch.observations:
            o0 = batch.observations[0]
            if _is_finite(o0.gt_ecef_x_m) and _is_finite(o0.gt_ecef_y_m) and _is_finite(o0.gt_ecef_z_m):
                tx, ty, tz = float(o0.gt_ecef_x_m), float(o0.gt_ecef_y_m), float(o0.gt_ecef_z_m)
                cb, inter_map, ibb_map = self._bias_candidates(batch, (tx, ty, tz), gated_only=True)
                self.x.zero_()
                self.x[0:3] = torch.tensor([tx, ty, tz], dtype=self.dtype, device=self.device)
                self.x[self.cb_idx] = cb
                self.x[self.cd_idx] = 0.0
                for sid in self.inter_ids:
                    iidx = self._inter_idx(sid)
                    if iidx is not None:
                        self.x[iidx] = inter_map.get(sid, 0.0)
                for sid, sig in self.ibb_keys:
                    bidx = self._ibb_idx(sid, sig)
                    if bidx is not None:
                        self.x[bidx] = ibb_map.get((sid, sig), 0.0)
                self.last_time = float(batch.time_gps_s)
                self.initialized = True
                return True

        coarse = coarse_position_ls(
            observations=batch.observations,
            cno_min_dbhz=self.cfg.cno_min_dbhz,
            elev_min_deg=self.cfg.elev_min_deg,
            max_iter=self.cfg.max_init_iter,
        )
        if coarse is None:
            return False
        cx, cy, cz, cb0 = coarse
        cb, inter_map, ibb_map = self._bias_candidates(batch, (cx, cy, cz), gated_only=True)
        self.x.zero_()
        self.x[0:3] = torch.tensor([cx, cy, cz], dtype=self.dtype, device=self.device)
        self.x[self.cb_idx] = cb if _is_finite(cb) else cb0
        self.x[self.cd_idx] = 0.0
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                self.x[iidx] = inter_map.get(sid, 0.0)
        for sid, sig in self.ibb_keys:
            bidx = self._ibb_idx(sid, sig)
            if bidx is not None:
                self.x[bidx] = ibb_map.get((sid, sig), 0.0)
        self.last_time = float(batch.time_gps_s)
        self.initialized = True
        return True

    def _pdop_from_h(self, h_rows: Sequence[torch.Tensor]) -> float:
        if len(h_rows) < 4:
            return float("nan")
        g = torch.stack([torch.stack([h[0], h[1], h[2], torch.tensor(1.0, dtype=self.dtype, device=self.device)]) for h in h_rows], dim=0)
        try:
            q = torch.linalg.inv(g.transpose(0, 1) @ g)
            pdop_sq = float((q[0, 0] + q[1, 1] + q[2, 2]).detach().cpu().item())
            if pdop_sq <= 0.0 or not math.isfinite(pdop_sq):
                return float("nan")
            return math.sqrt(pdop_sq)
        except RuntimeError:
            return float("nan")

    def _pdop_r_inflation(self, pdop: float) -> float:
        if not self.cfg.use_pdop_r_inflation or not math.isfinite(pdop):
            return 1.0
        if pdop <= self.cfg.pdop_threshold:
            return 1.0
        extra = pdop - self.cfg.pdop_threshold
        factor = 1.0 + self.cfg.pdop_r_inflation_alpha * extra
        return min(self.cfg.pdop_r_inflation_max, max(1.0, factor))

    def process_epoch(
        self,
        batch: EpochBatch,
        learned_r_diag: Optional[torch.Tensor],
        learned_bias: Optional[torch.Tensor],
    ) -> StepResult:
        if not self._init_if_needed(batch):
            pred = torch.zeros((3,), dtype=self.dtype, device=self.device)
            gt = torch.tensor([0.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
            return StepResult(False, pred, gt, float(batch.time_gps_s), int(batch.epoch), str(batch.split), 0, 0, float("nan"), float("nan"), float("nan"), float("nan"))

        dt = 0.0 if self.last_time is None else float(batch.time_gps_s - self.last_time)
        self._predict(dt)
        self.last_time = float(batch.time_gps_s)
        self._apply_limits()

        cand_h: List[torch.Tensor] = []
        cand_v: List[torch.Tensor] = []
        cand_r: List[torch.Tensor] = []
        rejected = 0

        for i, obs in enumerate(batch.observations):
            sid = int(obs.gnss_id_raw)
            if sid not in self.system_ids:
                continue
            if not self._gate_measurement(obs.cno_dbhz, obs.elev_deg):
                continue

            rho, los = geometric_range((obs.sat_pos_x_m, obs.sat_pos_y_m, obs.sat_pos_z_m), (float(self.x[0]), float(self.x[1]), float(self.x[2])))
            iidx = self._inter_idx(sid)
            bidx = self._ibb_idx(sid, int(obs.sig_id))

            h = torch.zeros((self.nx,), dtype=self.dtype, device=self.device)
            h[0] = -los[0]
            h[1] = -los[1]
            h[2] = -los[2]
            h[self.cb_idx] = 1.0
            if iidx is not None:
                h[iidx] = 1.0
            if bidx is not None:
                h[bidx] = 1.0

            if learned_bias is not None and i < int(learned_bias.shape[0]):
                l_bias = learned_bias[i].to(self.dtype)
            else:
                l_bias = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            l_r = None
            if learned_r_diag is not None and i < int(learned_r_diag.shape[0]):
                l_r = learned_r_diag[i]

            inter = self.x[iidx] if iidx is not None else torch.tensor(0.0, dtype=self.dtype, device=self.device)
            ibb = self.x[bidx] if bidx is not None else torch.tensor(0.0, dtype=self.dtype, device=self.device)
            pred = torch.tensor(rho - obs.sat_bias_m, dtype=self.dtype, device=self.device) + self.x[self.cb_idx] + inter + ibb
            meas = torch.tensor(obs.pr_m, dtype=self.dtype, device=self.device) + l_bias
            innov = meas - pred
            r_var = self._measurement_variance_pr(obs.pr_noise_m, l_r)
            cand_h.append(h)
            cand_v.append(innov)
            cand_r.append(r_var)

        pdop = self._pdop_from_h(cand_h)
        r_inflation = self._pdop_r_inflation(pdop)

        if self.cfg.use_clock_jump_compensation and cand_v:
            step_m = CLIGHT * (self.cfg.clock_jump_ms / 1000.0)
            vals = [float(v.detach().cpu().item()) for v in cand_v]
            jump = round(_median(vals) / step_m) * step_m
            if abs(jump) > 0.0:
                self.x[self.cb_idx] = self.x[self.cb_idx] + jump
                for j in range(len(cand_v)):
                    cand_v[j] = cand_v[j] - jump

        accepted_h: List[torch.Tensor] = []
        accepted_v: List[torch.Tensor] = []
        accepted_r: List[torch.Tensor] = []
        scored: List[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for h, innov, r_var in zip(cand_h, cand_v, cand_r):
            r_eff = r_var * r_inflation
            ph = self.P @ h
            s_val = r_eff + torch.dot(h, ph)
            sigma = torch.sqrt(torch.clamp(s_val, min=1.0e-12))
            norm = float((torch.abs(innov) / sigma).detach().cpu().item())
            scored.append((norm, innov, r_eff, h))
            gate = float(self.cfg.innovation_gate_sigma) * float(sigma.detach().cpu().item())
            if abs(float(innov.detach().cpu().item())) > gate:
                rejected += 1
                continue
            if self.cfg.use_fde and abs(float(innov.detach().cpu().item())) > float(self.cfg.fde_sigma) * float(sigma.detach().cpu().item()):
                rejected += 1
                continue
            accepted_h.append(h)
            accepted_v.append(innov)
            accepted_r.append(r_eff)

        used = 0
        if accepted_v:
            H = torch.stack(accepted_h, dim=0)
            v = torch.stack(accepted_v, dim=0)
            Rdiag = torch.stack(accepted_r, dim=0)
            ok = self._batch_update(H, v, Rdiag)
            if ok:
                used = int(v.shape[0])
            else:
                rejected += int(v.shape[0])

        if used == 0 and self.cfg.starvation_fallback_enabled and scored:
            scored.sort(key=lambda t: t[0])
            fb_h: List[torch.Tensor] = []
            fb_v: List[torch.Tensor] = []
            fb_r: List[torch.Tensor] = []
            for norm, innov, r_var, h in scored:
                if norm > self.cfg.starvation_max_norm_innov:
                    break
                fb_h.append(h)
                fb_v.append(innov)
                fb_r.append(r_var * self.cfg.starvation_r_scale)
                if len(fb_v) >= max(1, int(self.cfg.starvation_min_updates)):
                    break
            if fb_v:
                H = torch.stack(fb_h, dim=0)
                v = torch.stack(fb_v, dim=0)
                Rdiag = torch.stack(fb_r, dim=0)
                ok = self._batch_update(H, v, Rdiag)
                if ok:
                    used = int(v.shape[0])
                else:
                    rejected += int(v.shape[0])

        pred_ecef = self.x[:3]
        gt = _gt_tensor_from_batch(batch, self.device, self.dtype)
        enu = _enu_error(pred_ecef, gt)

        # Truncated BPTT across epochs.
        self.x = self.x.detach().clone()
        self.P = self.P.detach().clone()
        self._apply_limits()

        return StepResult(
            ok=True,
            pred_ecef=pred_ecef,
            gt_llh=gt,
            time_gps_s=float(batch.time_gps_s),
            epoch=int(batch.epoch),
            split=str(batch.split),
            used_obs=used,
            rejected_obs=rejected,
            enu_e_m=enu[0],
            enu_n_m=enu[1],
            enu_u_m=enu[2],
            enu_3d_m=math.sqrt(enu[0] * enu[0] + enu[1] * enu[1] + enu[2] * enu[2]),
        )


def _gt_tensor_from_batch(batch: EpochBatch, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not batch.observations:
        return torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    o0 = batch.observations[0]
    return torch.tensor([float(o0.gt_lat_deg), float(o0.gt_lon_deg), float(o0.gt_h_m)], dtype=dtype, device=device)


def _enu_error(pred_ecef: torch.Tensor, gt_llh: torch.Tensor) -> Tuple[float, float, float]:
    x = float(pred_ecef[0].detach().cpu().item())
    y = float(pred_ecef[1].detach().cpu().item())
    z = float(pred_ecef[2].detach().cpu().item())
    lat = float(gt_llh[0].detach().cpu().item())
    lon = float(gt_llh[1].detach().cpu().item())
    h = float(gt_llh[2].detach().cpu().item())
    if not (_is_finite(lat) and _is_finite(lon) and _is_finite(h)):
        return float("nan"), float("nan"), float("nan")
    return ecef_to_enu(x, y, z, lat, lon, h)


def geodetic_from_pred(pred_ecef: torch.Tensor) -> Tuple[float, float, float]:
    return ecef_to_geodetic(
        float(pred_ecef[0].detach().cpu().item()),
        float(pred_ecef[1].detach().cpu().item()),
        float(pred_ecef[2].detach().cpu().item()),
    )
