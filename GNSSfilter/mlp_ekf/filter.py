"""MLP-Kalmannet EKF implementation (Python, no external deps)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .coarse_init import coarse_position_ls, geometric_range
from .config import MLPEKFConfig
from .geo import ecef_to_enu, ecef_to_geodetic
from .linalg import eye, mat_add, mat_mul, mat_transpose, solve_linear_system, zeros
from .models import EpochBatch, FilterResultRow, UpdateDebug

CLIGHT = 299792458.0


def _is_finite(value: float) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    if n % 2 == 1:
        return arr[n // 2]
    return 0.5 * (arr[n // 2 - 1] + arr[n // 2])


class MLPEKF:
    """State: [x,y,z,vx,vy,vz,cb,cd,inter_gnss...,ibb...]."""

    def __init__(
        self,
        cfg: Optional[MLPEKFConfig] = None,
        system_ids: Optional[Sequence[int]] = None,
        ibb_keys: Optional[Sequence[Tuple[int, int]]] = None,
        ref_sys_id: Optional[int] = None,
    ) -> None:
        self.cfg = cfg or MLPEKFConfig()
        systems = sorted(set(int(s) for s in (system_ids or [0])))
        if not systems:
            systems = [0]
        self.system_ids = systems
        if ref_sys_id is None:
            self.ref_sys_id = 0 if 0 in systems else systems[0]
        else:
            self.ref_sys_id = int(ref_sys_id)
            if self.ref_sys_id not in systems:
                self.ref_sys_id = systems[0]

        self.inter_ids = [sid for sid in systems if sid != self.ref_sys_id]
        self.inter_map: Dict[int, int] = {sid: i for i, sid in enumerate(self.inter_ids)}

        keys = sorted(set((int(k[0]), int(k[1])) for k in (ibb_keys or [])))
        self.ibb_keys = keys
        self.ibb_map: Dict[Tuple[int, int], int] = {k: i for i, k in enumerate(self.ibb_keys)}

        self.cb_idx = 6
        self.cd_idx = 7
        self.inter_start = 8
        self.ibb_start = self.inter_start + len(self.inter_ids)
        self.nx = 8 + len(self.inter_ids) + len(self.ibb_keys)

        self.x = [0.0] * self.nx
        self.P = zeros(self.nx, self.nx)
        self.initialized = False
        self.last_time: Optional[float] = None
        self._reset_covariance()

    def _inter_idx(self, sys_id: int) -> Optional[int]:
        slot = self.inter_map.get(int(sys_id))
        if slot is None:
            return None
        return self.inter_start + slot

    def _ibb_idx(self, sys_id: int, sig_id: int) -> Optional[int]:
        slot = self.ibb_map.get((int(sys_id), int(sig_id)))
        if slot is None:
            return None
        return self.ibb_start + slot

    def _reset_covariance(self) -> None:
        self.P = zeros(self.nx, self.nx)
        for i in range(3):
            self.P[i][i] = self.cfg.init_pos_std_m**2
        for i in range(3, 6):
            self.P[i][i] = self.cfg.init_vel_std_mps**2
        self.P[self.cb_idx][self.cb_idx] = self.cfg.init_cb_std_m**2
        self.P[self.cd_idx][self.cd_idx] = self.cfg.init_cd_std_mps**2
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                self.P[iidx][iidx] = self.cfg.init_cb_std_m**2
        for key in self.ibb_keys:
            bidx = self._ibb_idx(key[0], key[1])
            if bidx is not None:
                self.P[bidx][bidx] = self.cfg.init_cb_std_m**2

    def _predict(self, dt: float) -> None:
        if dt <= 0.0:
            return

        freeze_vel = self.cfg.freeze_velocity_state
        freeze_cd = self.cfg.freeze_clock_drift_state

        if not freeze_vel:
            self.x[0] += self.x[3] * dt
            self.x[1] += self.x[4] * dt
            self.x[2] += self.x[5] * dt
        else:
            self.x[3], self.x[4], self.x[5] = 0.0, 0.0, 0.0

        if not freeze_cd:
            self.x[self.cb_idx] += self.x[self.cd_idx] * dt
        else:
            self.x[self.cd_idx] = 0.0

        F = eye(self.nx)
        if not freeze_vel:
            F[0][3] = dt
            F[1][4] = dt
            F[2][5] = dt
        if not freeze_cd:
            F[self.cb_idx][self.cd_idx] = dt

        Q = zeros(self.nx, self.nx)
        for i in range(3):
            Q[i][i] = self.cfg.sig_qp**2 * dt
        for i in range(3, 6):
            Q[i][i] = self.cfg.sig_qv**2 * dt
        Q[self.cb_idx][self.cb_idx] = self.cfg.sig_qcb**2 * dt
        if not freeze_cd:
            Q[self.cd_idx][self.cd_idx] = self.cfg.sig_qcd**2 * dt
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                Q[iidx][iidx] = self.cfg.sig_q_inter**2 * dt
        for key in self.ibb_keys:
            bidx = self._ibb_idx(key[0], key[1])
            if bidx is not None:
                Q[bidx][bidx] = self.cfg.sig_q_ibb**2 * dt

        FP = mat_mul(F, self.P)
        self.P = mat_add(mat_mul(FP, mat_transpose(F)), Q)

    def _measurement_variance_pr(self, obs_noise_m: float, learned_r_scale: Optional[float]) -> float:
        floor_var = self.cfg.pr_noise_floor_m * self.cfg.pr_noise_floor_m
        if _is_finite(obs_noise_m):
            base_var = max(obs_noise_m * obs_noise_m, floor_var)
        else:
            base_var = floor_var
        if learned_r_scale is not None and _is_finite(learned_r_scale):
            scale = max(self.cfg.learned_r_min_scale, min(self.cfg.learned_r_max_scale, float(learned_r_scale)))
            return max(base_var * scale, floor_var)
        return base_var

    def _is_state_bad(self) -> bool:
        for v in self.x:
            if not math.isfinite(v):
                return True
        pos_norm = math.sqrt(self.x[0] * self.x[0] + self.x[1] * self.x[1] + self.x[2] * self.x[2])
        if pos_norm > 1.0e8:
            return True
        max_cb = 10.0 * self.cfg.max_clock_bias_m
        max_cd = 10.0 * self.cfg.max_clock_drift_mps
        if abs(self.x[self.cb_idx]) > max_cb:
            return True
        if abs(self.x[self.cd_idx]) > max_cd:
            return True
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None and abs(self.x[iidx]) > max_cb:
                return True
        for key in self.ibb_keys:
            bidx = self._ibb_idx(key[0], key[1])
            if bidx is not None and abs(self.x[bidx]) > max_cb:
                return True
        return False

    def _apply_state_limits(self) -> None:
        max_v = abs(self.cfg.max_speed_mps)
        self.x[3] = max(-max_v, min(max_v, self.x[3]))
        self.x[4] = max(-max_v, min(max_v, self.x[4]))
        self.x[5] = max(-max_v, min(max_v, self.x[5]))
        max_cb = abs(self.cfg.max_clock_bias_m)
        max_cd = abs(self.cfg.max_clock_drift_mps)
        self.x[self.cb_idx] = max(-max_cb, min(max_cb, self.x[self.cb_idx]))
        self.x[self.cd_idx] = max(-max_cd, min(max_cd, self.x[self.cd_idx]))
        for sid in self.inter_ids:
            iidx = self._inter_idx(sid)
            if iidx is not None:
                self.x[iidx] = max(-max_cb, min(max_cb, self.x[iidx]))
        for key in self.ibb_keys:
            bidx = self._ibb_idx(key[0], key[1])
            if bidx is not None:
                self.x[bidx] = max(-max_cb, min(max_cb, self.x[bidx]))

    def _scalar_update(self, h: List[float], innovation: float, r_var: float) -> bool:
        ph = [0.0] * self.nx
        for i in range(self.nx):
            s = 0.0
            for j in range(self.nx):
                s += self.P[i][j] * h[j]
            ph[i] = s
        s_val = r_var + sum(h[i] * ph[i] for i in range(self.nx))
        if s_val <= 1e-12 or not math.isfinite(s_val):
            return False
        k = [ph[i] / s_val for i in range(self.nx)]
        for i in range(self.nx):
            self.x[i] += k[i] * innovation
        i_kh = eye(self.nx)
        for i in range(self.nx):
            for j in range(self.nx):
                i_kh[i][j] -= k[i] * h[j]
        tmp = mat_mul(i_kh, self.P)
        p_new = mat_mul(tmp, mat_transpose(i_kh))
        for i in range(self.nx):
            for j in range(self.nx):
                p_new[i][j] += k[i] * r_var * k[j]
        for i in range(self.nx):
            for j in range(i + 1, self.nx):
                m = 0.5 * (p_new[i][j] + p_new[j][i])
                p_new[i][j] = m
                p_new[j][i] = m
        self.P = p_new
        return True

    def _batch_update(self, H: List[List[float]], v: List[float], r_diag: List[float]) -> bool:
        m = len(v)
        if m == 0:
            return False

        # PHt = P * H'
        PHt = zeros(self.nx, m)
        for i in range(self.nx):
            for j in range(m):
                s = 0.0
                for k in range(self.nx):
                    s += self.P[i][k] * H[j][k]
                PHt[i][j] = s

        # S = H * PHt + R
        S = zeros(m, m)
        for i in range(m):
            for j in range(m):
                s = 0.0
                for k in range(self.nx):
                    s += H[i][k] * PHt[k][j]
                if i == j:
                    s += r_diag[i]
                S[i][j] = s

        S_inv = self._invert_matrix(S)
        if S_inv is None:
            return False

        # K = PHt * inv(S)
        K = zeros(self.nx, m)
        for i in range(self.nx):
            for j in range(m):
                s = 0.0
                for k in range(m):
                    s += PHt[i][k] * S_inv[k][j]
                K[i][j] = s

        # x = x + K * v
        x_new = self.x[:]
        for i in range(self.nx):
            dx = 0.0
            for j in range(m):
                dx += K[i][j] * v[j]
            x_new[i] += dx

        # Joseph form covariance update:
        # P = (I-KH)P(I-KH)' + K R K'
        I = eye(self.nx)
        KH = zeros(self.nx, self.nx)
        for i in range(self.nx):
            for j in range(self.nx):
                s = 0.0
                for k in range(m):
                    s += K[i][k] * H[k][j]
                KH[i][j] = s
        IKH = zeros(self.nx, self.nx)
        for i in range(self.nx):
            for j in range(self.nx):
                IKH[i][j] = I[i][j] - KH[i][j]

        left = mat_mul(IKH, self.P)
        p_new = mat_mul(left, mat_transpose(IKH))

        # Add K R K'
        KR = zeros(self.nx, m)
        for i in range(self.nx):
            for j in range(m):
                KR[i][j] = K[i][j] * r_diag[j]
        add_term = mat_mul(KR, mat_transpose(K))
        p_new = mat_add(p_new, add_term)

        # Symmetrize.
        for i in range(self.nx):
            for j in range(i + 1, self.nx):
                mval = 0.5 * (p_new[i][j] + p_new[j][i])
                p_new[i][j] = mval
                p_new[j][i] = mval

        self.x = x_new
        self.P = p_new
        return True

    def _gate_measurement(self, cno_dbhz: float, elev_deg: float) -> bool:
        if not _is_finite(cno_dbhz) or cno_dbhz < self.cfg.cno_min_dbhz:
            return False
        if not _is_finite(elev_deg) or elev_deg < self.cfg.elev_min_deg:
            return False
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
            rho, _ = geometric_range(
                (obs.sat_pos_x_m, obs.sat_pos_y_m, obs.sat_pos_z_m),
                rx_xyz,
            )
            cb_i = obs.pr_m - rho + obs.sat_bias_m
            if not _is_finite(cb_i):
                continue
            all_vals.append(cb_i)
            by_sys[sid].append(cb_i)
            key = (sid, int(obs.sig_id))
            if key in by_key:
                by_key[key].append(cb_i)

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

    def _inter_state_string(self) -> str:
        items: List[str] = []
        for sid in self.inter_ids:
            idx = self._inter_idx(sid)
            val = self.x[idx] if idx is not None else 0.0
            items.append(f"{sid}:{val:.6f}")
        return ";".join(items)

    def _ibb_state_string(self) -> str:
        items: List[str] = []
        for sid, sig in self.ibb_keys:
            idx = self._ibb_idx(sid, sig)
            val = self.x[idx] if idx is not None else 0.0
            items.append(f"{sid}/{sig}:{val:.6f}")
        return ";".join(items)

    def _invert_matrix(self, a: List[List[float]]) -> Optional[List[List[float]]]:
        n = len(a)
        inv = [[0.0] * n for _ in range(n)]
        for col in range(n):
            b = [0.0] * n
            b[col] = 1.0
            x = solve_linear_system([row[:] for row in a], b)
            if x is None:
                return None
            for row in range(n):
                inv[row][col] = x[row]
        return inv

    def _estimate_pdop(self, pr_cache: Sequence[Tuple[int, float, float, List[float]]]) -> float:
        if len(pr_cache) < 4:
            return float("nan")
        nmat = [[0.0] * 4 for _ in range(4)]
        for _, _, _, h in pr_cache:
            g = [h[0], h[1], h[2], 1.0]
            for r in range(4):
                for c in range(4):
                    nmat[r][c] += g[r] * g[c]
        inv = self._invert_matrix(nmat)
        if inv is None:
            return float("nan")
        pdop_sq = inv[0][0] + inv[1][1] + inv[2][2]
        if pdop_sq <= 0.0 or not math.isfinite(pdop_sq):
            return float("nan")
        return math.sqrt(pdop_sq)

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
        learned_r_diag: Optional[Sequence[float]] = None,
        learned_bias: Optional[Sequence[float]] = None,
    ) -> Tuple[FilterResultRow, UpdateDebug]:
        init_ok = True
        coarse_state = None

        if self._is_state_bad():
            self.initialized = False
            self._reset_covariance()
            self.x = [0.0] * self.nx

        truth_init: Optional[Tuple[float, float, float, float, Dict[int, float], Dict[Tuple[int, int], float]]] = None
        if self.cfg.init_mode.lower() == "truth" and batch.observations:
            gt0 = batch.observations[0]
            if _is_finite(gt0.gt_ecef_x_m) and _is_finite(gt0.gt_ecef_y_m) and _is_finite(gt0.gt_ecef_z_m):
                tx, ty, tz = gt0.gt_ecef_x_m, gt0.gt_ecef_y_m, gt0.gt_ecef_z_m
                cb, inter_map, ibb_map = self._bias_candidates(batch, (tx, ty, tz), gated_only=True)
                truth_init = (tx, ty, tz, cb, inter_map, ibb_map)

        if not self.initialized:
            if truth_init is not None:
                tx, ty, tz, cb, inter_map, ibb_map = truth_init
                self.x = [0.0] * self.nx
                self.x[0], self.x[1], self.x[2] = tx, ty, tz
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
                self.initialized = True
                self.last_time = batch.time_gps_s
                coarse_state = [tx, ty, tz, cb]
            else:
                coarse = coarse_position_ls(
                    observations=batch.observations,
                    cno_min_dbhz=self.cfg.cno_min_dbhz,
                    elev_min_deg=self.cfg.elev_min_deg,
                    max_iter=self.cfg.max_init_iter,
                )
                if coarse is None:
                    init_ok = False
                else:
                    cx, cy, cz, cb0 = coarse
                    cb, inter_map, ibb_map = self._bias_candidates(batch, (cx, cy, cz), gated_only=True)
                    self.x = [0.0] * self.nx
                    self.x[0], self.x[1], self.x[2] = cx, cy, cz
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
                    self.initialized = True
                    self.last_time = batch.time_gps_s
                    coarse_state = [cx, cy, cz, self.x[self.cb_idx]]

        if self.initialized:
            dt = 0.0 if self.last_time is None else (batch.time_gps_s - self.last_time)
            self._predict(dt)
            self._apply_state_limits()
            self.last_time = batch.time_gps_s

        pr_innovations: List[float] = []
        rejected = 0
        used = 0
        pdop = float("nan")
        r_inflation = 1.0

        pr_cache: List[Tuple[int, float, float, List[float]]] = []
        if self.initialized:
            for i, obs in enumerate(batch.observations):
                sid = int(obs.gnss_id_raw)
                if sid not in self.system_ids:
                    continue
                if not self._gate_measurement(obs.cno_dbhz, obs.elev_deg):
                    continue
                rho, los = geometric_range(
                    (obs.sat_pos_x_m, obs.sat_pos_y_m, obs.sat_pos_z_m),
                    (self.x[0], self.x[1], self.x[2]),
                )
                iidx = self._inter_idx(sid)
                bidx = self._ibb_idx(sid, int(obs.sig_id))
                inter = self.x[iidx] if iidx is not None else 0.0
                ibb = self.x[bidx] if bidx is not None else 0.0
                net_bias = 0.0
                if learned_bias is not None and i < len(learned_bias):
                    net_bias = float(learned_bias[i])
                pred = rho + self.x[self.cb_idx] + inter + ibb - obs.sat_bias_m
                innov = (obs.pr_m + net_bias) - pred
                h = [0.0] * self.nx
                h[0], h[1], h[2] = -los[0], -los[1], -los[2]
                h[self.cb_idx] = 1.0
                if iidx is not None:
                    h[iidx] = 1.0
                if bidx is not None:
                    h[bidx] = 1.0
                lr = None
                if learned_r_diag is not None and i < len(learned_r_diag):
                    lr = float(learned_r_diag[i])
                r_var = self._measurement_variance_pr(obs.pr_noise_m, lr)
                pr_cache.append((i, innov, r_var, h))

            pdop = self._estimate_pdop(pr_cache)
            r_inflation = self._pdop_r_inflation(pdop)

            if self.cfg.use_clock_jump_compensation and pr_cache:
                step_m = CLIGHT * (self.cfg.clock_jump_ms / 1000.0)
                jump = round(_median([x[1] for x in pr_cache]) / step_m) * step_m
                if abs(jump) > 0.0:
                    self.x[self.cb_idx] += jump
                    for j in range(len(pr_cache)):
                        i0, innov0, r0, h0 = pr_cache[j]
                        pr_cache[j] = (i0, innov0 - jump, r0, h0)

            scored_cache: List[Tuple[float, float, float, List[float]]] = []
            accepted_h: List[List[float]] = []
            accepted_v: List[float] = []
            accepted_r: List[float] = []
            for _, innov, r_var, h in pr_cache:
                r_var_eff = r_var * r_inflation
                ph = [sum(self.P[ii][jj] * h[jj] for jj in range(self.nx)) for ii in range(self.nx)]
                s_val = r_var_eff + sum(h[ii] * ph[ii] for ii in range(self.nx))
                sigma = math.sqrt(max(s_val, 1e-12))
                norm = abs(innov) / max(sigma, 1e-12)
                scored_cache.append((norm, innov, r_var_eff, h))
                if abs(innov) > self.cfg.innovation_gate_sigma * sigma:
                    rejected += 1
                    continue
                if self.cfg.use_fde and abs(innov) > self.cfg.fde_sigma * sigma:
                    rejected += 1
                    continue
                accepted_h.append(h)
                accepted_v.append(innov)
                accepted_r.append(r_var_eff)
                pr_innovations.append(innov)

            if accepted_v:
                ok = self._batch_update(accepted_h, accepted_v, accepted_r)
                if ok:
                    used = len(accepted_v)
                    self._apply_state_limits()
                else:
                    rejected += len(accepted_v)
                    used = 0

            if used == 0 and self.cfg.starvation_fallback_enabled and scored_cache:
                scored_cache.sort(key=lambda t: t[0])
                fb_h: List[List[float]] = []
                fb_v: List[float] = []
                fb_r: List[float] = []
                picked = 0
                for norm, innov, r_var, h in scored_cache:
                    if norm > self.cfg.starvation_max_norm_innov:
                        break
                    fb_h.append(h)
                    fb_v.append(innov)
                    fb_r.append(r_var * self.cfg.starvation_r_scale)
                    picked += 1
                    if picked >= max(1, int(self.cfg.starvation_min_updates)):
                        break
                if fb_v:
                    ok = self._batch_update(fb_h, fb_v, fb_r)
                    if ok:
                        used = len(fb_v)
                        pr_innovations.extend(fb_v)
                        self._apply_state_limits()

        pred_lat_deg, pred_lon_deg, pred_h_m = ecef_to_geodetic(self.x[0], self.x[1], self.x[2])
        gt_obs = batch.observations[0]
        gt_lat = gt_obs.gt_lat_deg
        gt_lon = gt_obs.gt_lon_deg
        gt_h = gt_obs.gt_h_m
        if _is_finite(gt_lat) and _is_finite(gt_lon) and _is_finite(gt_h):
            e, n, u = ecef_to_enu(self.x[0], self.x[1], self.x[2], gt_lat, gt_lon, gt_h)
            enu_3d = math.sqrt(e * e + n * n + u * u)
        else:
            e, n, u, enu_3d = float("nan"), float("nan"), float("nan"), float("nan")

        row = FilterResultRow(
            epoch=batch.epoch,
            split=batch.split,
            week=batch.week,
            tow=batch.tow,
            time_gps_s=batch.time_gps_s,
            x_m=self.x[0],
            y_m=self.x[1],
            z_m=self.x[2],
            vx_mps=self.x[3],
            vy_mps=self.x[4],
            vz_mps=self.x[5],
            ref_sys_id=int(self.ref_sys_id),
            cb_m=self.x[self.cb_idx],
            cd_mps=self.x[self.cd_idx],
            inter_gnss_m=self._inter_state_string(),
            ibb_m=self._ibb_state_string(),
            used_obs=used,
            rejected_obs=rejected,
            pdop=pdop,
            r_inflation=r_inflation,
            gt_lat_deg=gt_lat,
            gt_lon_deg=gt_lon,
            gt_h_m=gt_h,
            pred_lat_deg=pred_lat_deg,
            pred_lon_deg=pred_lon_deg,
            pred_h_m=pred_h_m,
            enu_e_m=e,
            enu_n_m=n,
            enu_u_m=u,
            enu_3d_m=enu_3d,
            init_ok=1 if init_ok else 0,
        )
        dbg = UpdateDebug(
            pr_innovations=pr_innovations,
            rejected_count=rejected,
            used_count=used,
            pdop=pdop,
            r_inflation=r_inflation,
            init_ok=init_ok,
            coarse_state=coarse_state,
        )
        return row, dbg


def run_filter(
    batches: Iterable[EpochBatch],
    cfg: Optional[MLPEKFConfig] = None,
    learned_by_epoch: Optional[Dict[int, Tuple[Sequence[float], Sequence[float]]]] = None,
) -> Tuple[List[FilterResultRow], List[UpdateDebug]]:
    batch_list = list(batches)
    system_ids: Set[int] = set()
    signals_by_sys: Dict[int, Set[int]] = {}
    for b in batch_list:
        for o in b.observations:
            sid = int(o.gnss_id_raw)
            sig = int(o.sig_id)
            system_ids.add(sid)
            signals_by_sys.setdefault(sid, set()).add(sig)

    systems = sorted(system_ids) if system_ids else [0]
    ref_sys_id = 0 if 0 in systems else systems[0]
    ibb_keys: List[Tuple[int, int]] = []
    for sid in systems:
        sigs = sorted(signals_by_sys.get(sid, set()))
        if len(sigs) <= 1:
            continue
        ref_sig = sigs[0]
        for sig in sigs:
            if sig == ref_sig:
                continue
            ibb_keys.append((sid, sig))

    ekf = MLPEKF(cfg or MLPEKFConfig(), systems, ibb_keys=ibb_keys, ref_sys_id=ref_sys_id)
    rows: List[FilterResultRow] = []
    debugs: List[UpdateDebug] = []
    for batch in batch_list:
        rr: Optional[Sequence[float]] = None
        bb: Optional[Sequence[float]] = None
        if learned_by_epoch is not None and int(batch.epoch) in learned_by_epoch:
            rr, bb = learned_by_epoch[int(batch.epoch)]
        row, dbg = ekf.process_epoch(batch, rr, bb)
        rows.append(row)
        debugs.append(dbg)
    return rows, debugs
