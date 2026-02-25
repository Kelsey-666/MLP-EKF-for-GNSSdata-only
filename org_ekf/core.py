"""Core MATLAB EKF logic migrated to Python."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np

from .models import GnssMeasurements, KFState, ObsId, ObsMap, ObservationBlock, StateIndex
from .constants import Constants
from .utils import code_svids, ecef2llh, ecef2ned_rot, clean_struct, ismember_indices


@dataclass
class FormObservationResult:
    pr: ObservationBlock
    dr: ObservationBlock
    kf_state: KFState
    pdop: float


def get_obs_map(gnss: GnssMeasurements) -> tuple[KFState, ObsId, GnssMeasurements, ObsMap]:
    xt_id_sv = code_svids(gnss.sv_id, gnss.gnss_id, np.zeros_like(gnss.sv_id), np.zeros_like(gnss.sv_id))
    obs_id_sv = np.unique(xt_id_sv)

    ext_id = code_svids(gnss.sv_id, gnss.gnss_id, gnss.sig_id, np.zeros_like(gnss.sv_id))
    obs_id_xt = np.unique(ext_id)

    inter_gnss = np.unique(gnss.gnss_id)
    if inter_gnss.size > 0:
        inter_gnss = inter_gnss[inter_gnss != inter_gnss[0]]

    ibb_full = code_svids(np.zeros_like(gnss.sv_id), gnss.gnss_id, gnss.sig_id, np.zeros_like(gnss.sv_id))
    ibb_list: list[int] = []
    for gnss_id in np.unique(gnss.gnss_id):
        sig_ids = np.unique(gnss.sig_id[gnss.gnss_id == gnss_id])
        if sig_ids.size > 1:
            for sig in sig_ids[1:]:
                ibb_list.append(int(code_svids(np.array([0]), np.array([gnss_id]), np.array([sig]), np.array([0]))[0]))
    obs_id_ibb = np.array(sorted(set(ibb_list)), dtype=np.int64)

    map_sv = ismember_indices(xt_id_sv, obs_id_sv)
    map_inter = ismember_indices(gnss.gnss_id, inter_gnss) if inter_gnss.size > 0 else np.full_like(gnss.gnss_id, -1)
    map_ibb = ismember_indices(ibb_full, obs_id_ibb) if obs_id_ibb.size > 0 else np.full_like(gnss.gnss_id, -1)
    map_chn = ismember_indices(ext_id, obs_id_xt)
    obs_map = ObsMap(sv=map_sv, interGnss=map_inter, ibb=map_ibb, chn=map_chn)

    pos = np.array([0, 1, 2], dtype=int)
    vel = np.array([3, 4, 5], dtype=int)
    posvel = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    clock = np.array([6, 7], dtype=int)
    num_states = 8

    if obs_id_ibb.size > 0:
        ibb_idx = np.arange(num_states, num_states + obs_id_ibb.size, dtype=int)
        num_states += obs_id_ibb.size
    else:
        ibb_idx = np.array([], dtype=int)

    if inter_gnss.size > 0:
        inter_idx = np.arange(num_states, num_states + inter_gnss.size, dtype=int)
        num_states += inter_gnss.size
    else:
        inter_idx = np.array([], dtype=int)

    state_indx = StateIndex(
        pos=pos,
        vel=vel,
        posvel=posvel,
        clock=clock,
        ibb=ibb_idx,
        inter_gnss=inter_idx,
    )
    kf_state = KFState(
        x=np.zeros((num_states,), dtype=float),
        P=np.zeros((num_states, num_states), dtype=float),
        state_indx=state_indx,
        num_sv=int(obs_id_sv.size),
        num_obs=int(obs_id_xt.size),
        inter_gnss=int(inter_gnss.size),
        ibb=int(obs_id_ibb.size),
        num_states=int(num_states),
    )
    obs_id = ObsId(
        xtIdSv=obs_id_sv,
        xtId=obs_id_xt,
        interGnss=inter_gnss.astype(int),
        ibb=obs_id_ibb,
    )
    gnss.xtId = ext_id
    gnss.xtIdSv = xt_id_sv
    return kf_state, obs_id, gnss, obs_map


def init_state_and_cov(init_pos: np.ndarray, kf_state: KFState) -> KFState:
    p = np.zeros((kf_state.num_states,), dtype=float)
    x = np.zeros((kf_state.num_states,), dtype=float)
    x[kf_state.state_indx.posvel] = np.concatenate([(np.asarray(init_pos, dtype=float).reshape(3) + np.random.randn(3) * 2.0), np.zeros((3,))])
    p[kf_state.state_indx.posvel] = 1000.0**2
    p[kf_state.state_indx.clock[0]] = 1000.0**2
    p[kf_state.state_indx.clock[1]] = 100.0**2
    if kf_state.state_indx.ibb.size > 0:
        p[kf_state.state_indx.ibb] = 1000.0**2
    if kf_state.state_indx.inter_gnss.size > 0:
        p[kf_state.state_indx.inter_gnss] = 1000.0**2
    return KFState(
        x=x,
        P=np.diag(p),
        state_indx=kf_state.state_indx,
        num_sv=kf_state.num_sv,
        num_obs=kf_state.num_obs,
        inter_gnss=kf_state.inter_gnss,
        ibb=kf_state.ibb,
        num_states=kf_state.num_states,
    )


def predict_kf(kf_state: KFState, dt: float) -> tuple[KFState, np.ndarray]:
    s = kf_state.state_indx
    q = np.zeros((kf_state.num_states,), dtype=float)
    phi_diag = np.ones((kf_state.num_states,), dtype=float)

    accel_h = 5.0
    accel_v = 0.05
    llh = ecef2llh(kf_state.x[s.pos])
    n2e = ecef2ned_rot(llh[0], llh[1]).T
    qs = n2e @ np.diag([accel_h, accel_h, accel_v]) @ n2e.T
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    qpv = np.block([[0.25 * dt4 * qs, 0.5 * dt3 * qs], [0.5 * dt3 * qs.T, dt2 * qs]])

    h_0 = 17e-3
    h_2 = 10.0 * h_0
    sf = 2.0 * h_0
    sg = 8.0 * np.pi * np.pi * h_2
    dt2by2 = dt * dt / 2.0
    dt3by6 = 2.0 * dt * dt2by2 / 3.0

    if s.ibb.size > 0:
        q[s.ibb] = (0.01**2) * dt
    if s.inter_gnss.size > 0:
        q[s.inter_gnss] = (0.01**2) * dt

    phi = np.diag(phi_diag)
    phi[0, 3] = dt
    phi[1, 4] = dt
    phi[2, 5] = dt

    q_mat = np.diag(q)
    q_mat[np.ix_(s.posvel, s.posvel)] = qpv
    phi[s.clock[0], s.clock[1]] = dt
    q_mat[np.ix_(s.clock, s.clock)] = np.array([[sf * dt + sg * dt3by6, sf * dt2by2], [sf * dt2by2, sf * dt]])

    p_new = phi @ kf_state.P @ phi.T + q_mat
    x_new = phi @ kf_state.x
    return (
        KFState(
            x=x_new,
            P=p_new,
            state_indx=kf_state.state_indx,
            num_sv=kf_state.num_sv,
            num_obs=kf_state.num_obs,
            inter_gnss=kf_state.inter_gnss,
            ibb=kf_state.ibb,
            num_states=kf_state.num_states,
        ),
        phi,
    )


def _update_kf_eqns(
    x: np.ndarray,
    p: np.ndarray,
    h_in: np.ndarray,
    z_in: np.ndarray,
    r_in: np.ndarray,
    use_fde: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep_ix = np.where(~np.isnan(z_in))[0]
    x_saved = x.copy()
    p_saved = p.copy()

    it = 1
    calc_update = True
    dx = np.zeros((x.size,), dtype=float)
    sinnov = np.zeros((0, 0), dtype=float)
    zres = np.full(z_in.shape, np.nan, dtype=float)
    sres = np.zeros((0,), dtype=float)

    while calc_update and it <= 4 and keep_ix.size > 0:
        if it > 1:
            x = x_saved.copy()
            p = p_saved.copy()

        zres = np.full(z_in.shape, np.nan, dtype=float)
        r = r_in[np.ix_(keep_ix, keep_ix)]
        h = h_in[keep_ix, :]
        z = z_in[keep_ix]

        pht = p @ h.T
        sinnov = h @ pht + r
        try:
            k = pht @ np.linalg.inv(sinnov)
        except np.linalg.LinAlgError:
            sinnov = sinnov + np.eye(sinnov.shape[0]) * 1e-9
            k = pht @ np.linalg.inv(sinnov)

        p = p - k @ pht.T
        p = 0.5 * (p + p.T)
        dx = k @ z
        x = x + dx

        zres[keep_ix] = z - h @ dx
        eye_h = np.eye(len(z))
        sres_sq = np.diag(r - (eye_h - h @ k) @ h @ p @ h.T)
        sres_sq = np.where(sres_sq < 0.0, 0.0, sres_sq)
        sres = np.sqrt(sres_sq)

        resok = np.abs(zres[keep_ix]) < (sres * 3.0)
        if use_fde and np.any(~resok):
            worst_local = int(np.argmax(np.abs(zres[keep_ix])))
            keep_ix = np.delete(keep_ix, worst_local)
            it += 1
        else:
            calc_update = False

    return x, p, dx, sinnov, zres, sres


def update_kf(
    kf_state: KFState,
    h: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    use_fde: bool,
) -> tuple[KFState, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, p, dx, sinnov, zres, sres = _update_kf_eqns(kf_state.x, kf_state.P, h, z, r, use_fde)
    return (
        KFState(
            x=x,
            P=p,
            state_indx=kf_state.state_indx,
            num_sv=kf_state.num_sv,
            num_obs=kf_state.num_obs,
            inter_gnss=kf_state.inter_gnss,
            ibb=kf_state.ibb,
            num_states=kf_state.num_states,
        ),
        dx,
        sinnov,
        zres,
        sres,
    )


def form_observations(gnss: GnssMeasurements, ix: np.ndarray, kf_state: KFState, obs_map: ObsMap) -> FormObservationResult:
    x = kf_state.x.copy()
    p = kf_state.P
    wgs84_w = 7292115.1467e-11
    sol = Constants().c

    user_pos_e = x[kf_state.state_indx.pos]
    sat_pos_e = gnss.sat_pos_E[ix, :].T
    sat_vel_e = gnss.sat_vel_E[ix, :].T

    delta_pos = (gnss.sat_pos_E[ix, :] - user_pos_e.reshape(1, 3)).T
    theta = wgs84_w * (np.linalg.norm(delta_pos, axis=0) / sol)

    sat_pos_e = np.vstack(
        [
            sat_pos_e[0, :] * np.cos(theta) + sat_pos_e[1, :] * np.sin(theta),
            -sat_pos_e[0, :] * np.sin(theta) + sat_pos_e[1, :] * np.cos(theta),
            sat_pos_e[2, :],
        ]
    )

    delta_pos = sat_pos_e - user_pos_e.reshape(3, 1)
    rng = np.sqrt(np.sum(delta_pos**2, axis=0))
    negative_los = -(delta_pos.T / rng.reshape(-1, 1))

    n = len(ix)
    hpos = np.hstack([negative_los, np.zeros((n, 3)), np.ones((n, 1)), np.zeros((n, 1))])

    sat_code = gnss.sv_id[ix] + 256 * gnss.gnss_id[ix]
    _, uix = np.unique(sat_code, return_index=True)
    if len(uix) >= 4:
        a = hpos[uix, 0:3]
        try:
            temp = np.linalg.inv(a.T @ a)
            pdop = float(np.sqrt(np.sum(np.diag(temp[0:3, 0:3]))))
        except np.linalg.LinAlgError:
            pdop = float("nan")
    else:
        pdop = float("nan")

    hbias = np.zeros((n, kf_state.ibb + kf_state.inter_gnss), dtype=float)
    ibb_cols = obs_map.ibb[ix]
    ibb_rows = np.where(ibb_cols >= 0)[0]
    for rr in ibb_rows:
        hbias[rr, ibb_cols[rr]] = 1.0

    inter_cols = obs_map.interGnss[ix]
    inter_rows = np.where(inter_cols >= 0)[0]
    for rr in inter_rows:
        hbias[rr, kf_state.ibb + inter_cols[rr]] = 1.0

    satclk = gnss.sat_bias[ix]

    pr_r = np.diag(np.square(gnss.pr_noise[ix]))
    pr_h = np.hstack([hpos, hbias])
    pred = rng.reshape(-1) + (pr_h[:, kf_state.state_indx.clock[0] :] @ x[kf_state.state_indx.clock[0] :]) - satclk
    pr_z = gnss.pr[ix] - pred

    user_vel = x[kf_state.state_indx.vel]
    delta_vel = sat_vel_e - user_vel.reshape(3, 1)
    range_rate = np.sum(delta_vel * (-negative_los.T), axis=0) - gnss.sat_drift[ix]

    dr_h = np.zeros((n, kf_state.num_states), dtype=float)
    dr_h[:, kf_state.state_indx.vel] = negative_los
    dr_h[:, kf_state.state_indx.clock[1]] = 1.0
    dr_z = (-gnss.dr[ix] - range_rate) - dr_h[:, kf_state.state_indx.clock[0] :] @ x[kf_state.state_indx.clock[0] :]
    dr_r = np.diag(np.square(gnss.dr_noise[ix]))

    ms_jump = np.round(np.nanmedian(pr_z) / (sol * 0.001)) * sol * 0.001
    pr_z = pr_z - ms_jump
    x[kf_state.state_indx.clock[0]] = x[kf_state.state_indx.clock[0]] + ms_jump

    med_pr = np.nanmedian(pr_z)
    if np.abs(med_pr) > 1000.0:
        pr_z = pr_z - med_pr
        x[kf_state.state_indx.clock[0]] = x[kf_state.state_indx.clock[0]] + med_pr

    med_dr = np.nanmedian(dr_z)
    if np.abs(med_dr) > 100.0:
        dr_z = dr_z - med_dr
        x[kf_state.state_indx.clock[1]] = x[kf_state.state_indx.clock[1]] + med_dr

    kf_state_new = KFState(
        x=x,
        P=p,
        state_indx=kf_state.state_indx,
        num_sv=kf_state.num_sv,
        num_obs=kf_state.num_obs,
        inter_gnss=kf_state.inter_gnss,
        ibb=kf_state.ibb,
        num_states=kf_state.num_states,
    )
    return FormObservationResult(
        pr=ObservationBlock(H=pr_h, z=pr_z, R=pr_r),
        dr=ObservationBlock(H=dr_h, z=dr_z, R=dr_r),
        kf_state=kf_state_new,
        pdop=pdop,
    )


def allocate_output_struct(num_epoch: int, kf_state: KFState) -> dict[str, np.ndarray | dict]:
    return {
        "state_indx": kf_state.state_indx,
        "x": np.full((num_epoch, kf_state.num_states), np.nan, dtype=float),
        "std": np.full((num_epoch, kf_state.num_states), np.nan, dtype=float),
        "stdPosNed": np.full((num_epoch, 3), np.nan, dtype=float),
        "week": np.full((num_epoch,), np.nan, dtype=float),
        "tow": np.full((num_epoch,), np.nan, dtype=float),
        "num_obs": np.full((num_epoch,), np.nan, dtype=float),
        "numSat": np.full((num_epoch,), np.nan, dtype=float),
        "pdop": np.full((num_epoch,), np.nan, dtype=float),
        "prRes": np.full((num_epoch, kf_state.num_obs), np.nan, dtype=float),
        "drRes": np.full((num_epoch, kf_state.num_obs), np.nan, dtype=float),
        "prInnov": np.full((num_epoch, kf_state.num_obs), np.nan, dtype=float),
    }


def pedestrian_kalman_filter(gnss: GnssMeasurements) -> dict:
    if gnss.gnss_id.size == 0:
        warnings.warn("No input data found")
        return {}

    num_epoch = gnss.tow.size
    epoch_ix = np.full((num_epoch, 2), -1, dtype=int)
    for i, ep in enumerate(gnss.epoch):
        ee = int(ep) - 1
        if ee < 0 or ee >= num_epoch:
            continue
        if epoch_ix[ee, 0] == -1:
            epoch_ix[ee, 0] = i
        epoch_ix[ee, 1] = i

    kf_state, obs_id, gnss, obs_map = get_obs_map(gnss)
    out = allocate_output_struct(num_epoch, kf_state)
    out["state_indx"] = kf_state.state_indx

    kf_state = init_state_and_cov(gnss.pos_E[0, :], kf_state)
    first_epoch = True

    for epoch in range(num_epoch):
        if not first_epoch:
            dt = float(gnss.tow[epoch] - gnss.tow[epoch - 1])
            kf_state, _ = predict_kf(kf_state, dt)

        start, end = int(epoch_ix[epoch, 0]), int(epoch_ix[epoch, 1])
        valid_epoch = start >= 0 and end >= start
        ix = np.arange(start, end + 1, dtype=int) if valid_epoch else np.array([], dtype=int)

        if valid_epoch:
            unique_gnss = np.unique(gnss.gnss_id[ix])
            unique_sat = np.unique(gnss.gnss_id[ix] * 256 + gnss.sv_id[ix])
            if unique_sat.size < 3 + unique_gnss.size:
                valid_epoch = False
            else:
                first_epoch = False

        if valid_epoch:
            frm = form_observations(gnss, ix, kf_state, obs_map)
            pr, dr = frm.pr, frm.dr
            kf_state = frm.kf_state
            out["pdop"][epoch] = frm.pdop
            out["prInnov"][epoch, obs_map.chn[ix]] = pr.z

            kf_state, dx, _, dr_res, _ = update_kf(kf_state, dr.H, dr.z, dr.R, True)
            pr_z = pr.z - pr.H @ dx
            out["drRes"][epoch, obs_map.chn[ix]] = dr_res
            kf_state, _, _, pr_res, _ = update_kf(kf_state, pr.H, pr_z, pr.R, True)
            out["prRes"][epoch, obs_map.chn[ix]] = pr_res

        llh = ecef2llh(kf_state.x[kf_state.state_indx.pos])
        ren = ecef2ned_rot(llh[0], llh[1])

        out["x"][epoch, :] = kf_state.x
        out["std"][epoch, :] = np.sqrt(np.clip(np.diag(kf_state.P), 0.0, None))
        ppos = kf_state.P[np.ix_(kf_state.state_indx.pos, kf_state.state_indx.pos)]
        out["stdPosNed"][epoch, :] = np.sqrt(np.clip(np.diag(ren @ ppos @ ren.T), 0.0, None))
        out["tow"][epoch] = gnss.tow[epoch]
        out["week"][epoch] = gnss.week[epoch]
        if valid_epoch:
            out["num_obs"][epoch] = float(ix.size)
            out["numSat"][epoch] = float(np.unique(gnss.gnss_id[ix] * 256 + gnss.sv_id[ix]).size)
        else:
            out["num_obs"][epoch] = 0.0
            out["numSat"][epoch] = 0.0

        try:
            _ = np.linalg.cholesky(kf_state.P + np.eye(kf_state.P.shape[0]) * 1e-12)
        except np.linalg.LinAlgError:
            warnings.warn("Positive definite matrix check failed")

    out = clean_struct(out)
    out["state_indx"] = kf_state.state_indx
    out["obsId"] = obs_id
    return out

