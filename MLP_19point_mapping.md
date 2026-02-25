# MLP-Kalmannet: 19-Point Compatibility Mapping (Phase-1)

This file records how the current implementation in `D:\MLPkalmannet` handles the 19 known mismatches against LF-style workflow.

## Scope of this phase
- Implemented: data bridge + residual builder + CSV adapter + EKF migration and runner.
- Not implemented in this phase: training/validation/export scripts.

## Mapping

1. **No RINEX obs + eph chain**
- Handling: `Data/mlp_preprocess.py` consumes `raw_observation.csv` precomputed sat state and builds canonical obs rows.

2. **No residual field in source data**
- Handling: `Data/mlp_preprocess.py` computes `residual_m = pr_m - pr_pred_m` from coarse LS state.

3. **Q constant/time-varying mismatch**
- Handling: `GNSSfilter/mlp_ekf/config.py` exposes `q_mode` (`constant`/`dynamic`).

4. **GT format mismatch with LF reader**
- Handling: bridge emits canonical `Data/mlp_compat/mlp_gt.csv` with fixed columns.

5. **GT missing/unstable time alignment path**
- Handling: bridge preserves/recovers `week/tow/time_gps_s` and aligns by `epoch`.

6. **Label coordinate mismatch (ECEF vs lat/lon/h)**
- Handling: bridge outputs both ECEF and `gt_lat_deg/gt_lon_deg/gt_h_m`; lat/lon/h auto-derived if absent.

7. **Observation layout mismatch (flat rows vs epoch matrix)**
- Handling: bridge emits epoch-tagged canonical rows; `mlp_ekf/adapter.py` groups rows into epoch batches.

8. **GNSS/signal ID mapping mismatch**
- Handling: bridge writes `gnss_id_raw` and `gnss_system` via fixed map `{0:GPS,2:GAL,3:BDS,6:GLO}`.

9. **Config declared but not wired risk**
- Handling: EKF config fields are loaded in `mlp_ekf/config.py` and used in filter path.

10. **State definition mismatch (clock states)**
- Handling: migrated EKF uses per-system clock bias/drift states; missing systems are late-bootstrapped.

11. **Initialization mismatch (truth init vs coarse init)**
- Handling: default config uses `init_mode = coarse` (LF-style).

12. **Update chain mismatch (Doppler+PR vs PR-main)**
- Handling: default `use_doppler_update = false`; PR update is mainline.

13. **Measurement noise source mismatch**
- Handling: PR/DR floor and observation noise are both supported; learned-R hook remains in interface.

14. **Outlier handling mismatch (FDE)**
- Handling: `use_fde` exists but defaults OFF in LF-style baseline.

15. **Clock jump compensation mismatch**
- Handling: available in config, default OFF for LF-style alignment.

16. **QC entry mismatch**
- Handling: gating uses `cno_min_dbhz` + `elev_min_deg` + innovation gate.

17. **Train/val split mismatch**
- Handling: bridge writes chronological `train/val/test`; `Data/mlp_make_epoch_window.py` supports fixed windows.

18. **Data leakage risk (GT in raw)**
- Handling: model feature list in manifest excludes GT and audit columns; bridge keeps GT fields only for supervision/eval.

19. **Evaluation frame mismatch (NED vs ENU)**
- Handling: EKF metrics are ENU RMSE in `mlp_ekf/metrics.py`.

## Main outputs
- `Data/mlp_compat/mlp_observations.csv`
- `Data/mlp_compat/mlp_epochs.csv`
- `Data/mlp_compat/mlp_gt.csv`
- `Data/mlp_compat/mlp_manifest.json`
- `GNSSfilter/mlp_ekf/output/mlp_filter_states.csv`
- `GNSSfilter/mlp_ekf/output/mlp_filter_debug.json`
- `GNSSfilter/mlp_ekf/output/mlp_filter_metrics.json`
