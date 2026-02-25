# org_ekf

Python migration of `D:\MLPkalmannet\Data\pedestrian-kalman-filter`.

Implemented MATLAB-equivalent blocks:

- `GetObsMap` -> `core.get_obs_map`
- `InitStateAndCov` -> `core.init_state_and_cov`
- `PredictKf` -> `core.predict_kf`
- `FormObservations` -> `core.form_observations`
- `UpdateKf` -> `core.update_kf`
- `PedestrianKalmanFilter` -> `core.pedestrian_kalman_filter`
- Utility functions in `utils.py` (`codeSVIDs`, `ecef2llh`, `ecef2nedRot`, `ecef2nedError`, `cleanStruct`)

Run test segment (original epochs 2001-4000):

```powershell
python D:\MLPkalmannet\org_ekf\run_org_ekf.py `
  --raw-csv D:\MLPkalmannet\Data\raw_observation.csv `
  --gt-csv D:\MLPkalmannet\Data\gt_processed.csv `
  --epoch-start 2001 `
  --epoch-end 4000 `
  --out-dir D:\MLPkalmannet\org_results
```

Results are saved under `D:\MLPkalmannet\org_results\org_ekf_test2000_*`.

