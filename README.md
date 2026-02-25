# MLPKalmannet (Rebuild Phase-1)

This repository rebuild contains the first-stage LF-style adaptation using your own data and EKF:
- Data bridge + residual generation
- CSV epoch adapter
- Migrated EKF mainline (PR update by default)

## Paths
- Raw observations: `Data/raw_observation.csv`
- Ground truth: `Data/gt_processed.csv`
- Data bridge script: `Data/mlp_preprocess.py`
- Window split helper: `Data/mlp_make_epoch_window.py`
- EKF package: `GNSSfilter/mlp_ekf`
- EKF runner: `GNSSfilter/mlp_ekf/run_mlp_filter.py`

## Quick Start

1. Build bridge files
```bash
python Data/mlp_preprocess.py --raw Data/raw_observation.csv --gt Data/gt_processed.csv --output-dir Data/mlp_compat
```

2. Run EKF
```bash
python GNSSfilter/mlp_ekf/run_mlp_filter.py --observations Data/mlp_compat/mlp_observations.csv --config GNSSfilter/mlp_ekf/default_config.json --output-dir GNSSfilter/mlp_ekf/output
```

3. Optional fixed window split (e.g., 2000 train + 2000 test)
```bash
python Data/mlp_make_epoch_window.py --input-observations Data/mlp_compat/mlp_observations.csv --input-epochs Data/mlp_compat/mlp_epochs.csv --input-gt Data/mlp_compat/mlp_gt.csv --output-dir Data/mlp_train2000_test2000 --train-epochs 2000 --test-epochs 2000
```

## Notes
- This phase intentionally does not include training scripts yet.
- See `MLP_19point_mapping.md` for detailed mismatch handling.
