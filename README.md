# MLPKalmannet

MLPKalmannet is a GNSS learning-filter integration project built around:
- CSV-based GNSS data preprocessing,
- an EKF backend adapted for learning-based noise/bias correction,
- and training/evaluation/export scripts for reproducible experiments.

The repository follows an LF-style workflow, but uses project-local data interfaces and naming.

---

## Framework Overview

The end-to-end pipeline has four stages:
1. Data bridge (`raw/gt` CSV -> model-ready epoch tables).
2. EKF baseline / EKF + learned correction execution.
3. Model training and checkpoint selection.
4. Acceptance, trajectory reconstruction, and model export.

---

## Repository Structure

- `Data/`
  - Raw and processed CSV assets.
  - Preprocessing scripts (`mlp_preprocess.py`, `mlp_make_epoch_window.py`).
- `GNSSfilter/mlp_ekf/`
  - Adapted EKF implementation and runner.
- `mlp_train/`
  - Dataset, model, loss, and training/evaluation pipeline.
- `config/`
  - EKF and training configuration JSON files.
- `train.py`
  - Training entry.
- `accept.py`
  - Acceptance/evaluation entry.
- `reconstruct.py`
  - Rebuild EKF trajectory using trained checkpoint.
- `export.py`
  - Export TorchScript model + metadata.

---

## Environment Setup

Recommended: Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not used in your local setup, install core packages manually:
`torch`, `numpy`, `pandas`, `scipy`, `matplotlib`.

---

## Private Data Policy and Input Files

This project uses private GNSS data in internal experiments.
Therefore, private raw data and private GT data are intentionally not published to GitHub.

Before running the pipeline, place your own files at:
- `Data/YOUR RAW-DATA.csv`
- `Data/YOUR GT-DATA.csv`

### Raw CSV expected fields (project schema)
Typical required columns include:
- `epoch`, `week`, `tow`
- `gnss_id`, `sv_id`, `sig_id`
- `pr_m`, `dr_mps`
- `sat_pos_E_x_m/sat_pos_E_y_m/sat_pos_E_z_m`
- `sat_vel_E_x_mps/sat_vel_E_y_mps/sat_vel_E_z_mps`
- `sat_bias_m`, `sat_drift_mps`
- `elev_rad` or `elev_deg`, `azim_rad` or `azim_deg`
- `cno` (or `cno_dbhz`)

### GT CSV expected fields (project schema)
Typical required columns include:
- `epoch`, `week`, `tow`
- `gt_pos_E_x_m`, `gt_pos_E_y_m`, `gt_pos_E_z_m`
- optional: `gt_vel_E_x_mps`, `gt_vel_E_y_mps`, `gt_vel_E_z_mps`

---

## Quick Start

### 1. Build bridge files
```bash
python MLP_Preprocess.py --raw "Data/YOUR RAW-DATA.csv" --gt "Data/YOUR GT-DATA.csv" --output-dir Data/mlp_compat
```

### 2. Optional fixed window split (e.g., 2000 train + 2000 test)
```bash
python Data/mlp_make_epoch_window.py --input-observations Data/mlp_compat/mlp_observations.csv --input-epochs Data/mlp_compat/mlp_epochs.csv --input-gt Data/mlp_compat/mlp_gt.csv --output-dir Data/mlp_train2000_test2000 --train-epochs 2000 --test-epochs 2000
```

### 3. Run EKF baseline
```bash
python MLP_Filter.py --observations Data/mlp_train2000_test2000/mlp_observations.csv --config config/MLP-EKF-seed44.json --output-dir GNSSfilter/mlp_ekf/output
```

### 4. Train model
```bash
python train.py --config config/MLP-Train-seed44.json
```

### 5. Acceptance / evaluation
```bash
python accept.py --checkpoint trained_model/<your_run>/<best_checkpoint>.pth --observations Data/mlp_train2000_test2000/mlp_observations.csv --ekf-config config/MLP-EKF-seed44.json --split test --output-dir mlp_results
```

### 6. Reconstruct trajectory
```bash
python reconstruct.py --checkpoint trained_model/<your_run>/<best_checkpoint>.pth --observations Data/mlp_train2000_test2000/mlp_observations.csv --ekf-config config/MLP-EKF-seed44.json --split test --output mlp_results/reconstruct_test.csv
```

### 7. Export model
```bash
python export.py --checkpoint trained_model/<your_run>/<best_checkpoint>.pth --output trained_model/mlp/exported_model.ts
```

---

## Notes

- Default project experiments commonly use seed44-oriented configs.
- Keep preprocessing, EKF config, training config, and evaluation config consistent for fair comparison.
- For detailed compatibility and mismatch handling, see `MLP_19point_mapping.md`.

## Public Dataset Source
For reproducible experiments with public navigation products, this project uses BKG IGS BRDC resources:
- Directory: `https://igs.bkg.bund.de/root_ftp/IGS/BRDC/`
- This set with `week=1900, tow~130926-132119` corresponds to `2016-06-06 (DOY 158)`.
- Available file (multi-GNSS navigation + broadcast clock):
  `https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2016/158/BRDC00WRD_U_20161580000_01D_MN.rnx.gz`
