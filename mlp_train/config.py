"""Config structures for MLP-Kalmannet training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainConfig:
    observations_csv: str = "Data/mlp_train2000_test2000/mlp_observations.csv"
    ekf_config: str = "GNSSfilter/mlp_ekf/default_config.json"
    save_dir: str = "trained_model/mlp"
    model_name: str = "mlp_model"
    train_split: str = "train"
    val_split: str = "test"
    test_split: str = "test"
    model_type: str = "mlp"  # mlp | gru
    input_size: int = 4
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 128, 64])
    output_size: int = 2
    gru_hidden_size: int = 96
    gru_num_layers: int = 1
    gru_dropout: float = 0.0
    gru_bidirectional: bool = False
    batch_size_train: int = 16
    batch_size_eval: int = 1
    epochs: int = 40
    lr: float = 2.0e-4
    eval_every: int = 1
    device: str = "cpu"
    seed: int = 42
    multi_seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    grad_clip_norm: float = 2.0
    checkpoint_every: int = 20
    init_mode_train: Optional[str] = "truth"
    init_mode_eval: Optional[str] = "truth"
    max_train_steps_per_epoch: int = -1
    best_model_by: str = "score"  # val_3d_rmse_m | score
    early_stop_patience: int = 12
    early_stop_min_delta: float = 0.1
    early_stop_min_epochs: int = 20
    learned_r_min_scale: float = 0.2
    learned_r_max_scale: float = 20.0
    learned_bias_abs_max_m: float = 30.0
    residual_winsor_q_low: float = 0.01
    residual_winsor_q_high: float = 0.99
    loss_alpha: float = 0.75
    loss_gamma: float = 1.5
    loss_dynamic_gamma: bool = True
    loss_scale_factor: float = 1.0


def load_train_config(path: str) -> TrainConfig:
    cfg = TrainConfig()
    p = Path(path)
    if not p.exists():
        return cfg
    data: Dict[str, Any]
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def save_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
