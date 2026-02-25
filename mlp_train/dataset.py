"""Dataset adapters for MLP-Kalmannet training on preprocessed CSV observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from mlp_ekf import load_epoch_batches
from mlp_ekf.models import EpochBatch


FEATURE_COLUMNS = ("cno_dbhz", "azim_deg", "elev_deg", "residual_m")


@dataclass
class EpochSample:
    batch: EpochBatch
    features: torch.Tensor  # [n_obs, 4]
    gt_llh: torch.Tensor  # [3]


def _feature_tensor(batch: EpochBatch) -> torch.Tensor:
    rows: List[List[float]] = []
    for o in batch.observations:
        rows.append([float(o.cno_dbhz), float(o.azim_deg), float(o.elev_deg), float(o.residual_m)])
    if not rows:
        return torch.zeros((0, len(FEATURE_COLUMNS)), dtype=torch.double)
    return torch.tensor(rows, dtype=torch.double)


def _gt_tensor(batch: EpochBatch) -> torch.Tensor:
    if not batch.observations:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
    o0 = batch.observations[0]
    return torch.tensor([float(o0.gt_lat_deg), float(o0.gt_lon_deg), float(o0.gt_h_m)], dtype=torch.double)


def compute_norm_params(
    samples: Sequence[EpochSample],
    residual_winsor_q_low: float = 0.01,
    residual_winsor_q_high: float = 0.99,
) -> Dict[str, List[float]]:
    if not samples:
        return {"mean": [0.0, 0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0, 1.0]}
    all_rows = [s.features for s in samples if s.features.numel() > 0]
    if not all_rows:
        return {
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "clip_low": [float("-inf")] * 4,
            "clip_high": [float("inf")] * 4,
        }
    cat = torch.cat(all_rows, dim=0).clone()
    ql = max(0.0, min(0.5, float(residual_winsor_q_low)))
    qh = min(1.0, max(0.5, float(residual_winsor_q_high)))
    res_col = cat[:, 3]
    low_v = torch.quantile(res_col, ql)
    high_v = torch.quantile(res_col, qh)
    cat[:, 3] = torch.clamp(res_col, min=low_v, max=high_v)
    mean = cat.mean(dim=0)
    std = cat.std(dim=0)
    std = torch.where(std < 1.0e-6, torch.ones_like(std), std)
    clip_low = [float("-inf"), float("-inf"), float("-inf"), float(low_v)]
    clip_high = [float("inf"), float("inf"), float("inf"), float(high_v)]
    return {"mean": mean.tolist(), "std": std.tolist(), "clip_low": clip_low, "clip_high": clip_high}


def apply_norm(features: torch.Tensor, norm_params: Dict[str, List[float]]) -> torch.Tensor:
    if features.numel() == 0:
        return features.clone()
    clip_low = torch.tensor(norm_params.get("clip_low", [float("-inf")] * 4), dtype=torch.double).view(1, -1)
    clip_high = torch.tensor(norm_params.get("clip_high", [float("inf")] * 4), dtype=torch.double).view(1, -1)
    x = torch.maximum(features, clip_low)
    x = torch.minimum(x, clip_high)
    mean = torch.tensor(norm_params["mean"], dtype=torch.double).view(1, -1)
    std = torch.tensor(norm_params["std"], dtype=torch.double).view(1, -1)
    return (x - mean) / std


class MLPDataset(Dataset):
    """Epoch dataset with variable-length per-epoch observation layout."""

    def __init__(
        self,
        observations_csv: str,
        split: Optional[str] = None,
        is_train: bool = True,
        norm_params: Optional[Dict[str, List[float]]] = None,
        residual_winsor_q_low: float = 0.01,
        residual_winsor_q_high: float = 0.99,
    ) -> None:
        batches = load_epoch_batches(observations_csv)
        if split is not None:
            batches = [b for b in batches if b.split == split]
        raw_samples = [EpochSample(batch=b, features=_feature_tensor(b), gt_llh=_gt_tensor(b)) for b in batches]
        if is_train:
            self.norm_params = norm_params or compute_norm_params(
                raw_samples,
                residual_winsor_q_low=residual_winsor_q_low,
                residual_winsor_q_high=residual_winsor_q_high,
            )
        else:
            self.norm_params = norm_params or compute_norm_params(
                raw_samples,
                residual_winsor_q_low=residual_winsor_q_low,
                residual_winsor_q_high=residual_winsor_q_high,
            )
        self.samples: List[EpochSample] = []
        for s in raw_samples:
            self.samples.append(EpochSample(batch=s.batch, features=apply_norm(s.features, self.norm_params), gt_llh=s.gt_llh))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, EpochBatch]:
        s = self.samples[idx]
        return s.features, s.gt_llh, s.batch

    @property
    def system_ids(self) -> List[int]:
        ids = set()
        for s in self.samples:
            for o in s.batch.observations:
                ids.add(int(o.gnss_id_raw))
        return sorted(ids)

    @property
    def ref_sys_id(self) -> int:
        systems = self.system_ids
        if not systems:
            return 0
        return 0 if 0 in systems else systems[0]

    @property
    def ibb_keys(self) -> List[Tuple[int, int]]:
        by_sys: Dict[int, set[int]] = {}
        for s in self.samples:
            for o in s.batch.observations:
                sid = int(o.gnss_id_raw)
                sig = int(o.sig_id)
                by_sys.setdefault(sid, set()).add(sig)
        keys: List[Tuple[int, int]] = []
        for sid in sorted(by_sys.keys()):
            sigs = sorted(by_sys[sid])
            if len(sigs) <= 1:
                continue
            ref_sig = sigs[0]
            for sig in sigs:
                if sig == ref_sig:
                    continue
                keys.append((sid, sig))
        return keys


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, EpochBatch]]):
    features, labels, batches = zip(*batch)
    max_len = max((f.shape[0] for f in features), default=0)
    feat_dim = features[0].shape[1] if features and features[0].ndim == 2 else len(FEATURE_COLUMNS)
    n = len(features)
    padded = torch.zeros((n, max_len, feat_dim), dtype=torch.double)
    mask = torch.zeros((n, max_len), dtype=torch.bool)
    for i, f in enumerate(features):
        if f.shape[0] == 0:
            continue
        padded[i, : f.shape[0], :] = f
        mask[i, : f.shape[0]] = True
    labels_t = torch.stack(labels) if labels else torch.zeros((0, 3), dtype=torch.double)
    return padded, labels_t, mask, list(batches)


def split_datasets(
    observations_csv: str,
    train_split: str,
    val_split: str,
    test_split: str,
    residual_winsor_q_low: float = 0.01,
    residual_winsor_q_high: float = 0.99,
) -> Tuple[MLPDataset, MLPDataset, MLPDataset]:
    train_ds = MLPDataset(
        observations_csv,
        split=train_split,
        is_train=True,
        norm_params=None,
        residual_winsor_q_low=residual_winsor_q_low,
        residual_winsor_q_high=residual_winsor_q_high,
    )
    val_ds = MLPDataset(
        observations_csv,
        split=val_split,
        is_train=False,
        norm_params=train_ds.norm_params,
        residual_winsor_q_low=residual_winsor_q_low,
        residual_winsor_q_high=residual_winsor_q_high,
    )
    test_ds = MLPDataset(
        observations_csv,
        split=test_split,
        is_train=False,
        norm_params=train_ds.norm_params,
        residual_winsor_q_low=residual_winsor_q_low,
        residual_winsor_q_high=residual_winsor_q_high,
    )
    return train_ds, val_ds, test_ds
