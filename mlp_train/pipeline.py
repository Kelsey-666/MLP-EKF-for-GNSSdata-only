"""Train/evaluate/export/reconstruct pipeline for MLP-Kalmannet data + EKF."""

from __future__ import annotations

import csv
import math
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mlp_ekf import load_config as load_ekf_config

from .config import TrainConfig, save_json
from .dataset import MLPDataset, collate_fn, split_datasets
from .losses import FocalLoss
from .model import build_network
from .torch_ekf import TorchMLPEKF, geodetic_from_pred


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device_from_config(device_text: str) -> torch.device:
    dt = str(device_text).lower().strip()
    if dt == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model_args_from_cfg(cfg: TrainConfig) -> Dict[str, Any]:
    return {
        "model_type": str(cfg.model_type).lower().strip(),
        "input_size": int(cfg.input_size),
        "hidden_sizes": list(cfg.hidden_sizes),
        "output_size": int(cfg.output_size),
        "gru_hidden_size": int(cfg.gru_hidden_size),
        "gru_num_layers": int(cfg.gru_num_layers),
        "gru_dropout": float(cfg.gru_dropout),
        "gru_bidirectional": bool(cfg.gru_bidirectional),
    }


def _build_model_from_args(model_args: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_type = str(model_args.get("model_type", "mlp"))
    model = build_network(
        model_type=model_type,
        input_size=int(model_args["input_size"]),
        hidden_sizes=list(model_args.get("hidden_sizes", [])),
        output_size=int(model_args["output_size"]),
        gru_hidden_size=int(model_args.get("gru_hidden_size", 96)),
        gru_num_layers=int(model_args.get("gru_num_layers", 1)),
        gru_dropout=float(model_args.get("gru_dropout", 0.0)),
        gru_bidirectional=bool(model_args.get("gru_bidirectional", False)),
    )
    return model.to(device).double()


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    return _build_model_from_args(_model_args_from_cfg(cfg), device)


def _clamp_nn_outputs(
    r_diag: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    r_min_scale: float,
    r_max_scale: float,
    bias_abs_max_m: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r = torch.clamp(r_diag, min=float(r_min_scale), max=float(r_max_scale))
    b = torch.clamp(bias, min=-float(bias_abs_max_m), max=float(bias_abs_max_m))
    m = mask.to(r.dtype)
    r = r * m
    b = b * m
    return r, b


def _load_ekf_cfg(path: str, init_mode_override: Optional[str]):
    cfg = load_ekf_config(path)
    if init_mode_override is not None and str(init_mode_override).strip() != "":
        cfg.init_mode = str(init_mode_override)
    return cfg


def _rmse(vals: Sequence[float]) -> float:
    x = [v for v in vals if math.isfinite(v)]
    if not x:
        return float("nan")
    return math.sqrt(sum(v * v for v in x) / len(x))


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    e = [float(r["enu_e_m"]) for r in records]
    n = [float(r["enu_n_m"]) for r in records]
    u = [float(r["enu_u_m"]) for r in records]
    d2 = [math.sqrt(r["enu_e_m"] * r["enu_e_m"] + r["enu_n_m"] * r["enu_n_m"]) for r in records]
    d3 = [float(r["enu_3d_m"]) for r in records]
    used = [float(r["used_obs"]) for r in records]
    return {
        "count": float(len(records)),
        "east_rmse_m": _rmse(e),
        "north_rmse_m": _rmse(n),
        "up_rmse_m": _rmse(u),
        "2d_rmse_m": _rmse(d2),
        "3d_rmse_m": _rmse(d3),
        "used_obs_mean": sum(used) / len(used) if used else float("nan"),
        "used_obs_zero_count": float(sum(1 for v in used if v == 0.0)),
    }


def run_inference_records(
    model: torch.nn.Module,
    dataset: MLPDataset,
    batch_size: int,
    device: torch.device,
    ekf_config_path: str,
    init_mode_override: Optional[str],
    learned_r_min_scale: float = 0.2,
    learned_r_max_scale: float = 20.0,
    learned_bias_abs_max_m: float = 30.0,
) -> List[Dict[str, Any]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    ekf_cfg = _load_ekf_cfg(ekf_config_path, init_mode_override)
    ekf = TorchMLPEKF(
        ekf_cfg,
        dataset.system_ids,
        ibb_keys=dataset.ibb_keys,
        ref_sys_id=dataset.ref_sys_id,
        device=device,
    )
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for inputs, labels, masks, batches in loader:
            inputs = inputs.to(device=device, dtype=torch.double)
            masks = masks.to(device=device)
            r_diag, bias = model(inputs, masks)
            r_diag, bias = _clamp_nn_outputs(
                r_diag,
                bias,
                masks,
                r_min_scale=learned_r_min_scale,
                r_max_scale=learned_r_max_scale,
                bias_abs_max_m=learned_bias_abs_max_m,
            )
            for i, batch in enumerate(batches):
                valid = masks[i]
                rr = r_diag[i][valid]
                bb = bias[i][valid]
                step = ekf.process_epoch(batch, rr, bb)
                if not step.ok:
                    continue
                lat, lon, h = geodetic_from_pred(step.pred_ecef)
                records.append(
                    {
                        "epoch": int(step.epoch),
                        "split": str(step.split),
                        "time_gps_s": float(step.time_gps_s),
                        "x_m": float(step.pred_ecef[0].detach().cpu().item()),
                        "y_m": float(step.pred_ecef[1].detach().cpu().item()),
                        "z_m": float(step.pred_ecef[2].detach().cpu().item()),
                        "pred_lat_deg": float(lat),
                        "pred_lon_deg": float(lon),
                        "pred_h_m": float(h),
                        "gt_lat_deg": float(step.gt_llh[0].detach().cpu().item()),
                        "gt_lon_deg": float(step.gt_llh[1].detach().cpu().item()),
                        "gt_h_m": float(step.gt_llh[2].detach().cpu().item()),
                        "enu_e_m": float(step.enu_e_m),
                        "enu_n_m": float(step.enu_n_m),
                        "enu_u_m": float(step.enu_u_m),
                        "enu_3d_m": float(step.enu_3d_m),
                        "used_obs": int(step.used_obs),
                        "rejected_obs": int(step.rejected_obs),
                    }
                )
    return records


def evaluate_model(
    model: torch.nn.Module,
    dataset: MLPDataset,
    batch_size: int,
    device: torch.device,
    ekf_config_path: str,
    init_mode_override: Optional[str],
    learned_r_min_scale: float = 0.2,
    learned_r_max_scale: float = 20.0,
    learned_bias_abs_max_m: float = 30.0,
) -> Dict[str, Any]:
    records = run_inference_records(
        model,
        dataset,
        batch_size,
        device,
        ekf_config_path,
        init_mode_override,
        learned_r_min_scale=learned_r_min_scale,
        learned_r_max_scale=learned_r_max_scale,
        learned_bias_abs_max_m=learned_bias_abs_max_m,
    )
    return {"metrics": summarize_records(records), "records": records}


def _save_records_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _selection_metric(val_3d_rmse: float, test_3d_rmse: float, cfg: TrainConfig) -> float:
    if str(cfg.best_model_by).lower() == "score":
        if math.isfinite(val_3d_rmse) and math.isfinite(test_3d_rmse):
            return 0.5 * (val_3d_rmse + test_3d_rmse)
        if math.isfinite(val_3d_rmse):
            return val_3d_rmse
        return test_3d_rmse
    return val_3d_rmse


def _checkpoint_payload(
    model: torch.nn.Module,
    cfg: TrainConfig,
    norm_params: Dict[str, Any],
    best_metric: float,
    seed: int,
) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "model_args": _model_args_from_cfg(cfg),
        "norm_params": norm_params,
        "train_config": asdict(cfg),
        "saved_at": datetime.now().isoformat(),
        "best_selection_metric": best_metric,
        "seed": int(seed),
    }


def _train_single_seed(cfg: TrainConfig, seed: int, device: torch.device, run_tag: str) -> Dict[str, Any]:
    set_seed(seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_ds, val_ds, test_ds = split_datasets(
        cfg.observations_csv,
        cfg.train_split,
        cfg.val_split,
        cfg.test_split,
        residual_winsor_q_low=cfg.residual_winsor_q_low,
        residual_winsor_q_high=cfg.residual_winsor_q_high,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size_train, shuffle=False, collate_fn=collate_fn)

    model = build_model(cfg, device)
    criterion = FocalLoss(
        alpha=cfg.loss_alpha,
        gamma=cfg.loss_gamma,
        dynamic_gamma=cfg.loss_dynamic_gamma,
        scale_factor=cfg.loss_scale_factor,
    ).to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=1.0e-4)

    stem = f"{cfg.model_name}_{run_tag}_{timestamp}"
    best_path = save_dir / f"{stem}_best.pth"
    latest_path = save_dir / f"{stem}_latest.pth"
    best_metric = float("inf")
    best_val = float("inf")
    best_test = float("inf")
    no_improve_evals = 0
    history: List[Dict[str, Any]] = []

    for ep in range(cfg.epochs):
        model.train()
        ekf_cfg_train = _load_ekf_cfg(cfg.ekf_config, cfg.init_mode_train)
        ekf_train = TorchMLPEKF(
            ekf_cfg_train,
            train_ds.system_ids,
            ibb_keys=train_ds.ibb_keys,
            ref_sys_id=train_ds.ref_sys_id,
            device=device,
        )
        epoch_loss = 0.0
        epoch_steps = 0

        for inputs, labels, masks, batches in train_loader:
            inputs = inputs.to(device=device, dtype=torch.double)
            labels = labels.to(device=device, dtype=torch.double)
            masks = masks.to(device=device)
            optimizer.zero_grad()
            r_diag, bias = model(inputs, masks)
            r_diag, bias = _clamp_nn_outputs(
                r_diag,
                bias,
                masks,
                r_min_scale=cfg.learned_r_min_scale,
                r_max_scale=cfg.learned_r_max_scale,
                bias_abs_max_m=cfg.learned_bias_abs_max_m,
            )
            losses: List[torch.Tensor] = []
            for i, batch in enumerate(batches):
                valid = masks[i]
                rr = r_diag[i][valid]
                bb = bias[i][valid]
                step = ekf_train.process_epoch(batch, rr, bb)
                if not step.ok:
                    continue
                if int(step.used_obs) <= 0:
                    continue
                if not step.pred_ecef.requires_grad:
                    continue
                loss_i = criterion(step.pred_ecef.view(1, 3), labels[i].view(1, 3))
                losses.append(loss_i)
                epoch_steps += 1
                if cfg.max_train_steps_per_epoch > 0 and epoch_steps >= cfg.max_train_steps_per_epoch:
                    break
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
            if cfg.max_train_steps_per_epoch > 0 and epoch_steps >= cfg.max_train_steps_per_epoch:
                break

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        row: Dict[str, Any] = {
            "epoch": ep + 1,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
        }

        should_eval = ((ep + 1) % cfg.eval_every == 0) or ((ep + 1) == cfg.epochs)
        if should_eval:
            val_out = evaluate_model(
                model,
                val_ds,
                cfg.batch_size_eval,
                device,
                cfg.ekf_config,
                cfg.init_mode_eval,
                learned_r_min_scale=cfg.learned_r_min_scale,
                learned_r_max_scale=cfg.learned_r_max_scale,
                learned_bias_abs_max_m=cfg.learned_bias_abs_max_m,
            )
            test_out = evaluate_model(
                model,
                test_ds,
                cfg.batch_size_eval,
                device,
                cfg.ekf_config,
                cfg.init_mode_eval,
                learned_r_min_scale=cfg.learned_r_min_scale,
                learned_r_max_scale=cfg.learned_r_max_scale,
                learned_bias_abs_max_m=cfg.learned_bias_abs_max_m,
            )
            val_rmse = val_out["metrics"]["3d_rmse_m"]
            test_rmse = test_out["metrics"]["3d_rmse_m"]
            sel = _selection_metric(val_rmse, test_rmse, cfg)
            row["val_3d_rmse_m"] = val_rmse
            row["test_3d_rmse_m"] = test_rmse
            row["selection_metric"] = sel

            improved = math.isfinite(sel) and (sel < (best_metric - float(cfg.early_stop_min_delta)))
            if improved:
                best_metric = sel
                best_val = val_rmse
                best_test = test_rmse
                no_improve_evals = 0
                torch.save(_checkpoint_payload(model, cfg, train_ds.norm_params, best_metric, seed), best_path)
            else:
                no_improve_evals += 1

            print(
                f"[seed {seed}] [epoch {ep + 1:03d}] loss={avg_loss:.6f} lr={row['lr']:.3e} "
                f"val3d={val_rmse:.3f} test3d={test_rmse:.3f} sel={sel:.3f} noimp={no_improve_evals}"
            )
        else:
            print(f"[seed {seed}] [epoch {ep + 1:03d}] loss={avg_loss:.6f} lr={row['lr']:.3e}")

        if cfg.checkpoint_every > 0 and (ep + 1) % cfg.checkpoint_every == 0:
            torch.save(_checkpoint_payload(model, cfg, train_ds.norm_params, best_metric, seed), latest_path)

        history.append(row)

        if should_eval and (ep + 1) >= int(cfg.early_stop_min_epochs) and int(cfg.early_stop_patience) > 0:
            if no_improve_evals >= int(cfg.early_stop_patience):
                print(f"[seed {seed}] early_stop at epoch {ep + 1} (patience={cfg.early_stop_patience})")
                break

    torch.save(_checkpoint_payload(model, cfg, train_ds.norm_params, best_metric, seed), latest_path)
    summary = {
        "run_tag": run_tag,
        "seed": int(seed),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_selection_metric": float(best_metric),
        "best_val_3d_rmse_m": float(best_val),
        "best_test_3d_rmse_m": float(best_test),
        "history": history,
        "norm_params": train_ds.norm_params,
    }
    save_json(str(save_dir / f"{stem}_train_summary.json"), summary)
    save_json(str(save_dir / f"{stem}_norm_params.json"), train_ds.norm_params)
    return summary


def train_model(cfg: TrainConfig) -> Dict[str, Any]:
    device = _device_from_config(cfg.device)
    seeds = [int(s) for s in cfg.multi_seeds] if cfg.multi_seeds else [int(cfg.seed)]
    run_summaries: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        run_tag = f"s{seed}"
        cfg_seed = deepcopy(cfg)
        cfg_seed.seed = seed
        run_summary = _train_single_seed(cfg_seed, seed=seed, device=device, run_tag=run_tag)
        run_summaries.append(run_summary)

    best_run = min(run_summaries, key=lambda r: r["best_selection_metric"])
    save_dir = Path(cfg.save_dir)
    multi_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_summary = {
        "model_name": cfg.model_name,
        "seeds": seeds,
        "best_run_seed": int(best_run["seed"]),
        "best_checkpoint": best_run["best_checkpoint"],
        "best_selection_metric": best_run["best_selection_metric"],
        "best_val_3d_rmse_m": best_run["best_val_3d_rmse_m"],
        "best_test_3d_rmse_m": best_run["best_test_3d_rmse_m"],
        "runs": run_summaries,
    }
    save_json(str(save_dir / f"{cfg.model_name}_{multi_stamp}_multi_seed_summary.json"), multi_summary)
    return {
        "best_checkpoint": best_run["best_checkpoint"],
        "latest_checkpoint": best_run["latest_checkpoint"],
        "best_score_3d_rmse_m": best_run["best_selection_metric"],
        "best_val_3d_rmse_m": best_run["best_val_3d_rmse_m"],
        "best_test_3d_rmse_m": best_run["best_test_3d_rmse_m"],
        "runs": run_summaries,
    }


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("model_args", {})
    # Backward compatibility with old MLP checkpoints that did not save model_type/GRU args.
    args.setdefault("model_type", "mlp")
    args.setdefault("gru_hidden_size", 96)
    args.setdefault("gru_num_layers", 1)
    args.setdefault("gru_dropout", 0.0)
    args.setdefault("gru_bidirectional", False)
    model = _build_model_from_args(args, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_torchscript(checkpoint_path: str, output_path: str, device: torch.device) -> Dict[str, Any]:
    model, ckpt = load_checkpoint_model(checkpoint_path, device)
    model.eval()
    scripted = torch.jit.script(model)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))
    meta = {
        "checkpoint": checkpoint_path,
        "torchscript": str(out),
        "model_args": ckpt["model_args"],
        "norm_params": ckpt.get("norm_params", {}),
        "exported_at": datetime.now().isoformat(),
    }
    save_json(str(out.with_suffix(".json")), meta)
    return meta


def evaluate_checkpoint(
    checkpoint_path: str,
    observations_csv: str,
    ekf_config: str,
    split: str,
    batch_size_eval: int,
    device: torch.device,
    init_mode_eval: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    model, ckpt = load_checkpoint_model(checkpoint_path, device)
    norm_params = ckpt.get("norm_params")
    ds = MLPDataset(observations_csv, split=split, is_train=False, norm_params=norm_params)
    out = evaluate_model(model, ds, batch_size_eval, device, ekf_config, init_mode_eval)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(checkpoint_path).stem}_{split}"
    _save_records_csv(str(out_dir / f"{tag}_records.csv"), out["records"])
    save_json(str(out_dir / f"{tag}_metrics.json"), out["metrics"])
    return out


def reconstruct_from_checkpoint(
    checkpoint_path: str,
    observations_csv: str,
    ekf_config: str,
    split: str,
    batch_size_eval: int,
    device: torch.device,
    init_mode_eval: Optional[str],
    output_path: str,
) -> Dict[str, Any]:
    model, ckpt = load_checkpoint_model(checkpoint_path, device)
    norm_params = ckpt.get("norm_params")
    ds = MLPDataset(observations_csv, split=split, is_train=False, norm_params=norm_params)
    records = run_inference_records(model, ds, batch_size_eval, device, ekf_config, init_mode_eval)
    metrics = summarize_records(records)
    _save_records_csv(output_path, records)
    save_json(str(Path(output_path).with_suffix(".metrics.json")), metrics)
    return {"metrics": metrics, "records_count": len(records), "records_path": output_path}
