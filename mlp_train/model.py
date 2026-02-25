"""LF-compatible models: per-observation MLP or GRU outputs R_diag and bias."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_sizes: Iterable[int]) -> tuple[nn.Sequential, int]:
    hs: List[int] = list(hidden_sizes)
    layers: List[nn.Module] = []
    cur_dim = in_dim
    for h in hs:
        layers.append(nn.Linear(cur_dim, h))
        layers.append(nn.ReLU())
        cur_dim = h
    return nn.Sequential(*layers), cur_dim


def _apply_output_mask(r_diag: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None):
    if mask is not None:
        r_diag = r_diag * mask.to(r_diag.dtype)
        bias = bias * mask.to(bias.dtype)
    return r_diag, bias


class BasicModel(nn.Module):
    """Per-observation MLP."""

    def __init__(self, input_size: int, hidden_sizes: Iterable[int], output_size: int = 2) -> None:
        super().__init__()
        self.backbone, out_dim = _build_mlp(input_size, hidden_sizes)
        self.fc_out = nn.Linear(out_dim, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, N, F]
        b, n, f = x.shape
        y = self.backbone(x.reshape(b * n, f))
        out = self.fc_out(y).reshape(b, n, -1)
        r_diag = self.softplus(out[:, :, 0]) + 1.0e-6
        bias = out[:, :, 1]
        return _apply_output_mask(r_diag, bias, mask)


class GRUModel(nn.Module):
    """Per-observation GRU encoder + MLP head."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        output_size: int = 2,
        gru_hidden_size: int = 96,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.0,
        gru_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=int(gru_hidden_size),
            num_layers=int(gru_num_layers),
            dropout=float(gru_dropout) if int(gru_num_layers) > 1 else 0.0,
            bidirectional=bool(gru_bidirectional),
            batch_first=True,
        )
        head_in = int(gru_hidden_size) * (2 if bool(gru_bidirectional) else 1)
        self.head, head_out = _build_mlp(head_in, hidden_sizes)
        self.fc_out = nn.Linear(head_out, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, N, F]
        y, _ = self.gru(x)
        b, n, d = y.shape
        h = self.head(y.reshape(b * n, d))
        out = self.fc_out(h).reshape(b, n, -1)
        r_diag = self.softplus(out[:, :, 0]) + 1.0e-6
        bias = out[:, :, 1]
        return _apply_output_mask(r_diag, bias, mask)


def build_network(
    model_type: str,
    input_size: int,
    hidden_sizes: Iterable[int],
    output_size: int = 2,
    gru_hidden_size: int = 96,
    gru_num_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_bidirectional: bool = False,
) -> nn.Module:
    mode = str(model_type).strip().lower()
    if mode == "gru":
        return GRUModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            gru_dropout=gru_dropout,
            gru_bidirectional=gru_bidirectional,
        )
    return BasicModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
