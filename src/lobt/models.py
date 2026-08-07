"""Models: logistic regression, MLP baselines, and the LOB transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

N_CLASSES = 3


class LogisticBaseline(nn.Module):
    """Multinomial logistic regression on the flattened window."""

    def __init__(self, window: int, n_features: int, n_horizons: int) -> None:
        super().__init__()
        self.heads = nn.Linear(window * n_features, N_CLASSES * n_horizons)
        self.n_horizons = n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = self.heads(x.reshape(b, -1))
        return out.reshape(b, self.n_horizons, N_CLASSES)


class MLPBaseline(nn.Module):
    """2-hidden-layer MLP on the flattened window."""

    def __init__(
        self,
        window: int,
        n_features: int,
        n_horizons: int,
        hidden: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window * n_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], N_CLASSES * n_horizons),
        )
        self.n_horizons = n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = self.net(x.reshape(b, -1))
        return out.reshape(b, self.n_horizons, N_CLASSES)


class LOBTransformer(nn.Module):
    """Encoder-only transformer over a window of LOB snapshots.

    Each snapshot (40 features) is embedded to d_model; learned positional
    embeddings over the window; pre-norm encoder layers; mean-pool; one
    classification head per horizon.
    """

    def __init__(
        self,
        window: int = 100,
        n_features: int = 40,
        n_horizons: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.Linear(d_model, N_CLASSES * n_horizons)
        self.n_horizons = n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        out = self.heads(h)
        return out.reshape(x.shape[0], self.n_horizons, N_CLASSES)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, window: int, n_features: int, n_horizons: int, **kw):
    if name == "logistic":
        return LogisticBaseline(window, n_features, n_horizons)
    if name == "mlp":
        return MLPBaseline(window, n_features, n_horizons)
    if name == "transformer":
        return LOBTransformer(
            window=window, n_features=n_features, n_horizons=n_horizons, **kw
        )
    raise ValueError(f"unknown model {name}")
