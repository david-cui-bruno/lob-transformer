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


class ConvStem(nn.Module):
    """DeepLOB-style convolutional feature extractor over the raw book.

    Input (b, T, 40) viewed as (b, 1, T, 40). Three stages:
    1. (1,2) convs pair price with size at each level/side -> (b, c, T, 20)
    2. (1,2) convs pair bid with ask per level              -> (b, c, T, 10)
    3. (1,10) conv fuses all 10 levels                      -> (b, c, T, 1)
    Temporal (k,1) convs with LeakyReLU throughout, as in Zhang et al. 2019.
    Output: (b, T, c) sequence for the transformer encoder.
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()

        def block(in_c, out_c, kw):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=(1, kw), stride=(1, kw)),
                nn.LeakyReLU(0.01),
                nn.Conv2d(out_c, out_c, kernel_size=(4, 1), padding=(2, 0)),
                nn.LeakyReLU(0.01),
                nn.Conv2d(out_c, out_c, kernel_size=(4, 1), padding=(1, 0)),
                nn.LeakyReLU(0.01),
            )

        self.s1 = block(1, channels, 2)
        self.s2 = block(channels, channels, 2)
        self.s3 = block(channels, channels, 10)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        h = x.unsqueeze(1)          # (b, 1, T, 40)
        h = self.s1(h)
        h = self.s2(h)
        h = self.s3(h)              # (b, c, T', 1)
        h = h.squeeze(-1).transpose(1, 2)  # (b, T', c)
        return h


class ConvTransformer(nn.Module):
    """ConvStem + transformer encoder (v2 model, DeepLOB-inspired stem)."""

    def __init__(
        self,
        window: int = 100,
        n_features: int = 40,
        n_horizons: int = 5,
        channels: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = ConvStem(channels)
        self.proj = nn.Linear(channels, d_model)
        # stem temporal convs shift length slightly; size pos for window + margin
        self.pos = nn.Parameter(torch.zeros(1, window + 8, d_model))
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
        h = self.stem(x)                       # (b, T', c)
        h = self.proj(h) + self.pos[:, : h.shape[1]]
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
    if name == "convtransformer":
        return ConvTransformer(
            window=window, n_features=n_features, n_horizons=n_horizons, **kw
        )
    raise ValueError(f"unknown model {name}")
