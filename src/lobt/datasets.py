"""Windowed torch datasets over FI-2010-style feature/label streams."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedLOBDataset(Dataset):
    """Sliding windows of T snapshots -> label at the window's last timestep.

    Sample i covers timesteps [i, i+T) and predicts labels[i+T-1] for each
    horizon. Windows never cross stream boundaries because each dataset wraps
    exactly one contiguous stream.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: dict[int, np.ndarray],
        window: int = 100,
        horizons: tuple[int, ...] = (10, 20, 30, 50, 100),
    ) -> None:
        if len(x) != len(next(iter(y.values()))):
            raise ValueError("features and labels must be aligned")
        if len(x) < window:
            raise ValueError(f"stream shorter ({len(x)}) than window ({window})")
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.horizons = horizons
        self.y = np.stack([y[k] for k in horizons], axis=1).astype(np.int64)
        self.window = window

    def __len__(self) -> int:
        return len(self.x) - self.window + 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.x[i : i + self.window]
        labels = self.y[i + self.window - 1]
        return torch.from_numpy(w.copy()), torch.from_numpy(labels.copy())


def class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"class missing from labels: counts={counts}")
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
