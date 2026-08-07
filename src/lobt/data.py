"""FI-2010 data loading (DeepLOB distribution variant).

Files: Train_Dst_NoAuction_DecPre_CF_7.txt (days 1-7) and
Test_Dst_NoAuction_DecPre_CF_{7,8,9}.txt (days 8, 9, 10), each a matrix of
shape (149, n_samples):

- rows 0..39   : LOB top-10 levels, repeating (ask_p, ask_sz, bid_p, bid_sz)
                 per level, DecPre-normalized. These are our model inputs.
- rows 40..143 : handcrafted features from Ntakaris et al. (unused here; the
                 point is to learn from the raw book).
- rows 144..148: labels for horizons k = 10, 20, 30, 50, 100 encoded
                 1=up, 2=stationary, 3=down.

We remap labels to ours: DOWN=0, STATIONARY=1, UP=2.

Loading 600MB of text is slow, so each txt is cached as .npy on first load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .labels import DOWN, STATIONARY, UP

HORIZONS = (10, 20, 30, 50, 100)
N_LOB_FEATURES = 40

TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILES = (
    "Test_Dst_NoAuction_DecPre_CF_7.txt",
    "Test_Dst_NoAuction_DecPre_CF_8.txt",
    "Test_Dst_NoAuction_DecPre_CF_9.txt",
)

# FI-2010 encoding -> ours
_LABEL_MAP = {1: UP, 2: STATIONARY, 3: DOWN}


def _cached_matrix(path: Path) -> np.ndarray:
    """Load a (149, n) matrix, caching the parsed text as .npy."""
    cache = path.with_suffix(".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache, mmap_mode="r")
    mat = np.loadtxt(path)
    if mat.ndim != 2 or mat.shape[0] != 149:
        raise ValueError(f"{path.name}: expected (149, n), got {mat.shape}")
    mat = mat.astype(np.float32)
    np.save(cache, mat)
    return np.load(cache, mmap_mode="r")


def _split_features_labels(mat: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return (features (n, 40) float32, {horizon: labels (n,) int64})."""
    x = np.ascontiguousarray(mat[:N_LOB_FEATURES].T, dtype=np.float32)
    labels: dict[int, np.ndarray] = {}
    for i, k in enumerate(HORIZONS):
        raw = mat[144 + i].astype(np.int64)
        y = np.empty_like(raw)
        for src, dst in _LABEL_MAP.items():
            y[raw == src] = dst
        bad = ~np.isin(raw, list(_LABEL_MAP))
        if bad.any():
            raise ValueError(f"unexpected label values: {np.unique(raw[bad])}")
        labels[k] = y
    return x, labels


def detect_segments(x: np.ndarray, thresh: float = 0.2) -> list[tuple[int, int]]:
    """Split a concatenated multi-stock stream into per-stock segments.

    FI-2010 files concatenate 5 stocks back-to-back with no marker. Segment
    boundaries show up as large relative jumps in the best-ask price (the
    stocks trade at very different price scales). Verified against the known
    structure: train -> 5 segments, each test day -> 5 segments.
    """
    a = x[:, 0].astype(np.float64)
    jumps = np.where(np.abs(np.diff(a)) / a[:-1] > thresh)[0]
    edges = [0, *(jumps + 1), len(x)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def load_fi2010(
    data_dir: str | Path,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, dict[int, np.ndarray]]:
    """Load (train_x, train_y, test_x, test_y).

    Train = days 1-7 file. Test = concatenation of the three test files
    (days 8-10), in chronological order.
    """
    data_dir = Path(data_dir)
    train_x, train_y = _split_features_labels(_cached_matrix(data_dir / TRAIN_FILE))
    xs, ys = [], []
    for name in TEST_FILES:
        x, y = _split_features_labels(_cached_matrix(data_dir / name))
        xs.append(x)
        ys.append(y)
    test_x = np.concatenate(xs, axis=0)
    test_y = {k: np.concatenate([y[k] for y in ys], axis=0) for k in HORIZONS}
    return train_x, train_y, test_x, test_y


def train_val_split_segmented(
    x: np.ndarray,
    y: dict[int, np.ndarray],
    segments: list[tuple[int, int]],
    val_frac: float = 0.1,
) -> tuple[
    list[np.ndarray],
    list[dict[int, np.ndarray]],
    list[np.ndarray],
    list[dict[int, np.ndarray]],
]:
    """Chronological split within EACH segment: the tail val_frac of every
    stock's stream is validation. Returns per-segment lists so windows can
    never cross a stock boundary or the train/val boundary.
    """
    tr_xs, tr_ys, va_xs, va_ys = [], [], [], []
    for s, e in segments:
        cut = s + int((e - s) * (1.0 - val_frac))
        tr_xs.append(x[s:cut])
        tr_ys.append({k: v[s:cut] for k, v in y.items()})
        va_xs.append(x[cut:e])
        va_ys.append({k: v[cut:e] for k, v in y.items()})
    return tr_xs, tr_ys, va_xs, va_ys


def train_val_split(
    x: np.ndarray, y: dict[int, np.ndarray], val_frac: float = 0.1
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, dict[int, np.ndarray]]:
    """Chronological split: last val_frac of the train stream is validation."""
    n = len(x)
    cut = int(n * (1.0 - val_frac))
    return (
        x[:cut],
        {k: v[:cut] for k, v in y.items()},
        x[cut:],
        {k: v[cut:] for k, v in y.items()},
    )


def synthetic_tape(
    n: int = 5000, seed: int = 0
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Small synthetic LOB stream with FI-2010-shaped features and labels
    computed by our own label math. Used by tests; never for results.
    """
    from .labels import fi2010_labels, label_index_range

    rng = np.random.default_rng(seed)
    mid = 100.0 * np.cumprod(1.0 + rng.normal(0, 5e-5, size=n))
    spread = 0.02 + 0.005 * rng.random(n)
    x = np.zeros((n, N_LOB_FEATURES), dtype=np.float32)
    for lvl in range(10):
        ask = mid + spread / 2 + 0.01 * lvl
        bid = mid - spread / 2 - 0.01 * lvl
        x[:, 4 * lvl + 0] = ask
        x[:, 4 * lvl + 1] = rng.exponential(100, n)
        x[:, 4 * lvl + 2] = bid
        x[:, 4 * lvl + 3] = rng.exponential(100, n)
    labels: dict[int, np.ndarray] = {}
    for k in HORIZONS:
        full = np.full(n, STATIONARY, dtype=np.int64)
        lab = fi2010_labels(mid, k=k, alpha=1e-5)
        start, stop = label_index_range(n, k)
        full[start:stop] = lab
        labels[k] = full
    return x, labels


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
