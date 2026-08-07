"""FI-2010 style label computation for mid-price direction prediction.

Reference: Ntakaris et al. 2018, "Benchmark Dataset for Mid-Price Forecasting of
Limit Order Book Data with Machine Learning Methods" (arXiv:1705.03233).

Label definition (their Eq. 5-6, the "smoothed" variant used by DeepLOB):

    m_minus(t) = mean(mid[t-k+1 .. t])        (past window, inclusive of t)
    m_plus(t)  = mean(mid[t+1 .. t+k])        (future window, exclusive of t)
    l(t)       = (m_plus(t) - m_minus(t)) / m_minus(t)

    label = up (2)         if l >  alpha
            stationary (1) if -alpha <= l <= alpha
            down (0)       if l < -alpha

Only timestamps with a full past and future window get labels; callers must
align features to the labeled index range [k-1, n-k).
"""

from __future__ import annotations

import numpy as np

DOWN, STATIONARY, UP = 0, 1, 2


def mid_price(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    """Mid price from best bid/ask arrays (float64 for label math)."""
    return (best_bid.astype(np.float64) + best_ask.astype(np.float64)) / 2.0


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Rolling mean of window k; out[i] = mean(x[i .. i+k-1]). Length n-k+1."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(x) < k:
        return np.empty(0, dtype=np.float64)
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    return (c[k:] - c[:-k]) / k


def fi2010_labels(mid: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Smoothed direction labels for horizon k and threshold alpha.

    Returns an array of length n - 2*k + 1... conceptually; concretely:
    labels[i] corresponds to original index t = i + (k - 1), valid while a
    full future window exists, i.e. t <= n - k - 1.

    Output length: n - 2k + 1 (empty if n < 2k).
    """
    n = len(mid)
    if n < 2 * k:
        return np.empty(0, dtype=np.int64)
    # m_minus[t] over t in [k-1, n-1]; take t up to n-k-1
    m_minus_all = rolling_mean(mid, k)          # index i -> t = i + k - 1
    m_minus = m_minus_all[: n - 2 * k + 1]
    # m_plus[t] = mean(mid[t+1 .. t+k]) for t in [k-1, n-k-1]
    m_plus_all = rolling_mean(mid, k)           # index j -> window starts at j
    m_plus = m_plus_all[k:]                     # start j = t+1 = k ... n-k
    rel = (m_plus - m_minus) / m_minus
    labels = np.full(rel.shape, STATIONARY, dtype=np.int64)
    labels[rel > alpha] = UP
    labels[rel < -alpha] = DOWN
    return labels


def label_index_range(n: int, k: int) -> tuple[int, int]:
    """Original-timestamp range [start, stop) that fi2010_labels covers."""
    if n < 2 * k:
        return (0, 0)
    return (k - 1, n - k)
