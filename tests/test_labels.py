"""Unit tests for FI-2010 label math."""

import numpy as np
import pytest

from lobt.labels import (
    DOWN,
    STATIONARY,
    UP,
    fi2010_labels,
    label_index_range,
    mid_price,
    rolling_mean,
)


class TestRollingMean:
    def test_basic(self):
        out = rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        np.testing.assert_allclose(out, [1.5, 2.5, 3.5])

    def test_k1_identity(self):
        x = np.array([3.0, 1.0, 4.0])
        np.testing.assert_allclose(rolling_mean(x, 1), x)

    def test_short_input(self):
        assert len(rolling_mean(np.array([1.0]), 2)) == 0

    def test_bad_k(self):
        with pytest.raises(ValueError):
            rolling_mean(np.array([1.0]), 0)


class TestMidPrice:
    def test_mid(self):
        np.testing.assert_allclose(
            mid_price(np.array([99.0, 100.0]), np.array([101.0, 102.0])),
            [100.0, 101.0],
        )


class TestFi2010Labels:
    def test_flat_series_is_stationary(self):
        mid = np.full(50, 100.0)
        labels = fi2010_labels(mid, k=5, alpha=0.001)
        assert len(labels) == 50 - 10 + 1
        assert (labels == STATIONARY).all()

    def test_monotonic_up(self):
        mid = np.linspace(100.0, 110.0, 60)  # steadily rising ~0.17% per step
        labels = fi2010_labels(mid, k=5, alpha=0.0001)
        assert (labels == UP).all()

    def test_monotonic_down(self):
        mid = np.linspace(110.0, 100.0, 60)
        labels = fi2010_labels(mid, k=5, alpha=0.0001)
        assert (labels == DOWN).all()

    def test_step_change_detected_at_boundary(self):
        # 30 flat at 100, then 30 flat at 101: labels near the step must be UP.
        mid = np.concatenate([np.full(30, 100.0), np.full(30, 101.0)])
        k = 5
        labels = fi2010_labels(mid, k=k, alpha=0.001)
        start, stop = label_index_range(len(mid), k)
        # t=29 is the last pre-step index: future window fully at 101 -> UP
        assert labels[29 - start] == UP
        # far before the step: both windows flat at 100 -> STATIONARY
        assert labels[10 - start] == STATIONARY
        # far after the step: both windows flat at 101 -> STATIONARY
        assert labels[50 - start] == STATIONARY

    def test_length_and_range_consistency(self):
        n, k = 100, 10
        mid = 100.0 + np.sin(np.arange(n) / 5.0)
        labels = fi2010_labels(mid, k=k, alpha=0.001)
        start, stop = label_index_range(n, k)
        assert stop - start == len(labels)
        assert start == k - 1
        assert stop == n - k

    def test_too_short_returns_empty(self):
        assert len(fi2010_labels(np.full(9, 100.0), k=5, alpha=0.01)) == 0
        assert label_index_range(9, 5) == (0, 0)

    def test_alpha_boundary_is_stationary(self):
        # Construct exact rel == alpha: must be STATIONARY (inclusive bound).
        k = 1
        alpha = 0.01
        mid = np.array([100.0, 101.0])  # rel = (101 - 100)/100 = alpha exactly
        labels = fi2010_labels(mid, k=k, alpha=alpha)
        assert len(labels) == 1
        assert labels[0] == STATIONARY

    def test_matches_naive_reference(self):
        rng = np.random.default_rng(7)
        mid = 100.0 * np.cumprod(1.0 + rng.normal(0, 1e-4, size=300))
        k, alpha = 7, 2e-5
        fast = fi2010_labels(mid, k=k, alpha=alpha)
        start, stop = label_index_range(len(mid), k)
        naive = []
        for t in range(start, stop):
            m_minus = mid[t - k + 1 : t + 1].mean()
            m_plus = mid[t + 1 : t + k + 1].mean()
            rel = (m_plus - m_minus) / m_minus
            naive.append(UP if rel > alpha else DOWN if rel < -alpha else STATIONARY)
        np.testing.assert_array_equal(fast, np.array(naive))
