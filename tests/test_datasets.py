"""Tests for data loading, windowing, and split hygiene."""

import numpy as np
import pytest
import torch

from lobt.data import (
    HORIZONS,
    detect_segments,
    synthetic_tape,
    train_val_split,
    train_val_split_segmented,
)
from lobt.datasets import MultiStreamLOBDataset, WindowedLOBDataset, class_weights


@pytest.fixture(scope="module")
def tape():
    return synthetic_tape(n=2000, seed=1)


class TestSyntheticTape:
    def test_shapes(self, tape):
        x, y = tape
        assert x.shape == (2000, 40)
        for k in HORIZONS:
            assert y[k].shape == (2000,)
            assert set(np.unique(y[k])) <= {0, 1, 2}

    def test_book_is_ordered(self, tape):
        x, _ = tape
        # ask prices increase with level; bid prices decrease; ask > bid at L0
        assert (x[:, 0] < x[:, 4]).all()  # ask L0 < ask L1
        assert (x[:, 2] > x[:, 6]).all()  # bid L0 > bid L1
        assert (x[:, 0] > x[:, 2]).all()  # ask L0 > bid L0


class TestWindowing:
    def test_len_and_alignment(self, tape):
        x, y = tape
        ds = WindowedLOBDataset(x, y, window=100)
        assert len(ds) == 2000 - 100 + 1
        w, labels = ds[0]
        assert w.shape == (100, 40)
        assert labels.shape == (len(HORIZONS),)
        # label of window [0,100) must be the label at t=99
        assert labels[0].item() == y[HORIZONS[0]][99]

    def test_last_window(self, tape):
        x, y = tape
        ds = WindowedLOBDataset(x, y, window=100)
        w, labels = ds[len(ds) - 1]
        np.testing.assert_allclose(w.numpy()[-1], x[-1])
        assert labels[2].item() == y[HORIZONS[2]][-1]

    def test_rejects_short_stream(self, tape):
        x, y = tape
        with pytest.raises(ValueError):
            WindowedLOBDataset(x[:50], {k: v[:50] for k, v in y.items()}, window=100)

    def test_rejects_misaligned(self, tape):
        x, y = tape
        with pytest.raises(ValueError):
            WindowedLOBDataset(x[:-1], y, window=100)


class TestSplitHygiene:
    def test_chronological_no_overlap(self, tape):
        x, y = tape
        tx, ty, vx, vy = train_val_split(x, y, val_frac=0.1)
        assert len(tx) + len(vx) == len(x)
        # boundary: train ends exactly where val begins
        np.testing.assert_allclose(np.concatenate([tx, vx]), x)
        # windows built per-split can never span the boundary by construction
        ds_t = WindowedLOBDataset(tx, ty, window=100)
        ds_v = WindowedLOBDataset(vx, vy, window=100)
        w_last, _ = ds_t[len(ds_t) - 1]
        w_first, _ = ds_v[0]
        np.testing.assert_allclose(w_last.numpy()[-1], tx[-1])
        np.testing.assert_allclose(w_first.numpy()[0], vx[0])


class TestClassWeights:
    def test_inverse_frequency(self):
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)
        w = class_weights(y)
        assert w.shape == (3,)
        assert w[2] > w[1] > w[0]
        assert torch.isclose(w.mean(), torch.tensor(1.0))

    def test_missing_class_raises(self):
        with pytest.raises(ValueError):
            class_weights(np.array([0, 0, 1]))


class TestSegments:
    def _two_stock_stream(self):
        x1, y1 = synthetic_tape(n=600, seed=2)
        x2, y2 = synthetic_tape(n=400, seed=3)
        x2 = x2.copy()
        x2[:, [0, 2]] *= 3.0  # different price scale, like FI-2010 stocks
        x = np.concatenate([x1, x2])
        y = {k: np.concatenate([y1[k], y2[k]]) for k in HORIZONS}
        return x, y

    def test_detects_boundary(self):
        x, _ = self._two_stock_stream()
        segs = detect_segments(x)
        assert segs == [(0, 600), (600, 1000)]

    def test_segmented_split_stays_within_stocks(self):
        x, y = self._two_stock_stream()
        segs = detect_segments(x)
        tr_xs, tr_ys, va_xs, va_ys = train_val_split_segmented(x, y, segs, val_frac=0.1)
        assert len(tr_xs) == len(va_xs) == 2
        # each stock contributes its own tail to val
        np.testing.assert_allclose(va_xs[0], x[540:600])
        np.testing.assert_allclose(va_xs[1], x[960:1000])

    def test_multistream_never_crosses_boundary(self):
        x, y = self._two_stock_stream()
        segs = detect_segments(x)
        xs = [x[s:e] for s, e in segs]
        ys = [{k: v[s:e] for k, v in y.items()} for s, e in segs]
        ds = MultiStreamLOBDataset(xs, ys, window=100)
        # total windows = per-segment windows summed, NOT (n - w + 1) of the concat
        assert len(ds) == (600 - 100 + 1) + (400 - 100 + 1)
        # the first window of stream 2 starts exactly at the boundary
        w_first_s2, _ = ds[600 - 100 + 1]
        np.testing.assert_allclose(w_first_s2.numpy()[0], x[600])
        # no window mixes both price scales
        w_last_s1, _ = ds[600 - 100]
        assert w_last_s1.numpy()[:, 0].max() < x[600:, 0].min()

    def test_short_stream_skipped(self):
        x, y = self._two_stock_stream()
        xs = [x[:600], x[600:650]]  # second too short for window=100
        ys = [
            {k: v[:600] for k, v in y.items()},
            {k: v[600:650] for k, v in y.items()},
        ]
        ds = MultiStreamLOBDataset(xs, ys, window=100)
        assert ds.skipped == 1
        assert len(ds) == 600 - 100 + 1

    def test_all_labels_matches_iteration(self):
        x, y = self._two_stock_stream()
        segs = detect_segments(x)
        xs = [x[s:e] for s, e in segs]
        ys = [{k: v[s:e] for k, v in y.items()} for s, e in segs]
        ds = MultiStreamLOBDataset(xs, ys, window=100)
        labels = ds.all_labels(0)
        assert len(labels) == len(ds)
        sampled = np.array([ds[i][1][0].item() for i in range(0, len(ds), 97)])
        np.testing.assert_array_equal(sampled, labels[::97])
