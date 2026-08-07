"""Tests for data loading, windowing, and split hygiene."""

import numpy as np
import pytest
import torch

from lobt.data import HORIZONS, synthetic_tape, train_val_split
from lobt.datasets import WindowedLOBDataset, class_weights


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
