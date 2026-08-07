"""Model shape tests and a tiny overfit sanity check."""

import numpy as np
import torch

from lobt.models import (
    LOBTransformer,
    LogisticBaseline,
    MLPBaseline,
    build_model,
    param_count,
)
from lobt.train import macro_f1


class TestShapes:
    def test_all_models_output_shape(self):
        x = torch.randn(4, 100, 40)
        for name in ["logistic", "mlp", "transformer"]:
            m = build_model(name, window=100, n_features=40, n_horizons=5)
            out = m(x)
            assert out.shape == (4, 5, 3), name

    def test_transformer_param_budget(self):
        m = LOBTransformer(window=100, n_features=40, n_horizons=5)
        n = param_count(m)
        assert 100_000 < n < 5_000_000, n


class TestMacroF1:
    def test_perfect(self):
        conf = np.diag([10, 20, 30])
        assert macro_f1(conf) == 1.0

    def test_all_wrong(self):
        conf = np.array([[0, 10, 0], [0, 0, 10], [10, 0, 0]])
        assert macro_f1(conf) == 0.0

    def test_known_value(self):
        # binaryish case: class 0: tp=8 fp=2 fn=2 -> f1=0.8; class1: tp=2 fp=2 fn=2 -> 0.5
        conf = np.array([[8, 2, 0], [2, 2, 0], [0, 0, 0]])
        f1 = macro_f1(conf)
        assert abs(f1 - (0.8 + 0.5 + 0.0) / 3) < 1e-9


class TestOverfitTiny:
    def test_transformer_overfits_100_samples(self):
        """A 2-layer transformer must overfit 100 fixed samples: proves the
        training path (forward, loss, backward) learns at all."""
        torch.manual_seed(0)
        x = torch.randn(100, 20, 40)
        y = torch.randint(0, 3, (100, 1))
        m = LOBTransformer(window=20, n_features=40, n_horizons=1,
                           d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(200):
            logits = m(x)
            loss = loss_fn(logits[:, 0], y[:, 0])
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = (m(x)[:, 0].argmax(-1) == y[:, 0]).float().mean().item()
        assert acc > 0.95, f"failed to overfit: acc={acc}"
