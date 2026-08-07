# lob-transformer

A transformer trained to predict short-horizon mid-price direction from raw
limit order book states, benchmarked against classical baselines on
**FI-2010** (the standard public LOB benchmark) under a fixed, leakage-checked
evaluation protocol.

Companion project to [orderbook](https://github.com/david-cui-bruno/orderbook):
that repo builds the matching engine; this one learns from the data such an
engine produces.

## Results (FI-2010, test days 8-10, macro F1 at horizon k=10)

| model | params | test F1 k10 |
|---|---|---|
| logistic regression | 60k | 0.267 |
| MLP 256-128 | 1.06M | 0.360 |
| **transformer** | **210k** | **0.607 ± 0.013** (3 seeds) |

+25 F1 over the MLP with 5x fewer parameters. Ablations show the gain is the
sequence treatment, not capacity (2 layers ≈ 4 layers) or context length
(window 10 ≈ window 100). Full analysis, including an honest gap statement
vs published DeepLOB numbers, in [docs/findings.md](docs/findings.md).

## A leakage bug you'll hit if you do this naively

FI-2010's files concatenate **5 stocks** with no delimiter. Naive sliding
windows cross stock boundaries (price jumps up to 174% mid-window). This repo
detects segment boundaries, splits train/val per stock, and makes
boundary-crossing windows structurally impossible. Details in
[docs/findings.md](docs/findings.md).

## Task

- Input: last T=100 book snapshots, 10 levels/side: (price, size) = 40 features.
- Target: smoothed mid-price direction (down/stationary/up) at horizons
  k ∈ {10, 20, 30, 50, 100}, FI-2010 label definition (Ntakaris et al. 2018).
- Protocol: train days 1-7, test days 8-10 (DeepLOB split); val = chronological
  tail of each stock's train stream; test evaluated once per final model;
  class-weighted loss; model selection on val k=10 macro F1 only.

## Model

Encoder-only transformer, 210k params: linear embed 40→64, learned positions,
4 pre-norm layers (4 heads, ff 256), mean-pool, per-horizon heads, trained
jointly on all horizons. AdamW, warmup + cosine, early stopping. ~11 min per
run on Apple-Silicon MPS.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # 33 tests
# put FI-2010 DecPre files in data/ (see src/lobt/data.py header)
./scripts/run_all.sh                  # baselines + transformer seed 0
./scripts/run_rest.sh                 # seeds 1-2 + ablations
.venv/bin/python scripts/report.py    # results table
```

Every number in the docs comes from a committed `results/*/summary.json`.

## Verification

- 33 unit tests: label math (vs naive reference), windowing, split hygiene,
  segment handling, model shapes, and a train-loop overfit check.
- `scripts/validate_labels.py`: dataset labels empirically match future
  mid-price moves (UP +7.6e-4, DOWN -7.2e-4, STAT ~0).
- No claims of profitability. This is a supervised prediction benchmark.
