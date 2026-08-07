# lob-transformer: spec

## Goal

Train a transformer to predict short-horizon mid-price direction from raw limit
order book states, and beat classical baselines under an honest, published-
protocol evaluation. This is the ML sequel to the `orderbook` project: that repo
proves I can build the matching engine; this one proves I can learn from its
data.

Resume-bullet shape this must produce (numbers TBD by real runs):

> Trained a N-param transformer on X.XM LOB snapshots (FI-2010 benchmark)
> predicting mid-price direction at horizon k; F1 XX.X vs XX.X logistic /
> XX.X MLP baselines under the published anchored-walk-forward protocol;
> ablations attribute +X.X F1 to depth attention vs flattened features.

## Task definition

- **Input**: the last `T=100` LOB snapshots, each the top 10 levels per side:
  (price, size) for bids and asks = 40 features per snapshot.
- **Target**: direction of the smoothed mid-price change over horizon
  `k ∈ {10, 20, 50, 100}` events: down / stationary / up (3 classes).
  Smoothing and thresholding follow the FI-2010 label definition (Ntakaris et
  al. 2018): compare the mean of the next k mids `m+(t)` to the mean of the
  previous k mids `m-(t)`; label by relative change vs threshold α.
- **Metric**: macro F1 (headline, robust to class imbalance) plus accuracy,
  per-class precision/recall, and confusion matrices.

## Dataset

**Primary: FI-2010** (Ntakaris et al. 2018), the standard public LOB benchmark:
~4M events, 5 Nasdaq Nordic stocks, 10 trading days, 10 levels per side,
pre-normalized variants (z-score, min-max, decimal precision) with labels for
all horizons included. Widely mirrored; used by DeepLOB (Zhang et al. 2019) and
dozens of successors, so external reference numbers exist to sanity-check my
pipeline (DeepLOB reports macro F1 ≈ 83 at k=10 under z-score normalization,
train days 1-7 / test days 8-10).

**Fallback if FI-2010 mirrors are unreachable**: LOBSTER free sample files
(AAPL/AMZN/GOOG/INTC/MSFT, 1 day, levels 1/5/10). Same loader interface; labels
computed by our own code using the FI-2010 formula. External comparability is
weaker (single day), which the writeup must state plainly.

**Synthetic smoke set**: tiny deterministic tape generated in-repo for unit
tests and CI, so tests never depend on the big download.

## Evaluation protocol (fixed before training; no peeking)

1. **Split**: FI-2010 standard "anchored" setup: train on days 1-7, test on
   days 8-10 (the split used by DeepLOB, so published numbers are comparable).
   10% of train (by time, tail) is validation for early stopping and model
   selection. No shuffling across the time boundary.
2. **Model selection**: pick checkpoint by validation macro F1 at k=10 only.
   Test set is touched once per final model.
3. **Baselines and model share one harness**: identical data, labels, splits,
   and metric code, so comparisons are apples-to-apples.
4. **Seeds**: 3 seeds for the headline model; report mean ± std.
5. **Class imbalance**: report class distribution; use class-weighted loss.

## Models

### Baseline 1: multinomial logistic regression
On the flattened current snapshot (40 features) and on the flattened window
(100×40). Establishes the "linear floor".

### Baseline 2: MLP
2 hidden layers (256, 128), ReLU, dropout 0.2, on the flattened window.
Establishes the "nonlinear but structure-blind floor".

### Main: LOB transformer (~1-3M params, tuned to MPS budget)
- Per-snapshot embedding: linear 40 → d_model (64).
- Learned positional embedding over the T=100 sequence.
- 4 encoder layers, 4 heads, d_ff 256, pre-norm, dropout 0.1.
- Mean-pool over time → linear head → 3 logits (one head per horizon,
  trained jointly with summed weighted losses).
- AdamW, lr 3e-4 with cosine decay, warmup 500 steps, batch 256, bf16 on MPS
  where stable, gradient clipping 1.0, early stopping patience 5 evals.

### Ablations (each answers one interview question)
1. **Window length**: T ∈ {10, 50, 100} — how much history matters?
2. **Depth attention vs flattened**: replace per-snapshot linear embed of
   structured (level, side) features with a shuffled-feature control — does the
   model exploit book structure or just correlations?
3. **Capacity**: 2 vs 4 layers — is the gain from attention or just params?

## Repo layout

```
lob-transformer/
  docs/spec.md            this file
  src/lobt/
    data.py               download/verify/load FI-2010, LOBSTER fallback, synthetic
    labels.py             FI-2010 label math (also used for LOBSTER/synthetic)
    datasets.py           torch Dataset/DataLoader with windowing
    models/{linear.py,mlp.py,transformer.py}
    train.py              training loop (MPS), checkpointing, metrics JSONL
    eval.py               test-set evaluation, confusion matrices, results JSON
  tests/                  label math, windowing, split hygiene, shape checks
  scripts/                run_baselines.sh, run_main.sh, run_ablations.sh
  results/                committed JSON summaries + plots (not checkpoints)
```

## Milestones with checks

| # | milestone | check |
|---|-----------|-------|
| 1 | data + labels | unit tests pass; label distribution within a few pp of published FI-2010 stats |
| 2 | split hygiene | test proves no window crosses the train/test day boundary |
| 3 | baselines | logistic + MLP reproduce ballpark literature numbers (logistic ≈ 40s-60s F1 range) |
| 4 | transformer | beats both baselines on val F1 at k=10 |
| 5 | final eval | single test run; compare to DeepLOB published F1; 3 seeds |
| 6 | ablations | 3 ablations, one-line conclusion each |
| 7 | writeup | README results table + docs/findings.md; every number from a committed results JSON |

## Honesty rules

- Never tune on test. Val only.
- Report worse-than-published outcomes as-is; a transformer losing to DeepLOB's
  CNN-LSTM at equal budget is a finding, not a failure.
- No claims of profitability or trading strategy: this is a supervised
  prediction benchmark, and the writeup says so.

## Hardware budget

Apple Silicon MPS (torch 2.11, verified available). ~1-3M param model,
~2-4M training windows: minutes-to-low-hours per run, feasible for
3 seeds + ablations overnight if needed.
