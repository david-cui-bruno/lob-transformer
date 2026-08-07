# Findings

All numbers are macro F1 on the FI-2010 test days (8-10), computed once per
final model from `results/*/summary.json`. Headline horizon k=10.

## Main result

| model | params | test F1 k10 |
|---|---|---|
| logistic (flattened window) | 60k | 0.267 |
| MLP 256-128 (flattened window) | 1.06M | 0.360 |
| transformer (3 seeds) | 210k | **0.607 ± 0.013** |

The transformer beats the MLP by +25 F1 points with 5x fewer parameters. The
gap is architectural: the MLP sees the same 100x40 window but must learn
temporal structure through a flat 4000-dim projection, while attention treats
the window as a sequence.

Per-seed: 0.609 / 0.619 / 0.592. Val-test gap is small (val 0.587-0.596),
so model selection on val did not overfit.

## Ablations (seed 0, same budget)

| variant | test F1 k10 | delta vs base |
|---|---|---|
| base (window 100, 4 layers) | 0.609 | — |
| window 10 | 0.624 | +0.015 |
| window 50 | 0.603 | -0.006 |
| shuffled feature order | 0.612 | +0.003 |
| 2 layers | 0.607 | -0.002 |

1. **Recent history dominates.** A 10-snapshot window slightly *beats* 100.
   Most predictive signal at k=10 lives in the last few book states; long
   context is not where the transformer's advantage comes from. The advantage
   over the MLP persists at equal window sizes, so it is the sequence
   treatment, not the receptive field.
2. **Feature order is irrelevant** (+0.003). Expected: the first layer is a
   learned linear embedding, which can absorb any fixed permutation. This
   ablation is a *control* confirming the pipeline doesn't secretly depend on
   column layout, not evidence about book structure.
3. **Two layers suffice** (-0.002). Capacity is not the bottleneck at this
   data scale; the win over baselines is not parameter count (the MLP has 5x
   more).

## The leakage bug worth knowing about

FI-2010's distribution files concatenate **5 stocks** back to back with no
delimiter. A naive sliding window (which is what most quick reproductions do)
produces windows that span two different stocks: the price scale jumps up to
174% mid-window, and labels near boundaries are computed across the seam.

We detect segment boundaries (relative best-ask jumps > 20%, verified: exactly
5 train segments and 3x5 test segments with per-stock scales matching across
days) and window each segment independently: train/val splits are taken per
stock, and no window can cross a stock boundary by construction
(`MultiStreamLOBDataset`). The empirical label check
(`scripts/validate_labels.py`) confirms UP/DOWN labels correspond to
+7.6e-4 / -7.2e-4 mean future mid moves per stock.

## Honest comparison to published work

DeepLOB (Zhang et al. 2019) reports ~0.83 F1 at k=10 on this benchmark. We
get 0.61. Differences: they train a CNN+LSTM for ~100 epochs with heavier
feature extraction; our budget is 15 epochs / ~11 minutes on Apple-Silicon
MPS, and our per-segment hygiene discards boundary windows. We did not
attempt to close this gap; the project's aim is a clean, verified pipeline
with defensible baselines and ablations, not SOTA. The remaining gap is a
known cost of budget and architecture, and closing it is future work
(CNN feature stem, longer training, learning-rate sweeps).

## Limitations

- FI-2010 is Nasdaq Nordic 2010: results say nothing about modern US markets.
- Direction classification is not a trading strategy; no costs, latency, or
  fill modeling. No profitability claims.
- Segment detection is heuristic (price-jump threshold), though verified
  against the known 5-stock structure.
