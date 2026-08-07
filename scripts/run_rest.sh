#!/bin/bash
# Seeds 1-2 for the headline model, then the 3 ablations. Sequential (one MPS).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON="--model transformer --epochs 15 --eval-every 1000 --patience 6"
$PY -m lobt.train $COMMON --seed 1 --out results/transformer_s1
$PY -m lobt.train $COMMON --seed 2 --out results/transformer_s2
# Ablation 1: window length
$PY -m lobt.train $COMMON --seed 0 --window 10 --out results/abl_window10
$PY -m lobt.train $COMMON --seed 0 --window 50 --out results/abl_window50
# Ablation 2: shuffled feature order (destroys book structure locality)
$PY -m lobt.train $COMMON --seed 0 --shuffle-features --out results/abl_shuffle
# Ablation 3: capacity (2 layers instead of 4)
$PY -m lobt.train $COMMON --seed 0 --n-layers 2 --out results/abl_layers2
echo ALL_RUNS_DONE
