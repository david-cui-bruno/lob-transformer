#!/bin/bash
# Sequential runs (single MPS device). Baselines then transformer seed 0.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "JCODE_PROGRESS {\"message\":\"logistic baseline\"}"
$PY -m lobt.train --model logistic --out results/logistic_s0 --epochs 4 --lr 1e-3 --eval-every 1500
echo "JCODE_CHECKPOINT {\"message\":\"logistic done\"}"
echo "JCODE_PROGRESS {\"message\":\"mlp baseline\"}"
$PY -m lobt.train --model mlp --out results/mlp_s0 --epochs 8 --eval-every 1500
echo "JCODE_CHECKPOINT {\"message\":\"mlp done\"}"
echo "JCODE_PROGRESS {\"message\":\"transformer seed 0\"}"
$PY -m lobt.train --model transformer --out results/transformer_s0 --epochs 15 --eval-every 1000 --patience 6
echo "JCODE_CHECKPOINT {\"message\":\"transformer s0 done\"}"
