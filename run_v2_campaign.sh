#!/bin/bash
# Fair-budget control + conv seeds, sequential to avoid MPS contention
.venv/bin/python -m lobt.train --model transformer --out results/transformer_long_s0 --epochs 60 --patience 10 --seed 0
.venv/bin/python -m lobt.train --model convtransformer --out results/convtransformer_long_s1 --epochs 60 --patience 10 --seed 1
.venv/bin/python -m lobt.train --model convtransformer --out results/convtransformer_long_s2 --epochs 60 --patience 10 --seed 2
echo "CAMPAIGN DONE"
