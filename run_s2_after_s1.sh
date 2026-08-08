#!/bin/bash
# wait for s1 to finish, then run s2
while [ ! -f /Users/davidcui824/lob-transformer/results/convtransformer_long_s1/summary.json ]; do
  sleep 60
done
cd /Users/davidcui824/lob-transformer
.venv/bin/python -m lobt.train --model convtransformer --out results/convtransformer_long_s2 --epochs 60 --patience 10 --seed 2 > runs_convtransformer_long_s2.log 2>&1
