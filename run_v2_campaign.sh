#!/bin/bash
# Resumable campaign: skips any run whose summary.json already exists.
# Safe to re-run after a crash/sleep. Runs sequentially (single MPS device).
cd /Users/davidcui824/lob-transformer

run() {
  local model=$1 seed=$2 out=$3
  if [ -f "results/$out/summary.json" ]; then
    echo "SKIP $out (already complete)"
    return
  fi
  echo "START $out"
  rm -rf "results/$out"
  .venv/bin/python -m lobt.train --model "$model" --out "results/$out" \
    --epochs 60 --patience 10 --seed "$seed" > "runs_$out.log" 2>&1
  if [ -f "results/$out/summary.json" ]; then
    echo "DONE $out"
  else
    echo "FAILED $out"
  fi
}

run convtransformer 2 convtransformer_long_s2
run transformer 1 transformer_long_s1
run transformer 2 transformer_long_s2
echo "CAMPAIGN COMPLETE"
