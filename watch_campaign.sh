#!/bin/bash
# Watch the lob-transformer v2 campaign; exit 0 when all 3 runs have summaries.
# If the driver dies with runs missing, restart run_v2_campaign.sh (it is resumable).
cd /Users/davidcui824/lob-transformer
need="convtransformer_long_s2 transformer_long_s1 transformer_long_s2"
while true; do
  done_all=1
  for r in $need; do
    [ -f "results/$r/summary.json" ] || done_all=0
  done
  if [ "$done_all" = 1 ]; then
    echo "JCODE_PROGRESS {\"message\": \"campaign complete: all 3 summaries present\"}"
    exit 0
  fi
  if ! pgrep -f "lobt.train" > .watch_pgrep_tmp; then
    echo "JCODE_PROGRESS {\"message\": \"no train process; restarting resumable campaign\"}"
    nohup ./run_v2_campaign.sh > runs_v2_campaign_restart.log 2>&1 &
    sleep 30
  fi
  # progress: latest line of whichever run log is active
  latest=$(ls -t runs_convtransformer_long_s2.log runs_transformer_long_s1.log runs_transformer_long_s2.log 2> .watch_ls_err | head -1)
  if [ -n "$latest" ]; then
    line=$(tail -1 "$latest" | tr '"' "'" | cut -c1-160)
    echo "JCODE_PROGRESS {\"message\": \"$latest :: $line\"}"
  fi
  sleep 120
done
