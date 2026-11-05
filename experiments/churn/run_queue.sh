#!/usr/bin/env bash
# churn100 seeds 3-9 then churn200 seeds 0-9, at most 4 concurrent (Mike: CPU load).
cd "$(dirname "$0")/../.."
{ for s in 3 4 5 6 7 8 9; do echo "$s 100 churn100-s$s"; done
  for s in 1 2 4 7 9 5 3 8 0 6; do echo "$s 200 churn200-s$s"; done; } |
xargs -P 4 -L 1 sh -c 'PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py churn "$0" --generations "$1" --name "$2" > experiments/10seed/logs/$2.log 2>&1'
