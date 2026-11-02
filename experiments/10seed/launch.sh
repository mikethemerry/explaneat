#!/usr/bin/env bash
# Launch all 10 seeds of one dataset in parallel, one process per seed.
# Usage: experiments/10seed/launch.sh heart
set -euo pipefail
cd "$(dirname "$0")/../.."
ds="$1"
mkdir -p experiments/10seed/logs
for s in 0 1 2 3 4 5 6 7 8 9; do
  PYTHONHASHSEED=0 nohup uv run python experiments/10seed/run_seed.py "$ds" "$s" \
    > "experiments/10seed/logs/${ds}-s${s}.log" 2>&1 &
  echo "$ds s$s pid $!"
done
