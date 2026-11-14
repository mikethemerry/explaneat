#!/usr/bin/env bash
# Run the study's seeds in at most 3 lanes (3 training processes at a time), seeds 0-4 per task.
# Usage: experiments/ushape_saddle/launch.sh u saddle [both]
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/ushape_saddle/logs
jobs_list=()
for t in "$@"; do for s in 0 1 2 3 4; do jobs_list+=("$t $s"); done; done
lane() {  # lane k runs every 3rd job
  local k=$1; local i=0
  for j in "${jobs_list[@]}"; do
    if (( i % 3 == k )); then
      set -- $j
      PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
        uv run python experiments/ushape_saddle/run_seed.py "$1" "$2" > "experiments/ushape_saddle/logs/$1-s$2.log" 2>&1 || true
    fi
    i=$((i+1))
  done
}
for k in 0 1 2; do lane $k & done
wait
echo "all lanes done"
