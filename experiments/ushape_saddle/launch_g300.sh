#!/usr/bin/env bash
# Longer runs on the both dataset: seeds 0-4 of shape-C rerun from scratch to 300 generations
# (identical config; generations 0-99 replay the originals), at most 3 training processes.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/ushape_saddle/logs
seeds=(0 1 2 3 4)
lane() {
  local k=$1; local i=0
  for s in "${seeds[@]}"; do
    if (( i % 3 == k )); then
      PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
        uv run python experiments/ushape_saddle/run_seed.py both "$s" --generations 300 --name "shape-C-s$s-g300" \
        > "experiments/ushape_saddle/logs/both-s$s-g300.log" 2>&1 || true
    fi
    i=$((i+1))
  done
}
for k in 0 1 2; do lane $k & done
wait
echo "all lanes done"
