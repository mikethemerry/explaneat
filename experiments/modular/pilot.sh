#!/bin/bash
# Timing/debug pilot (PLAN.md: pilot results are not analysed for outcomes).
cd "$(dirname "$0")/../.."
mkdir -p experiments/modular/logs/pilot
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
for spec in "mux11 plain" "mux11 modules" "monk2 plain" "monk2 modules"; do
  set -- $spec
  uv run python experiments/modular/run.py "$1" "$2" 0 --out experiments/modular/results/pilot \
    > "experiments/modular/logs/pilot/$1-$2-s0.log" 2>&1 &
done
wait
