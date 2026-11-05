#!/bin/bash
# Timing pilot 2 at the candidate budget (not analysed for outcomes).
cd "$(dirname "$0")/../.."
mkdir -p experiments/modular/logs/pilot2
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
run() {
  uv run python experiments/modular/run.py "$1" "$2" 0 --generations 300 --pop-size 150 \
    --stage-switch "$3" --out experiments/modular/results/pilot2 \
    > "experiments/modular/logs/pilot2/$1-$2-s0.log" 2>&1
}
run mux11 modules 150 &
run mux11 sham 150 &
run monk2 modules "" &
run monk2 topology "" &
wait
