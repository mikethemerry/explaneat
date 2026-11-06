#!/bin/bash
# Re-run completed main runs with the post-fix code (RESULTS.md deviation 1)
# into results/verify and compare their detail logs with results/main.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p experiments/modular/results/verify experiments/modular/logs/verify
for spec in "monk2 sham 0" "mux11 modules 1"; do
  set -- $spec
  uv run python experiments/modular/run.py "$1" "$2" "$3" --quiet --out experiments/modular/results/verify \
    > "experiments/modular/logs/verify/$1-$2-s$3.log" 2>&1
  if cmp -s "experiments/modular/results/verify/$1-$2-s$3.detail.json" "experiments/modular/results/main/$1-$2-s$3.detail.json"; then
    echo "$1 $2 $3 detail IDENTICAL"; else echo "$1 $2 $3 detail DIFFERS"; fi
done
