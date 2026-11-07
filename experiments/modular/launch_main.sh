#!/bin/bash
# Main study (PLAN.md "Budget"): 2 tasks x 4 arms x seeds 0-9, seed-major
# order, at most 4 concurrent processes with 1 thread each. Runs whose
# summary JSON already exists are skipped (restartable).
cd "$(dirname "$0")/../.."
OUT=experiments/modular/results/main
LOGS=experiments/modular/logs/main
mkdir -p "$OUT" "$LOGS"
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
jobs_file=$(mktemp)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for task in monk2 mux11; do
    for arm in plain modules sham topology; do
      [ -f "$OUT/$task-$arm-s$seed.json" ] || echo "$task $arm $seed" >> "$jobs_file"
    done
  done
done
echo "$(date) launching $(wc -l < "$jobs_file") runs" >> "$LOGS/launch.log"
xargs -P 4 -L 1 sh -c 'uv run python experiments/modular/run.py "$0" "$1" "$2" --out '"$OUT"' > '"$LOGS"'/"$0"-"$1"-s"$2".log 2>&1; echo "$(date) done $0 $1 $2 exit $?" >> '"$LOGS"'/launch.log' < "$jobs_file"
rm -f "$jobs_file"
echo "$(date) all done" >> "$LOGS/launch.log"
