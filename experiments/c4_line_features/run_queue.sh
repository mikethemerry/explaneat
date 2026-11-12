#!/bin/bash
# Queue of the line-features base test: at most $MAXJ runs at once (other engineers' jobs share
# the machine). Order: the heavy arm on both bases (seeds 0-2, interleaved), then the default arm,
# then the per-line arm (default config only). Usage: MAXJ=2 ./run_queue.sh
set -u
cd "$(dirname "$0")/../.."
MAXJ=${MAXJ:-2}
RUNS=(
  "lines heavy 0" "lines_cells heavy 0" "lines heavy 1" "lines_cells heavy 1" "lines heavy 2" "lines_cells heavy 2"
  "lines default 0" "lines_cells default 0" "lines default 1" "lines_cells default 1" "lines default 2" "lines_cells default 2"
  "perline default 0" "perline default 1" "perline default 2"
)
for r in "${RUNS[@]}"; do
  set -- $r
  n="$1-$2-s$3"
  if [ -f "experiments/c4_line_features/logs/$n.json" ]; then echo "skip $n (done)"; continue; fi
  while [ "$(pgrep -f "python experiments/c4_line_features/run_seed" | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  echo "$(date '+%F %T') start $n"
  PYTHONHASHSEED=0 uv run python experiments/c4_line_features/run_seed.py "$3" --inputs "$1" --config "$2" \
    > "experiments/c4_line_features/logs/$n.out" 2>&1 &
  sleep 20
done
wait
echo "$(date '+%F %T') all done"
