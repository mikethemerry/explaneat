#!/bin/bash
# The exact run queue of the complex-real study, at most $MAXJ concurrent (default 2 while the
# MNIST runs use the machine). Logs: experiments/complex_real/logs/<name>.out
set -u
cd "$(dirname "$0")/../.."
MAXJ=${MAXJ:-2}
JOBS=(
  "krk 0 --variant default" "krk 1 --variant default" "krk 2 --variant default"
  "krk 0 --variant tuned" "krk 1 --variant tuned"
  "c4 0 --variant default" "c4 1 --variant default" "c4 2 --variant default"
  "c4 0 --variant tuned" "c4 1 --variant tuned"
)
for j in "${JOBS[@]}"; do
  set -- $j
  name="$1-$4-s$2"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  echo "$(date '+%F %T') start $name"
  PYTHONHASHSEED=0 uv run python experiments/complex_real/run_seed.py $j > "experiments/complex_real/logs/$name.out" 2>&1 &
done
wait
echo "$(date '+%F %T') all done"
