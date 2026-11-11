#!/bin/bash
# Main runs of the Connect-4 capacity study: the pilot config chosen on validation AUC
# (p18-h10-w015-e50: 10 started hidden units, weight_init_stdev 0.15, 50 backprop epochs per
# generation, pop 150), trained on the full training split, 80 generations, seeds 0-2.
# At most $MAXJ study runs at once. Usage: MAXJ=2 ./run_main.sh [seeds...]
set -u
cd "$(dirname "$0")/../.."
MAXJ=${MAXJ:-2}
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
for s in "${SEEDS[@]}"; do
  while [ "$(pgrep -f "bin/python3 experiments/c4_capacity/run_seed" | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  n="main-h10-e50-s$s"
  echo "$(date '+%F %T') start $n"
  PYTHONHASHSEED=0 uv run python experiments/c4_capacity/run_seed.py "$s" --name "$n" --generations 80 \
    --set NEAT.pop_size=150 --set DefaultGenome.num_hidden=10 --set DefaultGenome.weight_init_stdev=0.15 \
    --epochs 50 > "experiments/c4_capacity/logs/$n.out" 2>&1 &
  sleep 15
done
wait
echo "$(date '+%F %T') all done"
