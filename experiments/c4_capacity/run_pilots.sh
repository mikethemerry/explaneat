#!/bin/bash
# Pilot queue of the Connect-4 capacity study: pop 150, 50 generations, seed 0, training on 90% of
# the training split, monitor AUC on the held-out 10% of the training split (test never read).
# At most $MAXJ study runs on the machine at once (counted with pgrep, so separate queues share it). Usage: MAXJ=2 ./run_pilots.sh <pilot name> ...   (no names = all)
set -u
cd "$(dirname "$0")/../.."
MAXJ=${MAXJ:-2}
P="--generations 50 --val-frac 0.1 --set NEAT.pop_size=150"
args() {
  case "$1" in
    p0-base) echo "$P" ;;
    p1-h10) echo "$P --set DefaultGenome.num_hidden=10" ;;
    p2-h10-e20) echo "$P --set DefaultGenome.num_hidden=10 --epochs 20" ;;
    p3-grow) echo "$P --set DefaultGenome.node_add_prob=0.8 --set DefaultGenome.conn_add_prob=0.9" ;;
    p4-tuned) echo "$P --tuned --threshold-step 0.1" ;;
    p5-h20-e20) echo "$P --set DefaultGenome.num_hidden=20 --epochs 20" ;;
    p6-grow-e20) echo "$P --set DefaultGenome.node_add_prob=0.8 --set DefaultGenome.conn_add_prob=0.9 --epochs 20" ;;
    p7-h10-e20-pop400) echo "--generations 50 --val-frac 0.1 --set NEAT.pop_size=400 --set DefaultGenome.num_hidden=10 --epochs 20" ;;
    p8-h10-e50) echo "$P --set DefaultGenome.num_hidden=10 --epochs 50" ;;
    p9-h20-e50) echo "$P --set DefaultGenome.num_hidden=20 --epochs 50" ;;
    p10-h10-e20-g150) echo "--generations 150 --val-frac 0.1 --set NEAT.pop_size=150 --set DefaultGenome.num_hidden=10 --epochs 20" ;;
    p11-h10-e20-nodel) echo "$P --set DefaultGenome.num_hidden=10 --epochs 20 --set DefaultGenome.conn_delete_prob=0.05 --set DefaultGenome.node_delete_prob=0.05" ;;
    p12-h10-w015) echo "$P --set DefaultGenome.num_hidden=10 --set DefaultGenome.weight_init_stdev=0.15" ;;
    p13-h10-w015-e20) echo "$P --set DefaultGenome.num_hidden=10 --set DefaultGenome.weight_init_stdev=0.15 --epochs 20" ;;
    p14-w015) echo "$P --set DefaultGenome.weight_init_stdev=0.15" ;;
    p15-grow-w015) echo "$P --set DefaultGenome.node_add_prob=0.8 --set DefaultGenome.conn_add_prob=0.9 --set DefaultGenome.weight_init_stdev=0.15" ;;
    p16-h20-w015-e20) echo "$P --set DefaultGenome.num_hidden=20 --set DefaultGenome.weight_init_stdev=0.15 --epochs 20" ;;
    p17-h10-w015-e20-tuned) echo "$P --set DefaultGenome.num_hidden=10 --set DefaultGenome.weight_init_stdev=0.15 --epochs 20 --tuned --threshold-step 0.1" ;;
    p18-h10-w015-e50) echo "$P --set DefaultGenome.num_hidden=10 --set DefaultGenome.weight_init_stdev=0.15 --epochs 50" ;;
    p19-h5-w015-e20) echo "$P --set DefaultGenome.num_hidden=5 --set DefaultGenome.weight_init_stdev=0.15 --epochs 20" ;;
    p20-lrwarm) echo "$P --lr-warm" ;;
    p21-lrwarm-h10-w015-e20) echo "$P --lr-warm --set DefaultGenome.num_hidden=10 --set DefaultGenome.weight_init_stdev=0.15 --epochs 20" ;;
    p22-lrwarm-grow-tuned) echo "$P --lr-warm --set DefaultGenome.node_add_prob=0.8 --set DefaultGenome.conn_add_prob=0.9 --tuned --threshold-step 0.1" ;;
    p23-lrwarm-grow-e20) echo "$P --lr-warm --set DefaultGenome.node_add_prob=0.8 --set DefaultGenome.conn_add_prob=0.9 --epochs 20" ;;
    p24-lrwarm-grow-disj3) echo "$P --lr-warm --set DefaultGenome.node_add_prob=0.8 --set DefaultGenome.conn_add_prob=0.9 --set DefaultGenome.compatibility_disjoint_coefficient=3.0 --set DefaultSpeciesSet.compatibility_threshold=2.0" ;;
    *) echo "unknown pilot $1" >&2; exit 1 ;;
  esac
}
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(p0-base p1-h10 p2-h10-e20 p3-grow p4-tuned p5-h20-e20 p6-grow-e20)
for n in "${NAMES[@]}"; do
  while [ "$(pgrep -f "bin/python3 experiments/c4_capacity/run_seed" | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  echo "$(date '+%F %T') start $n"
  PYTHONHASHSEED=0 uv run python experiments/c4_capacity/run_seed.py 0 --name "$n" $(args "$n") \
    > "experiments/c4_capacity/logs/$n.out" 2>&1 &
  sleep 15
done
wait
echo "$(date '+%F %T') all done"
