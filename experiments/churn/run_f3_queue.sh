#!/bin/bash
# F3 runs (GRAFT_F_G_PLAN.md amendment 2), same 4-process limit as run_fg_queue.sh.
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
LOGS=experiments/churn/logs_fg
heavy() {
  m=$(ps aux | grep "[e]xperiments/modular/run.py" | grep -v "uv run\|sh -c" | wc -l)
  c=$(ps aux | grep "[e]xperiments/churn/run_\(earlygen\|protect\|monk2graft\|evolve_graft\)" | grep -v "uv run\|sh -c" | wc -l)
  echo $((m + c))
}
for seed in 0 1 2; do
  sleep $((RANDOM % 30 + 30))
  while [ "$(heavy)" -ge 4 ]; do sleep 20; done
  echo "$(date) start f3_s$seed" >> $LOGS/queue.log
  (uv run python experiments/churn/run_evolve_graft.py --seed $seed > $LOGS/f3_s$seed.log 2>&1; echo "$(date) done f3_s$seed exit $?" >> $LOGS/queue.log) &
  sleep 60
done
wait
