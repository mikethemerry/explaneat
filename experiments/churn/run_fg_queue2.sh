#!/bin/bash
# Second F/G queue (after the protection-key fix): concurrency is enforced by fg_governor.sh
# (newest processes are SIGSTOPped while modular + F/G > 4); jobs start 2 min apart.
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
LOGS=experiments/churn/logs_fg
while read -r name args; do
  [ -z "$name" ] && continue
  echo "$(date) start $name: $args" >> $LOGS/queue.log
  (uv run python $args > $LOGS/$name.log 2>&1; echo "$(date) done $name exit $?" >> $LOGS/queue.log) &
  sleep 120
done <<JOBS
d2_es experiments/churn/run_protect.py --context d2 --regime es --P 0,25,100
d2_e500 experiments/churn/run_protect.py --context d2 --regime e500 --P 0,25,100
f3_s0 experiments/churn/run_evolve_graft.py --seed 0
f3_s1 experiments/churn/run_evolve_graft.py --seed 1
f3_s2 experiments/churn/run_evolve_graft.py --seed 2
JOBS
wait
