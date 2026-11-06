#!/bin/bash
# Studies F/G job queue (GRAFT_F_G_PLAN.md): launches each job when the machine has
# fewer than 4 heavy processes (modular study + these), 1 thread each, PYTHONHASHSEED=0.
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
LOGS=experiments/churn/logs_fg
heavy() {
  m=$(ps aux | grep "[e]xperiments/modular/run.py" | grep -v "uv run\|sh -c" | wc -l)
  c=$(ps aux | grep "[e]xperiments/churn/run_\(earlygen\|protect\|monk2graft\)" | grep -v "uv run\|sh -c" | wc -l)
  echo $((m + c))
}
while read -r name args; do
  [ -z "$name" ] && continue
  while [ "$(heavy)" -ge 4 ]; do sleep 20; done
  echo "$(date) start $name: $args" >> $LOGS/queue.log
  (uv run python $args > $LOGS/$name.log 2>&1; echo "$(date) done $name exit $?" >> $LOGS/queue.log) &
  sleep 60
done <<JOBS
f1_es experiments/churn/run_earlygen.py --stage evaluate --regime es
c2_e500 experiments/churn/run_protect.py --context c2 --regime e500 --P 0,25,100
c2_es experiments/churn/run_protect.py --context c2 --regime es --P 0,25,100
d2_es experiments/churn/run_protect.py --context d2 --regime es --P 0,25,100
d2_e500 experiments/churn/run_protect.py --context d2 --regime e500 --P 0,25,100
f2_es experiments/churn/run_monk2graft.py --stage evaluate --regime es
f2_e500 experiments/churn/run_monk2graft.py --stage evaluate --regime e500
f1_e500 experiments/churn/run_earlygen.py --stage evaluate --regime e500
JOBS
wait
echo "$(date) all done" >> $LOGS/queue.log
