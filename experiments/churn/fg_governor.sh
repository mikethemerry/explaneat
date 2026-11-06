#!/bin/bash
# Keeps (modular heavy processes + running F/G processes) <= 4: SIGSTOPs the newest F/G
# processes when over, SIGCONTs the oldest stopped ones when slots free.
LOG=/Users/mike/dev/explaneat/experiments/churn/logs_fg/governor.log
while true; do
  m=$(ps aux | grep "[e]xperiments/modular/run.py" | grep -v "uv run\|sh -c" | wc -l | tr -d ' ')
  pids=$(ps -eo pid,lstart,command | grep "[e]xperiments/churn/run_\(earlygen\|protect\|monk2graft\|evolve_graft\)\.py" | grep -v "uv run\|sh -c" | awk '{print $1}')
  allowed=$((4 - m)); [ $allowed -lt 0 ] && allowed=0
  i=0
  for p in $(echo "$pids" | sort -n); do
    st=$(ps -o stat= -p $p | cut -c1)
    if [ $i -lt $allowed ]; then
      [ "$st" = "T" ] && kill -CONT $p && echo "$(date) CONT $p (modular $m)" >> $LOG
    else
      [ "$st" != "T" ] && kill -STOP $p && echo "$(date) STOP $p (modular $m)" >> $LOG
    fi
    i=$((i + 1))
  done
  sleep 15
done
