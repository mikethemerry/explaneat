#!/usr/bin/env bash
# The five long runs in three lanes (at most 3 training processes):
#   lane 0: shape-C s0, then churn s0;  lane 1: shape-C s1, then churn s1;  lane 2: shape-C s2.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/long_run/logs
run() { PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
          uv run python experiments/long_run/run_long.py "$1" "$2" > "experiments/long_run/logs/$1-s$2.log" 2>&1; }
(run both 0; run churn 0) &
(run both 1; run churn 1) &
(run both 2) &
wait
echo "all lanes done"
