#!/usr/bin/env bash
# CPU only, two processes at most (two threads each). Logs in logs/.
set -u
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
( for pool in 10k 50k; do for part in lr l1 tree hgb; do $PY run_baselines.py --part $part --pool $pool; done; done ) > logs/main.log 2>&1 &
( for pool in 10k 50k; do $PY run_baselines.py --part ebm --pool $pool; done ) > logs/ebm.log 2>&1 &
wait
