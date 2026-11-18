#!/bin/sh
# 4 arms x seeds 0-2, 15 minutes each, 2 processes at a time (CPU, 1 thread each).
cd "$(dirname "$0")"
for s in 0 1 2; do for arm in fixed phase selfadapt halving; do echo "$arm $s"; done; done |
  xargs -P 2 -n 2 sh -c 'uv run python run.py --arm "$0" --seed "$1" --minutes ${MINUTES:-15} > logs/"$0"_s"$1".out 2>&1'
