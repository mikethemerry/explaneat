#!/bin/bash
# Run a queue of MNIST runs one after another (one process at a time per queue).
#   experiments/mnist/launch.sh mnist10:0 mnist38:0 mnist38:1
# Each run resumes from its checkpoint if one exists. stdout goes to
# ~/.explaneat/mnist_runs/<task>-s<seed>/stdout.log (outside the repo).
cd "$(dirname "$0")/../.." || exit 1
for spec in "$@"; do
  task="${spec%%:*}"; seed="${spec##*:}"
  dir="$HOME/.explaneat/mnist_runs/${task}-s${seed}"
  mkdir -p "$dir"
  PYTHONHASHSEED=0 uv run python experiments/mnist/run_seed.py "$task" "$seed" >> "$dir/stdout.log" 2>&1
  echo "$(date '+%F %T') ${task}-s${seed} exit $?" >> "$HOME/.explaneat/mnist_runs/queue.log"
done
