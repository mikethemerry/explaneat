#!/bin/bash
# One queue of MNIST-10 linear-start runs, one at a time (run two queues at most).
#   experiments/mnist/launch_linear_start.sh "ls-full-s0 0 --init full" "ls-lr-s1 1 --init lr"
# Each spec is "<name> <seed> <linear_start.py flags>". Resumes from a checkpoint if one exists.
# stdout: ~/.explaneat/mnist_runs/<name>/stdout.log (outside the repo).
cd "$(dirname "$0")/../.." || exit 1
RUNS="${EXPLANEAT_MNIST_RUNS:-$HOME/.explaneat/mnist_runs}"
for spec in "$@"; do
  name="${spec%% *}"
  mkdir -p "$RUNS/$name"
  # shellcheck disable=SC2086
  PYTHONHASHSEED=0 uv run python experiments/mnist/linear_start.py $spec >> "$RUNS/$name/stdout.log" 2>&1
  echo "$(date '+%F %T') $name exit $?" >> "$RUNS/linear_start_queue.log"
done
