#!/bin/bash
# Train the whole grid for every task, one process at a time (CPU is shared).
cd "$(dirname "$0")/../.."
for t in "$@"; do
  uv run python experiments/explanation_barrier/train.py "$t"
done
