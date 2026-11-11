#!/bin/bash
# Complexity measures for every imported ladder network, one process, 2 threads.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
uv run python experiments/explanation_barrier/complexity.py \
  additive 4 hinges3 4 threeway 8 pairwise 8x8 parity2 4 parity3 8 parity4 8 hinges6 8 rule 16x16 additive 16 \
  c4 4 c4 8 c4 16 c4 32 c4 32x16
