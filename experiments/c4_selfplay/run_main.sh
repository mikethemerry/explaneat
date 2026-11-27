#!/usr/bin/env bash
# Main self-play runs: three seeds in parallel (one core each; torch uses 1 thread per run).
# Design choices (from the 5-round pilot's timings, docs/plans/2026-11-25-c4-selfplay.md Task 9):
#   400 games x 200 sims per round, replay buffer of 5 rounds, 5 generations x 20 epochs per round,
#   100 rounds (about 500 generations of continuous lineage), 8 temperature moves, no Dirichlet noise.
# Resumable: rerunning this script continues each seed from its checkpoint.
#
#   bash experiments/c4_selfplay/run_main.sh            # launch (or resume) all three seeds
#   tail -f ~/.explaneat/c4_selfplay_runs/logs/main-s0.log
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="${EXPLANEAT_HOME:-$HOME/.explaneat}/c4_selfplay_runs/logs"
mkdir -p "$LOG_DIR"
for seed in 0 1 2; do
  PYTHONHASHSEED=0 caffeinate -i uv run python experiments/c4_selfplay/run.py \
    --name "main-s${seed}" --seed "$seed" --rounds 100 --games 400 --sims 200 \
    --gens 5 --epochs 20 --buffer 5 --temp-moves 8 \
    >> "$LOG_DIR/main-s${seed}.log" 2>&1 &
done
wait
