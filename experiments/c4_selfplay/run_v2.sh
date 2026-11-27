#!/usr/bin/env bash
# Self-play v2 runs: three seeds (v2-s0/1/2), each training its population in 4 worker processes.
# PLACEHOLDER budget: the numbers are set after the v2 pilot. Design choices (run.py's v2 options):
#   --rounds 200 --games 600 --sims 400   more and better-searched games per round, longer lineage
#   --gens 5 --epochs 20 --buffer 6       as main, with a slightly longer replay buffer
#   --temp-moves 8 --dirichlet 1.0,0.25   sampled openings plus root noise, for more varied positions
#   --value-mix 0.5                       label = half game outcome, half MCTS root value (less noisy)
#   --holdout-frac 0.1                    10% of the newest round's positions held out of its
#                                         training; the champion is the best genome on them
#                                         (picks for generalisation, not for fitting the buffer)
#   --workers 4                           parallel backprop, bitwise identical to serial
#   --species-band 4,10,0.1,0.3           keep 4-10 species (the main pilot collapsed to one):
#                                         threshold moves 0.1 per generation, never below 0.3
#   --set DefaultGenome.num_hidden=10     start with 10 homologous hidden units (shared ids 1..10)
#   --set DefaultGenome.node_delete_prob=0.05   keep hidden units from being pruned away quickly
#   --set DefaultSpeciesSet.compatibility_threshold=1.0   starting threshold nearer the band
# Resumable: rerunning this script continues each seed from its checkpoint.
#
#   bash experiments/c4_selfplay/run_v2.sh             # launch (or resume) all three seeds
#   tail -f ~/.explaneat/c4_selfplay_runs/logs/v2-s0.log
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="${EXPLANEAT_HOME:-$HOME/.explaneat}/c4_selfplay_runs/logs"
mkdir -p "$LOG_DIR"
for seed in 0 1 2; do
  PYTHONHASHSEED=0 caffeinate -i uv run python experiments/c4_selfplay/run.py \
    --name "v2-s${seed}" --seed "$seed" --rounds 200 --games 600 --sims 400 \
    --gens 5 --epochs 20 --buffer 6 --temp-moves 8 --dirichlet 1.0,0.25 \
    --value-mix 0.5 --holdout-frac 0.1 --workers 4 --species-band 4,10,0.1,0.3 \
    --set DefaultGenome.num_hidden=10 --set DefaultGenome.node_delete_prob=0.05 \
    --set DefaultSpeciesSet.compatibility_threshold=1.0 \
    >> "$LOG_DIR/v2-s${seed}.log" 2>&1 &
done
wait
