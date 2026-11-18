#!/usr/bin/env bash
# After the MNIST 10-class run (results/sweep_p50_m4_e5) stops: prune, then the overnight warm start.
#   a. scarcity: lambda x5, deletion mutations only, same 10k subset, validation guard 0.01 -> results/mnist10_pruned
#      (its finalize evaluates the pruned best-by-validation genome on test once and fine-tunes it: the headline network)
#   c. warm start of the pruned population on all 50k training rows, successive halving (2 / top 20% +8 / leaders 20),
#      plateau 30, cap 12 h of wall time -> results/mnist10_ws50k_halving
# The report for the headline network is built separately (CPU) while c runs.
set -euo pipefail
cd "$(dirname "$0")"
until [ -f results/sweep_p50_m4_e5/selected_genome.json ]; do sleep 30; done
sleep 5
uv run python run_long.py --seed 0 --warm-start results/sweep_p50_m4_e5/checkpoint.pkl --schedule scarcity \
  --n-train 10000 --epochs 5 --penalty 0.001 --mut-scale 4 --scarcity-penalty-scale 5 --scarcity-max-gens 10 \
  --val-guard 0.01 --device auto --threads 2 --out results/mnist10_pruned > logs/mnist10_pruned.log 2>&1
touch results/mnist10_pruned/DONE
uv run python run_long.py --seed 0 --warm-start results/mnist10_pruned/checkpoint.pkl --schedule halving \
  --epochs 5 --penalty 0.001 --mut-scale 4 --train-chunk 0 --patience 30 --min-delta 0.001 --max-gens 400 \
  --max-minutes 720 --device auto --threads 2 --out results/mnist10_ws50k_halving > logs/mnist10_ws50k_halving.log 2>&1
