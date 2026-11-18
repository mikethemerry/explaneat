#!/usr/bin/env bash
# Runs every configuration reported in RESULTS.md, at most 3 processes at once.
# Device: auto (MPS when available, else CPU); baselines and fine-tuning run on CPU.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
{
for lam in 0.001 0.005; do for seed in 0 1 2; do
  echo "--dataset synthetic --penalty $lam --seed $seed --pop 30 --gens 60 --epochs 2"
done; done
for lam in 0.001 0.005; do for seed in 0 1 2; do
  echo "--dataset digits38 --penalty $lam --seed $seed --pop 30 --gens 40 --epochs 5"
done; done
for seed in 0 1; do
  echo "--dataset digits10 --penalty 0.001 --seed $seed --pop 30 --gens 60 --epochs 5"
done
} | xargs -P 3 -I{} sh -c 'uv run python run.py {} --device auto --threads 2 > "logs/$(echo {} | tr -d " -" ).log" 2>&1'
