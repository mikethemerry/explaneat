#!/usr/bin/env bash
# MNIST runs (28x28, levels 28/14/7). Device: auto (MPS); baselines and fine-tuning on CPU.
# Usage: ./run_mnist.sh 38   (3 vs 8: lambda 0.001 seeds 0-2, lambda 0.005 seed 0)
#        ./run_mnist.sh 10   (10-class: lambda 0.001)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [ "${1:-38}" = "38" ]; then
{
for seed in 0 1 2; do echo "--dataset mnist38 --penalty 0.001 --seed $seed --pop 30 --gens 40 --epochs 2"; done
echo "--dataset mnist38 --penalty 0.005 --seed 0 --pop 30 --gens 40 --epochs 2"
} | xargs -P 4 -I{} sh -c 'uv run python run.py {} --device auto --threads 2 > "logs/$(echo {} | tr -d " -" ).log" 2>&1'
else
{
for seed in ${SEEDS:-0}; do echo "--dataset mnist10 --penalty 0.001 --seed $seed --pop 30 --gens ${GENS:-40} --epochs 2"; done
} | xargs -P 2 -I{} sh -c 'uv run python run.py {} --device auto --threads 2 > "logs/$(echo {} | tr -d " -" ).log" 2>&1'
fi
