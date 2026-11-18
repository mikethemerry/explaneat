#!/usr/bin/env bash
# Build the report draft: the pruned, unpruned and 50k warm-start MNIST 10-class networks from their
# decomposition records, then 3 vs 8 vs noise, then the earlier networks from the cache. CPU only; test is never read here.
set -euo pipefail
cd "$(dirname "$0")"
R=../results
uv run python build_report.py --cache \
  --record ../explanations/mnist10_pruned.json --record-run $R/mnist10_pruned --record-results $R/mnist10_pruned/headline_results.json \
  --record-title "MNIST 10-class (ExConNEAT, pruned)" \
  --record ../explanations/mnist10_unpruned.json --record-run $R/sweep_p50_m4_e5 --record-results $R/sweep_p50_m4_e5/results.json \
  --record-title "MNIST 10-class (ExConNEAT, unpruned)" \
  --record ../explanations/mnist10_ws50k.json --record-run $R/mnist10_ws50k_halving --record-results $R/mnist10_ws50k_halving/results.json \
  --record-title "MNIST 10-class (ExConNEAT, 50k warm start)" \
  --three8noise ../explanations/three8noise_run_s0.json --out convneat_report_draft.html
