#!/usr/bin/env bash
# Sigmoid hidden units on the both dataset: seeds 0-4 of shape-C, 300 generations, the original
# config except hidden_activation = sigmoid (experiments named shape-C-sig-s<n>). At most 2
# training processes while the ReLU 300-generation job (worktrees/ushape) runs, 3 after it ends.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/ushape_saddle/logs
mine() { pgrep -f "bin/python3 experiments/ushape_saddle/run_seed.py both .* --name shape-C-sig-s" | wc -l | tr -d ' '; }
limit() { if pgrep -f "worktrees/ushape/.venv/bin/python3 experiments/ushape_saddle/run_seed.py" >/dev/null; then echo 2; else echo 3; fi; }
for s in 0 1 2 3 4; do
  while (( $(mine) >= $(limit) )); do sleep 30; done
  PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
    uv run python experiments/ushape_saddle/run_seed.py both "$s" --generations 300 \
      --hidden-activation sigmoid --name "shape-C-sig-s$s" \
      > "experiments/ushape_saddle/logs/both-sig-s$s.log" 2>&1 &
  sleep 20
done
wait
echo "all sigmoid runs done"
