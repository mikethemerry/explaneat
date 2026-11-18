#!/usr/bin/env bash
# ConvNEAT MNIST long run on a Linux box with an NVIDIA GPU (RTX 5080 = Blackwell sm_120: CUDA 12.8, torch >= 2.7).
# See README.md in this folder. Run from anywhere; paths resolve to the repo.
#
#   ./run_rig.sh setup                 # separate venv .venv-rig with CUDA 12.8 torch (does not touch the repo's .venv)
#   ./run_rig.sh check                 # GPU visible? conv tests pass? MNIST cache present?
#   ./run_rig.sh bench                 # per-step timing on the GPU (bench/mnist_scale_cuda.json)
#   ./run_rig.sh calibrate             # one generation (pop 24) at 1/2/4/8 workers: pick WORKERS
#   ./run_rig.sh start SEED [args...]  # fresh run, rig defaults (pop 50, 4x structural mutation, 5 epochs, full 50k)
#   ./run_rig.sh resume DIR [args...]  # continue a run directory (e.g. one copied from the Mac)
#   ./run_rig.sh pause DIR             # checkpoint and exit after the current generation
#   ./run_rig.sh pack                  # tar the results + logs to copy back
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(dirname "$HERE")"                # experiments/conv_neat
REPO="$(cd "$EXP/../.." && pwd)"
VENV="${VENV:-$REPO/.venv-rig}"
PY="$VENV/bin/python"
WORKERS="${WORKERS:-6}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EXP/logs"

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)
    command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$PY" "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$PY" numpy scipy scikit-learn matplotlib pytest
    "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
    ;;
  check)
    "$PY" - <<'EOF'
import os, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available: driver >= 570 and the cu128 wheel are needed for an RTX 50xx"
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(64, 3, 28, 28, device="cuda"); w = torch.randn(4, 3, 5, 5, device="cuda", requires_grad=True)
torch.nn.functional.conv2d(x, w, padding=2).sum().backward(); torch.cuda.synchronize(); print("conv2d fwd/bwd ok")
p = os.path.expanduser("~/.explaneat/data/mnist_784.npz")
print("MNIST cache", p, "present" if os.path.exists(p) else "MISSING (copy it, or the first run fetches it from OpenML)")
EOF
    "$PY" -m pytest "$REPO/tests/test_conv" -q -p no:cacheprovider --noconftest -o addopts="" -o filterwarnings=
    ;;
  bench)
    "$PY" "$EXP/bench_mnist_scale.py" --device cuda --threads 1 "$@"
    ;;
  calibrate)
    for w in 1 2 4 8; do
      out="$(mktemp -d)"
      t0=$(date +%s)
      "$PY" "$EXP/run_long.py" --seed 0 --pop 24 --max-gens 1 --finetune 0 --train-chunk 0 \
        --device cuda --threads 1 --workers "$w" --out "$out" > "$out/log.txt" 2>&1
      echo "workers=$w: one generation of 24 minimal genomes, full 50k x 5 epochs: $(( $(date +%s) - t0 )) s (incl. startup)"
      rm -rf "$out"
    done
    echo "Genomes grow (x3-10 the cost per step by generation 100); pick the smallest WORKERS near the fastest."
    ;;
  start)
    seed="${1:?seed}"; shift
    out="$EXP/results/mnist10rig_p50_m4_e5_full_s$seed"
    nohup "$PY" "$EXP/run_long.py" --seed "$seed" --pop 50 --epochs 5 --mut-scale 4 --batch 256 --train-chunk 0 \
      --device cuda --threads 1 --workers "$WORKERS" --out "$out" "$@" > "$EXP/logs/$(basename "$out").log" 2>&1 &
    echo "started pid $! -> $out (log: logs/$(basename "$out").log)"
    ;;
  resume)
    dir="${1:?run directory, e.g. experiments/conv_neat/results/mnist10long_gap_s0}"; shift
    dir="$(cd "$dir" && pwd)"
    seed="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['args']['seed'])" "$dir/config.json")"
    nohup "$PY" "$EXP/run_long.py" --resume --seed "$seed" --out "$dir" --device cuda --threads 1 \
      --workers "$WORKERS" "$@" >> "$EXP/logs/$(basename "$dir").log" 2>&1 &
    echo "resumed pid $! -> $dir"
    ;;
  pause)
    touch "${1:?run directory}/STOP"; echo "will checkpoint and exit after the current generation"
    ;;
  pack)
    cd "$EXP"
    tar czf "$REPO/convneat_rig_results.tgz" --exclude='*.pkl' results/mnist10rig_* results/mnist10long_* logs/mnist10* 2>/dev/null || true
    tar czf "$REPO/convneat_rig_checkpoints.tgz" results/mnist10rig_*/checkpoint.pkl results/mnist10long_*/checkpoint.pkl 2>/dev/null || true
    ls -la "$REPO"/convneat_rig_*.tgz
    ;;
  *)
    sed -n '2,12p' "$0"
    ;;
esac
