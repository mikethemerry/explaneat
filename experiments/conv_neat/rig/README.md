# ConvNEAT MNIST long run on the RTX 5080 rig

This is the same experiment as the Mac run (`run_long.py`; protocol in `../RESULTS.md`, section 11b), on a Linux box with an
NVIDIA RTX 5080. Everything runs through `run_rig.sh` in this folder.

## Why the rig

The Mac benchmark is in `../bench_mnist_scale.py` and `../bench/mnist_scale_*.json`. On MPS, a genome's training step is
**overhead-bound**: about 1.3 ms per PropNEAT layer for launches and Python, so batch 8 costs the same as batch 128. From
batch 512 it is compute-bound, at about 44 us per image per epoch for a complexity-50 genome. Two processes saturate the
Mac GPU. Pop 50 × the full 50k images × 5 epochs would take about 15 min per generation on the Mac, and pop 150 about 50 min.

On CUDA the per-launch cost is about 3 to 4 times lower, and cuDNN handles these small convolutions better. The
main gain, though, comes from **training several genomes at once** in worker processes (`--workers N`,
`explaneat/conv/parallel.py`). Each worker drives the GPU from its own CPU core. Every genome's training rng is
seeded by the population, so the results do not depend on the number of workers (bit-exact on CPU; tested).

Expected gain (an estimate, not measured): about 4 to 5 times per process, and about 8 to 15 times in aggregate with 4 to 8 workers. At 10
times, the recommended config (pop 50, 50k, 5 epochs; below) takes about 1.5 to 2 min per generation. Run
`calibrate` and `bench` first, and put the measured numbers in RESULTS.md.

## 1. Environment (a separate venv)

The repo pins `torch>=2.2,<2.3`, and on Linux `pyproject.toml` routes torch to the **CPU** wheel index. An RTX 50xx
(Blackwell, sm_120) needs CUDA 12.8 wheels and **torch >= 2.7**. So do **not** use `uv sync` / `uv run` for this. The
rig gets its own venv, `.venv-rig`, holding only what `explaneat.conv` needs:

```bash
git clone <repo> explaneat && cd explaneat && git checkout conv-neat-prototype   # or copy the worktree
curl -LsSf https://astral.sh/uv/install.sh | sh                                  # if uv is missing
experiments/conv_neat/rig/run_rig.sh setup     # uv venv .venv-rig (py3.12) + torch (cu128 index) + numpy scipy sklearn matplotlib pytest
experiments/conv_neat/rig/run_rig.sh check     # GPU name/capability (expect (12, 0)), a conv fwd/bwd, MNIST cache, tests/test_conv
```

* You need an NVIDIA driver >= 570 (the CUDA 12.8 minimum for Blackwell).
* The ConvNEAT code uses only stable torch APIs (`conv2d`, `adaptive_avg_pool2d`, `interpolate`, Adam,
  `torch.where`). It has no `torch.load`/`save`, no `torch.compile` and no private APIs. `tests/test_conv` passes on torch 2.15 / numpy 2.5
  (checked on the Mac, CPU).
* Checkpoints hold only numpy and plain Python. One written on the Mac (torch 2.2, numpy 1.26) resumed under torch 2.15 /
  numpy 2.5 with 2 CPU workers.
* `explaneat/__init__.py` tries to import the rest of ExplaNEAT (neat-python, sqlalchemy, ...). In this venv those imports
  fail with an ImportWarning, which is harmless: `explaneat.conv` is self-contained.

## 2. MNIST cache

Copy it from the Mac. Otherwise the first run fetches it once from OpenML (about 15 MB, through `sklearn.fetch_openml`):

```bash
scp mac:~/.explaneat/data/mnist_784.npz ~/.explaneat/data/       # mkdir -p ~/.explaneat/data first
```

## 3. Calibrate, then run

```bash
experiments/conv_neat/rig/run_rig.sh bench        # per-step times at batch 8..1024 -> bench/mnist_scale_cuda.json
experiments/conv_neat/rig/run_rig.sh calibrate    # one generation (24 minimal genomes, 50k x 5 epochs) at 1/2/4/8 workers
WORKERS=6 experiments/conv_neat/rig/run_rig.sh start 0      # fresh: results/mnist10rig_p50_m4_e5_full_s0, log in logs/
WORKERS=6 experiments/conv_neat/rig/run_rig.sh start 1      # second seed (if the GPU and CPU have room; else run after seed 0)
tail -f experiments/conv_neat/logs/mnist10rig_p50_m4_e5_full_s0.log
```

**Recommended rig settings** (the `start` defaults; from the Mac's hyperparameter check, `../RESULTS.md` 11b):

* population 50, **4× structural mutation** (`--mut-scale 4`), **5 epochs**, batch 256. On the Mac, 5 epochs beat 20
  and 50 per minute of wall time by a wide margin, and this config passed logistic regression (0.910) by generation
  20 on a 10k subset;
* the **full 50k** training pool every generation (`--train-chunk 0`), which the Mac could not afford (it used a fixed
  10k subset): the first thing the rig can add;
* λ 0.001, train fitness, plateau N 30 / min Δ 0.001, cap 400, head GAP + linear.

The projection is uncertain until `calibrate` runs:

* about 980 Adam steps per genome per generation; at a complexity of about 60, about 18 ms per step on the Mac
  alone, so about 15 min per generation of 50 on MPS;
* on the 5080 with 4 to 8 workers, at about 10×, roughly 1.5 to 2 min per generation, so 200 generations in about
  5 to 7 h per seed;
* a 20k subset (`--n-train 20000`) about halves that.

**Workers.** Choose the smallest number that is near the fastest in `calibrate`; expect 4 to 8.

* Each worker holds the data on the GPU (about 0.2 GB) plus a CUDA context (about 0.5 GB), so 16 GB of VRAM is not the limit.
* CPU cores are the limit: keep workers × seeds ≤ physical cores − 1.
* For more speed per worker, try batch 512 (`--batch 512`), which halves the steps. On the Mac, batch 256 was the
  compromise between steps per generation (new genes need Adam steps to prove themselves) and throughput.

**Other arms.** Extra arguments go to `run_long.py`:

* `--head grid2`: 2×2 pooling before the linear head;
* `--train-chunk 20000`;
* `--fitness val`: the earlier runs' rule.

## 4. Resume

Resume a rig run after an interruption, or **continue a Mac run on the rig**:

```bash
# on the Mac: pause the run cleanly (it checkpoints after the current generation), then copy its directory
touch experiments/conv_neat/results/sweep_p50_m4_e5/STOP          # or mnist10long_gap_s0 (the paused pop-100 run)
rsync -a experiments/conv_neat/results/sweep_p50_m4_e5 rig:explaneat/experiments/conv_neat/results/
# on the rig
WORKERS=6 experiments/conv_neat/rig/run_rig.sh resume experiments/conv_neat/results/sweep_p50_m4_e5
```

* A resumed run keeps its experiment settings from the checkpoint (population, chunking, epochs, batch, λ, seed).
* Device and workers come from the command line.
* Checkpoints are written every 5 generations (`--checkpoint-every`), so a crash loses at most 5 generations.
* `log.jsonl` is truncated back to the checkpoint on resume.
* GPU kernels are not bit-deterministic, so a resumed run is statistically, not bitwise, the same run.

## 5. Copy results back

```bash
experiments/conv_neat/rig/run_rig.sh pack     # convneat_rig_results.tgz (no checkpoints) + convneat_rig_checkpoints.tgz
scp rig:explaneat/convneat_rig_results.tgz . && tar xzf convneat_rig_results.tgz -C experiments/conv_neat/
```

Each run directory holds:

| file | content |
|---|---|
| `config.json` | arguments, ConvNEAT config, torch version, host, split row indices |
| `log.jsonl` | one row per generation |
| `checkpoint.pkl` | population state |
| `results.json`, `selected_genome.json` | at the end: stop reason, test and fine-tuned accuracy, parameters, wall time |

Each `log.jsonl` row records:

* best and mean validation accuracy;
* best and mean training accuracy;
* the monitored (best-by-train) genome's validation accuracy;
* the selected genome's validation accuracy, complexity and parameters;
* the plateau counter;
* sizes, wall time, host, device and workers.

Then on the Mac, `uv run python experiments/conv_neat/summarise_long.py` redraws the curves.

## Not implemented (possible next steps)

* **CUDA-graph capture of the training step.** Each genome's step has static shapes, and Adam can use
  `capturable=True`. Capture would remove most launch overhead, perhaps another 2 to 3 times on small genomes. It is untested,
  because there was no GPU here.
* **`torch.compile`.** It does not pay: every genome is a new graph, and compiling takes 10 to 40 s, longer than the
  genome's training.
* **Batching a population's genomes into grouped convolutions.** This would reach the GPU's real throughput, but it is
  a larger refactor of the PropNEAT engine.
