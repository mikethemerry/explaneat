# MNIST-10 linear sanity check

**Question.** The mnist10 PropNEAT runs reached test accuracy 0.74–0.80, while multinomial
LR reached 0.91 on the same rows. A genome with every pixel wired to every output and no
hidden nodes is multinomial LR. If PropNEAT's own training step cannot fit that genome to
about LR, multi-output training is broken.

**Answer.** The training path is not broken. Two things limited the earlier runs: the
sparse structure the search kept, and the slow optimiser schedule.

Script: `linear_check.py`. Raw output: `logs/linear_check.json` (per-epoch losses) and
`logs/linear_check_evolved.json`. Test: `tests/test_core/test_multiclass_linear.py`.

## Setup

- **Data:** mnist10 from `data.py`: the same 10,000 train rows and the full 10,000-row
  standard test set.
- **Config:** the same, `configs/mnist10.ini` via `run_seed.load_config`, with
  `OutputSinkGenome` and identity outputs. Only `initial_connection` was changed, to
  `full_direct`: 7,840 connections, no hidden nodes, and weights and biases drawn from
  N(0, 1) as in the runs.
- **Training step (`propneat_gen`):** `BackpropPopulation.backpropagate` itself, on a
  one-genome population. It uses `SoftmaxCrossEntropyCriterion`, `mask_phantom=True`, and
  5 epochs per call, with a fresh `Adadelta(lr=1.5)` each call, as in one generation of the
  runs. The weights are written back to the genome after each call.
- **References:**
  - `torch_gen`: `X @ W + b` with `F.cross_entropy`, starting from the same weights and
    using the same fresh-Adadelta-every-5 schedule;
  - `*_cont`: one Adadelta for all epochs;
  - `torch_adam`: Adam with lr 0.01.

## Results

### Fully connected genome

| arm | epochs | train acc | train CE | test acc |
|---|---|---|---|---|
| propneat_gen (the PropNEAT step) | 500 (= 100 gens × 5) | 0.8974 | 0.408 | 0.8731 |
| | 2,500 (= 100 gens × 25) | 0.9505 | 0.187 | 0.8934 |
| | 5,000 | 0.9654 | 0.134 | 0.8954 |
| torch_gen (nn.Linear, same schedule) | 5,000 | 0.9654 | 0.134 | 0.8954 |
| propneat_cont / torch_cont | 5,000 | 0.9652 | 0.134 | 0.8950 |
| torch_adam | 5,000 | 0.9999 | 0.015 | 0.8821 |
| sklearn LR (`baselines.json`, L2 C=1) | — | 0.9682 | 0.128 | 0.9097 |

- **Per-epoch loss:** PropNEAT and the plain linear model are identical.
  - With the same schedule, the maximum |difference| in train CE over all 1,000
    generations is 0.0.
  - With one continuous Adadelta, it is 0.0 over all 5,000 epochs.
  - At initialisation, NeuralNeat and `X @ W + b` agree exactly.
  - So the input → output mapping, the output order, the loss, the output activation and
    the write-back are all correct.
- **Train fit:** the train fit reaches LR's (CE 0.134 vs 0.128; accuracy 0.965 vs 0.968).
- **Test gap:** the 0.015 test-accuracy gap to sklearn is regularisation. sklearn's LR uses
  L2 with C = 1. PropNEAT's loss has no penalty, and it starts from N(0, 1) weights on 784
  pixels. Unregularised Adam, trained further, overfits to 0.88. This is not a
  training-path bug.

### The evolved genomes, trained further with their structure fixed

Each run's saved best genome was trained for 3,000 more epochs on its own connections.

| run | conns / inputs used | test acc as run | + 3,000 epochs, PropNEAT step | + 3,000 epochs, Adam |
|---|---|---|---|---|
| mnist10-s0 | 568 / 414 | 0.7522 | 0.7746 (train 0.765) | 0.7792 (train 0.774) |
| mnist10-s1 | 536 / 390 | 0.7372 | 0.7644 (train 0.763) | 0.7700 (train 0.772) |
| mnist10-e25-s0 | 741 / 500 | 0.8039 | 0.8068 (train 0.804) | 0.8087 (train 0.809) |

Even with Adam to near convergence, these structures stay at about 0.77–0.81. Train and
test accuracy are equal, so the models underfit. With 54–74 connections per output (against 784 in a full readout),
the structure itself is the ceiling.

## What it means for the earlier MNIST-10 runs

**The search was under-structured and under-trained, not broken.** The low accuracy has
two causes:

1. **Structure (the main cause).** The runs start from `partial_direct 0.1`, about 78
   pixels per output, and evolution kept the genomes sparse: 536–741 connections against
   the 7,840 of a full linear model. A sparse linear readout of that size cannot pass about
   0.81, however long it trains. Connections are added by mutation one at a time, and with
   one species and 2 elites, a new connection survives only if it beats the retrained
   champion within the same 5 epochs.
2. **Training budget (secondary).** The PropNEAT step is slow on this problem, because
   Adadelta is re-created every generation and the weights start at N(0, 1). The full
   linear genome needs about 500 epochs (100 generations × 5) to reach 0.87 test, and
   about 2,500 to reach 0.89. With 5 epochs per generation, the champion lineage gets about
   500 epochs in a 100-generation run.

   Resetting the optimiser is not the issue: the fresh-every-5 schedule and continuous
   Adadelta end at the same loss, and the fresh one is ahead early.

The README's suggestion to start mnist10 from `full_direct` should, by this check, give
about 0.87 test accuracy after 100 × 5 epochs and about 0.89 after 100 × 25, before any
hidden structure. Closing the last 0.015 to sklearn would need weight decay or a smaller
weight init. That was not run.

## Reproduce

```bash
PYTHONHASHSEED=0 uv run python experiments/mnist/linear_check.py --epochs 5000 --every 100      # ~8 min, 1 thread
PYTHONHASHSEED=0 uv run python experiments/mnist/linear_check.py --evolved mnist10-s0,mnist10-s1,mnist10-e25-s0 \
    --epochs 3000 --every 500 --out experiments/mnist/logs/linear_check_evolved.json
uv run pytest tests/test_core/test_multiclass_linear.py    # sklearn digits, 64 -> 10
```

The regression test uses sklearn digits (64 inputs, 10 classes) and makes two checks:
- **Parity:** after 20 PropNEAT generations, the genome's weights equal those of
  `nn.Linear` trained on the same schedule, to within 1e-9.
- **Accuracy (`slow`):** after 400 generations × 5 epochs, train accuracy is above 0.97 and
  test accuracy is within 0.03 of sklearn LR.

**Test-only fix found on the way.** `test_multiclass.py::test_single_output_mapping_unchanged`
saved `NodeMapping._align_outputs` through the class, which unwraps the `staticmethod`. It then
restored the bare function. Every later multi-output `NeuralNeat` in the same pytest process
then raised a `TypeError`. The library code was not affected, and nor were the runs. The test
now saves and restores the raw `staticmethod`.
