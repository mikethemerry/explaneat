# MNIST PropNEAT runs: candidates for "not explained within our approach"

**Exploratory, not pre-registered.** Nothing here was planned in advance or counts towards
any pre-registered threshold. The aim is to train PropNEAT models on MNIST that are
candidates for a model we cannot explain within our approach and budget, so the
explanation tools can be tried on them later. **No explanation was attempted here.** This
directory only reports training results and structure statistics that indicate how hard
explanation is likely to be.

## Summary

Under this protocol, **none of the runs is a good candidate for "not explained within our
approach"**. Every best genome is close to a sparse linear model:
- each has 0–5 hidden ReLU units, each fed by 1–5 pixels;
- 88–100% of its connections are direct pixel → output weights.

The 3-vs-8 models match logistic regression: test AUC 0.993–0.994 vs LR 0.9937 and a
small GBM 0.9997.

The 10-class models are under-trained: test accuracy 0.74–0.80 vs multinomial LR 0.91 and
GBM 0.96. Their single hidden unit, where there is one, is always on, so it is linear on
the data.

Two reasons stand out, both visible in the logs:
1. **One species.** Every run has a single species (two at most), so only 2 elites are kept.
   The elite champion is retrained every generation, and new structure rarely beats it. In
   mnist38-s0 the same champion genome (key 15891) leads from generation 64 to 99, and in
   the 500-generation run it leads until generation 313. The 500-generation
   run (a strict continuation of mnist38-s0) ends with 4 hidden units, a test AUC 0.0002
   higher, and no added depth.
2. **Training budget.** With 5 backprop epochs per generation, mnist10 is still improving at
   generation 99. Five times the epochs (mnist10-e25-s0) reaches 0.80 test accuracy, but
   the network shape stays the same.

**For a harder candidate:**
- **Raise speciation.** Lower `compatibility_threshold` or add a species target, so
  structural innovations survive.
- **Train longer per generation.**
- **Grow structure.** Initialise with hidden nodes, or raise `node_add_prob` and the bias
  init.
- **Start mnist10 from `full_direct`.**

These are suggestions only; none was run. The existing models are imported and loadable in
the UI/MCP, and every tool check passes.

| run | test | hidden | conns | inputs used | genome id |
|---|---|---|---|---|---|
| mnist38-s0 | AUC 0.9937 | 5 | 133 | 123/196 | `dbc4cbe9-968b-42d7-9b97-d3f6ae5b0abd` |
| mnist38-s1 | AUC 0.9939 | 1 | 113 | 112/196 | `0a7fa35b-a747-4bab-9a16-d519af74f210` |
| mnist38-s2 | AUC 0.9932 | 0 | 112 | 112/196 | `6187d7cf-ef21-45c5-8b6d-b9d4e8c57403` |
| mnist38-g500-s0 (extension) | AUC 0.9939 | 4 | 131 | 122/196 | `7a51a92b-d432-4682-8a41-8e5669ae6e95` |
| mnist10-s0 | acc 0.7522 | 1 | 568 | 414/784 | `f9d70c2b-5d5f-4c5c-b666-315571bb9777` |
| mnist10-s1 | acc 0.7372 | 0 | 536 | 390/784 | `d362cbf2-585c-4595-a501-f75c1d6a6fda` |
| mnist10-e25-s0 (extension) | acc 0.8039 | 1 | 741 | 500/784 | `e3ff56b5-326e-459b-9b52-c02e1224cac2` |

**Extensions.** These were added because the protocol runs were far inside the time budget.
They are labelled as extensions and are not part of the requested design:
- **`mnist38-g500-s0`:** the same seed and config as mnist38-s0, for 500 generations. Its
  first 100 generations are identical to mnist38-s0.
- **`mnist10-e25-s0`:** mnist10 seed 0 with 25 backprop epochs per generation instead of 5
  (`--epochs 25`).

## Data

Source: OpenML `mnist_784` v1 via `sklearn.datasets.fetch_openml`, cached outside the repo
at `~/.explaneat/data/mnist_784.npz` (uint8 pixels and int labels). OpenML row order is the
standard MNIST order: rows 0–59,999 are the standard training set and rows
60,000–69,999 the standard test set. Code: `data.py`.

| | `mnist38` (binary) | `mnist10` (10-class) |
|---|---|---|
| Classes | 3 (y = 0) vs 8 (y = 1) | 0–9 |
| Inputs | 28×28 → 14×14 by 2×2 mean pooling = 196 | 784 |
| Scaling | raw / 255 → [0, 1], then pooled; **no z-scoring** | raw / 255 → [0, 1]; no z-scoring |
| Train | **all** 11,982 standard-train 3s and 8s (no subset was needed: about 8 s per generation) | stratified seeded subset of standard train, 10,000 rows (1,000 per class) |
| Test | all 1,984 standard-test 3s and 8s | the full standard test set, 10,000 rows |

- **Scaling:** [0, 1] keeps every input a pixel intensity, which is easier to read in an
  explanation. Constant border pixels are left in.
- **Subset seed:** `SUBSET_SEED = 12345`, independent of the evolution seed, so all seeds see
  the same rows. The OpenML row indices are recorded in the DB dataset's
  `additional_metadata`.

## Training

Harness: `run_seed.py`. Seeding follows `experiments/10seed/run_seed.py`: python `random`,
numpy and torch are seeded before the population is built; 1 torch thread (and 1
BLAS/OpenMP thread); deterministic algorithms; `PYTHONHASHSEED=0` is required.

- **Storage:** in memory (`BackpropPopulation`). Nothing is written to the database during a
  run.
- **Phantom weights:** `mask_phantom = True`, the library default. The stored genome is the
  trained network. The 10-seed study ran unmasked.
- **Training schedule:** 5 backprop epochs per generation, full batch, Adadelta with lr 1.5,
  as in the 10-seed study.
- **Best genome:** the best *training* fitness over all generations; there is no validation
  selection. The test split is only logged: each generation, the champion's test metric is
  written to the CSV.

| | `mnist38` | `mnist10` |
|---|---|---|
| NEAT config | `configs/mnist38.ini` = `experiments/10seed/configs/heart.ini` with `num_inputs = 196` | `configs/mnist10.ini` = heart.ini with `num_inputs = 784`, `num_outputs = 10`, `initial_connection = partial_direct 0.1` |
| Output | 1 sigmoid | 10 identity outputs (logits) |
| Backprop loss | BCE (the default) | softmax cross-entropy (`explaneat.core.multiclass.SoftmaxCrossEntropyCriterion`) |
| Fitness | train ROC AUC (`auc_fitness`) | **−(train cross-entropy)** (`make_multiclass_fitness("neg_ce")`). Chosen over accuracy because it is smooth, is the quantity backprop minimises, and does not tie |
| Genome class | `DefaultGenome` | `OutputSinkGenome`: add-connection never starts at an output (see below) |
| Population / generations | 250 / 100 | 250 / 100 |

**Initial connections (mnist10).** `full_direct` would start every genome with 7,840
connections. `partial_direct 0.1` starts with about 784, each output reading about 78
random pixels. Unused pixels can be connected later by add-connection mutations.

### Multi-output support (library changes, additive)

Binary runs are bit-identical to `origin/main`. A 4-generation mnist38 run (masked and
unmasked) gives byte-identical populations and StructureNetwork outputs under both code
trees.

- **`explaneat/core/multiclass.py` (new):**
  - `SoftmaxCrossEntropyCriterion`: takes `[N, k]` logits and the population's `[N, 1]`
    float labels. `BackpropPopulation` needed no change.
  - `make_multiclass_fitness`
  - `OutputSinkGenome` and `use_output_sink_genome(config)`. neat-python allows
    output → hidden connections, so with several outputs, one output could feed another
    through a hidden node.
- **`NeuralNeat`:** with `num_outputs > 1` (`NodeMapping._align_outputs`), every output is
  moved to one final layer, in `output_keys` order. Previously the network returned only the
  layer containing node 0, so outputs at other depths were lost, and hidden nodes at node
  0's depth were mixed into the output. A genome whose output feeds a node is rejected
  (`GenomeNotValidError`). Single-output genomes never enter this code.
- **`StructureNetwork.forward`:** with more than one output, the whole network is run and
  `[N, k]` is returned in `output_node_ids` order. An output with no path from an input is
  the constant activation(bias), as in NeuralNeat.
- **MCP `compute_performance`:** crashed on k-output models (a shape broadcast error). It now
  reports argmax accuracy, per-class accuracy, softmax cross-entropy and OvR macro AUC for
  them. The single-output path is unchanged.
- **Tests:** `tests/test_core/test_multiclass.py`, covering:
  - column order;
  - NeuralNeat ↔ StructureNetwork parity on 6 random multi-output genomes;
  - unconnected outputs;
  - rejection of non-sink outputs;
  - criterion and fitness;
  - a population training step;
  - the single-output path being untouched.

  Also `tests/test_mcp/test_multiclass_performance.py`.

### Reproduce

```bash
PYTHONHASHSEED=0 uv run python experiments/mnist/run_seed.py mnist38 0     # seeds 0-2
PYTHONHASHSEED=0 uv run python experiments/mnist/run_seed.py mnist10 0     # seeds 0-1
experiments/mnist/launch.sh mnist10:0 mnist38:0 ...                         # a queue, one at a time
uv run python experiments/mnist/baselines.py
DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/mnist/import_run.py mnist38-s0
DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/mnist/analyze.py mnist38-s0
uv run python experiments/mnist/summarize.py                                # the tables below
```

**Resume:** rerun the same command. A checkpoint every 5 generations
(`~/.explaneat/mnist_runs/<name>/ckpt.pkl.gz`) holds the population, species, indexers,
ancestry and all RNG states. A 4-generation run interrupted after generation 2 and resumed
reproduced the uninterrupted run exactly.

**Outputs:**
- In the repo:
  - `logs/<name>.csv`: one row per generation;
  - `logs/<name>.json`: the summary;
  - `logs/<name>.import.json`: the DB ids and size change;
  - `logs/<name>.analysis.json`: parity, tools and structure;
  - `baselines.json`.
- Outside the repo, in `~/.explaneat/mnist_runs/<name>/`:
  - `final.pkl.gz`: the last evaluated population, species, parent keys and best genome;
  - `best_genome.pkl.gz`;
  - `parents.csv.gz`: the full parent log, in the lineage-sidecar format;
  - `stdout.log`.

## Timings

Measured on an Apple M4 Pro, 1 thread per run, with up to 3 runs plus another agent's work
on the machine.

- **One generation (first timing):**
  - mnist38 on all 11,982 rows: about 4–5 s.
  - mnist10 on 10,000 rows: about 12–15 s, before machine contention.
  - mnist10 with 25 epochs: about 55 s.
- **Cost driver:** most of the cost is full-batch forward/backward passes over 250 genomes.
  The networks stay small, so the time per generation barely grows.
- **Whole runs (wall clock):**
  - mnist38: 11–14 min per 100 generations; the 500-generation run took 95 min;
  - mnist10: 40–57 min per 100 generations;
  - mnist10-e25: 156 min.

  All were far inside the 6–8 h budget, so no subsetting was needed for mnist38.

The per-run columns (first-5 / last-5 seconds per generation) are in the results table.

## Results

Train and test metrics are for the best genome (best train fitness), from the in-memory
NeuralNeat. They are recomputed from the DB through StructureNetwork in the parity check
below.

| run | train | test | best gen | wall (min) | s/gen first-5 / last-5 | genome id |
|---|---|---|---|---|---|---|
| mnist10-e25-s0 | acc 0.7987, CE 0.6284 | acc 0.8039, CE 0.6373 | 99 | 156 | 77.4 / 79.6 | `e3ff56b5-326e-459b-9b52-c02e1224cac2` |
| mnist10-s0 | acc 0.7400, CE 0.7905 | acc 0.7522, CE 0.7683 | 99 | 40 | 18.6 / 28.1 | `f9d70c2b-5d5f-4c5c-b666-315571bb9777` |
| mnist10-s1 | acc 0.7365, CE 0.8195 | acc 0.7372, CE 0.8013 | 99 | 57 | 22.3 / 37.5 | `d362cbf2-585c-4595-a501-f75c1d6a6fda` |
| mnist38-g500-s0 | AUC 0.9929, acc 0.9696 | AUC 0.9939, acc 0.9688 | 499 | 95 | 6.1 / 9.2 | `7a51a92b-d432-4682-8a41-8e5669ae6e95` |
| mnist38-s0 | AUC 0.9924, acc 0.9659 | AUC 0.9937, acc 0.9682 | 99 | 12 | 5.2 / 8.1 | `dbc4cbe9-968b-42d7-9b97-d3f6ae5b0abd` |
| mnist38-s1 | AUC 0.9921, acc 0.9664 | AUC 0.9939, acc 0.9708 | 99 | 11 | 4.7 / 6.8 | `0a7fa35b-a747-4bab-9a16-d519af74f210` |
| mnist38-s2 | AUC 0.9918, acc 0.9659 | AUC 0.9932, acc 0.9667 | 99 | 14 | 6.2 / 8.7 | `6187d7cf-ef21-45c5-8b6d-b9d4e8c57403` |

**Baselines** (`baselines.py`, the same rows; `baselines.json`):

| task | model | train | test |
|---|---|---|---|
| mnist38 | logistic_regression | AUC 0.9930, acc 0.9703 | AUC 0.9937, acc 0.9693 |
| mnist38 | gbm_small | AUC 0.9999, acc 0.9982 | AUC 0.9997, acc 0.9914 |
| mnist10 | logistic_regression | acc 0.9682, CE 0.1282 | acc 0.9097, CE 0.3408 |
| mnist10 | gbm_small | acc 1.0000, CE 0.0019 | acc 0.9628, CE 0.1334 |

## Structure (how hard explanation is likely to be)

All counts are for the best genome's phenotype, the nodes on an input → output path:

- **Hidden activity:** a hidden ReLU unit counts as *dead* if its output is 0 on every
  training row, *always-on* if it is > 0 on every row (it is then linear on the data), and
  *switching* otherwise. "1%" uses tolerances of ≤ 1% / ≥ 99% of rows instead.
- **Fan-in:** the number of enabled incoming connections.
- **Depth:** the number of layers after the input layer.

| run | hidden | depth | conns (direct in->out) | inputs used | fan-in hidden mean/max | switching / always-on / dead (strict) | same at 1% | parity | tools |
|---|---|---|---|---|---|---|---|---|---|
| mnist10-e25-s0 | 1 | 2 | 741 (739) | 500/784 | 1.0/1 | 0/1/0 | 0/1/0 | ok | ok |
| mnist10-s0 | 1 | 2 | 568 (566) | 414/784 | 1.0/1 | 0/1/0 | 0/1/0 | ok | ok |
| mnist10-s1 | 0 | 1 | 536 (536) | 390/784 | 0.0/0 | 0/0/0 | 0/0/0 | ok | ok |
| mnist38-g500-s0 | 4 | 2 | 131 (116) | 122/196 | 2.8/5 | 3/0/1 | 3/0/1 | ok | ok |
| mnist38-s0 | 5 | 2 | 133 (117) | 123/196 | 2.2/4 | 4/0/1 | 4/0/1 | ok | ok |
| mnist38-s1 | 1 | 2 | 113 (111) | 112/196 | 1.0/1 | 0/0/1 | 0/0/1 | ok | ok |
| mnist38-s2 | 0 | 1 | 112 (112) | 112/196 | 0.0/0 | 0/0/0 | 0/0/0 | ok | ok |

- **Species:** every run had 1 species, apart from 2 briefly in mnist38-s2.
- **Depth:** all genomes have depth ≤ 2, meaning inputs → (hidden) → output, with no
  hidden → hidden connections.
- **mnist10 hidden units:** the single hidden unit in mnist10-s0 and mnist10-e25-s0 is fed
  by one pixel and is always on, so it is linear on the training rows.
- **Per-unit detail:** active fractions per hidden unit, and the input ids used, are in
  `logs/<name>.analysis.json`.

**What this means for explanation (not attempted):**
- **mnist38:** each model is a weighted pixel template, read directly from about 110–120
  direct weights, plus at most 4 switching hinge units on 1–5 pixels each.
- **mnist10:** each model is a 10-way linear template on about 400–500 pixels. The size is
  in the number of direct weights (about 540–740), not in any interaction.
- **Difficulty:** within the approach, the difficulty is breadth (many inputs per output),
  not depth or non-linearity.

## Database import, parity and tool checks

`import_run.py` writes only new rows:
- **Dataset and split:** `mnist38_14x14` v`n11982` and `mnist10_784` v`n10000`, each with one
  `DatasetSplit`. `task_type` is `classification`; the class names are `["3","8"]` and
  `"0"`–`"9"`. X stores only the rows used, as float64: the train rows, then the test rows.
- **Experiments:** one per run, named `<task>-s<seed>` (extensions are named as below).
  `config_json["training"]` records `mask_phantom`, `output_activation` (identity for
  mnist10), the criterion and the fitness. `hardware_info.run_params` holds the training
  parameters.
- **Population:** one `Population` row, for the last evaluated generation (99, or 499 for
  the 500-generation run), with its `Species` rows and all 250 genomes. Each genome's NEAT
  parent keys are in `Genome.mutation_history` (`parent1_key`, `parent2_key`,
  `is_elite_copy`, `species_key`). The parents themselves are not in the DB, so
  `parent1_id`/`parent2_id` are NULL. The full parent log is in
  `~/.explaneat/mnist_runs/<name>/parents.csv.gz`.
- **Best genome:** in every run it is the final population's champion (same key and weights),
  so it has no separate row.
- **Results:** `Result` rows hold the best genome's in-memory train and test metrics.

**Parity (`analyze.py`):**
- The best genome is loaded the way the UI/MCP load it: `load_genome_and_config` → ExplaNEAT
  phenotype → StructureNetwork.
- On the DB dataset/split, it reproduces the in-memory train and test metrics **exactly**
  (difference 0.0) for every run, including the 10-output identity case.
- NeuralNeat outputs from the DB genome are bit-equal to those from the in-memory genome.
- StructureNetwork and NeuralNeat outputs differ by at most about 4e-15 (summation order).

**Tools:** the following ran without errors on every imported genome, with no annotations:
- MCP `get_phenotype`, `get_model_state`, `classify_nodes`, `get_coverage` and
  `compute_performance`;
- the UI's REST routes `GET /api/genomes/{id}`, `/phenotype` and `/model`.

Their results are in `logs/<name>.analysis.json`. The only breakage was `compute_performance`
on 10 outputs (fixed above). Nothing broke for 196 or 784 inputs.

**DB size** (`pg_database_size(explaneat_dev)`, measured by the importer immediately
before and after each import). Other agents were writing to the same DB, so only the
per-import deltas are attributable to this work:

mnist10-e25-s0: DB +5.2 MB (10323 -> 10328 MB)
mnist10-s0: DB +16.0 MB (10293 -> 10309 MB)
mnist10-s1: DB +5.2 MB (10313 -> 10318 MB)
mnist38-g500-s0: DB +1.4 MB (10320 -> 10321 MB)
mnist38-s0: DB +5.7 MB (10281 -> 10286 MB)
mnist38-s1: DB +1.4 MB (10288 -> 10289 MB)
mnist38-s2: DB +1.4 MB (10292 -> 10293 MB)

The total is about 36 MB. It includes the two datasets: mnist38, 13,966 × 196 float64, and
mnist10, 20,000 × 784 float64, stored compressed by Postgres TOAST. On disk outside the repo:
- `~/.explaneat/mnist_runs`: 19 MB;
- `~/.explaneat/data`: 29 MB, the MNIST cache and the OpenML download.

### Opening the genomes in the UI

```bash
uv run python -m explaneat api        # then open http://127.0.0.1:8000/genome/<genome id>
```

Or go to Experiments → `mnist38-s0` (etc.) → best genome. In the MCP, use `get_best_genome`
with the experiment id, or the genome ids above.

The mnist10 phenotypes have about 800 nodes (784 inputs), so the graph view is crowded.
