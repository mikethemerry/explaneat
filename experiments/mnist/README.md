# MNIST PropNEAT runs: candidates for "not explained within our approach"

**Exploratory, not pre-registered.** Nothing here was planned in advance or counts towards
any pre-registered threshold. The aim is to train PropNEAT models on MNIST that are
candidates for a model we cannot explain within our approach and budget, so the
explanation tools can be tried on them later. **No explanation was attempted here.** This
directory only reports training results and structure statistics that indicate how hard
explanation is likely to be.

## Summary

@@SUMMARY@@

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
  - NeuralNeat ↔ StructureNetwork parity on 10 random multi-output genomes;
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

@@TIMINGS@@

## Results

Train and test metrics are for the best genome (best train fitness), from the in-memory
NeuralNeat. They are recomputed from the DB through StructureNetwork in the parity check
below.

@@RESULTS@@

## Structure (how hard explanation is likely to be)

All counts are for the best genome's phenotype, the nodes on an input → output path:

- **Hidden activity:** a hidden ReLU unit counts as *dead* if its output is 0 on every
  training row, *always-on* if it is > 0 on every row (it is then linear on the data), and
  *switching* otherwise. "1%" uses tolerances of ≤ 1% / ≥ 99% of rows instead.
- **Fan-in:** the number of enabled incoming connections.
- **Depth:** the number of layers after the input layer.

@@STRUCTURE@@

@@INTERPRETATION@@

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

@@DB@@

### Opening the genomes in the UI

```bash
uv run python -m explaneat api        # then open http://127.0.0.1:8000/genome/<genome id>
```

Or go to Experiments → `mnist38-s0` (etc.) → best genome. In the MCP, use `get_best_genome`
with the experiment id, or the genome ids above.

The mnist10 phenotypes have about 800 nodes (784 inputs), so the graph view is crowded.
