# ExplaNEAT — reproducibility bundle (paper explanations)

Static export of the three explained models. **No database is required**: the
genomes and explanations are static JSON, and each model directory ships the
exact preprocessed split so `load.py` runs offline.

## Datasets

The three datasets are public UCI benchmarks — Adult / Census Income, Cleveland
Heart Disease, and MONK's Problem 2 — reconstructed with the given split
(`random_state = 42`, `test_size = 0.2`; see each `split.json`). MONK-2 is the
full 432-configuration input-space enumeration (deduplicated from the raw PMLB
file; see `scripts/ingest_monk2.py`). The exact z-normalised arrays used for
evaluation are included as `data.npz` so no download is needed to verify.

## Per-model files (`adult/`, `heart/`, `monk2/`)

- `genome.json` — full evolved genome: nodes (id, type, bias, activation,
  aggregation, response) and connections (from, to, weight, enabled). Same
  fields the app serialises to load a genome.
- `phenotype.json` — the active pruned phenotype (reachable nodes + enabled
  connections only).
- `operations.json` — the complete ordered event stream (node splits, identity
  inserts, renames, annotations) with parameters; replaying it reconstructs the
  annotated model state.
- `annotations.json` — resolved annotation hierarchy (name, entry/exit/subgraph
  nodes, hypothesis, parent/children).
- `evidence.json` — evidence records per annotation (closed-form formulas, SHAP,
  ablation studies) with payloads and narratives.
- `split.json` — dataset name, test_size, random_state, train/test sizes,
  stratify flag, and the StandardScaler params (mean/scale).
- `performance.json` — train/test AUC and accuracy (Table 1 test AUC recorded).
- `neat_config.ini` — complete NEAT config (so a `neat.Config` can be built
  offline).
- `data.npz` — z-normalised `X_train/y_train/X_test/y_test` for `load.py`.

## Reproduce the numbers (no DB)

```bash
uv run python reproducibility/load.py adult   # or heart / monk2
```

`load.py` rebuilds the network with the repo's phenotype/NeuralNeat builder from
`genome.json` + `neat_config.ini`, runs a forward pass on `data.npz`, and prints
train/test AUC + accuracy — which should match `performance.json`.

- **Coverage**: replay `operations.json` on the phenotype (repo
  `ModelStateEngine`) to obtain the annotated model state, then count covered
  nodes against the annotation subgraphs (repo `CoverageComputer`).
- **AUC**: as above via `load.py`.
- **Ablation** (e.g. Adult native-country edges): load `genome.json`, disable
  the documented connections on an in-memory copy, and re-run the forward pass
  on the test arrays — the delta reproduces the `evidence.json` ablation record.

## `scripts/`

The analysis scripts used to produce the paper's numbers (baselines, ablation
bootstrap CIs, densified twin, path-length checks, MONK-2 ingest). Author-
identifying strings have been redacted for anonymous review.
