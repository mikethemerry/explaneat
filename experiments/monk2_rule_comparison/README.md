# MONK-2: does a glass-box model's explanation state the rule? (exploratory, 22 Nov 2026)

MONK-2's label is "exactly two of the six attributes take their first value" (asserted on every row by
`fit_models.py`). PropNEAT's original MONK-2 network was explained as a count ramp and an at-least-three gate
composed in the root. This study asks whether the usual glass-box comparators state the same rule in their
explanations, on the same split. Not pre-registered.

## Protocol

- **Data** (`extract_data.py`, layer 2): the comparator suite's MONK-2 split (`experiments/benchmarks/common.py`
  `CLASSIFICATION["monk2"]`, `explaneat_v4`), read in a READ ONLY transaction with
  `experiments/10seed/compute_runs.ReadOnlySession` (the loader `experiments/benchmarks/export_data.py` uses), into
  `data/monk2_split.json`: the unscaled one-hot columns, labels, train/test indices and the split's stored
  StandardScaler. `fit_models.py` rebuilds the scaled arrays as `compute_runs.load_split` does and asserts their hash
  equals `experiments/benchmarks/results/data_manifest.json` (the suite's arrays, exactly).
- **Inputs.** one-hot: the 17 scaled one-hot columns; raw: the six attributes as their values, recovered from the
  one-hot columns; count: the number of attributes at their first value (one engineered feature).
- **Models** (`fit_models.py`, `random_state=42` everywhere, EBM `n_jobs=2` as the suite):
  1. `ebm_onehot`: EBM on one-hot with the settings the suite's CV chose for MONK-2, **read** from
     `experiments/benchmarks/results/classification/monk2.json` (whichever version is current; when the widened-grid
     rerun lands there, rerun this script); its test AUC is asserted equal to the suite's;
  2. `ebm_raw`: EBM on the raw attributes, nominal, all 15 pairwise interactions;
  3. `tree_onehot`, `tree_raw`: decision tree, unlimited depth (raw attributes as ordered integers);
  4. `tree_count`, `ebm_count`: tree (unlimited depth) and EBM without interactions on the count alone.
- **Measured.** Test AUC and accuracy (the test rows read once); explanation size (EBM: terms, main vs pairs, the
  fewest terms holding 80% and 95% of the total `term_importances()`; tree: depth, leaves); whether the explanation
  states the rule:
  - count models: the tree's leaves / the EBM's intercept plus count term give class 1 exactly at count 2 →
    "only given the count";
  - EBMs without the count: the model's log-odds on the training rows grouped by the count (mean, min, max per
    count); if they peak at 2 and separate the classes → "only given the count" (the rule is in the sum of terms,
    readable only by regrouping by the count), else "no";
  - trees without the count: no split reads the count → "no";
  - PropNEAT's original network: read from its committed plain-language layer
    (`experiments/10seed/plain_language/monk2-paper.json`): the root's children with nonzero contribution; "yes"
    when they are one linear unit and one gated unit, each reading all six attributes (the count ramp and the
    at-least-three gate). Its test AUC is the suite's `results/propneat.json`.

## Run

```
PGOPTIONS='-c default_transaction_read_only=on' uv run python experiments/monk2_rule_comparison/extract_data.py
OMP_NUM_THREADS=2 /Users/mike/dev/explaneat/.claude/worktrees/conv-neat/.venv/bin/python \
    experiments/monk2_rule_comparison/fit_models.py      # needs `interpret` (InterpretML 0.7.8), ~45 s
uv run python thesis/ch7/exp1/monk2_rule_table.py         # thesis table tab:exp1-monk2-rules
uv run python thesis/ch7/exp1/make_figures.py ebm-monk2-count ebm-monk2-pairs   # thesis figures (options)
```

Both scripts reran byte-identical (22 Nov 2026).

## Outputs

- `data/monk2_split.json`: the split (layer-2 cache).
- `results/monk2_rules.json`: every number above per model, the EBM settings and their source file, the per-count
  log-odds, `ebm_raw`'s per-row training log-odds and its 15 pair terms (value 1 first on both axes), and the
  original network's parts and active units.
- `results/monk2_rules.csv`: one row per model (test AUC, accuracy, size, rule).
