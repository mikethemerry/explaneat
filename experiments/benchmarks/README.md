# One comparator suite for every Chapter 7 dataset (exploratory, 19 Nov 2026; grids widened 21-22 Nov)

The same comparators, tuned the same way, for every dataset of Experiment 1 (the original Heart, MONK-2
and Adult networks, the Churn case `churn-case-s10`, WHAS500 with GBSG2 as a side note, Connect-4 cells
and lines, synthetic U / saddle / U + saddle, XOR) and Experiment 2 (Adult, Churn, Heart-100), on the
exact split each PropNEAT run used. Adult, Churn and Heart share one split between the two experiments
(`experiment` "1,2" in `benchmarks.csv`). Not pre-registered.

## Protocol

- **Data.** Each set's database split, read in READ ONLY transactions (`export_data.py`; Heart, MONK-2,
  Adult and Churn from `explaneat_v4`, the rest from `explaneat_dev`), inputs scaled with the split's
  stored StandardScaler, exactly as `experiments/10seed/compute_runs.load_split` (manifest with row
  counts and a hash: `results/data_manifest.json`). Survival: the Cox case's split files and
  standardisation (`experiments/survival/cox_case`). Heart-100 uses the Heart split; the Churn case uses
  the churn100 split.
- **Tuning.** 5-fold stratified CV on the training rows only (shuffle, seed 42), scoring ROC AUC
  (survival: stratified on the event indicator, scoring Harrell's C), refit on all training rows. The
  test rows are read once per model. Nothing is selected on test.
- **Models.** Binary classification (`classification.py`): L2 and L1 logistic regression, decision tree,
  random forest (500 trees), HistGradientBoosting, EBM (InterpretML 0.7.8), MLP (one or two hidden
  layers). Survival (`survival.py`): unpenalised Cox PH (lifelines), L2 Cox, elastic-net Cox, random
  survival forest and gradient-boosted survival (scikit-survival). The comparators do not see the
  networks' validation rows.
- **Grids** (design choices, the same for every dataset; `STAGES` in `classification.py` and
  `survival.py`; table below). Where the full product grid is too large, the search runs in stages on the
  same folds (`search.py`): a later stage tunes its parameters with the earlier stages' choices fixed (a
  one-pass coordinate search); a parameter tuned in two stages takes its value from the later one. The
  boosting iteration count is scored from staged predictions of one fit per fold: with no early stopping
  and no subsampling, the model after k iterations is the model fitted with k iterations
  (`check_staged.py`: identical CV scores, `results/check_staged.json`).
- **Grid check** (`check_grids.py`, run by `make_csv.py`; `results/grid_check.md`): every chosen value's
  place in its grid. A value at an end of its grid counts as an edge unless it is the end of the
  parameter's range (`common.DOMAIN_ENDS`: max_depth None or 1, min_samples_leaf 1, l2 0, interactions
  0, max_features 1 or all, one iteration, two leaves, MLP alpha 0, l1_ratio 1; for the EBM also all
  feature pairs and more bins than any feature has values, `classification.range_ends`), or an interior
  value has exactly the same CV score (a flat optimum such as CV AUC 1.0 on MONK-2, where the tie-break
  put the choice at the edge). The check exits non-zero while any edge remains.
- **Metrics.** Test AUC with a 95% percentile interval over 10,000 bootstrap resamples of the test rows
  (`RandomState(42)`, the 10-seed study's index matrix, the same for every model of a dataset), and test
  accuracy. Survival: Harrell's C (same bootstrap), Uno's C, mean time-dependent AUC at 1, 3, 5 years
  and the integrated Brier score from `explaneat/survival/metrics.py`.
- **PropNEAT rows** (`propneat.py`). Experiment 1: each explained network (including the original Adult
  network and the Churn case) recomputed from its genome (read-only) on the cached arrays; train and test
  AUC equal the studies' run records to 1e-9 (asserted). WHAS500 / GBSG2: scored from the genome pickles
  in `survival.py`, equal to the run records. XOR: the 30 reduced networks from
  `experiments/xor/xor_results.json`. Experiment 2: from the runs CSVs (median and range of ten seeds;
  the best seed by training AUC with its interval).

## Grids

| Model | Stage | Grid |
|---|---|---|
| L2 / L1 logistic regression | 1 | C 1e-4 … 1e4, half decades (17 values); liblinear seeded |
| Decision tree | 1 | max_depth 1-8, 10, 12, 15, 20, None × min_samples_leaf 1, 2, 5, 10, 20, 50, 100, 200 |
| Random forest | 1 | max_depth 3, 5, 8, 12, 20, None × min_samples_leaf 1, 2, 5, 10, 20 (max_features sqrt) |
| | 2 | max_features 1, sqrt, 0.5, None |
| Gradient boosting | 1 | learning_rate 0.003, 0.01, 0.03, 0.1, 0.3, 1 × max_leaf_nodes 2, 3, 7, 15, 31, 63, 127 × max_iter 1 … 10,000 (staged; min_samples_leaf 20, l2 0) |
| | 2 | min_samples_leaf 1, 5, 20, 50, 100 × l2_regularization 0, 0.1, 1, 10, 100 |
| | 3 | max_leaf_nodes 2 … 511 |
| | 4 | max_iter 1 … 30,000 (staged) |
| EBM | 1 | interactions 0, 5, 10, 20, 50, 100, 200 (max_bins 256) |
| | 2 | max_bins 16, 32, 64, 256, 1024, 4096, 16384, 32768 |
| MLP | 1 | alpha 0, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100 (one layer of 32) |
| | 2 | one layer of 2 … 1024 units; two layers (32, 16) … (1024, 512), and (2048, 1024) on Adult and Connect-4 cells only (below) |
| | 3 | alpha again |
| Cox L2 | 1 | alpha 1e-3 … 1e6, half decades |
| Cox elastic net | 1 | l1_ratio 1e-4 … 1 (12 values) × alpha 1e-4 … 100, half decades |
| Random survival forest | 1 | max_depth 1, 2, 3, 5, 8, 12, None × min_samples_leaf 1, 2, 5, 10, 20, 50, 100 |
| | 2 | max_features 1, sqrt, 0.5, None |
| Gradient-boosted survival | 1 | learning_rate 0.001 … 1 (7) × max_depth 1-5 × n_estimators 1 … 30,000 (staged) |

The grids were widened from those of `experiments/references` (that run is kept in
`results/narrow_grids/`) in rounds on 21-22 Nov 2026: after each round every choice at a grid edge was
widened in that direction and the affected models refitted. The widening followed the CV choices only
(the test AUCs were printed, but no choice and no grid was set on them). The rounds were stopped on
22 Nov with these exceptions, listed in `results/grid_check.md`:

- **Connect-4 cells, EBM interactions = 200**, the top of the grid (the set has 861 feature pairs). CV
  AUC still rises with the number of pairs (`results/classification/c4_cells.json`, EBM stage 1); more pairs were not fitted
  (cost: one fit at 200 pairs is about half an hour on 54,045 rows).
- **MLP architectures**: the (2048, 1024) option was added in the last round and fitted only on Adult and
  Connect-4 cells, where (1024, 512) had been the widest choice; neither chose it. The other seven sets'
  MLP grids stop at (1024, 512), which none of them chose.

So the statement "the grids were fixed in advance and no selected value lies at a grid boundary" does
not hold as it stands: the grids were set before the final fits but widened after earlier CV choices,
the MLP grid is not quite the same on every set, and one choice (Connect-4 cells EBM) is at its edge.

## Run

```
uv run python experiments/benchmarks/export_data.py          # cache/*.npz (not committed)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BENCH_JOBS=4
/Users/mike/dev/explaneat/.claude/worktrees/conv-neat/.venv/bin/python \
    experiments/benchmarks/classification.py [dataset ...]  # needs `interpret` (EBM)
uv run --extra survival python experiments/benchmarks/survival.py
uv run python experiments/benchmarks/propneat.py             # read-only DB
uv run python experiments/benchmarks/make_csv.py             # benchmarks.csv, results/reuse.json, results/grid_check.md
OMP_NUM_THREADS=2 <venv with interpret>/python experiments/benchmarks/heart_splits.py   # heart_splits.json (reads benchmarks.csv)
uv run python experiments/benchmarks/compare_grids.py        # results/before_after.{csv,md}
<venv with interpret>/python experiments/benchmarks/check_staged.py      # results/check_staged.json
```

Runtime from scratch: several hours each for Adult and Connect-4 cells (gradient boosting and the EBM),
minutes for the small sets. Use one BLAS / OpenMP thread per worker: with more, the boosting and MLP
fits oversubscribe the cores and run many times slower (the results do not depend on it).
`classification.py` skips a model whose stored search equals the current one; when the grids change it
refits only the stages that changed and, within a stage, only the new grid points
(`search.search(previous=...)`; every fit is seeded, so a stored CV score is what a refit gives). The
committed results were built that way over the widening rounds.

**Determinism.** Two fresh runs of the final configuration on Heart, each from an empty results directory
(`BENCH_RESULTS=<scratch dir>`), were identical in every field but the timings, and gave the same choices
and test AUCs as the committed file (`check_determinism.py` -> `results/determinism_heart.json`; the
committed file differs only in bookkeeping: `range_ends`, added after those models were fitted, and the
MLP search, which in the fresh runs offers (2048, 1024)). `heart_splits.py` reran byte-identical.

## Outputs

- `benchmarks.csv`: one row per dataset and model (PropNEAT rows included): experiment, dataset, model,
  CV-chosen params, CV score, test metric, value, 95% interval, accuracy, train value; survival rows
  (`test_metric` harrell_c) add Uno's C, td-AUC and IBS. The survival results are also in their own file,
  `results/survival.json`; the classification results in `results/classification/`.
- `results/classification/<dataset>.json` (every stage's grid, choice and CV table), `results/survival.json`,
  `results/propneat.json`.
- `results/grid_check.md`: every CV choice and its place in its grid (generated).
- `results/before_after.md` / `.csv`: every comparator before and after the widening, the best comparator
  per dataset, the PropNEAT-versus-comparator relations that flip, gap closed, the Heart split study.
- `results/reuse.json`: every earlier comparator number on these splits, set beside this run's.
- `results/narrow_grids/`: the 19 Nov run on the narrower grids (classification, survival, its
  `benchmarks.csv`, `heart_splits.json`, `reuse.json`), kept for comparison.

## Reused versus recomputed

Everything was recomputed under one protocol; earlier numbers are set beside this run's in
`results/reuse.json`. All earlier tuned results (`references/heart_benchmark/bench.json`,
`references/results/baselines.json`, `results/narrow_grids/`) used narrower grids; where the choice did
not change, the numbers agree. Untuned LR (C = 1) and default gradient boosting in `churn/baselines.json`,
`ushape_saddle/references.json`, `c4_capacity/baselines.json`, `c4_line_features/baselines.json`, and the
Cox case's ridge (alpha tuned on validation rows) and forest (fixed settings), are a different protocol.
The unpenalised Cox PH and the selected networks' C agree with `survival/cox_case/references.csv`.

## Notes

- Widening changed the comparators most where the data are large or strongly non-additive: Connect-4
  cells (tree, EBM, MLP and boosting all gain), MONK-2 (random forest, boosting and the EBM now close to
  or at a perfect test AUC) and the synthetic U (boosting with stumps). See `results/before_after.md`.
- On Heart's fixed split boosting and the EBM lose test AUC with the wider grids (CV chose other
  settings); over the 100 random splits (`heart_splits.json`) the comparators average about the same.
- WHAS500: CV C is flat across the survival models, so the ranking on 100 test patients is noisy.
