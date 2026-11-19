# One comparator suite for every Chapter 7 dataset (exploratory, 19 Nov 2026)

The same comparators, tuned the same way, for every dataset of Experiment 1 (Heart and MONK-2 paper
genomes, WHAS500, Connect-4 cells and lines, synthetic U / saddle / U + saddle, XOR) and Experiment 2
(Adult, Churn, Heart-100), on the exact split each PropNEAT run used. Not pre-registered.

## Protocol

- **Data.** Each set's database split, read in READ ONLY transactions (`export_data.py`; Heart, MONK-2,
  Adult and Churn from `explaneat_v4`, the rest from `explaneat_dev`), inputs scaled with the split's
  stored StandardScaler, exactly as `experiments/10seed/compute_runs.load_split` (manifest with row
  counts and a hash: `results/data_manifest.json`). Survival: the Cox case's split files and
  standardisation (`experiments/survival/cox_case`). Heart-100 uses the Heart split.
- **Tuning.** 5-fold stratified CV on the training rows only (shuffle, seed 42), scoring ROC AUC
  (survival: stratified on the event indicator, scoring Harrell's C), refit on all training rows. The
  test rows are read once per model. Nothing is selected on test.
- **Models.** Binary classification (`classification.py`): L2 and L1 logistic regression, decision tree,
  random forest (500 trees), HistGradientBoosting, EBM (InterpretML 0.7.8), one-hidden-layer MLP. The grids
  are those of `experiments/references/baselines.py` (L2 LR, GB) and
  `experiments/references/heart_benchmark/bench.py` (the other five), the same for every dataset.
  Survival (`survival.py`): unpenalised Cox PH (lifelines), L2 Cox, elastic-net Cox, random survival
  forest and gradient-boosted survival (scikit-survival). The comparators do not see the networks'
  validation rows.
- **Metrics.** Test AUC with a 95% percentile interval over 10,000 bootstrap resamples of the test rows
  (`RandomState(42)`, the 10-seed study's index matrix, the same for every model of a dataset), and test
  accuracy. Survival: Harrell's C (same bootstrap), Uno's C, mean time-dependent AUC at 1, 3, 5 years
  and the integrated Brier score from `explaneat/survival/metrics.py`.
- **PropNEAT rows** (`propneat.py`). Experiment 1: each explained network recomputed from its genome
  (read-only) on the cached arrays; train and test AUC equal the studies' run records to 1e-9 (asserted).
  WHAS500 / GBSG2: scored from the genome pickles in `survival.py`, equal to the run records. XOR: the 30
  reduced networks from `experiments/xor/xor_results.json`. Experiment 2: from the runs CSVs (median and
  range of ten seeds; the best seed by training AUC with its interval).

## Run

```
uv run python experiments/benchmarks/export_data.py          # cache/*.npz (not committed)
OMP_NUM_THREADS=2 /Users/mike/dev/explaneat/.claude/worktrees/conv-neat/.venv/bin/python \
    experiments/benchmarks/classification.py [dataset ...]  # needs `interpret` (EBM); ~2.5 h, Adult EBM ~1.5 h
uv run python experiments/benchmarks/survival.py
uv run python experiments/benchmarks/propneat.py
uv run python experiments/benchmarks/make_csv.py             # benchmarks.csv, results/reuse.json
```

## Outputs

- `benchmarks.csv`: one row per dataset and model (PropNEAT rows included): experiment, dataset, model,
  CV-chosen params, CV score, test metric, value, 95% interval, accuracy, train value; survival rows add
  Uno's C, td-AUC and IBS.
- `results/classification/<dataset>.json`, `results/survival.json`, `results/propneat.json`.
- `results/reuse.json`: every earlier comparator number on these splits, set beside this run's.

## Reused versus recomputed

Everything was recomputed under one protocol; earlier numbers were checked (`results/reuse.json`):

- **Identical (same protocol):** `references/heart_benchmark/bench.json` (Heart: L1 LR, tree, RF, EBM,
  MLP; test and CV AUC), `references/results/baselines.json` gradient boosting (Heart, MONK-2, Adult,
  Churn; test and CV AUC), `survival/cox_case/references.csv` unpenalised Cox PH and the selected
  networks' C (WHAS500, GBSG2).
- **Same test AUC, different CV protocol:** `baselines.json` L2 LR chose C with unstratified KFold; the
  stratified CV picks the same C on all four sets.
- **Different protocol, superseded here:** untuned LR (C = 1) and default gradient boosting in
  `churn/baselines.json`, `ushape_saddle/references.json`, `c4_capacity/baselines.json`,
  `c4_line_features/baselines.json` (tuned GB on Connect-4 cells: 0.962 vs 0.944 default); the Cox case's
  ridge (alpha tuned on validation rows) and forest (fixed settings).

## Notes

- WHAS500 L2 Cox chose alpha = 1000 (strong shrinkage; its IBS is worse); CV C is flat (0.759-0.761)
  across the survival models, so the ranking on 100 test patients is noisy.
- MONK-2 is "exactly two of six attributes": linear models and EBM (pairwise interactions only) fail
  (test AUC 0.42-0.52); the paper network and the MLP are perfect.
- Connect-4 L2 LR chose C = 100 (grid edge); the CV AUC is flat in C there.
