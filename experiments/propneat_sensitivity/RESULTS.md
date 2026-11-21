# PropNEAT hyperparameter sensitivity check: results (21 Nov 2026)

Design: `PLAN.md` (committed before any run, with one amendment after the CV runs). The numbers are in
the generated files and nowhere else: `cv.csv`, `selection.json`, `final.csv`, `summary.json`, and the
tables in `summary.md` (written by `summarise.py`). The thesis tables `tab:exp1-sensitivity` and
`tab:exp1-sensitivity-cv` are built from these files by `thesis/ch7/exp1/sensitivity_table.py`.
Rerun the scripts before quoting anything.

## Commands (in order)

```
uv run python experiments/propneat_sensitivity/prepare.py            # configs/*.ini, folds.json (DB read-only)
uv run python experiments/propneat_sensitivity/launch.py dryrun      # runtime estimate (dryrun/)
uv run python experiments/propneat_sensitivity/launch.py check cv repeat   # 4 + 240 + 5 runs, 6 at a time
uv run python experiments/propneat_sensitivity/choose.py             # cv.csv, selection.json, repeat check
uv run python experiments/propneat_sensitivity/launch.py final       # selected config, all training rows, seeds 0-4
uv run python experiments/propneat_sensitivity/summarise.py          # final.csv, summary.json, summary.md
uv run python thesis/ch7/exp1/sensitivity_table.py                   # thesis/ch7/exp1/tables/exp1-sensitivity.tex
cd thesis/ch7/exp1 && latexmk -pdf preview_sensitivity.tex && latexmk -c preview_sensitivity.tex
```

`launch.py` sets `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1` and `PGOPTIONS=-c default_transaction_read_only=on`
and skips any job whose result file exists. One run is one process (`run_one.py`); each writes its result
JSON and a separate `.meta.json` (wall clock, commit).

## Checks

- **Harness = fixed runs.** The fixed configuration on all training rows with seed 0 reproduces each
  dataset's stored seed-0 run exactly: generation and key of the selected genome, train and test AUC (to
  1e-9), hidden nodes and enabled connections (`summary.json`, `harness_check_seed0`). So the in-memory
  loop is the fixed runs' training path, and the tuned and fixed numbers are comparable.
- **Determinism.** The repeat CV (`repeat/`, saddle, `e20_adult_wd0.01`, all five folds) is byte-identical
  to `runs/` (`selection.json`, `_repeat_check`). It was run with the amended `run_one.py`
  (`skip_invalid_genomes=True`), the `runs/` files with the original, so it also shows the flag changes
  nothing in a run without an invalid genome. `choose.py`, `summarise.py` and `sensitivity_table.py` each
  reran byte-identical.
- **Test rows** are loaded only in the `full` runs (final and harness check); no choice reads them.

## What the results say (read with `summary.md`)

- **The CV surface is flat on U and saddle.** There the spread of the twelve configurations' mean fold
  AUC is smaller than the fold-to-fold SD of a single configuration. On U the two best means are equal at
  three decimals and the tie rule picked the simpler one. **Churn and U + saddle are less flat**: on both,
  20 backprop epochs per generation lift the mean fold AUC (on Churn mainly with the Heart family; on
  U + saddle every 20-epoch configuration is near the top and the lowest means are 5-epoch ones).
- **No dataset's CV choice is its fixed configuration**, but on U and saddle the fixed configuration's CV
  mean is within the same narrow band as the selected one.
- **Tuned vs fixed test AUC.** Churn: tuning raises the median test AUC, with smaller networks. U: about the
  same. Saddle: the selected configuration tests slightly lower than the fixed one (the CV gain did not carry
  over). U + saddle: tuning raises the median and narrows the range.
- **Conclusions against the comparators** (`benchmarks.csv` as committed on this branch; it may be updated
  by the widened-grid comparator rerun, and the table reads whatever the file holds):
  - Churn: the gap to gradient boosting narrows but remains; no seed, fixed or tuned, reaches gradient
    boosting. The Experiment 2 conclusion (PropNEAT below gradient boosting on Churn) stands.
  - Synthetic sets: every tuned seed beats gradient boosting on all three sets (on U + saddle tuning turns
    "most seeds" into all five). The conclusion "PropNEAT matches or beats gradient boosting on the synthetic
    shapes" stands and is, if anything, firmer. Against the best comparator per set the picture is unchanged
    in kind: about level on U and saddle (within the seed range), below the MLP on U + saddle.
- **Answer:** tuning PropNEAT by cross-validation over these three factors does not change any conclusion of
  the fixed-configuration runs. It moves Churn closer to, not past, gradient boosting.

## Runtime

Dry-run estimate (PLAN.md): about 6-7 h. Actual: 269 runs (4 check, 240 CV, 5 repeat, 20 final) from
15:12 to 21:36 on 21 Nov 2026 at 6 processes, while other jobs kept the machine's load average at about
50-330 on 14 cores (each run's wall clock is in its `.meta.json`). Synthetic runs with the Adult/Churn
family grow larger networks and took several times the dry-run extrapolation.

## Deviations from the plan

- One CV run (`both / e5_heart_wd0.01 / fold 1`) stopped on a genome with no output path; rerun with
  `skip_invalid_genomes=True` (PLAN.md, amendment). The first repeat CV (old code) was deleted unchecked and
  rerun with the new code.
- `select.py` was renamed `choose.py` (a script named `select.py` shadows Python's `select` module, which
  `subprocess` imports).

## Files

`common.py` (settings, read-only loader), `prepare.py`, `run_one.py`, `launch.py`, `choose.py`,
`summarise.py`; `configs/<dataset>_<family>.ini`; `folds.json`; `dryrun/`; `runs/<dataset>/<config>_fold<k>_s0.json`
(CV), `runs/<dataset>/<config>_full_s<seed>.json` (final and harness check); `repeat/`; `cv.csv`;
`selection.json`; `final.csv`; `summary.json`; `summary.md`. Logs (`logs/`) are not committed.
