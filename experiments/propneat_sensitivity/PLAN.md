# PropNEAT hyperparameter sensitivity check (plan, fixed before any run, 21 Nov 2026)

A sidebar to Experiment 1, decided on 21 Nov 2026. **Question:** would tuning PropNEAT's hyperparameters by
cross-validation change the conclusions of the fixed-configuration runs?

**Datasets:** Churn and the three synthetic sets (U, saddle, U + saddle). Adult and Connect-4 are
excluded on compute grounds; the thesis says so explicitly.

## Fixed configurations (the baselines)

Read from the run records and configs (`experiments/thesis_run_config/RESULTS.md`, finding 1):

| dataset | fixed runs | config | family | epochs | weight decay |
|---|---|---|---|---|---|
| Churn | churn100 (`experiments/churn/runs_churn100.csv`), `experiments/10seed/run_seed.py churn` | `experiments/churn/configs/churn.ini` | Adult/Churn | 5 | 0 |
| U, saddle, U + saddle | `experiments/ushape_saddle/runs.csv`, `experiments/ushape_saddle/run_seed.py` | `experiments/ushape_saddle/configs/syn10.ini` | Heart | 5 | 0 |

Both: population 250, 100 generations, AUC fitness on the training rows, ReLU hidden / sigmoid output,
`full_direct` start, weight init SD 1 (weight and bias mutation off), unmasked (`mask_phantom=False`),
full-batch Adadelta lr 1.5, best genome = max training fitness over all generations (ties: earliest
generation, then lowest key).

## Factors (full factorial, 12 configurations)

1. Backprop epochs per generation: {5, 20}.
2. Structural mutation family: **Heart** (node add/delete 0.4/0.2, conn add/delete 0.5/0.3,
   compatibility threshold 3, disjoint coefficient 1) or **Adult/Churn** (0.7/0.1, 0.68/0.15, 5, 2).
3. Weight penalty (`BackpropPopulation(weight_decay=...)`, Adadelta's L2 term on every trained
   parameter): {0, 1e-3, 1e-2}.

Everything else is the dataset's fixed config. `prepare.py` writes `configs/<dataset>_<family>.ini`: the
fixed-run config with only the six family keys replaced (the fixed family's file is the fixed config,
byte for byte). Configuration ids: `e<epochs>_<family>_wd<decay>`; the fixed one is `e5_adult_wd0`
(Churn) and `e5_heart_wd0` (synthetic).

## Selection (training rows only)

- Folds: `StratifiedKFold(5, shuffle=True, random_state=42)` over the split's training rows in stored
  order (the comparator suite's splitter), written to `folds.json`.
- For each dataset, configuration and fold: one evolutionary run, seed 0, trained on the other four
  folds (the scaler fit on those rows), best genome by training fitness as above, scored by ROC AUC on
  the held-out fold. 4 x 12 x 5 = 240 runs. Test rows are never loaded into a CV run.
- Configuration score = mean of the five fold AUCs (SD reported).
- Choice per dataset: the highest score. **Tie rule:** scores equal when rounded to 3 decimals are
  tied; among tied configurations the simplest wins: fewer epochs, then lower weight decay, then the
  Heart family.
- `choose.py` writes `cv.csv` (12 x 4 rows: per-fold AUC, mean, SD, network size) and `selection.json`.

## Final runs and comparison

- The chosen configuration, all training rows (the split's stored scaler, asserted), seeds 0-4, test
  AUC read once per seed (`final.csv`, with network size: hidden nodes, enabled connections, inputs).
- Compared with: the fixed runs' test AUC, seeds 0-4, from their run records (not rerun); and the
  comparator suite (`experiments/benchmarks/benchmarks.csv`, which may be updated by the widened-grid
  rerun on branch `thesis/wide-grids`; the generator reads whatever the file holds).
- Harness check (`launch.py check`): the fixed configuration on all training rows, seed 0, per
  dataset, must reproduce the fixed run record's selected generation, key and test AUC (the in-memory
  loop equals the stored run). If a dataset's choice is its fixed configuration, its final runs are
  reruns of the fixed runs and are checked against them seed by seed.

## Determinism

`PYTHONHASHSEED=0`, one torch thread per process (`OMP_NUM_THREADS=1`), deterministic torch algorithms,
`random` / numpy / torch seeded before the population is built. In memory only
(`BackpropPopulation`; no database writes); data read in READ ONLY transactions
(`PGOPTIONS=-c default_transaction_read_only=on` and `SET TRANSACTION READ ONLY`). Each run writes its
result JSON and, separately, a `.meta.json` (wall clock, commit), so results can be compared byte for
byte. Repeat check: the CV of `saddle / e20_adult_wd0.01` (all non-default factor levels) is run a
second time into `repeat/` and must be byte-identical to `runs/`.

## Compute estimate (3-generation dry run, `launch.py dryrun`, 6 processes at once)

`dryrun/*/*.meta.json`: 3 generations took 6.5 s (Churn e5 Adult), 9.4 s (Churn e20 Heart), 10.1 s
(Churn e20 Adult wd 1e-2), 3.8 s (U e5 Heart), 5.8 s (U e20 Heart), 6.3 s (U e20 Adult), about 2 s of
each being start-up. Networks grow over a run, so the per-generation cost rises: the fixed runs' own
records (100 generations, 5 epochs) took 0.18-0.41 h for Churn (4,000 rows, run 4 at a time) and
about 0.03 h for the synthetic sets, i.e. about 4-5x (Churn) and 1.7x (synthetic) the dry-run
extrapolation. Estimate per run: Churn CV ~15 min at 5 epochs and ~25-30 min at 20; synthetic ~1.7 min
and ~3 min. CV total ~28 CPU-hours, at 6 processes ~5 h wall; final + check + repeat ~1 h. **About
6-7 h in all**, longer if the machine is shared (the comparator rerun is also running).

## Outputs

`PLAN.md`, `common.py`, `prepare.py`, `run_one.py`, `launch.py`, `choose.py`, `summarise.py`,
`configs/`, `folds.json`, `runs/<dataset>/*.json`, `repeat/`, `cv.csv`, `selection.json`, `final.csv`,
`summary.json`, `RESULTS.md` (commands). Thesis table: `thesis/ch7/exp1/sensitivity_table.py` ->
`tab:exp1-sensitivity` (not added to the Overleaf upload yet).

## Amendment (21 Nov 2026, after the CV runs, before any choice or final run)

One CV run (`both / e5_heart_wd0.01 / fold 1`) stopped with `GenomeNotValidError`: a genome with no
path to the output reached backprop, which raises by default (the fixed runs' code path would have
stopped the same way; none of them met such a genome). `run_one.py` now passes
`skip_invalid_genomes=True`: such a genome stays untrained and `auc_fitness` scores it 0.5. The flag
only replaces the raise, so every run that met no invalid genome is unchanged. That run is redone with
the flag, and the repeat CV (`repeat/`) is rerun with the new code: it must equal `runs/` (run with
the old code) byte for byte, which checks determinism and that the flag changes nothing else. (The
first repeat CV, with the old code, was deleted unchecked to make room for this rerun.)
