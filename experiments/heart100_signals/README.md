# Heart-100 signal matrix (Experiment 2, ten Heart-100 networks)

The Heart-100 counterpart of `experiments/churn_signals/` and `experiments/adult_signals/`: one row per
(seed, signal) for every signal of a shared data basis, so "how much signal" (the SD of a network's
component, in logit) can be compared across the three datasets. Read-only on `explaneat_v4` (through
the cache of `experiments/linear_betas/betas.py`). Method and thresholds: the docstring of `signals.py`.

## Run

```bash
DATABASE_URL=postgresql://localhost/explaneat_v4 EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 \
OMP_NUM_THREADS=1 uv run python experiments/linear_betas/betas.py     # builds the cache (forward passes)
OMP_NUM_THREADS=1 uv run python experiments/heart100_signals/signals.py
```

## Method in brief

- Basis `Phi` (242 training rows, 128 columns): `lin:<variable>` (a number's z-column; a category's
  level offsets against its most common level), `shape:<number>` (hinges at the training quartiles;
  vessels at 1 and 2; residualised on all linear columns), `int:<a>*<b>` for every pair of variables
  that a switching unit of one of the ten networks reads together (drivers and modifiers in
  `experiments/heart100_audit/features.csv`; products of a number's z and median hinge, and of 0/1
  level indicators with at least 10 rows; residualised on all linear and bend columns).
- Each network's logit is projected on `[1, Phi]` (least squares, train). R^2 train 0.980 to 0.998,
  test 0.865 to 0.983 (`projection.csv`; 61 test rows).
- `sd_train`: SD over the training rows of a network's component of the signal; `used`: `sd_train >= 0.10`
  logit, as churn and adult. `sd_robust`: the same after clipping to the 1st-99th percentiles.
- The data's own version: ridge logistic fit of the labels on `Phi` (C by 5-fold CV log-loss);
  `data_component_sd` and `data_cv_logloss_gain` (CV log-loss lost without the block; > 0: relevant).

## Files

| file | content |
|---|---|
| `matrix.csv` | seed x signal: `sd_train`, `sd_robust`, `used`, `corr_with_data`, number of basis columns |
| `signals.csv` | one row per signal: kind, variables, columns, data component SD, data CV log-loss gain, seeds using it, median SD |
| `projection.csv` | per network: R^2 of the projection (train, test), AUC of the network and of its projection |
| `signals.py` | the script |

The linear terms' betas (coefficients of the linear block, per real unit) are in
`experiments/linear_betas/betas_heart100.csv`.
