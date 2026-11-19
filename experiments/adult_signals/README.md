# Adult signal catalogue (Experiment 2, ten Adult networks)

A light version of `experiments/churn_signals/` for the ten frozen-v4 Adult networks
(`experiments/10seed/runs.csv`, `dataset = adult`). Exploratory, read-only on `explaneat_v4`.
The method and every threshold are in the docstring of `catalogue.py`.

**Outcome direction.** The dataset `adult_ohe_v2` stores class 1 as `<=50k` (class names
`['>50k', '<=50k']`, asserted on load), so the positive class is an income of **$50,000 a year or
less** and every logit, component and description here is about the chance of a *low* income.

## Run

```bash
DATABASE_URL=postgresql://localhost/explaneat_v4 \
EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \
uv run python experiments/adult_signals/catalogue.py            # about 4 min with the caches
#                                                     --refresh  also re-reads the networks (read-only)
```

## Method in brief

- Basis `Phi` (135 columns on the 39,073 training rows): linear terms per variable (a one-hot
  category is one block of offsets; marital status and household role together; occupation and
  work class together, since their missing-value levels are the same people), hinges for age,
  hours, capital gain, capital loss and the census weight (residualised on all linear columns),
  education levels beyond the years-of-schooling line, and four interactions read by the
  explanations' units (residualised on all linear and hinge columns).
- Each network's logit is projected on `[1, Phi]` (least squares, train); the block's component is
  the network's realisation of the signal. The data's own version: ridge logistic fit of the labels
  on `Phi`; relevance = 5-fold CV log-loss gain of the block.
- Used: component SD >= 0.10 logit after clipping to the 1st-99th percentiles (rare-group offsets
  of tens of logits would otherwise dominate; such seeds are `rare_only`).
- Realisation code (non-linear signals): H a single hinge, G a single gate (the unit also reads
  another variable), S*k* split across *k* units, /*n*p in *n* explanation parts. Units that read
  one yes/no input or one one-hot level are linear by the binary one-step rule
  (`explaneat/analysis/plain_language_linearity.py`) and never count as carriers.
- Route (linear signals): `edge` (direct edge to the output through identity nodes only),
  `units` (only through ReLU units), `edge+units`.
- Test numbers (paired bootstrap, 2,000 resamples of the 9,769 test rows, `RandomState(42)`) are
  computed once and never used to choose anything.

## Files

| file | content |
|---|---|
| `signals.csv` | one row per signal: short name, one-line description, kind, variables, data relevance, seeds using it, seeds whose test gain CI is above 0, median SD, notes on realisation variants, `description_check` (the description's direction checked on the data's own fit) |
| `matrix.csv` | seed x signal: SD (raw and clipped), used, `rare_only`, realisation code, route, AUC lost when the component is removed (train, test with CI), carrier units and parts |
| `projection.csv` | per network: R^2 of the projection on train and test, AUC of the network and of its projection |
| `catalogue.py` | the script |
| `cache/` | forward passes per network, the data's fit, `meta.json` (outcome, knots, description checks) |
