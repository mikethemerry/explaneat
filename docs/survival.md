# Survival analysis in ExplaNEAT (Cox proportional hazards)

User guide for `explaneat.survival`. Background, literature and the verified
reference implementation: `docs/survival-in-explaneat.md` (the 2026-08-20
brief). Validation results: `experiments/survival/RESULTS.md`.

## 1. The model

An evolved network outputs one scalar, the **log-risk** `g(x)`, through an
**identity** output activation. It is trained on the Cox partial likelihood,
and a Breslow baseline hazard is then fitted on the training rows:

$$h(t \mid x) = h_0(t)\, e^{g(x)}, \qquad S(t \mid x) = S_0(t)^{\exp g(x)}, \qquad S_0(t) = e^{-H_0(t)}$$

$$H_0(t) = \sum_{t_i \le t,\ \delta_i = 1} \frac{d_i}{\sum_{j \in R(t_i)} e^{g(x_j)}}$$

This is a Cox model with the linear predictor $\beta^\top x$ replaced by the
evolved network. Nothing else changes, so the comparison against Cox isolates
one question: does a non-linear, evolved index help?

The model inherits proportional hazards. `g` has no time input, so the hazard
ratio between two patients, $\exp(g(x_1) - g(x_2))$, is constant over
follow-up, and predicted survival curves never cross. `g` is identified only
up to an additive constant, which the baseline absorbs. Only differences in
`g` mean anything, and a model's `g` must always be paired with its own
baseline.

The objective, with Efron's correction for tied event times (the default, and
what lifelines and R `survival` use), is

$$\mathcal{L}(g) = -\frac{1}{D}\sum_{k}\Big[\sum_{i \in D_k} g(x_i) - \sum_{l=0}^{d_k-1} \log\Big(\sum_{j \in R(\tau_k)} e^{g(x_j)} - \tfrac{l}{d_k}\sum_{i \in D_k} e^{g(x_i)}\Big)\Big]$$

where $D$ is the number of events. The loss couples every event to its whole
risk set, so it is not a mean of per-row terms. ExplaNEAT trains full-batch,
which is the regime where it is exact. Censored rows contribute no outer
term, but they sit in the denominators of every earlier event.

## 2. Training

### 2.1 What is configured where

| Setting | Where it lives | Survival value |
|---|---|---|
| Output activation | `config.output_activation` on the NEAT config (`neuralneat.set_output_activation(config, "identity")`). Absent means sigmoid, the classification default, unchanged. | `identity` |
| Backprop loss | `BackpropPopulation(..., criterion=CoxCriterion("efron"))` | Efron Cox loss, full batch |
| Target | `ys` is `[N, 2]`: column 0 the observed time (> 0), column 1 the event (1) or censoring (0) indicator | `make_survival_targets(time, event)` |
| Fitness | `make_cox_fitness("efron", complexity_penalty=λ)` | `-loss - λ·k/D` |
| Phantom mask | `BackpropPopulation(mask_phantom=True)` (the default) | on |
| Invalid genomes | `BackpropPopulation(skip_invalid_genomes=True)` (opt-in) | on |

In DB mode, `DatabaseBackpropPopulation` records the non-default settings on
the experiment row as `config_json["training"]`:

```json
{"mask_phantom": true, "output_activation": "identity",
 "criterion": {"name": "cox_ph", "ties": "efron"}}
```

`load_neat_config` reads `output_activation` back, so every genome loaded from
the DB (by the API, the MCP server, the CLI and the coverage code) is rebuilt
with the identity output it was trained with. `resume_from_db` also restores
the Cox criterion. Classification experiments have no `output_activation` key
and load exactly as before.

### 2.2 Fitness: `-loss`, never `1/loss`

The classification convention `fitness = 1/loss` must not be used for Cox. The
Cox loss is a log-likelihood on a different scale, `1/loss` misbehaves near
zero, and `fitness_threshold` loses its meaning. ExplaNEAT-Cox uses
`fitness = -loss`.

The floor is known in advance. A constant-output genome scores
`-null_cox_loss(risk_sets)`, which with Breslow ties is the mean of
$\log|R(t_i)|$ over events. It is data-dependent: about 5.7 on WHAS500 and 6.0
on GBSG2 for the full data, and lower on a training fold, because the risk
sets are smaller. So:

- Set `fitness_threshold = 999` (unreachable) and stop on generations, as
  `experiments/survival/configs/survival.ini` does.
- To stop on fitness, set the threshold relative to the null loss
  (`-null_cox_loss + margin`), never to a number copied from a classification
  config.
- Fitness at or below `-null_cox_loss` means "no signal found". That is a
  better diagnostic than any absolute threshold.

**Parsimony.** `complexity_penalty = λ` subtracts `λ·k/D`, where `k` is the
number of enabled connections plus hidden nodes and `D` the number of training
events. With `λ = 1` this is the per-event AIC (`(-2·pll + 2k) / 2D`). The
runner uses `λ = 1` by default. The penalty applies to selection only;
backprop minimises the plain Cox loss.

### 2.3 The runner

`experiments/survival/run_seed.py` is seeded and reproducible (python/numpy/
torch seeds, `PYTHONHASHSEED=0`, one torch/BLAS thread, deterministic
algorithms). It records the config, the git commit and the environment in the
run JSON.

```bash
# public dataset bundled with scikit-survival (no download)
PYTHONHASHSEED=0 uv run python experiments/survival/run_seed.py gbsg2 0
# known-truth simulation (run this first on any new installation)
PYTHONHASHSEED=0 uv run python experiments/survival/run_seed.py sim-nonlinear 0
# a local cohort file
PYTHONHASHSEED=0 uv run python experiments/survival/run_seed.py cohort 0 \
    --csv /secure/local/cohort.csv --time-col fu_years --event-col cvd_event \
    --features age,sex,sbp,tchdl,diabetes,smoking --categorical sex \
    --split-file /secure/local/folds/fold0.json \
    --horizons 1,3,5 --calibration-horizon 5 --out /secure/local/runs
```

Per run, it:

1. makes or reads event-stratified train/val/test index arrays;
2. standardises on the training rows;
3. evolves the network: backprop and the penalised fitness use the training
   fold, and the model is selected by early stopping over generation
   champions. Each generation's training-fitness champion is scored by its
   plain Cox loss on the validation fold, and the champion with the best
   validation loss is kept (`--selection champion`);
4. fits the Breslow baseline on the training rows;
5. fits the comparator: scikit-survival `CoxPHSurvivalAnalysis` with Efron
   ties, the same features and the same folds, ridge `alpha` tuned on the same
   validation fold by validation Cox loss;
6. evaluates both on the test fold.

`--db` switches to `DatabaseBackpropPopulation`. That mode selects on
training fitness, because the DB loop has no validation selection.
`experiments/survival/study.py` drives repeated splits and aggregates them.

Three settings in the runner and config matter. Two earlier protocol versions
lost C against Cox without them (`experiments/survival/RESULTS.md` §4):

- **Select over champions, not the whole population.** If the best genome is
  picked by validation loss over every genome of every generation (about
  9,000 candidates against about 45 validation events), the choice rewards
  luck. The selected genome was under-trained, sometimes worse than the null
  model on the training fold.
- **Score validation with the plain Cox loss.** The AIC term corrects for
  in-sample optimism, so it belongs to the training fitness only.
- **Use `weight_init_stdev = 0.1`.** Cox coefficients per SD are usually
  below 1 in absolute value. An N(0, 1) weight on every new connection is a
  perturbation that 5 Adadelta epochs per generation cannot undo.

### 2.4 In code

```python
from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.neuralneat import set_output_activation
from explaneat.survival import CoxCriterion, make_cox_fitness, make_survival_targets

set_output_activation(config, "identity")
y = make_survival_targets(time, event)                     # [N, 2]
pop = BackpropPopulation(config, X_train, y_train,
                         criterion=CoxCriterion("efron"), skip_invalid_genomes=True)
# validation early stopping over generation champions (as run_seed.py does):
# from experiments.survival.run_seed import ChampionTracker
# tracker = ChampionTracker(config, X_val, y_val); pop.reporters.add(tracker)
pop.run(make_cox_fitness("efron", complexity_penalty=1.0), n=60, nEpochs=5)
best = pop.best_genome            # or tracker.best with the tracker

from explaneat.core.neuralneat import NeuralNeat
from explaneat.survival import fit_survival_predictor
import torch
g = lambda X: NeuralNeat(best, config).forward(torch.as_tensor(X)).detach().numpy().ravel()
pred = fit_survival_predictor(g, X_train, y_train[:, 0], y_train[:, 1])
pred.survival(X_test, [365, 1825])     # S(t|x)
pred.risk_at(X_test, 1825)             # 1 - S(5y|x), the PREDICT-style risk
```

## 3. Explaining a survival network

The explanation machinery is unchanged: splits, identity nodes, annotations,
compositions and coverage all work on the phenotype. What changes is how the
explanation is read.

**The output is the log-risk.** With an identity output,
$g(x) = b_0 + \sum_k v_k a_k(x)$ over the terms feeding the output node. Each
term is an additive **log hazard ratio**. Between two patients it contributes
the HR $\exp(v_k(a_k(x_1) - a_k(x_2)))$, constant over follow-up. `b_0`
carries no meaning (it is absorbed into $h_0$).

**Formulas.** `get_formula` on an annotation that reaches the output gives
the closed form of the log-risk (no sigmoid). Every engine (`NeuralNeat`,
`StructureNetwork`, `ActivationExtractor`, `AnnotationFunction`, the formula
renderer and the MCP/API evidence tools) follows the configured output
activation. The parity tests in `tests/test_core/test_engine_parity.py` check
this.

**Hypotheses in hazard-ratio terms.** `explaneat.survival.explain` turns
annotation structure into clinical readings:

- A direct edge `x_j → output` with weight `w` is a log-linear Cox term:
  HR = $e^{w}$ per SD, or $e^{w/\mathrm{sd}_j}$ per raw unit.
- A ReLU hinge `relu(w·x_j + b)` with output weight `v` adds log-HR
  `v·w` per SD of `x_j` on one side of the threshold $x_j = -b/w$ (quoted in
  raw units), and nothing on the other. This is a piecewise-linear effect with
  a knot, the kind of shape a clinician would draw.
- `contribution_summary` gives the spread of a term over the data as HRs
  against the median patient (p5/p25/p75/p95). A term with a large weight but
  no spread explains little.

Write leaf hypotheses in these units, for example: "the hinge on node 12 adds
log-HR +0.004 per positive node above 3 nodes, and nothing below".

**State the PH caveat in every explanation.** The explanation describes a
time-constant multiplicative effect because the model can represent nothing
else. It is not evidence that the effect is constant in the data. Test PH on
the comparator (Schoenfeld residuals) and say what you found.

`compute_performance` on a survival dataset (MCP and API) returns the survival
bundle (section 4) instead of AUC/accuracy, and includes the PH caveat text.

## 4. Metrics

Pre-specify all of these before running anything. Compute them for both
models, on the same folds.

| Metric | Function | Notes |
|---|---|---|
| Harrell's C | `metrics.harrell_c` | Report it for comparability. It is biased upward under heavy censoring, so do not lead with it. |
| Uno's IPCW C | `metrics.uno_c` (truncated at τ inside both folds' event support) | Lead with this one. |
| Time-dependent AUC | `metrics.time_dependent_auc` | Cumulative/dynamic AUC at the horizons, plus its integrated mean. |
| IPCW Brier at horizons | `metrics.brier_at` | Needs $S(t \mid x)$, so needs the Breslow baseline. |
| Integrated Brier score | `metrics.integrated_brier` | Discrimination and calibration together. |
| Calibration at a horizon | `metrics.calibration_at_horizon` | Predicted 1 − S(h \mid x) vs Kaplan-Meier observed risk by decile; expected/observed ratio. This is the PREDICT-style 5-year check. |
| Cox loss vs null | `cox_ph_loss`, `null_cox_loss` | On the test fold. |

`metrics.survival_metrics(...)` returns the whole bundle for one fold.
`survival.evidence.survival_performance` is what `compute_performance` calls.

**Concordance is computed on `g(x)` for both models.** There is no
distribution-to-risk transformation to choose, so the "C-hacking" problem
(Sonabend et al. 2022) cannot arise. Say so in the methods.

Each metric is verified against scikit-survival in `tests/test_survival/`.

## 5. Pitfalls

**Proportional hazards is inherited, not tested.** If Cox underperforms
because hazards are not proportional (crossing curves, effects that wane over
time), a Cox-loss network will not fix it: both models carry the same
handicap. Check PH on the comparator with Schoenfeld residuals. Stratified
Cox, or a matched network per stratum, is the standard middle option when PH
holds within groups but not across them. Find out whether the published
equations stratify anything (e.g. by sex). A non-PH model (Cox-Time,
discrete-time) is a different study, not this comparison.

**C-hacking and unfair comparisons:**

- Use identical folds: the same index arrays, not "the same seed".
- Tune both models or neither.
- Use identical preprocessing.
- Use repeated splits and report paired deltas with their spread. Expected
  differences are about 0.01 in C, smaller than single-split noise.
- Pre-register the metric list and the horizons.

**Events per parameter.** The effective sample size is the number of events,
not rows. A network with `k` free parameters fitted on `D` events needs
`D/k` large. The usual guidance for prediction models is at least 10–20
events per parameter. Keep the parsimony penalty on, and report `k`,
`D/k` and the number of inputs used. `run_seed.py` records these for every
run.

**Risk sets per fold.** Build `CoxRiskSets` from each fold's training rows.
`CoxCriterion` and the fitness functions do this from the `ys` they are given.
Never build risk sets on the full data and subset them: that leaks test rows
into the training denominators.

**Competing risks.** A CVD outcome with non-CVD death treated as censoring
gives a cause-specific hazard. Its $1 - S$ is not the cumulative incidence.
State which one is reported, and use the same definition as the comparator
equation.

**Ties.** Use the same tie method for both models (Efron here). Time recorded
in days or months produces many ties.

## 6. Bringing a cohort in (PREDICT / VAREANZ)

The cohort data is under HDEC/DAP approval. The rules in this section are
part of the method, not advice.

**Where the data lives:**

- [ ] The data stays inside the approved secure environment. The analysis
  runs there, and the ExplaNEAT software is brought to the data (e.g. as a
  container image). No row-level data leaves it.
- [ ] Never commit cohort data, derived row-level files, fold files containing
  real indices, or run outputs with row-level predictions to git. Keep `--out`
  and `--split-file` outside the repository. The runner records only the file
  name of `--csv`, never its path. Check `git status` before every commit.
- [ ] Use a **local** database if the DB mode is used: SQLite inside the
  environment (`DATABASE_URL=sqlite:///...`), or the environment's own
  Postgres. Never use a shared or remote database. For SQLite, import
  `explaneat.db.sqlite_compat` before `db.create_all()`; it renders the
  JSONB/UUID columns. `experiments/survival/explain_setup.py` and `xn.py`
  show the pattern.
- [ ] Never send cohort data, or anything derived from patients, to external
  services. That includes LLM-assisted annotation: annotation tools see model
  structure and fitted parameters only, never rows. Use a local model or
  annotate manually if the environment forbids external calls (HDEC 4.2).
  Evidence tools that compute on rows (`compute_viz_data`, `compute_shap`,
  `get_input_distribution`) must run inside the environment, and only
  aggregates may leave it, under the release rules of the data agreement.
- [ ] Outputs (metrics, formulas, explanations) go to VIEW for review before
  release.

**File format:**

- [ ] CSV or Parquet, one row per person: `--csv path --time-col --event-col
  --features ... --categorical ...`.
- [ ] `--features` lists the predictors explicitly, with no extra columns, and
  matches the comparator equation's predictor list.
- [ ] Missing values raise unless `--impute median` is passed. Pre-register
  the imputation rule (or impute upstream, as the comparator did) and apply
  the same rule to both models.

**Columns and definitions to pin down with Katrina and Rod before analysis:**

- [ ] **Index date** and **follow-up time** unit. Use years, so that the
  horizon is 5.
- [ ] **Outcome definition**: which CVD events count, from which sources
  (hospitalisations, mortality), and the event indicator value
  (`--event-value`).
- [ ] **Censoring definition**: end of study, death from other causes,
  emigration or loss to follow-up. State the competing-risk treatment
  (cause-specific vs Fine-Gray) to match the published equation.
- [ ] **Expected n and event count.** The event count bounds the defensible
  model size (events per parameter). Fix the parsimony setting from it before
  fitting.
- [ ] **Strata**: does the published equation stratify, e.g. by sex? If so,
  match it (one model per stratum, or a stratified comparator).

**Comparator Cox model:**

- [ ] The same predictors with the same transformations (splines,
  interactions) as the published PREDICT/VAREANZ equation, refitted on the
  same training folds. Also fit a plain linear Cox on the identical inputs the
  network sees, so that "non-linearity helps" is separated from "different
  predictors".
- [ ] Efron ties. The same standardisation fitted on training rows only.

**Folds:**

- [ ] Generate the event-stratified index arrays once
  (`survival_split_indices`, `save_split_indices`), store them inside the
  environment, and have every model read the same file (`--split-file`).
  Repeated splits (e.g. 10) with paired deltas.

**Metrics and horizon to pre-register:**

- [ ] Horizon: 5 years (`--horizons 1,3,5 --calibration-horizon 5`).
- [ ] Primary: Uno's C (τ = 5 years) and calibration at 5 years by decile
  (expected/observed ratio and the decile table).
- [ ] Secondary: Harrell's C, time-dependent AUC at 1/3/5 years, IPCW Brier
  at 5 years, IBS up to 5 years, test Cox loss vs null.
- [ ] Explanation outcome: structural and compositional coverage, the
  closed-form log-risk, and leaf hypotheses in HR terms, reviewed by
  clinicians (including Māori members of the team).
- [ ] PH diagnostics on the comparator (Schoenfeld residuals).
- [ ] Concordance computed on g(x) for both models.

**Before running on the cohort:**

- [ ] Run `uv run pytest tests/test_survival` and the simulation check
  (`study.py sim`) inside the environment. Unvalidated code is not run
  against patient data (HDEC 2.6).
