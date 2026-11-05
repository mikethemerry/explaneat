# Churn: can PropNEAT beat logistic regression and stay explainable?

Started 2026-11-05. This is separate from the pre-registered 10-seed experiment
(`experiments/10seed/PREREG.md`).

## Why churn

On Heart, MONK-2 and Adult, PropNEAT networks are reliably explainable, but only as good as
L2 logistic regression or worse. Churn has real nonlinear headroom. On the split below, test
AUC is:

| Model | Train AUC | Test AUC (95% CI) | Test acc |
|---|---|---|---|
| L2 logistic regression | 0.843 | **0.813** (0.775–0.848) | 0.862 |
| Gradient boosting (HistGB, defaults) | 1.000 | **0.932** (0.898–0.962) | 0.961 |

Source: `baselines.json`, written by `ingest_churn.py`. CIs come from 10,000 bootstrap
resamples of the test set, seed 42.

Questions:
1. Does PropNEAT close some of the 0.12 gap between logistic regression and gradient boosting?
2. Can the gain be explained? Which hinges and interactions does it come from?
3. Does the 51-category `state` one-hot produce rare-category switches like Adult's? If so,
   can they be identified from the explanation and repaired (prune → retrain), with the effect
   on test AUC measured?

## Data

- PMLB `churn` (pmlb 1.0.1.post3), 5,000 rows, 14.15% positive in train.
- **Dropped:** `phone number`, a unique per-row id.
- **One-hot:** `state` (51) and `area code` (3). `international plan` and `voice mail plan`
  stay binary; the 15 numeric columns are unchanged. That gives **71 inputs**.
- **Collinearity:** each `total * charge` column is an exact multiple of its `total * minutes`
  column, so after z-scoring the two are identical. They are kept deliberately: the explanation
  should show the network using one, the other, or both interchangeably.
- **Split:** stratified 80:20, seed 42 (4,000 / 1,000), StandardScaler fit on train.
- **DB ids:** raw `b7b31ff0-e76a-4492-972a-f12bb00d4ee3`, prepared `churn_onehot`
  `802129c2-b11b-43a5-b6af-8bcf80e4c252`, split `45962ea6-3c78-49f9-a01e-3665f6809b1e`.

## Training (pilot)

- **NEAT config:** `configs/churn.ini`. This is the Adult paper config
  (`experiments/10seed/configs/adult.ini`) with only `num_inputs` changed, 107 → 71. It is
  built for wide one-hot tabular data, with backprop-only weights, conn_add 0.68 and node_add 0.7.
- **Training settings:** population 250, 100 generations, 5 backprop epochs per generation, AUC
  fitness on train, Adadelta lr 1.5. There is no structural penalty.
- **Seeds:** 0, 1 and 2 for the pilot, via `experiments/10seed/run_seed.py churn <seed>` (the
  same seeded harness). Experiments are named `churn-pilot-s<seed>`.
- **Selection:** best training fitness; ties go to the earliest generation, then the lowest
  NEAT key.

## Decision rule after the pilot

- **PropNEAT test AUC clearly above logistic regression's 0.813:** run the full 10 seeds and
  explain all of them (`experiments/10seed/EXPLAIN_PROTOCOL.md` v3). Then do the repair study
  on any rare-category switches it finds.
- **Not clearly above:** report it. Then try a longer run (more generations) before abandoning
  churn, as on Adult, where 100 generations left about 0.024 of headroom against gradient
  boosting.

## Pilot result and next steps (2026-11-05)

The pilot (`runs_churn-pilot.csv`) scored test AUC 0.895 / 0.904 / 0.876 (seeds 0–2), against
logistic regression's 0.813 and gradient boosting's 0.932. That is clearly above logistic
regression, so the rule above applies.

At Mike's request, both of these run:
- **churn100:** 10 seeds × 100 generations. The pilot experiments were renamed
  `churn-pilot-s*` → `churn100-s*` (seeds 0–2). They are reused as-is, because seeded runs are
  deterministic. Seeds 3–9 are new, run as `run_seed.py churn <s> --name churn100-s<s>`.
- **churn200:** 10 seeds × 200 generations, run as
  `run_seed.py churn <s> --generations 200 --name churn200-s<s>`. Their first 100 generations
  should exactly reproduce churn100, which is checked.

At most 4 training runs go at once (CPU load). The churn100 networks are explained first,
then churn200, all with protocol v3.

Explanation orders were fixed before any structure was inspected, with
`numpy.random.default_rng(20261105)`: the first permutation is churn100, the second churn200.
Genomes are explained in that order as their runs finish.
- churn100: s4 s1 s0 s7 s8 s2 s9 s3 s6 s5
- churn200: s1 s2 s4 s7 s9 s5 s3 s8 s0 s6
