# Synthetic U-shape and saddle study: can PropNEAT find two clinical non-linear shapes, and can the explanation name them?

Exploratory, not pre-registered. Branch `exp/ushape-saddle`. Freeze 19 Nov 2026.

**Question (Mike).** Two shapes occur in medicine and have appeared in no real result so far:
(1) a U-shaped risk curve on one input (too low and too high both raise risk) in an otherwise
linear dataset; (2) a saddle-shaped risk surface between two inputs acting in opposing
directions (risk rises when one is high and the other low, falls when both are high or both
low: a signed `x1 * x2` interaction, a hyperbolic paraboloid). Does PropNEAT find them, and does
the frozen explanation method recover and name them?

## 1. Answer in brief

_To be written after the runs and the explanations._

## 2. Design (fixed before any network was trained)

### 2.1 Data (`data.py`, `calibrate.py`, `make_datasets.py`)

Three synthetic tabular datasets share the same 2,000 rows of 10 inputs (clinical-style names and
distributions in the style of the explanation ladder's family, `experiments/explanation_barrier/data.py`;
data seed 20261114) and the same linear backbone; they differ only in the nonlinear term:

| Input | Distribution (raw units) |
|---|---|
| `age` | normal(62, 12), clipped 30–95, integer years |
| `bmi` | normal(27.5, 5), clipped 15–50, one decimal |
| `sbp` | systolic blood pressure, mmHg: 132 + 18 (0.25 za + 0.97 N), clipped 85–210 |
| `heart_rate` | beats/min, normal(76, 13), clipped 40–140 |
| `sodium` | serum sodium, mmol/L, normal(139, 4), clipped 120–158, integer |
| `creatinine` | serum creatinine, µmol/L, log-normal around 85 (sd 0.3 in log, mildly raised by age), clipped 40–400 |
| `smoker`, `diabetes`, `male` | binary, p = 0.25 / 0.30 / 0.55 |
| `noise` | normal(0, 1), used by no ground truth |

Standardised quantities (fixed constants, not data estimates): za = (age − 62)/12, zb = (bmi − 27.5)/5,
zp = (sbp − 132)/18, zh = (heart_rate − 76)/13, zn = (sodium − 139)/4, zc = (log creatinine − log 85)/0.3.

- **Backbone (all three):** logit = −1.2 + 0.7 za + 0.5 zc + 0.3 zb + 0.8 smoker + 0.6 diabetes + 0.3 male.
- **U (`shape-A`, task `u`):** + β_U ((zn + 0.5)² − 1): a U on sodium with its minimum at zn = −0.5,
  i.e. **137 mmol/L** (the sweet spot; the −1 keeps the mean logit near the backbone's). Both low
  and high sodium raise risk. Sodium is not in the backbone.
- **Saddle (`shape-B`, task `saddle`):** − β_S zh zp: risk rises when heart rate is high and
  systolic pressure low (or the reverse), falls when both are high or both low. Neither input is in
  the backbone, so the interaction is the whole of their signal.
- **Both (`shape-C`, task `both`, optional third dataset):** the backbone plus both terms.
- Labels `y ~ Bernoulli(sigmoid(logit))`, one label seed per dataset. Split stratified 75:25,
  `random_state` 42: **1,500 train / 500 test**; inputs z-scored by a StandardScaler fit on the
  training rows (as the 10-seed study). Outcome share 0.40 / 0.39 / 0.42.

**Strength calibration (`calibrate.py`, before any run).** The brief asks that the oracle (the true
logit) beat logistic regression by at least 0.05 test AUC. The rule used, fixed before any network
was trained: the smallest grid value at which the oracle beats L2 logistic regression by at least
**0.10** (twice the floor) *and* gradient boosting beats logistic regression by at least 0.05 (so the
signal is learnable by a standard flexible learner at n = 1,500, and a PropNEAT null is not down to a
weak signal). Grid {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}; test AUCs:

| Task | β | Oracle | LR | GB | Oracle − LR | GB − LR | |
|---|---|---|---|---|---|---|---|
| U | 0.50 | 0.808 | 0.764 | 0.745 | 0.044 | −0.019 | |
| U | 0.75 | 0.825 | 0.745 | 0.772 | 0.080 | 0.027 | |
| U | **1.00** | 0.851 | 0.732 | 0.830 | 0.120 | 0.098 | **chosen** |
| U | 1.25 | 0.872 | 0.741 | 0.819 | 0.131 | 0.078 | |
| saddle | 1.00 | 0.786 | 0.721 | 0.747 | 0.065 | 0.026 | meets the brief's floor but GB barely beats LR |
| saddle | 1.25 | 0.795 | 0.692 | 0.721 | 0.104 | 0.029 | |
| saddle | **1.50** | 0.841 | 0.727 | 0.780 | 0.115 | 0.053 | **chosen** |
| saddle | 2.00 | 0.849 | 0.686 | 0.817 | 0.163 | 0.131 | |

So **β_U = 1.0, β_S = 1.5**; `shape-C` uses both. The full grid is in `calibration.json`. The first
pass of the calibration used the bare 0.05 floor (U 0.75, saddle 1.0); the learnability condition was
added, still before any run, because at those strengths gradient boosting itself hardly beat logistic
regression on the saddle (0.747 vs 0.721), which would have left a PropNEAT null uninterpretable.

**Database.** `explaneat_dev`, new rows only: datasets `shape_syn_37095` (A, `b4d7be76`),
`shape_syn_818e0` (B, `3e662ee6`), `shape_syn_90c30` (C, `519c1f60`), each with one split (`datasets.json`).
The names and descriptions are opaque: the explaining agent sees the label, the dataset name and the
inputs' clinical names, not which input carries which shape (as the ladder withheld its generating
functions). The mapping task ↔ label lives in this file and `data.py`, which the explaining agent is
told not to open.

### 2.2 PropNEAT runs (`run_seed.py`, `configs/syn10.ini`, `launch.sh`)

| Item | Setting |
|---|---|
| Config | `configs/syn10.ini`: the thesis Heart config (`experiments/10seed/configs/heart.ini`) with `num_inputs` 10, nothing else changed: population 250, ReLU hidden units, sigmoid output, `initial_connection = full_direct` (the linear start), node add/delete 0.4/0.2, conn add/delete 0.5/0.3 |
| Training | the 10-seed harness path (`DatabaseBackpropPopulation.run` with `auc_fitness`, 5 backprop epochs per generation, unmasked, 1 torch thread, `PYTHONHASHSEED=0`), imported from `experiments/10seed/run_seed.py` |
| Budget | **100 generations**, seeds **0–4**, per dataset (A and B; C if time allows) |
| Selection | best genome by training fitness within the run, ties to the earliest generation then the lowest NEAT key (the 10-seed rule); the test split is never used for selection |
| Ancestry | `track_ancestry` on (the population default) |
| Processes | at most 3 training processes at a time (`launch.sh`, three lanes) |
| Second arm | the brief suggested a second arm with `initial_connection = full_direct`; the thesis config family already starts `full_direct` (Heart, Adult and MONK-2 all do), so the main arm *is* the linear start and no second arm was run |

Experiments are named `shape-<A|B|C>-s<seed>`. A 2-generation timing dry run `shape-A-dryrun-s0` is
left in `explaneat_dev` as the record of the harness check.

### 2.3 Report per dataset

Test AUC per seed with a 10,000-resample paired bootstrap 95% CI (seed 42) against the oracle, L2
logistic regression and gradient boosting (`references.py`, same bootstrap); hidden units and
connections; the share of the LR → oracle gap closed, `(AUC − LR)/(oracle − LR)`, as the median over
seeds and for the best-by-train genome (`compute_runs.py` → `runs.csv`, `summary.json`).

### 2.4 Explanation and the ground-truth test

The best-by-train genome per dataset is explained with the frozen method (authoring skill v4,
protocol v4, strict mode with provenance, function-preserving operations only, 3-hour box, one
explaining agent per genome; `EXPLAIN_TASK.md`), rechecked (`verify_record.py`), blind-verified
(the verifier skill, one agent per genome), graded on the three levels
(`compute_levels.py --set ushape`) and given a plain-language draft (the plain-language skill).

**Ground truth (the point of the study), graded recovered / partial / not, as `gt_grades.json` of
the ladder:**
- **U:** does the hierarchy contain a part whose closed form is two opposing hinges on sodium (or an
  equivalent convex shape); does the hypothesis say both ends raise risk; does the stated minimum
  match 137 mmol/L (zn = −0.5)? Shape test: the leaf's response over sodium (PNG).
- **Saddle:** a ReLU network can only approximate zh·zp (for example by hinges on zh + zp and
  zh − zp, since (a + b)² − (a − b)² = 4ab). What structure did it use; did the explainer name it as
  an opposing-direction interaction between heart rate and systolic pressure; and how does the
  explained part probe against the concept `heart_rate * sbp` (ad-hoc expression) with
  `probe_concept`? Shape test: the 2-input heatmap (PNG).
- The backbone's terms are graded too (linear age, log-creatinine, bmi, the three steps; noise
  unused), as the ladder graded its additive task.

## 3. Performance (test split, n = 500; CIs are 10,000 bootstrap resamples, seed 42)

All 15 runs completed (100 generations each, about 2 minutes of wall clock per run; `runs.csv`,
`summary.json`, `references.json`). Gap closed = (AUC − LR) / (oracle − LR) on the test split.

### U (shape-A): dataset `shape-A`, n_test = 500

| Model | Test AUC [95% CI] | Gap closed (LR → oracle) | Hidden / conns | Inputs used |
|---|---|---|---|---|
| Oracle (true logit) | 0.851 [0.817, 0.883] | 1.00 | – | – |
| L2 logistic regression | 0.732 [0.684, 0.778] | 0.00 | – | – |
| Gradient boosting | 0.830 [0.793, 0.864] | 0.82 | – | – |
| PropNEAT seed 0 **(best by train)** | 0.844 [0.808, 0.877] | 0.94 | 10 / 31 | 8 of 10 |
| PropNEAT seed 1 | 0.850 [0.815, 0.883] | 0.99 | 15 / 39 | 10 of 10 |
| PropNEAT seed 2 | 0.849 [0.814, 0.881] | 0.98 | 11 / 35 | 8 of 10 |
| PropNEAT seed 3 | 0.839 [0.802, 0.873] | 0.89 | 9 / 26 | 8 of 10 |
| PropNEAT seed 4 | 0.839 [0.803, 0.872] | 0.90 | 12 / 37 | 9 of 10 |
| PropNEAT median of 5 | 0.844 (range 0.839–0.850) | 0.94 | 11 / 35 | |

### saddle (shape-B): dataset `shape-B`, n_test = 500

| Model | Test AUC [95% CI] | Gap closed (LR → oracle) | Hidden / conns | Inputs used |
|---|---|---|---|---|
| Oracle (true logit) | 0.841 [0.806, 0.874] | 1.00 | – | – |
| L2 logistic regression | 0.727 [0.681, 0.771] | 0.00 | – | – |
| Gradient boosting | 0.780 [0.739, 0.821] | 0.47 | – | – |
| PropNEAT seed 0 **(best by train)** | 0.835 [0.799, 0.870] | 0.94 | 10 / 34 | 8 of 10 |
| PropNEAT seed 1 | 0.832 [0.795, 0.867] | 0.92 | 16 / 41 | 9 of 10 |
| PropNEAT seed 2 | 0.823 [0.784, 0.859] | 0.84 | 11 / 33 | 8 of 10 |
| PropNEAT seed 3 | 0.830 [0.792, 0.864] | 0.90 | 10 / 33 | 9 of 10 |
| PropNEAT seed 4 | 0.825 [0.787, 0.860] | 0.86 | 11 / 34 | 9 of 10 |
| PropNEAT median of 5 | 0.830 (range 0.823–0.835) | 0.90 | 11 / 34 | |

### both (shape-C): dataset `shape-C`, n_test = 500

| Model | Test AUC [95% CI] | Gap closed (LR → oracle) | Hidden / conns | Inputs used |
|---|---|---|---|---|
| Oracle (true logit) | 0.884 [0.854, 0.913] | 1.00 | – | – |
| L2 logistic regression | 0.699 [0.650, 0.745] | 0.00 | – | – |
| Gradient boosting | 0.854 [0.820, 0.886] | 0.84 | – | – |
| PropNEAT seed 0 | 0.853 [0.817, 0.886] | 0.83 | 12 / 31 | 10 of 10 |
| PropNEAT seed 1 | 0.840 [0.803, 0.874] | 0.76 | 12 / 35 | 9 of 10 |
| PropNEAT seed 2 | 0.859 [0.824, 0.891] | 0.86 | 11 / 36 | 10 of 10 |
| PropNEAT seed 3 | 0.872 [0.840, 0.902] | 0.93 | 11 / 35 | 9 of 10 |
| PropNEAT seed 4 **(best by train)** | 0.873 [0.841, 0.903] | 0.94 | 8 / 35 | 9 of 10 |
| PropNEAT median of 5 | 0.859 (range 0.840–0.873) | 0.86 | 11 / 35 | |

**Reading.** On every dataset the five PropNEAT seeds sit within about 0.01 of each other and
close a median 94% (U), 90% (saddle) and 86% (both) of the LR → oracle gap; every seed beats
gradient boosting on the saddle (GB closes 47%: a 1,500-row boosted ensemble finds the signed
product poorly), and the medians match or exceed GB on the U (0.844 vs 0.830) and on both
(0.859 vs 0.854). The bootstrap CIs of PropNEAT and the oracle overlap on all three; LR's does not
overlap PropNEAT's on any. The networks are small: 8–16 hidden units, 26–41 connections, and
they leave 0–2 of the 10 inputs unconnected. Selection is by training fitness only; the
best-by-train genome per dataset (seed 0, seed 0, seed 4) is the one explained below.

## 4. Method for the explanation

_After the explanations._

## 5. What each explained model does

## 6. Ground-truth grades

## 7. The three levels and the verifier

## 8. Plain-language drafts

## 9. Caveats

## 10. Reproduce
