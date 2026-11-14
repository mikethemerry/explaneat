# Synthetic U-shape and saddle study: can PropNEAT find two clinical non-linear shapes, and can the explanation name them?

Exploratory, not pre-registered. Branch `exp/ushape-saddle`. Freeze 19 Nov 2026.

**Question (Mike).** Two shapes occur in medicine and have appeared in no real result so far:
(1) a U-shaped risk curve on one input (too low and too high both raise risk) in an otherwise
linear dataset; (2) a saddle-shaped risk surface between two inputs acting in opposing
directions (risk rises when one is high and the other low, falls when both are high or both
low: a signed `x1 * x2` interaction, a hyperbolic paraboloid). Does PropNEAT find them, and does
the frozen explanation method recover and name them?

## 1. Answer in brief

**Yes to both, on every dataset.** PropNEAT found both shapes. Its five seeds closed a median 94% (U), 90%
(saddle) and 86% (both) of the gap from logistic regression to the oracle, and it beat gradient
boosting on the saddle on every seed (section 3). The frozen method recovered and named the shapes in
all three explained genomes (section 6).

- **U on sodium.** Both genomes build the U from ReLU hinges on sodium.
  - **shape-A:** a low-sodium arm plus two high-sodium hinges. The typical-case floor is 135–137 mmol/L
    (truth: 137).
  - **shape-C:** one unit split into an asymmetric V, lowest at 138 mmol/L (truth: 137).
  - In both, the hypothesis says that low and high sodium both raise risk. The probe of the U part
    against the true term (zn + 0.5)² gives R² 0.95 (A) and 0.90 (C).
- **Saddle.** Both genomes build −zh·zp from hinges on combinations of the two inputs:
  - hinges on systolic pressure **minus** heart rate (in C, both signs);
  - a serial chain on pressure **plus** heart rate that lowers risk when both are high.
  - This is the (a − b)² − (a + b)² identity, done with hinges.

  Both explainers named it as an opposing-direction interaction: a fast pulse with low pressure, or high
  pressure with a slow pulse, raises risk, and both high lowers it. The whole-model heart-rate × pressure
  surface correlates 0.96 with the true surface in both genomes.
  - In **shape-C** one composition holds the saddle, scoring R² 0.78 against −(hr − 76)(sbp − 132).
  - In **shape-B** no single part is the saddle. It appears at the root, where two leaves and a direct
    heart-rate edge add up to it (R² 0.96 against the product plus linear terms).
- **What went wrong was the wording, not the shapes.** The blind verifiers refuted three statements on
  shape-A and two on shape-B, and none on shape-C. Every refuted statement is a clinical gloss on a
  shape, and every number reproduced:
  - shape-A: "normal sodium leaves all three hinges off"; "a higher BMI raises the risk steadily";
  - shape-B: "bmi only above a turn"; one heart-rate direction inside the bmi part.

  On shape-A the model is closer to the truth than the gloss: risk does climb through upper-normal
  sodium, as the true U does. Spurious structure shows up as small gated interactions that the truth
  lacks:
  - shape-A: bmi × low sodium, and two noise-column chains;
  - shape-B: an age ceiling above 80;
  - shape-C: age × high sodium, and low sodium widening the saddle gates.

  The truth's weakest term, the male step (β 0.3; LR coefficient 0.09–0.15), was missed in A and C.

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

One explaining agent per genome, as fixed in section 2.4:
- the authoring skill v4 and protocol v4;
- strict mode with provenance;
- function-preserving operations only;
- `EXPLAIN_TASK.md` verbatim, with a 3-hour box.

**The interruption.** The three agents started at 16:40 NZDT on 14 Nov. They were cut off at about 17:03,
with their structure, annotations, meanings, formulas and part of the hypotheses already stored. Three fresh
agents resumed from the stored database state and the scratch notes, under the same task and the same box.
Nothing was undone. They finished at 17:10–17:13, well inside the box. On resuming:
- shape-B re-ran one root query that had been computed before the last identity operation (stale
  provenance; same values).
- shape-C fixed a scope error. Six leaves had quoted `query_ablate`'s whole-split mean risk change as a
  mean over the rows the leaf acts on; they were re-queried with `where` and rewritten.

Every closing recheck (`verify recheck` v3) gave 0 mismatches and 0 needs_review.

After the explanations:
- **Verifier.** One blind verifier per record (`VERIFY_TASK.md`, the verifier skill, read-only). It saw
  neither the generating function nor the explainer's scratch work.
- **Levels.** `compute_levels.py --set ushape`.
- **Plain language.** `plain-language extract`, then a draft with the plain-language skill (section 8).
- **Ground truth.** Shape plots (`shape_plots.py`, `figures/`). An ad-hoc concept probe of the explained
  parts against the true terms (`gt_probe.py` → `gt_probe.json`): the training-row contribution of a part
  (or group of parts), switched off together, against `heart_rate * sbp`, the centred product −zh·zp, the
  product plus linear zh and zp, and (zn + 0.5)². The fit is affine, a·concept + b.

| | shape-A (U) | shape-B (saddle) | shape-C (both) |
|---|---|---|---|
| Genome / explanation | `2aac39c4` / `b5ff5e07` | `25f5830e` / `4d8e6b70` | `a1cbdd4a` / `f0f1da92` |
| Operations | 24 | 24 | 32 |
| Leaves / compositions | 4 / 3 | 5 / 1 | 8 / 2 |
| Splits / identity nodes | 7 / 0 | 5 / 3 | 8 / 4 |
| Coverage (structural / compositional) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Function preservation (max abs error) | 3.3e-16 | 4.4e-16 | 3.3e-16 |
| Closed-form max abs error | 0.018 | 0.044 | 0.057 |
| Numbers in the record (provenance-backed) | 365 (324) | 539 (515) | 373 (352) |
| Closing recheck | 0 mismatch, 0 review | 0 mismatch, 0 review | 0 mismatch, 0 review |

## 5. What each explained model does

The explainers' readings are in `explanations/<label>.json` (`overall_function`, `clinical_terms`, the
leaves). The weights below are on z-scored inputs; the thresholds come from typical-case sweeps.

**shape-A (U; seed 0, test AUC 0.844).** The model is a sigmoid of direct terms plus a sodium U.
- **Direct terms**, per sd: age 0.79, smoker 0.40, bmi 0.38, creatinine 0.36, diabetes 0.28, noise 0.17.
- **The U** is built from three hinges:
  - a low-sodium arm (units 357 → 3629), on below 134–135 mmol/L, whose switch-on point a higher bmi lowers;
  - a gentle high-sodium hinge (unit 219), on from 137–138;
  - a steep high-sodium hinge (unit 1180), on from 141–142.
- **Risk across sodium**, typical case: 77% at 130, 10% at 137, 92% at 145. Switching the sodium structure
  off lowers training AUC from 0.857 to 0.697.
- **Noise chains.** Two three-unit chains driven by the noise column, one gated off by diabetes and one by
  high creatinine, feed small terms into the low-sodium arm. They are described, not explained (AUC 0.8565
  vs 0.8532 without them).
- **Unused:** sbp and heart rate.

**shape-B (saddle; seed 0, test AUC 0.835).** The model is a linear score plus four hinged parts.
- **Linear score:** heart rate 1.29, age 0.81, creatinine 0.46, smoker 0.45, diabetes 0.43, male 0.21.
- **The saddle** is spread over three pieces:
  - `pressure_over_rate` (unit 440), a hinge on 1.13 sbp − 0.91 heart rate, raises risk steeply when
    pressure is high for the rate;
  - `pressure_rate_load` (units 534, 1197, 2442, 560, 7124, 7602), a serial chain on 1.92 sbp + 1.42 heart
    rate, lowers risk ever more steeply as both rise;
  - the direct heart-rate edge supplies the fast-rate, low-pressure arm.
- **Risk across heart rate**, typical case: at systolic 110 mmHg it rises from 3% to 90% as heart rate
  rises, and at 150 mmHg it falls.
- **Other parts:**
  - `rate_bmi_amplifier` carries bmi through a hinge that is nearly linear on most rows;
  - an age ceiling switches on above about 80 years.
- **Unconnected:** sodium and noise. Linearising every hidden unit lowers training AUC from 0.846 to 0.745.

**shape-C (both; seed 4, test AUC 0.873).** The model adds terms on the logit scale.
- **Linear terms:** age 0.98, creatinine 0.60, bmi 0.23, smoker 0.31, noise 0.17, plus a diabetes step.
- **`sodium_V`** (unit 174, split three ways) is abs(sodium − 138.7) with a 2.6× steeper high side
  (probe 0.998).
- **`bp_hr_saddle`** (a composition) holds:
  - `tachycardia_low_sbp_gate` (unit 999, on 2.36 hr − 1.11 sbp, best concept heart_rate / sbp 0.95);
  - `hypertension_low_hr_gate` (unit 1595, on 1.10 sbp − 0.93 hr, sbp − heart_rate 0.92);
  - `high_sbp_and_hr_penalty` (units 3040 → 3544, on about sbp + 1.5 hr, negative, sbp * heart_rate 0.99);
  - a small `sbp_hinge`.
  Switching the penalty chain off costs the most training AUC (0.896 to 0.783).
- **Spurious couplings:**
  - low sodium opens both mismatch gates at a smaller mismatch (unit 174 feeds 999 and 1595);
  - a gate (unit 1269) makes the high-sodium side open earlier in younger patients.
- **Unused:** the male input has no edge.

## 6. Ground-truth grades (`gt_grades.json`, `gt_probe.json`, `figures/`)

| Term | shape-A (U) | shape-B (saddle) | shape-C (both) |
|---|---|---|---|
| Sodium U, minimum 137 | **recovered**: three hinges; floor 135–137; rise from the minimum 5.1 logit at 128 (truth 5.1), 9.6 at 150 (truth 10.6); R² 0.95 vs (zn+0.5)² | (absent; sodium unconnected: recovered) | **recovered** as a V: minimum 138 (138.7 in the leaf); rise 3.6 at 128 (truth 5.1), 8.0 at 150 (truth 10.6); R² 0.90 |
| Saddle −1.5 zh zp | (absent; sbp and hr unconnected: recovered) | **recovered at the root**: hinge on sbp − hr, chain on sbp + hr, direct hr edge; named as opposing directions; surface r = 0.96 with the truth (95% interaction); the two saddle leaves vs `heart_rate * sbp` R² 0.47, vs −zh zp 0.31, vs product + linear 0.96 | **recovered**: composition `bp_hr_saddle`; hinges on hr − sbp, sbp − hr and a chain on sbp + hr; named as opposing directions; vs −zh zp R² 0.78 (0.80 with linear terms; raw `heart_rate * sbp` 0.03); surface r = 0.96 (96% interaction) |
| age, creatinine, smoker, diabetes | recovered | recovered (age with a spurious ceiling above 80) | recovered |
| bmi | partial (bmi × low-sodium interaction; "steadily" refuted) | partial (model nearly linear, record's "only above a turn" refuted) | recovered |
| male step (β 0.3) | not (0.06 edge into a hinge) | recovered (0.21) | not (no edge) |
| unused inputs | recovered (noise flagged as fitted noise) | recovered | recovered (noise 0.17 flagged as spurious) |
| **Score** | **6.5 / 8, recovered** | **7.5 / 8, recovered (at the root)** | **8 / 9, recovered** |

**How the ReLU networks built the saddle.** Neither network computes a product.
- **shape-C** uses all three pieces of the hinge identity (a − b)² − (a + b)² = −4ab: two opposing difference
  hinges (fast pulse for the pressure; high pressure for the pulse) that raise risk, and one sum chain that
  lowers it when both are high.
- **shape-B** uses one difference hinge (pressure high for the rate) and the sum chain. The missing
  difference arm (fast rate, low pressure) is covered by a large direct heart-rate edge that the sum chain
  cancels at high pressure.
- In both, the explained surface matches the true one (r = 0.96). The raw concept `heart_rate * sbp` is a
  poor probe of any single part: it is dominated by its linear terms. The centred product, or the product
  with linear terms, is the right ad-hoc concept.

**Shape plots.**
- `figures/<label>-sodium.png`: model response over sodium against the true U.
- `figures/<label>-saddle.png`: model surface, the part that moves most, and the true −1.5 zh zp.
- `figures/<label>-shape.json`: the swept values.

## 7. The three levels and the verifier

| | shape-A | shape-B | shape-C |
|---|---|---|---|
| Faithful (function preservation) | 3.3e-16 | 4.4e-16 | 3.3e-16 |
| Decomposed (leaves; granularity) | 4; 1.0 | 5; 1.0 | 8; 1.0 |
| Understood, probe v2 (m / p / c) | 3 / 1 / 0 (0.75) | 2 / 3 / 0 (0.40) | 4 / 3 / 1 (0.50) |
| Understood, explainer's own grades | 2 / 2 / 0 | 5 / 0 / 0 | 8 / 0 / 0 |
| Understood, verifier's leaf grades | 2 m, 1 p, 1 c | 3 m, 1 p, 1 c | 7 m, 1 p |
| Blind verifier | **refuted**: 3 refuted, 8 weakened, 14 upheld; 185 statements | **refuted**: 2 refuted, 9 weakened, 21 upheld, 1 flag; 112 statements | **weakened**: 0 refuted, 4 weakened, 18 upheld, 1 flag; 118 statements |

**Refuted statements.**
- **shape-A:**
  - `overall_function`, "normal sodium leaves all three hinges off": they are all off only at 135–137, and
    the gentle hinge is on for all 812 rows with sodium of 139 or more;
  - `clinical_terms`, "a higher BMI raises the risk steadily": at sodium 130, risk falls from 87% at BMI 16
    to 58% at BMI 42 (163 rows);
  - `low_creatinine_noise_chain`: the sign of a 0.007 mean risk change is reversed (the tool's
    with-minus-without difference was read as without-minus-with).
- **shape-B:**
  - `rate_bmi_amplifier`'s clinical sentence (heart rate raises risk through it, more so when pressure is
    high for the rate): there the part falls with heart rate;
  - "bmi raises risk only above a turn in the overweight range": the part is on at bmi 15 in 41% of rows.

Every number, weight, threshold and closed form reproduced in all three.

**Weakenings, chiefly:**
- threshold labels that do not fit the switch-on points ("upper end of the normal range" for 141–142);
- sweep end points outside the training range;
- shape-C's "jointly high" for what is a threshold on sbp + 1.5 hr: the penalty also acts on 249 of 394
  fast-pulse, low-pressure rows.

**The flags** (B, C) note that "both high lowers risk" is clinically odd and that the data agree. The
flag is on how the data were made.

On the probe (v2), C's `sodium_V` is graded closed-form-only: the explainer's own V concept fails the
simplicity condition, and the library sweep has no V on sodium. The verifier and the explainer both grade
it meaningful.

## 8. Plain-language drafts (`plain_language/<label>.json`, key `drafted`)

The claims were extracted read-only from each stored explanation:

| | Claims | Verified | Not supported |
|---|---|---|---|
| shape-A | 104 | 82 | 22 |
| shape-B | 67 | 59 | 8 |
| shape-C | 108 | 87 | 21 |

Drafts follow the plain-language skill (v5, readdown and titles for every part). They are lint-clean,
marked `drafted_by: ai-assisted` and `verified_by: null`, and every number in them is a claim slot.

| | Summary / readdown / detail sentences | Worth-checking flags | Proposed revisions |
|---|---|---|---|
| shape-A | 9 / 39 / 24 | 5 | 7 |
| shape-B | 17 / 38 / 18 | 3 | 6 |
| shape-C | 14 / 49 / 29 | 4 | 10 |

The whole-model heart-rate and pressure direction claims failed verification: about half the cases go each
way, as a saddle implies. The drafts therefore state the saddle only through interaction claims. For
example, in shape-C, heart rate from 59 to 93 beats/min takes risk from about 3% to 22% when systolic
pressure is at or below 132 mmHg, but from about 24% down to 1% above it; shape-B gives 10% → 55% and
71% → 5%.

**Sodium minimum.**
- shape-C: the V is placed lowest at about 138.5 mmol/L.
- shape-A: the U is placed lowest at about 134.5 mmol/L. That is the claim generator's typical-case
  estimate; the 1 mmol/L sweep of section 6 has the floor flat over 135–137.

Flags for review include:
- faster heart rate lowering risk at high pressure (B, C);
- fitted noise raising risk (A, C);
- higher BMI lowering risk at low sodium (A);
- high sodium counting more at younger ages (C);
- the male input ignored (C).

## 9. Caveats

- **Synthetic and exploratory.** One data draw, one explained genome per dataset (the best by training
  fitness), not pre-registered. The truth is known, so this tests recovery, not discovery.
- **Interrupted explanations.** The three explanations were resumed after an interruption (section 4).
  The resuming agents worked from the stored state and kept everything. Total wall clock was about
  35 minutes each, inside the 3-hour box.
- **Agent-verified only.** Not signed off; `mike_verification` is absent. The plain-language drafts are
  unverified.
- **Grades are by hand.** The ground-truth grades are assigned by hand from the records, the verdicts, `gt_probe.json`
  and the figures; they follow the ladder's `gt_grades.json` style. A term counts as recovered when the
  record states it in the right direction and shape. Gentle extra structure the truth lacks is listed as
  spurious, not as a miss.
- **Wording errors and the grades.** The verifiers' refutations are of wording, and they bear on the bmi
  grades (partial in A and B). The U and saddle grades stand because the shape statements themselves
  (both ends raise risk; opposing directions) were upheld.
- **The raw product is a poor saddle probe.** `heart_rate * sbp` (raw units) is mostly linear on this
  range. The R² against it understates a saddle part; the centred product and the product with linear
  terms are reported next to it.
- **Location of the U minimum.** The minimum's location depends on the held case. The figures use the
  median / most common value of every other input.

## 10. Reproduce

```bash
cd /Users/mike/dev/explaneat   # or the exp/ushape-saddle worktree
export DATABASE_URL=postgresql://localhost/explaneat_dev
uv run python experiments/ushape_saddle/make_datasets.py && uv run python experiments/ushape_saddle/calibrate.py
experiments/ushape_saddle/launch.sh u saddle both               # 15 runs, 3 lanes
uv run python experiments/ushape_saddle/references.py           # references.json
uv run python experiments/ushape_saddle/compute_runs.py         # runs.csv, summary.json (test AUCs, CIs)
# explanations: EXPLAIN_TASK.md per best-by-train genome (authoring skill v4) -> explanations/<label>.json
# blind verifiers: VERIFY_TASK.md (verifier skill) -> verdicts/<label>/
uv run python experiments/explanation_levels/compute_levels.py --set ushape    # levels/
EXPLANEAT_RECORDS_DIR=experiments/ushape_saddle/explanations EXPLANEAT_LAYERS_DIR=experiments/ushape_saddle/plain_language \
  uv run python -m explaneat plain-language extract --label <label> --dir experiments/ushape_saddle/plain_language
uv run python experiments/ushape_saddle/shape_plots.py <label>  # figures/
uv run python experiments/ushape_saddle/gt_probe.py             # gt_probe.json
uv run python scripts/thesis/thesis_experiment_tables.py        # thesis_exports/ch7/ch7_experiment1_cases.tex
```

## 11. Longer runs on the both dataset (`launch_g300.sh`, `score_g300.py`, `g300_scores.json`)

**Question.** In section 3, only 2 of the 5 shape-C seeds (3 and 4) found both shapes in 100 generations.
With another 100-200 generations, is "both" found more consistently?

**Design.** The five shape-C seeds (0-4) were rerun from scratch to 300 generations with the same
`run_seed.py`, config and dataset (`--generations 300`, experiments `shape-C-s{n}-g300` in
`explaneat_dev`; the original 100-generation runs are untouched). Runs replay exactly: for every seed the
best genome up to generation 100 of the rerun has the same NEAT key, generation and training fitness, bit
for bit, as the original run's best genome (e.g. seed 0: key 24178, generation 98, 0.8556646223493756),
so the generation-100 rows below are the section 3 genomes. Each run took about 7-8 minutes (3 lanes).

At generations 100, 200 and 300 (generation indices up to 99, 199, 299) the best-by-train genome so far
(highest training fitness, earliest generation, lowest key) is scored with no explanation:

- **U:** the model's logit as sodium goes from 128 to 150 mmol/L (1 mmol/L steps, every other input at the
  typical case, as in `shape_plots.py`); Pearson r with the true U term, and whether both ends rise (logit at
  128 and at 150 each more than 1 above the curve's minimum).
- **Saddle:** the logit on a 41 x 41 grid of heart rate x SBP (training 1st-99th percentiles, others typical),
  mean removed; Pearson r with the true -zh*zp surface, and the interaction share (variance left after the
  best additive f(heart rate) + g(SBP) fit; the truth's share is 1).
- **Found** when r >= 0.9 (fixed before scoring). On the 100-generation genomes this threshold gives the
  section 3 reading: U in all five, saddle only in seeds 3 and 4 (seeds 0-2 have saddle r 0.79-0.84).

Train AUC is on the 1,500 training rows, test AUC on the 500 test rows; species count is from the run log.

| seed | gen | best at gen | train AUC | test AUC | train - test | U r | both ends rise | saddle r | interaction share | U | saddle | both | hidden | conns | species |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 98 | 0.856 | 0.853 | +0.003 | 0.98 | yes | 0.84 | 0.75 | yes | no | no | 12 | 31 | 1 |
| 0 | 200 | 197 | 0.869 | 0.851 | +0.018 | 0.98 | yes | 0.82 | 0.77 | yes | no | no | 16 | 40 | 1 |
| 0 | 300 | 298 | 0.882 | 0.865 | +0.017 | 0.98 | yes | 0.88 | 0.85 | yes | no | no | 18 | 42 | 1 |
| 1 | 100 | 99 | 0.856 | 0.840 | +0.016 | 0.99 | yes | 0.79 | 0.76 | yes | no | no | 12 | 35 | 1 |
| 1 | 200 | 195 | 0.896 | 0.869 | +0.027 | 0.99 | yes | 0.96 | 0.97 | yes | yes | yes | 13 | 39 | 1 |
| 1 | 300 | 297 | 0.897 | 0.869 | +0.028 | 1.00 | yes | 0.96 | 0.97 | yes | yes | yes | 14 | 41 | 1 |
| 2 | 100 | 99 | 0.855 | 0.859 | -0.003 | 0.97 | yes | 0.83 | 0.77 | yes | no | no | 11 | 36 | 2 |
| 2 | 200 | 199 | 0.893 | 0.869 | +0.024 | 0.97 | yes | 0.91 | 0.90 | yes | yes | yes | 15 | 41 | 2 |
| 2 | 300 | 298 | 0.894 | 0.872 | +0.023 | 0.96 | yes | 0.93 | 0.93 | yes | yes | yes | 20 | 55 | 2 |
| 3 | 100 | 99 | 0.882 | 0.872 | +0.010 | 0.97 | yes | 0.92 | 0.93 | yes | yes | yes | 11 | 35 | 1 |
| 3 | 200 | 199 | 0.895 | 0.869 | +0.026 | 0.98 | yes | 0.96 | 0.96 | yes | yes | yes | 13 | 40 | 1 |
| 3 | 300 | 298 | 0.898 | 0.872 | +0.027 | 0.98 | yes | 0.95 | 0.94 | yes | yes | yes | 16 | 46 | 1 |
| 4 | 100 | 98 | 0.896 | 0.873 | +0.023 | 0.98 | yes | 0.96 | 0.96 | yes | yes | yes | 8 | 35 | 1 |
| 4 | 200 | 198 | 0.903 | 0.877 | +0.025 | 0.96 | yes | 0.97 | 0.97 | yes | yes | yes | 10 | 41 | 1 |
| 4 | 300 | 283 | 0.903 | 0.876 | +0.028 | 0.96 | yes | 0.98 | 0.97 | yes | yes | yes | 9 | 40 | 1 |

**Answer.** Yes, mostly. Both shapes are found by **2 of 5 seeds at generation 100, 4 of 5 at 200 and 4 of 5
at 300**. Seeds 1 and 2 found the saddle between generations 100 and 200 (saddle r 0.79 -> 0.96 and 0.83 ->
0.91). In each case the jump in training AUC (about +0.04) came with the saddle. The extra 100 generations to 300
added no new seed. Seed 0 is still short at 300 (saddle r 0.88, interaction share 0.85; it is moving
towards the saddle but is below the threshold). The U was found by every seed at every checkpoint, both ends rising by
3.5-10 logit units. No seed lost a shape it had: r stayed at or above 0.9 once reached (lowest later
value 0.91), and U r never fell below 0.96.

**Overfitting check (1,500 training rows).** Test AUC kept rising or held: from generation 100 to 300 it went
from 0.840-0.873 to 0.865-0.876 (median 0.859 -> 0.872), and no seed's test AUC fell by more than 0.003
between checkpoints. The train-test gap widened from -0.003-0.023 to 0.017-0.028. At 300 generations, training AUC
(0.882-0.903) is at or above the oracle's training AUC (0.893), while test AUC (0.865-0.876) stays below the
oracle's 0.884 (oracle gap 0.008). So the longer runs fit some noise, mildly, but the extra generations
mostly bought the saddle rather than overfitting: the generalisation gain from finding it is larger than
the loss. Networks grew modestly (hidden units 8-12 -> 9-20, connections 31-36 -> 40-55); the populations
stayed at one species (two for seed 2) throughout.

**Figures.** `figures/shape-C-s1-g300-sodium.png` and `figures/shape-C-s1-g300-saddle.png` (with
`-shape.json`) show seed 1 at generation 300, a seed that newly found both: U minimum at 135 mmol/L, saddle r
0.96, interaction share 0.97. The genome is not explained, so the parts panels are empty (`shape_plots.py
--genome <id> --task both`).

**Caveats.** Five seeds; one run per seed (a single trajectory per seed, not a distribution). The found
threshold (r >= 0.9) is a cut on a continuous score; seed 0 at 0.88 is close to it. The shape scores read
the model at one held case (the typical patient). These genomes have not been explained or verified.

```bash
experiments/ushape_saddle/launch_g300.sh                       # 5 runs to 300 generations, 3 lanes
uv run python experiments/ushape_saddle/score_g300.py          # g300_scores.json / .csv
uv run python experiments/ushape_saddle/shape_plots.py shape-C-s1-g300 --genome <genome_id> --task both
```

## 12. Sigmoid hidden units on the both dataset (`launch_sig.sh`, `score_gens.py`, `sig_gens.json`)

**Question (Mike).** Would sigmoid hidden units find both shapes more often than ReLU? A smooth unit can
make a U from two sigmoids and a saddle from smooth gates, so it is a fair question.

**Design.** Same dataset (shape-C), config (`configs/syn10.ini`), harness and seeds (0-4) as section 11,
300 generations, with one change: hidden units apply the standard logistic 1/(1 + e^-z) instead of ReLU
(`run_seed.py --hidden-activation sigmoid`, i.e. `config.hidden_activation = "sigmoid"`, recorded on the
experiment as `config_json["training"]["hidden_activation"]`; the output stays sigmoid; this is the
unscaled logistic, not neat-python's sigmoid(5z)). Experiments `shape-C-sig-s{n}` in `explaneat_dev`,
about 9-11 minutes each, at most 2 at a time while the section 11 runs were going. The code default is
unchanged: a 3-generation replay of `shape-C-s0` with the new code gave the same 750 genomes with the
same fitness, bit for bit, and the engines agree on every scored genome (NeuralNeat vs the evidence
forward pass, max |difference| < 1e-15).

Scoring is section 11's, rerun by `score_gens.py` (which reproduces section 11's generation-100 ReLU rows
exactly): best-by-train genome up to generations 99/199/299; U r and both ends rising > 1 logit over
128-150 mmol/L at the typical case; saddle r and interaction share on the 41 x 41 heart rate x SBP grid;
found = r >= 0.9. "U min" is where the model's sodium curve bottoms out (truth 137 mmol/L).

| seed | gen | best at gen | train AUC | test AUC | train - test | U r | both ends rise | U min | saddle r | interaction share | U | saddle | both | hidden | conns | species |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 99 | 0.817 | 0.822 | -0.005 | 0.96 | yes | 137 | 0.57 | 0.35 | yes | no | no | 13 | 42 | 1 |
| 0 | 200 | 199 | 0.852 | 0.848 | +0.003 | 0.96 | yes | 137 | 0.80 | 0.89 | yes | no | no | 19 | 58 | 1 |
| 0 | 300 | 299 | 0.887 | 0.875 | +0.012 | 0.97 | yes | 136 | 0.92 | 0.98 | yes | yes | yes | 19 | 58 | 1 |
| 1 | 100 | 99 | 0.798 | 0.789 | +0.009 | 0.92 | yes | 139 | -0.05 | 0.00 | yes | no | no | 12 | 38 | 1 |
| 1 | 200 | 199 | 0.811 | 0.807 | +0.004 | 0.98 | yes | 138 | 0.62 | 0.49 | yes | no | no | 12 | 42 | 1 |
| 1 | 300 | 299 | 0.853 | 0.850 | +0.004 | 0.98 | yes | 138 | 0.83 | 0.81 | yes | no | no | 12 | 41 | 1 |
| 2 | 100 | 99 | 0.795 | 0.788 | +0.006 | 0.95 | yes | 138 | -0.05 | 0.00 | yes | no | no | 14 | 40 | 2 |
| 2 | 200 | 199 | 0.804 | 0.801 | +0.004 | 0.98 | yes | 138 | -0.05 | 0.00 | yes | no | no | 15 | 42 | 2 |
| 2 | 300 | 299 | 0.808 | 0.805 | +0.002 | 0.98 | yes | 138 | 0.00 | 0.00 | yes | no | no | 16 | 47 | 2 |
| 3 | 100 | 99 | 0.815 | 0.815 | 0.000 | 0.93 | yes | 136 | 0.80 | 0.84 | yes | no | no | 6 | 32 | 1 |
| 3 | 200 | 199 | 0.866 | 0.851 | +0.015 | 0.87 | yes | 136 | 0.87 | 0.96 | no | no | no | 14 | 51 | 1 |
| 3 | 300 | 299 | 0.878 | 0.867 | +0.011 | 0.89 | yes | 136 | 0.87 | 0.96 | no | no | no | 14 | 47 | 1 |
| 4 | 100 | 99 | 0.794 | 0.799 | -0.005 | 0.92 | yes | 136 | 0.18 | 0.07 | yes | no | no | 7 | 32 | 1 |
| 4 | 200 | 199 | 0.834 | 0.831 | +0.004 | 0.91 | yes | 138 | 0.80 | 0.86 | yes | no | no | 13 | 51 | 1 |
| 4 | 300 | 299 | 0.877 | 0.854 | +0.023 | 0.81 | yes | 138 | 0.91 | 0.93 | no | yes | no | 15 | 55 | 1 |

**Answer: no, sigmoid units did worse.** Both shapes were found by **0 of 5 seeds at generation 100, 0 of
5 at 200 and 1 of 5 at 300** (seed 0), against ReLU's 2, 4 and 4 of 5 (section 11).

| hidden units | both found at 100 / 200 / 300 | U found at 100 / 200 / 300 | saddle found at 100 / 200 / 300 | median test AUC at 100 / 200 / 300 |
|---|---|---|---|---|
| ReLU (sections 3 and 11) | 2 / 4 / 4 | 5 / 5 / 5 | 2 / 4 / 4 | 0.859 / 0.869 / 0.872 |
| sigmoid (this section) | 0 / 0 / 1 | 5 / 4 / 3 | 0 / 0 / 2 | 0.799 / 0.831 / 0.854 |

- **U.** Every sigmoid seed had the U at generation 100 (r 0.92-0.96, minimum at 136-139 mmol/L), but
  shallower than ReLU's: the low end rose 1.1-2.3 logit above the minimum (ReLU 3.5-5.0). Two seeds then
  bent it away from the true shape while fitting other terms: seed 3 at 200 and 300 (r 0.87, 0.89) and
  seed 4 at 300 (r 0.81; both ends still rise, the high arm is now the lower one). With ReLU no seed lost the U.
- **Saddle.** No sigmoid seed reached r 0.9 by generation 200; seeds 0 and 4 crossed it only near the end
  (0.92, 0.91 at 300). Seed 2 never used heart rate and SBP together (r 0.00, no interaction) in 300
  generations, and seed 1 reached 0.83. The saddle seed 0 built is bounded: its surface spans about
  ±3 logit where the truth spans ±9, flattening at the corners as saturating units do
  (`figures/shape-C-sig-best-g300-saddle.png`).
- **Speed.** The sigmoid runs learn more slowly throughout: median test AUC at 300 generations (0.854)
  is below ReLU's at 100 (0.859). Seed 0's best genome is from the last generation at every checkpoint,
  so it was still improving at 300; more generations might add seeds, but at the same budget ReLU found
  both shapes in 4 of 5 seeds and sigmoid in 1.

**Overfitting check (1,500 training rows).** None to speak of. Test AUC rose at every checkpoint for every
seed (0.788-0.822 at 100, 0.801-0.851 at 200, 0.805-0.875 at 300), and the train-test gap stayed small:
-0.005 to +0.009 at 100, +0.002 to +0.023 at 300 (ReLU at 300: +0.017 to +0.028). The sigmoid models are
underfitting, not overfitting: training AUC at 300 (0.808-0.887) is below the oracle's training AUC
(0.893) for every seed. Networks were a little larger than ReLU's at 300 (hidden 12-19, connections
41-58; ReLU 9-20, 40-55), one species throughout (two for seed 2).

**Figures.** `figures/shape-C-sig-best-g300-sodium.png` and `figures/shape-C-sig-best-g300-saddle.png`:
seed 0 at generation 300, the best-by-train sigmoid seed (training AUC 0.887; U minimum 136 mmol/L, U r
0.97, saddle r 0.92, interaction share 0.98).

**Caveats.** Five seeds, one trajectory each; the config's mutation and backprop settings were tuned with
ReLU and were not retuned for sigmoid units (5 backprop epochs per generation may suit saturating units
less). The found threshold is a cut on a continuous score (seed 4's U at 0.81, seed 3's saddle at 0.87).
No explanations were run: the explanation tools' hinge readings assume ReLU (see the code note below).

**Code.** `config.hidden_activation` ∈ {relu (default), sigmoid} (`explaneat/core/neuralneat.py`
`set_hidden_activation`), restored by `load_neat_config`; the network structures carry it on hidden nodes
and in `metadata["hidden_activation"]`, so StructureNetwork, ActivationExtractor, AnnotationFunction,
formulas and the verify/plain-language evaluator compute the trained function
(`tests/test_core/test_engine_parity.py`). Not yet adapted, and still assuming ReLU hinges: the
plain-language claim generator and shape classes (hinge / threshold readings), the survival hazard-ratio
readings of hidden units (`survival/explain.py`), linearisation (`query_linearise`,
`linearise_annotations`), `unit_activity` (dead / always-on units), the regime map in the evidence
routes, and the free-text formula checker (`verify/freetext.py`).

```bash
experiments/ushape_saddle/launch_sig.sh                        # 5 sigmoid runs to 300 generations
uv run python experiments/ushape_saddle/score_gens.py --prefix shape-C-sig-s --gens 100 200 300 \
    --out sig_gens.json --log-name "both-sig-s{seed}" --plots shape-C-sig
```
