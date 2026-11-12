# Cox survival case (thesis Experiment 1)

2026-11-12, branch `exp/cox-case`. One PropNEAT-Cox model per public survival dataset
(GBSG2, WHAS500), compared with Cox references on the same split, and the selected network
explained to the three levels with the frozen method (authoring skill v4, strict mode with
provenance, function-preserving operations only), then challenged by a blind verifier.
Public data only (scikit-survival bundled); the research database `explaneat_dev`.

Method background: `docs/survival.md`; the earlier benchmark and protocol history:
`experiments/survival/RESULTS.md`; the three levels: `docs/annotation_coverage_model.md`
("Three levels of explanation") and `experiments/explanation_levels/RESULTS.md`.

## 1. Answer in brief

- **Both selected networks are sparse linear Cox models.** Under the survival configuration
  (identity output, Efron Cox loss, fitness = −loss − k/D, 60 generations) no seed of either
  dataset kept a hidden node. GBSG2 keeps 5 of 9 inputs (every seed converged to the same
  genome); WHAS500 keeps 8–12 of 14 (the selected seed: 9). That is the honest result of the
  frozen configuration, not a choice made on test data.
- **Performance is at parity with Cox and, on WHAS500, slightly ahead of every reference.**
  Test fold, selected network against ridge Cox / unpenalised Cox PH / random survival forest:
  GBSG2 Harrell's C 0.675 vs 0.680 / 0.680 / 0.700 (Uno's C 0.642 vs 0.632 / 0.632 / 0.669,
  IBS 0.114 vs 0.114 / 0.114 / 0.105); WHAS500 C 0.830 vs 0.819 / 0.818 / 0.808 (Uno's C
  0.833 vs 0.823 / 0.820 / 0.815, IBS 0.138 vs 0.149 / 0.141 / 0.150). One split: the
  differences are within single-split noise (`experiments/survival/RESULTS.md` §3.2 gives
  the repeated-split picture, parity).
- **The explanations are complete and verified.** Both: structural and compositional coverage
  1.0 under one root, function preservation 4.4e-16, every stated number a stored query
  (329 and 227 references; recheck 0 mismatches, 0 needs-review), every leaf self-probed.
  GBSG2: 3 leaves (tumour biology, nodal burden, hormone therapy); WHAS500: 4 leaves (age
  and body mass, admission haemodynamics, in-hospital course, index MI type). Every hypothesis
  is written as hazard ratios per clinical unit and carries the proportional-hazards caveat.
- **Three levels.** Faithful: 4.4e-16 on both. Decomposed: 3 and 4 leaves of 1–3 direct
  edges; granularity is undefined (it is a share of hidden units, and there are none).
  Understood: every leaf grades *meaningful* against the tabular concept library (7 of 7),
  but two of the seven are explained by their own input only after a one-sided reading (the
  GBSG2 tumour-biology sum, the WHAS500 haemodynamics sum: best single concepts at score
  0.91 and 0.83 are thresholds on one of their inputs). The explainers call those two
  "described by their formula".
- **Verifier:** see §6 (filled from `verdicts/`).

## 2. Design (fixed before any run; `run_case.py` docstring)

| Item | Setting |
|---|---|
| Data | `load_builtin("gbsg2")`: n = 686, 299 events, 9 model columns (one-hot `horTh`, `menostat`, `tgrade`), time in days. `load_builtin("whas500")`: n = 500, 215 events, 14 model columns, time in days (follow-up to death). |
| Split | Event-stratified 80/20 train/test, seed 42; then 20% of the training part held out as the validation fold (seed 42). GBSG2: 438 / 110 / 138 rows (191 / 48 / 60 events); WHAS500: 320 / 80 / 100 (138 / 34 / 43). Scaler fitted on the 438 / 320 training rows. The same index arrays (`splits/<ds>.splits.json`) for every seed and every reference. |
| Runs | `experiments/survival/run_seed.py`, unchanged: identity output, Efron Cox loss full batch, 5 Adadelta epochs per generation, phantom mask, `configs/survival.ini` (population 150, `weight_init_stdev` 0.1, ReLU hidden), 60 generations, fitness −loss − k/D (per-event AIC, k = enabled connections + hidden nodes). Seeds 0–4, the same budget each (12–13 s per run). |
| Selection | Within a run: the generation champion (by training fitness) with the lowest plain validation Cox loss (protocol v3). Across seeds: the seed with the lowest validation Cox loss. Test metrics were not looked at before selection (`select` reads them into `runs.csv` only after the choice is written to `selected.json`). |
| References | Ridge Cox (scikit-survival, Efron; α tuned on the validation fold by validation Cox loss: 1e-4 on GBSG2, 100 on WHAS500); unpenalised Cox PH (lifelines, Efron) with its Schoenfeld test; random survival forest (scikit-survival, 500 trees, min leaf 10, sqrt features, seed 42). All on the training rows, the same standardised features. |
| Metrics (test fold) | Harrell's C, Uno's C (τ inside both folds' event support), cumulative/dynamic AUC at 1, 3 and 5 years (365 / 1095 / 1825 days), IPCW Brier at the calibration horizon and IBS (25-point grid to the 80th test event percentile), calibration by decile (predicted 1 − S vs Kaplan-Meier observed; E/O ratio and the largest decile gap) at **5 years for GBSG2** (recurrence-free survival) and **3 years for WHAS500** (follow-up ends at 6.5 years and the 100-row test fold is thin by 5 years); test Cox loss. Concordance on g(x) for every model; the forest's risk score for the forest. |
| Explanation | Authoring skill v4, strict mode with provenance, function-preserving operations only (`rename_node`, `add_identity_node`, `annotate`), training split only, 3-hour box, one explaining agent per genome; hazard-ratio readings through the new `query_hazard_ratios` tool; concept probe and `set_meaning` on every leaf; `xn.py --recheck` last; `verify_record.py` writes the record. Then one blind verifier per genome (`explanation-verifier` skill), then `levels.py` (the functions of `experiments/explanation_levels/compute_levels.py`). |

## 3. Per seed (test fold)

The selected seed is in bold. Val / train Cox loss: Efron partial likelihood per event on
that fold; null = the constant model on the training fold (5.532 GBSG2, 5.234 WHAS500).

| Dataset | Seed | Val Cox loss | Train Cox loss | Harrell C | Uno C | td-AUC 1y / 3y / 5y | IBS | E/O (horizon) | Hidden | k | Inputs used | Events / param |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GBSG2 | 0 | 3.9802 | 5.381 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (5 y) | 0 | 5 | 5 | 38.2 |
| GBSG2 | **1** | 3.9802 | 5.381 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (5 y) | 0 | 5 | 5 | 38.2 |
| GBSG2 | 2 | 3.9802 | 5.381 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (5 y) | 0 | 5 | 5 | 38.2 |
| GBSG2 | 3 | 3.9802 | 5.381 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (5 y) | 0 | 5 | 5 | 38.2 |
| GBSG2 | 4 | 3.9802 | 5.381 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (5 y) | 0 | 5 | 5 | 38.2 |
| WHAS500 | **0** | 3.6001 | 4.766 | 0.830 | 0.833 | 0.848 / 0.913 / 0.862 | 0.138 | 0.885 (3 y) | 0 | 9 | 9 | 15.3 |
| WHAS500 | 1 | 3.6383 | 4.739 | 0.826 | 0.830 | 0.837 / 0.903 / 0.862 | 0.137 | 0.885 (3 y) | 0 | 8 | 8 | 17.3 |
| WHAS500 | 2 | 3.6262 | 4.764 | 0.810 | 0.816 | 0.814 / 0.887 / 0.830 | 0.149 | 0.874 (3 y) | 0 | 10 | 10 | 13.8 |
| WHAS500 | 3 | 3.6277 | 4.800 | 0.823 | 0.825 | 0.831 / 0.901 / 0.830 | 0.145 | 0.872 (3 y) | 0 | 12 | 12 | 11.5 |
| WHAS500 | 4 | 3.6391 | 4.800 | 0.826 | 0.832 | 0.850 / 0.907 / 0.837 | 0.143 | 0.866 (3 y) | 0 | 9 | 9 | 15.3 |

The five GBSG2 seeds end at the same genome (the same five edges; the validation losses
differ in the sixth decimal, so seed 1 is "selected" by a tie-break with no content). The
selected GBSG2 champion is generation 30's; the selected WHAS500 champion is generation 5's
(its 60-generation trace never beat it on validation). Source: `runs.csv`, `runs/*.json`.

## 4. References on the same split (test fold)

| Dataset | Model | Params | Harrell C | Uno C | td-AUC 1y / 3y / 5y | IBS | Brier at horizon | E/O (horizon) | Max decile gap | Test Cox loss |
|---|---|---|---|---|---|---|---|---|---|---|
| GBSG2 | PropNEAT-Cox (seed 1) | 5 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.240 | 0.966 (5 y) | 0.211 | 4.196 |
| GBSG2 | Ridge Cox (α = 1e-4) | 9 | 0.680 | 0.632 | 0.753 / 0.756 / 0.685 | 0.114 | 0.245 | 0.956 (5 y) | 0.338 | 4.205 |
| GBSG2 | Cox PH, unpenalised | 9 | 0.680 | 0.632 | 0.753 / 0.756 / 0.685 | 0.114 | 0.245 | 0.956 (5 y) | 0.338 | 4.205 |
| GBSG2 | Random survival forest | 500 trees | 0.700 | 0.669 | 0.720 / 0.751 / 0.752 | 0.105 | 0.222 | 0.920 (5 y) | 0.302 | — |
| WHAS500 | PropNEAT-Cox (seed 0) | 9 | 0.830 | 0.833 | 0.848 / 0.913 / 0.862 | 0.138 | 0.146 | 0.885 (3 y) | 0.453 | 3.588 |
| WHAS500 | Ridge Cox (α = 100) | 14 | 0.819 | 0.823 | 0.829 / 0.900 / 0.853 | 0.149 | 0.163 | 0.908 (3 y) | 0.324 | 3.715 |
| WHAS500 | Cox PH, unpenalised | 14 | 0.818 | 0.820 | 0.824 / 0.893 / 0.852 | 0.141 | 0.147 | 0.888 (3 y) | 0.414 | 3.623 |
| WHAS500 | Random survival forest | 500 trees | 0.808 | 0.815 | 0.833 / 0.883 / 0.866 | 0.150 | 0.159 | 0.885 (3 y) | 0.362 | — |

Validation Cox loss (the selection criterion, for the record): GBSG2 network 3.980, ridge and
plain Cox 3.992; WHAS500 network 3.600, ridge Cox 3.626, plain Cox 3.655 (the forest has no
partial likelihood; its validation Harrell's C is 0.697 and 0.720). On GBSG2 the forest leads
every discrimination and Brier number; on WHAS500 the network does. Calibration: all models
under-predict risk at the horizon on both test folds (E/O 0.87–0.97); the WHAS500 deciles are
noisy (10 patients each, one decile gap of 0.45 for the network, 0.32–0.41 for the others).

**Proportional hazards on the references (Schoenfeld, rank transform, training rows, lifelines
Cox PH).** GBSG2: `tgrade=III` p = 0.013, the other eight p ≥ 0.25. WHAS500: `sho=1`
p = 0.002, `mitype=1` 0.015, `cvd=1` 0.017, `los` 0.043; the rest p ≥ 0.17. So the
time-constant reading of grade III, shock, MI type and length of stay is the model's
assumption, not what the data say (`references.json`, `models.cox_ph.schoenfeld`).

Source: `references.py`, `references.json`, `references.csv`.

## 5. The explanations

Both records: `explanations/<label>.json` (the explaining agent's report with the
orchestrator's independent re-check: coverage, hierarchy, function preservation, closed-form
error, recheck). Levels: `levels/<label>.json`. Hazard-ratio readings of the raw phenotype
(script output, for this report): `hazard_readings/<tag>.json`; the explanations quote the
same quantities from their stored `query_hazard_ratios` records.

### 5.1 GBSG2 (`cox-gbsg2-s1`, genome `e4c62516`)

Closed form of the log-risk (standardised inputs; the constant is absorbed by the baseline):

g = 0.345·[tgrade=III] + 0.321·[tgrade=II] − 0.316·progrec + 0.293·pnodes − 0.206·[horTh=yes] + const

Refactoring: 9 input renames, 3 identity nodes on the output (one per leaf); no splits
(nothing to split). Leaves and root (structural and compositional coverage 1.0, hierarchy
complete, function preservation 4.4e-16, closed form at 3 s.f. off by at most 0.0035):

| Part | Inputs | What the model does (hazard ratios per clinical unit, other inputs fixed; training rows) | Understood |
|---|---|---|---|
| `tumour_biology` | tgrade=II, tgrade=III, progrec | hazard ×1.96 for grade II and ×2.26 for grade III against grade I; ×0.9984 per fmol/l of progesterone receptor (0.73 per SD; p95-vs-p5 patient 1.9) | meaningful, score 0.91 (best library concept `progrec > 29.5`, AUC 0.95, R² 0.30: the direction, not the value; "described by its three-term formula") |
| `nodal_burden` | pnodes | ×1.051 per positive lymph node, log-linear over the training range (p95 vs p5: 2.1) | meaningful, 1.00 (`pnodes`) |
| `hormone_therapy` | horTh=yes | ×0.653 for treated patients; an association in this cohort, not a treatment effect | meaningful, 1.00 (`horTh=yes`) |
| `cox_log_risk` (root) | the three exits; age, estrec, menostat=Post, tsize unconnected | the sum of the three parts and a constant; fixing the four unconnected inputs changes 0 of 438 training predictions; log-risk span p95 vs p5: HR 5.8 | — |

Training-split survival performance of the explained model: Harrell's C 0.682, Uno's C 0.651,
td-AUC 0.765 / 0.731 / 0.707 at 1 / 3 / 5 y, IBS 0.132, five-year calibration 50% predicted
vs 49% observed. These are the directions of the GBSG2 literature (nodes and grade raise the
hazard; progesterone receptor and tamoxifen lower it) and of the Cox PH reference (per-SD HR
1.34 nodes, 1.37 / 1.38 grade II / III, 0.73 progrec, 0.83 therapy; age 0.81, p = 0.085, is
the one reference term the network dropped that is not clearly null).

Explaining time 18 min. Self-probes: constructed grade I / II / III patients, therapy yes/no
cells, pnodes and progrec sweeps, age / tsize / estrec sweeps (no change), the all-parts
ablation (AUC against the event indicator 0.50). 329 numbers, all provenance-backed, 36
queries; recheck 0 / 0.

### 5.2 WHAS500 (`cox-whas500-s0`, genome `f96ebc8d`)

g = 0.372·age + 0.333·[chf=1] + 0.286·hr − 0.242·bmi − 0.219·[mitype=1] − 0.199·diasbp + 0.121·[sho=1] + 0.042·los − 0.029·sysbp + const

Refactoring: 14 renames, 4 identity nodes on the output; no splits. Coverage 1.0 / 1.0,
hierarchy complete, function preservation 4.4e-16, closed form off by at most 0.0014:

| Part | Inputs | What the model does (HR per clinical unit, other inputs fixed; training rows) | Understood |
|---|---|---|---|
| `age_and_body_mass` | age, bmi | ×1.026 per year of age (p95 vs p5: 3.2); ×0.957 per kg/m² of BMI (2.3) | meaningful, 0.93 (`age > 72`; age alone R² 0.82) |
| `admission_haemodynamics` | hr, diasbp, sysbp | ×1.0121 per beat/min (2.4); ×0.9908 per mm Hg diastolic (1.8); ×0.9991 per mm Hg systolic (1.1, near null) | meaningful by the library score 0.83 (`hr > 85`), but the explainer grades it partial in substance: hr alone R² 0.62, shock index 0.73; "hr − diasbp" R² 0.99 is the closed form restated |
| `in_hospital_course` | chf=1, sho=1, los | ×2.08 for congestive heart complications; ×1.94 for cardiogenic shock (fitted to 11 training patients); ×1.010 per day of stay | meaningful, 1.00 (`chf=1 and sho=1`, 7 rows; the explainer stored `chf=1`, score 0.99) |
| `index_mi_type` | mitype=1 | ×0.617 for a Q-wave against a non-Q-wave infarction; "described, not explained" (one input rescaled) | meaningful, 1.00 (`mitype=1`) |
| `cox_log_risk` (root) | the four exits; afb, av3, cvd, gender, miord unconnected | the sum of the four parts and a constant; log-risk span p95 vs p5: HR 27 | — |

Training-split performance: Harrell's C 0.770, td-AUC at 3 y 0.854. Ablation (AUC against
the event indicator, training rows): all parts 0.833; without age_and_body_mass 0.781,
without in_hospital_course 0.807, without admission_haemodynamics 0.822, without
index_mi_type 0.832. The Cox PH reference agrees in direction on every kept term (per-SD HR
age 1.92, chf 1.50, hr 1.32, bmi 0.81, mitype 0.72, diasbp 0.92, sho 1.14, los 1.06, sysbp
0.94); of the five dropped inputs, female sex (HR 0.82, p = 0.027) is the one the reference
finds at 0.05.

Explaining time 21 min. 227 numbers (168 provenance-backed, 59 allow-listed literals such
as node ids and probe settings), 42 references; recheck 0 / 0 after one rerun (two queries
computed before the identity-node operations were flagged and rerun with identical values).

## 6. The three levels and the verifier

| Label | Faithful: max abs err | Leaves | Granularity | Understood: meaningful / partial / closed-form only | Recheck | Verifier |
|---|---|---|---|---|---|---|
| cox-gbsg2-s1 | 4.4e-16 | 3 | n/a (0 hidden units) | 3 / 0 / 0 | 0 mismatch, 0 needs review | VERIFIER_GBSG2 |
| cox-whas500-s0 | 4.4e-16 | 4 | n/a (0 hidden units) | 4 / 0 / 0 | 0 mismatch, 0 needs review | VERIFIER_WHAS500 |

The probe grades are the library's (score = max(R², 2(AUC − 0.5)) of the best concept against
each leaf's exit). For a leaf of one direct edge the best concept is that input (score 1.0),
which is a description of the weight, not a finding; for the two multi-input sums the best
single concept is a threshold on the dominant input (0.91, 0.83), which the explainers
qualify as direction-only. Read the "understood" column with that in mind: these are linear
models whose parts are their inputs.

VERIFIER_SECTION

## 7. Caveats

- **No nonlinearity to explain.** Every seed of both datasets is linear. The explanations
  exercise the whole method (refactoring, provenance, self-probes, levels, verifier) on Cox
  models, but not hinges or splits; `experiments/survival/RESULTS.md` §5.2 reads hinges on a
  simulation with a known non-linear truth. A longer run or a weaker parsimony penalty was not
  tried because the configuration was fixed in advance and test numbers were already seen.
- **One split.** Paired differences against the references are single-split numbers (test
  folds of 138 and 100 rows); the repeated-split benchmark is the place for parity claims.
- **GBSG2 and WHAS500 test folds are not new to the method**: the same datasets were used
  under protocols v1–v3 (different splits).
- **Proportional hazards** is inherited; the Schoenfeld tests above flag grade III (GBSG2)
  and shock, MI type, CVD history and length of stay (WHAS500) on the reference model.
- **Granularity is undefined** for a network without hidden units; the "decomposed" level is
  then the leaf count and the fact that every leaf is 1–3 direct edges.
- **Tool notes from the explainers** (`tool_bugs` in the records): `query_probe` cannot set
  one-hot inputs whose names contain "=" (worked around with `hold="where:..."`);
  `query_enumerate` refuses a model with a continuous input; `query_ablate(direct=true)` finds
  no edges after `add_identity_node` (use `edges=[...->identity_*]`); `add_evidence` in strict
  mode refuses node ids in payload lists; `get_formula` has no provenance id; a probe-library
  sweep can pick a 7-row conjunction (`chf=1 and sho=1`) by AUC; duplicate evidence records
  cannot be deleted (amended as superseded instead). None blocked the explanations.
- **One test-split query sits in each genome's log** (`q_c3e50aeb46`, `q_69e548c451`): the
  loading script's round-trip check (`load_selected.py` calls `compute_performance` on the
  test split to confirm the DB-rebuilt genome reproduces the run's test C). The explainers
  saw it, did not quote it, and recorded it; it predates their first call.
- **Plain-language draft skipped.** The claim generator's `load_model` builds its model with
  the sigmoid risk scale and a flat 0/1 target, so a survival genome would need the
  horizon-risk scale and the event indicator wired in (as done for the verifier model in this
  branch) plus survival wording in the templates; that is a template change, so no draft.
- **Library changes made for this case** (tests added): the verifier model reduces a survival
  `[N, 2]` target to its event indicator, keeps the times, and gives survival genomes the
  horizon risk scale (`1 − S0(h)^exp(g)`, Breslow on the training rows) in place of a sigmoid;
  `output_terms` expands identity relays on the output; `query_hazard_ratios` is a
  provenance-stamped query tool. Classification genomes are unaffected.

## 8. Reproduce

```bash
uv sync --extra survival --extra dev
PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/run_case.py split
PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/run_case.py runs --jobs 2
uv run python experiments/survival/cox_case/run_case.py select
PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/references.py
uv run python experiments/survival/cox_case/load_selected.py           # explaneat_dev; writes ids.json
# one explaining agent per genome (authoring skill v4), then:
uv run python experiments/10seed/verify_record.py report_<label>.json --out-dir experiments/survival/cox_case/explanations
# one verifier agent per genome -> verdicts/<label>/
uv run python experiments/survival/cox_case/levels.py
uv run python experiments/survival/cox_case/tables.py                  # tables.md, thesis_exports/ch7/ch7_cox_case_table.tex
```
