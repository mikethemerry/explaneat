# Reference explanation for the Cox survival case

2026-11-15, branch `exp/cox-reference-explanation`. What the comparators of the Cox case
(`RESULTS.md` §4) say about the same data, in the units the network explanation uses, so
the explained PropNEAT-Cox networks (`cox-gbsg2-s1`, `cox-whas500-s0`) can be read against a
reference. The template for Chapter 8 (the network against the published PREDICT equation on
VARIANZ).

Script: `reference_explanation.py` (no database: the scikit-survival bundled data, the case's
`splits/`, `references.json`, `hazard_readings/`). Output: `reference_explanation.json`,
`figures/ref_*.png`, `thesis_exports/ch7/ch7_cox_reference_table.tex`. Nothing in
`RESULTS.md`, `references.*` or the explanation records was changed.

```bash
PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/reference_explanation.py   # ~2 min
```

## 1. Answer in brief

- **The network and Cox PH agree, input by input.** Every kept input has the Cox direction, and
  on GBSG2 every one of the network's five hazard ratios is inside the Cox 95% CI. On WHAS500 eight
  of nine are inside; age is the exception (network 1.026 per year, Cox 1.046, CI 1.029–1.064:
  same direction, about half the log-effect). Of the inputs the network dropped, Cox finds one at p < 0.05:
  WHAS500 female sex (0.666, CI 0.464–0.955, p = 0.027); the next is GBSG2 age (0.979 per year,
  p = 0.085).
- **The forest's GBSG2 edge comes from non-linear main effects, mainly age and nodes.** Not from an
  interaction (Friedman's H² ≤ 0.044 for every pair of its top five inputs; the strongest pair's
  product term in a Cox model: LR p = 0.74) and not from the non-proportional grade III effect
  (the forest does read it, fading from 6 months to about zero by 4.5 years, but letting Cox
  model it by stratification does not improve the test fold). Adding a spline for age and one for
  positive nodes to Cox PH (LR 45.7 on 8 df, p = 3e-7, training rows) brings the test fold from
  C 0.680 / Uno 0.632 / Brier at 5 y 0.245 to 0.694 / 0.666 / 0.220, against the forest's 0.704
  / 0.686 / 0.222. Age is U-shaped (risk highest under 40, lowest at 45–50); nodes rise steeply
  from 3 to 5 and keep rising to 12, then flatten; progesterone receptor flattens above about
  150 fmol/l.
- **Implication for the network.** The network dropped age because a straight line through a U
  is nearly flat (Cox per-year HR 0.979, p = 0.085); a hidden ReLU pair could have represented
  the young-age hinge, but no seed kept a hidden node under the case's parsimony penalty
  (`RESULTS.md` §7, "No nonlinearity to explain").
- **On WHAS500 the forest's reading is thresholds, not interactions** (age flat to 67 then a
  step; heart rate flat to 82 then rising to a plateau above 110; BMI protective only below 25;
  H² ≤ 0.008). With 320 training rows the forest is the weakest model on the test fold (C 0.808
  vs the network's 0.830).

## 2. Method

| Item | How |
|---|---|
| Rows | The case's training rows only (GBSG2 438, 191 events; WHAS500 320, 138 events); test fold only for the §4 scores |
| Cox PH | lifelines `CoxPHFitter(penalizer=0)`, Efron ties, on the standardised training rows: the `references.py` fit (coefficients asserted equal to `references.json`) |
| Per unit | **Converted from the standardised fit**: log HR per unit = coefficient / training SD, CI ends the same way, p unchanged. A refit on raw units gives the same coefficients (largest difference 7.8e-16 GBSG2, 6.7e-16 WHAS500) |
| PH test | Schoenfeld, rank transform, training rows: read from `references.json` (`models.cox_ph.schoenfeld`) |
| Network | `hazard_readings/<tag>.json` (`hr_per_unit`, `hr_per_sd`: the explanation's readings, not retyped). Rebuilding the log-risk from those readings reproduces the explanation's training Harrell C (0.682 GBSG2, 0.770 WHAS500) |
| Agreement | within: same direction, network HR inside the Cox 95% CI; outside: same direction, outside the CI; opposite: opposite sign; dropped: input not used by the network |
| Units | per year, per fmol/l, per node, per mm, per mmHg, per beat/min, per kg/m², per day; binary inputs as the named contrast (yes vs no, grade II / III vs I, female vs male, Q-wave vs non-Q-wave) |
| Forest | `references.py`'s random survival forest refitted with the same parameters (test Harrell C asserted equal: 0.700, 0.808). Permutation importance: training rows, Harrell C of the risk score, 10 repeats. PD: log cumulative hazard at the horizon (5 y GBSG2, 3 y WHAS500), averaged over the training rows, centred at the training median (binary: at 0), on which a Cox model draws a straight line. Friedman's H² on the risk at the horizon, 120 training rows (seed 0), all pairs among the top five |
| Forest edge | Cox variants on the training rows: a cubic B-spline (4 quantile knots, 4 extra df) per continuous input, one at a time and together; the forest's strongest pair as a product term; grade III as a stratum (no PH for grade III). LR tests on the training rows where nested; test-fold Harrell C and Uno C of the 5-year risk, IBS and Brier at 5 years from each model's own survival curves |

## 3. Side by side

### 3.1 GBSG2 (recurrence-free survival; network `cox-gbsg2-s1`)

Per clinical unit:

| Input | Unit | Cox PH HR per unit (95% CI) | p | PH test p | Network HR per unit | Network | Agreement |
|---|---|---|---|---|---|---|---|
| Age | per year | 0.9790 (0.9556–1.0029) | 0.085 | 0.249 | -- | dropped | dropped |
| Oestrogen receptor | per fmol/l | 1.0001 (0.9991–1.0011) | 0.857 | 0.382 | -- | dropped | dropped |
| Hormone therapy | yes vs no | 0.677 (0.494–0.926) | 0.015 | 0.927 | 0.653 | kept | within |
| Postmenopausal | post vs pre | 1.320 (0.828–2.106) | 0.244 | 0.568 | -- | dropped | dropped |
| Positive lymph nodes | per node | 1.0507 (1.0323–1.0694) | <0.001 | 0.960 | 1.0513 | kept | within |
| Progesterone receptor | per fmol/l | 0.9984 (0.9972–0.9997) | 0.013 | 0.501 | 0.9984 | kept | within |
| Tumour grade II | II vs I | 1.932 (1.038–3.597) | 0.038 | 0.336 | 1.960 | kept | within |
| Tumour grade III | III vs I | 2.158 (1.107–4.207) | 0.024 | 0.013 | 2.264 | kept | within |
| Tumour size | per mm | 1.0028 (0.9920–1.0137) | 0.617 | 0.657 | -- | dropped | dropped |

Per training SD (the scale of `RESULTS.md` §5):

| Input | Unit | Cox PH HR per SD (95% CI) | p | PH test p | Network HR per SD | Network | Agreement |
|---|---|---|---|---|---|---|---|
| Age | SD = 10.2 | 0.805 (0.629–1.030) | 0.085 | 0.249 | -- | dropped | dropped |
| Oestrogen receptor | SD = 164 | 1.016 (0.856–1.205) | 0.857 | 0.382 | -- | dropped | dropped |
| Hormone therapy | SD = 0.485 | 0.828 (0.711–0.964) | 0.015 | 0.927 | 0.814 | kept | within |
| Postmenopausal | SD = 0.495 | 1.147 (0.911–1.445) | 0.244 | 0.568 | -- | dropped | dropped |
| Positive lymph nodes | SD = 5.85 | 1.335 (1.204–1.481) | <0.001 | 0.960 | 1.340 | kept | within |
| Progesterone receptor | SD = 195 | 0.733 (0.573–0.938) | 0.013 | 0.501 | 0.729 | kept | within |
| Tumour grade II | SD = 0.477 | 1.369 (1.018–1.841) | 0.038 | 0.336 | 1.378 | kept | within |
| Tumour grade III | SD = 0.423 | 1.384 (1.044–1.835) | 0.024 | 0.013 | 1.412 | kept | within |
| Tumour size | SD = 13.7 | 1.039 (0.895–1.206) | 0.617 | 0.657 | -- | dropped | dropped |

The network reproduces the Cox model on its five inputs (within 5% of the Cox hazard ratio on
every one). Grade III fails the PH test (p = 0.013): its single hazard ratio is an average over
follow-up (§4.2).

### 3.2 WHAS500 (all-cause death; network `cox-whas500-s0`)

Per clinical unit:

| Input | Unit | Cox PH HR per unit (95% CI) | p | PH test p | Network HR per unit | Network | Agreement |
|---|---|---|---|---|---|---|---|
| Atrial fibrillation | yes vs no | 1.250 (0.800–1.954) | 0.328 | 0.763 | -- | dropped | dropped |
| Age | per year | 1.0462 (1.0285–1.0643) | <0.001 | 0.483 | 1.0261 | kept | outside |
| Complete heart block | yes vs no | 1.216 (0.327–4.524) | 0.770 | 0.577 | -- | dropped | dropped |
| Body mass index | per kg/m² | 0.9632 (0.9240–1.0040) | 0.077 | 0.412 | 0.9574 | kept | within |
| Congestive heart complications | yes vs no | 2.447 (1.663–3.600) | <0.001 | 0.617 | 2.084 | kept | within |
| History of CVD | yes vs no | 0.830 (0.536–1.283) | 0.401 | 0.017 | -- | dropped | dropped |
| Diastolic BP | per mmHg | 0.9960 (0.9814–1.0107) | 0.591 | 0.944 | 0.9908 | kept | within |
| Female sex | female vs male | 0.666 (0.464–0.955) | 0.027 | 0.881 | -- | dropped | dropped |
| Heart rate | per beat/min | 1.0119 (1.0038–1.0201) | 0.004 | 0.495 | 1.0121 | kept | within |
| Length of stay | per day | 1.0128 (0.9750–1.0520) | 0.513 | 0.043 | 1.0100 | kept | within |
| Recurrent MI | recurrent vs first | 0.986 (0.691–1.406) | 0.938 | 0.965 | -- | dropped | dropped |
| Q-wave MI | Q-wave vs non-Q | 0.484 (0.285–0.820) | 0.007 | 0.015 | 0.617 | kept | within |
| Cardiogenic shock | yes vs no | 2.052 (0.812–5.185) | 0.129 | 0.002 | 1.943 | kept | within |
| Systolic BP | per mmHg | 0.9981 (0.9896–1.0067) | 0.663 | 0.173 | 0.9991 | kept | within |

Per training SD:

| Input | Unit | Cox PH HR per SD (95% CI) | p | PH test p | Network HR per SD | Network | Agreement |
|---|---|---|---|---|---|---|---|
| Atrial fibrillation | SD = 0.351 | 1.081 (0.925–1.265) | 0.328 | 0.763 | -- | dropped | dropped |
| Age | SD = 14.4 | 1.920 (1.500–2.457) | <0.001 | 0.483 | 1.450 | kept | outside |
| Complete heart block | SD = 0.136 | 1.027 (0.859–1.227) | 0.770 | 0.577 | -- | dropped | dropped |
| Body mass index | SD = 5.55 | 0.812 (0.645–1.023) | 0.077 | 0.412 | 0.785 | kept | within |
| Congestive heart complications | SD = 0.454 | 1.501 (1.260–1.789) | <0.001 | 0.617 | 1.396 | kept | within |
| History of CVD | SD = 0.424 | 0.924 (0.768–1.111) | 0.401 | 0.017 | -- | dropped | dropped |
| Diastolic BP | SD = 21.5 | 0.917 (0.668–1.259) | 0.591 | 0.944 | 0.820 | kept | within |
| Female sex | SD = 0.492 | 0.819 (0.686–0.978) | 0.027 | 0.881 | -- | dropped | dropped |
| Heart rate | SD = 23.7 | 1.324 (1.094–1.602) | 0.004 | 0.495 | 1.330 | kept | within |
| Length of stay | SD = 4.18 | 1.055 (0.900–1.236) | 0.513 | 0.043 | 1.043 | kept | within |
| Recurrent MI | SD = 0.485 | 0.993 (0.836–1.180) | 0.938 | 0.965 | -- | dropped | dropped |
| Q-wave MI | SD = 0.453 | 0.720 (0.567–0.914) | 0.007 | 0.015 | 0.804 | kept | within |
| Cardiogenic shock | SD = 0.182 | 1.140 (0.963–1.350) | 0.129 | 0.002 | 1.129 | kept | within |
| Systolic BP | SD = 32.5 | 0.940 (0.712–1.241) | 0.663 | 0.173 | 0.972 | kept | within |

The network's terms are shrunk toward 1 against Cox on six of nine inputs (it is the generation-5
champion selected by validation loss, `RESULTS.md` §3), most for age (outside the CI). Four
inputs fail the PH test: shock (p = 0.002), Q-wave MI (0.015), CVD history (0.017, dropped) and
length of stay (0.043); the network keeps three of them as constant hazard ratios.

Figures: `figures/ref_forest_gbsg2.png`, `figures/ref_forest_whas500.png` (Cox HR with CI,
network HR as a diamond, dropped inputs as a cross at 1; per unit left, per SD right).

## 4. The forest's reading

### 4.1 Importance, shape and interaction

| Dataset | Top inputs (permutation importance, drop in training Harrell C) | Strongest pair (H²) |
|---|---|---|
| GBSG2 | pnodes 0.114, progrec 0.063, age 0.048, estrec 0.033, tsize 0.022 | pnodes × age 0.016 (largest absolute residual); progrec × age has the largest H², 0.044 |
| WHAS500 | age 0.104, chf 0.047, hr 0.029, bmi 0.023, sysbp 0.012 | chf × hr 0.008 |

Partial dependence (`figures/ref_pd_<ds>.png`, forest against the Cox and network lines on the
log cumulative hazard; 5-year / 3-year risk values in `reference_explanation.json`):

- **GBSG2.** Nodes: flat 1–3, steep 3–5, rising to 12, then flat (5-year risk 0.41 at 1–3 nodes,
  0.49 at 4, 0.64 at 10, 0.72 at 16); steeper than the Cox line at 4–12 nodes. Progesterone
  receptor: high risk below about 25 fmol/l (0.54–0.58), falling to about 0.42 by 120 fmol/l and
  flat after; the Cox and network lines keep falling to 400. Age: U-shaped, 5-year risk 0.59 at 36,
  minimum 0.44 at 47, 0.51 at 58–64; the Cox line falls with age, the network has no age term.
  Oestrogen receptor: small and irregular (risk 0.46–0.53), as Cox says (p = 0.86).
- **WHAS500.** Age: flat to 67 (3-year risk 0.26), a step to 0.43 by 74 and 0.53 by 87.
  Heart rate: flat to 82 (0.31), rising to 0.47 by 111. BMI: protective only below about 25
  (0.49 at 19 to 0.34 at 25, flat after). Congestive heart complications: 0.32 to 0.51.

The interactions are small on both datasets (2-D PD of the strongest pair:
`figures/ref_h2d_<ds>.png`; the GBSG2 heatmap is two main effects added: the young-age band and
the node gradient).

### 4.2 Grade III over time (GBSG2)

The forest's grade III vs not contrast (log cumulative hazard ratio, averaged over the training
rows) is 0.33 at 6 months, 0.18 at 1 year, 0.04 at 3 years and about 0 from 4.5 years
(`figures/ref_grade3_time_gbsg2.png`). Cox PH (log HR 0.77, grade III vs I) and the network
(0.82) hold it constant by assumption. This is the time-varying effect the Schoenfeld test flags;
the forest's level is lower throughout because a forest dilutes a binary input that it splits on
rarely (importance 0.005).

### 4.3 Where the GBSG2 advantage comes from

Test fold (138 rows, 60 events); the forest scored on its 5-year risk so every row is on the same
footing (its risk-score C is 0.700, `RESULTS.md` §4):

| Model (training rows) | LR vs linear (df, p) | Harrell C | Uno C | IBS | Brier at 5 y |
|---|---|---|---|---|---|
| Cox PH, linear | — | 0.680 | 0.632 | 0.1139 | 0.2450 |
| + spline age | 16.7 (4, 0.002) | 0.706 | 0.660 | 0.1123 | 0.2252 |
| + spline pnodes | 26.6 (4, 2e-5) | 0.671 | 0.644 | 0.1133 | 0.2385 |
| + spline progrec | 9.6 (4, 0.048) | 0.681 | 0.626 | 0.1116 | 0.2432 |
| + spline estrec | 2.9 (4, 0.58) | 0.663 | 0.617 | 0.1153 | 0.2517 |
| + spline tsize | 2.9 (4, 0.57) | 0.688 | 0.646 | 0.1131 | 0.2361 |
| + splines age and pnodes | 45.7 (8, 3e-7) | 0.694 | 0.666 | 0.1108 | 0.2199 |
| + splines, all five continuous | 63.1 (20, 2e-6) | 0.684 | 0.657 | 0.1083 | 0.2180 |
| + product pnodes × age | 0.1 (1, 0.74) | 0.681 | 0.629 | 0.1135 | 0.2457 |
| grade III stratified (no PH) | not nested | 0.664 | 0.621 | 0.1126 | 0.2515 |
| splines + product + grade III stratified | not nested | 0.661 | 0.648 | 0.1166 | 0.2561 |
| Random survival forest | — | 0.704 | 0.686 | 0.1054 | 0.2220 |

Reading: the training likelihood says age and nodes are non-linear (and progesterone receptor
marginally); the product term adds nothing; freeing grade III's PH assumption does not help the
test fold. With the age and nodes splines Cox PH closes most of the gap (Brier at 5 y equal to the
forest's, Uno C 0.666 vs 0.686, Harrell C 0.694 vs 0.704). So the forest's GBSG2 advantage is
**non-linearity in main effects** (age's young-patient risk and the shape of the node count), not
an interaction or the time-varying grade effect. One split, 60 test events: the C differences
between the spline variants are within noise; the LR tests on the training rows are the firmer
evidence.

## 5. For Chapter 8 (template)

The same three steps against PREDICT on VARIANZ: (1) the side-by-side table per clinical unit with
the reference's published hazard ratios (and CIs where published) in place of the refitted Cox,
the PH test where individual data allow it, and the network's readings from its explanation
record; (2) a flexible reference (forest) read by permutation importance, PD on the log
cumulative hazard and H²; (3) the gap attributed by adding one capability at a time to the Cox
reference (splines, a product term, a stratum), scored by training LR and test metrics.

## 6. Caveats

- Per-unit hazard ratios of continuous inputs are close to 1 by construction; read the per-SD
  table for size and the per-unit table for the clinical statement.
- Permutation importance on the training rows is in-sample for the forest (it saw those rows); it
  ranks inputs, it does not measure generalisation.
- The forest's PD averages over the training rows, including impossible combinations (grade III
  set on a grade II row in §4.2).
- Single split; the repeated-split comparison is `experiments/survival/RESULTS.md` §3.2.
