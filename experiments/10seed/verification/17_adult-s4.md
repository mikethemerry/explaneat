# Verification packet 17 · adult-s4

Pre-registered verification order: 17 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/c32e8fc8-410f-4126-ab7e-974c658ef3fc?experiment=b0915d36-d105-4e59-80ae-c5f209ab8f02&name=10seed-adult-s4), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 4 |
| Genome id | `c32e8fc8-410f-4126-ab7e-974c658ef3fc` |
| Experiment | 10seed-adult-s4 (b0915d36-d105-4e59-80ae-c5f209ab8f02); best at generation 99 of 100, NEAT key 20680 |
| Test AUC (runs csv) | 0.907 \[95% CI 0.901, 0.914\] |
| Train AUC | 0.91 |
| Size | 55 of 107 inputs connected; 15 hidden nodes; 81 enabled connections; depth 5; 13 skip connections + 48 direct input→output edges |
| Annotations | 9 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 15 of 15 |
| Claude's time | 45.2 min (2026-11-05T11:36:07+13:00 → 2026-11-05T12:21:22+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Almost a sparse logistic regression on 55 of 107 inputs (direct edges carry logit SD 3.37 of 3.85; mean-ablating every hidden unit: acc 0.856 -\> 0.838, AUC 0.910 -\> 0.895). One consequential nonlinearity: an age hinge turning the age slope into a U-shape (high-income share peaks mid-career; keeping only that unit recovers AUC 0.904). Remaining hidden units are affine relays, category lookups, or dead structure.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit_root`

|  |  |
|---|---|
| Annotation id | `ann_124` |
| Entry nodes | -2 (workclass:0), -3 (workclass:1), -4 (workclass:2), -8 (workclass:6), -11 (fnlwgt), -12 (education:0), -16 (education:12), -22 (education:4), -23 (education:5), -25 (education:7), -27 (education:9), -28 (education-num), -30 (marital-status:1), -31 (marital-status:2), -32 (marital-status:3), -39 (occupation:11), -41 (occupation:13), -42 (occupation:14), -44 (occupation:3), -45 (occupation:4), -46 (occupation:5), -48 (occupation:7), -57 (race:0), -60 (race:3), -62 (sex), -64 (capital-loss), -65 (hours-per-week), -68 (native-country:10), -69 (native-country:11), -71 (native-country:13), -73 (native-country:15), -74 (native-country:16), -78 (native-country:2), -80 (native-country:21), -90 (native-country:30), -93 (native-country:33), -94 (native-country:34), -98 (native-country:38), -99 (native-country:39), -103 (native-country:5), … (60 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_affine\_relay, age\_late\_career\_hinge, occupation8\_relays, occupation1\_relays, occupation6\_relationship1\_gate, relationship1\_indicator, occupation0\_country0\_gate, country32\_affine\_relay, dead\_relationship\_chain |
| Dimensionality (in → out, incl. descendants) | 116 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.247 e^{0.881 x_{age\_a} + 1.84 x_{capital-gain\_a} + 0.278 x_{capital-loss} + 0.977 x_{education-num} + 0.054 x_{education:0} - 0.0876 x_{education:12} + 0.135 x_{education:4} + 0.114 x_{education:5} - 0.0837 x_{education:7} - 0.123 x_{education:9} + 0.0864 x_{fnlwgt} + 0.287 x_{hours-per-week} + 0.056 x_{marital-status:1} + 0.703 x_{marital-status:2} + 0.0287 x_{marital-status:3} + 0.0122 x_{native-country:10} + 0.026 x_{native-country:11} + 0.00715 x_{native-country:13} + 0.833 x_{native-country:15} + 0.0168 x_{native-country:16} + 0.0443 x_{native-country:21} + 0.0445 x_{native-country:2} + 0.0565 x_{native-country:30} - 0.00425 x_{native-country:33} - 0.0102 x_{native-country:34} - 0.0344 x_{native-country:38} + 0.144 x_{native-country:39} + 0.0353 x_{native-country:5} + 0.0431 x_{native-country:9} + 0.0296 x_{occupation:11} + 0.0309 x_{occupation:13} - 0.0876 x_{occupation:14} - 0.0955 x_{occupation:3} + 0.139 x_{occupation:4} - 0.211 x_{occupation:5} - 0.18 x_{occupation:6\_a} - 0.166 x_{occupation:7} - 0.215 x_{occupation:8\_a} - 0.0305 x_{race:0} + 0.00223 x_{race:3} + 0.835 x_{relationship:0\_a} + 0.216 x_{relationship:4\_a} + 0.568 x_{relationship:5\_a} + 0.246 x_{sex} + 0.0215 x_{workclass:0} + 0.0781 x_{workclass:1} - 0.0477 x_{workclass:2} - 0.149 x_{workclass:6} - 1.22 \operatorname{age\_late\_career\_hinge}{\left(x_{age\_b},x_{native-country:4} \right)} + 1.6 \operatorname{capital\_gain\_affine\_relay}{\left(x_{capital-gain\_b} \right)} + 0.0752 \operatorname{country\_32\_affine\_relay}{\left(x_{native-country:32} \right)} - 0.00757 \operatorname{dead\_relationship\_chain}{\left(x_{relationship:5\_b},x_{native-country:19},x_{relationship:4\_b},x_{relationship:0\_b} \right)} - 0.334 \operatorname{occupation\_0\_country0\_gate}{\left(x_{native-country:0},x_{occupation:0} \right)} + 0.00473 \operatorname{occupation\_1\_relays\_0}{\left(x_{occupation:1} \right)} - 0.2 \operatorname{occupation\_1\_relays\_1}{\left(x_{occupation:1} \right)} - 0.0186 \operatorname{occupation\_6\_relationship1\_gate}{\left(x_{occupation:6\_b},x_{relationship:1\_a} \right)} - 1.1 \operatorname{occupation\_8\_relays\_0}{\left(x_{occupation:8\_b},x_{occupation:8\_c} \right)} - 0.542 \operatorname{occupation\_8\_relays\_1}{\left(x_{occupation:8\_b},x_{occupation:8\_c} \right)} - 0.00151 \operatorname{occupation\_8\_relays\_2}{\left(x_{occupation:8\_b},x_{occupation:8\_c} \right)} + 0.418 \operatorname{relationship\_1\_indicator}{\left(x_{relationship:1\_b} \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> ROOT (supported). Output 0 = sigmoid(z), z = 1.399 + 48 direct input terms + 12 hidden readouts (9 child annotations); 52 of the 107 one-hot/standardized inputs are unconnected and listed here as zero-effect entries. Class 1 is the 76% majority (presumably \<=50K), so negative weights push toward high income. The model is essentially a sparse logistic regression with one consequential nonlinearity. Direct edges carry logit SD 3.37 of 3.85 (mean-ablating them drops train AUC 0.910-\>0.518); all hidden readouts together carry SD 1.32 and mean-ablating every hidden unit costs only acc 0.856-\>0.838, AUC 0.910-\>0.895. The only nonlinearity that matters is age\_late\_career\_hinge (4647): keeping it alone recovers acc 0.850 / AUC 0.904. It turns the direct age slope (-0.88) into a U-shaped age effect with its minimum at age z\~0.8. The other children are lookups or affine relays: capital\_gain\_affine\_relay steepens the capital-gain slope to -2.54/SD; relationship1\_indicator (-1.03 for relationship:1), occupation0\_country0\_gate (+0.92 for occupation:0 unless native-country:0), occupation8\_relays (+0.45) and occupation1\_relays (+0.37) are category offsets; country32\_affine\_relay and the native-country:4 part of the age hinge memorise rare countries; occupation6\_relationship1\_gate is near-inert (\<=0.10 logit); dead\_relationship\_chain (613-\>2245-\>8678-\>170) and node 2114 are identically 0 on every valid input. Children combine ADDITIVELY in the logit (each child exit has one readout edge into node 0; no child feeds another). Largest exact logit-space attributions (mean \|phi\|, full train): relationship:0 0.82, education-num 0.72, marital-status:2 0.70, capital-gain 0.68, age 0.51, relationship:1 0.39, relationship:5 0.24, sex 0.23, occupation:8 0.21, hours-per-week 0.18.

**Evidence (4 records, DB):**

- `analytical_formula`: (no narrative)
- `ablation_study`: (no narrative)
- `shap_importance`: (no narrative)
- `performance_metrics`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "h_k are the child-annotation exits. On the train domain 1870, 1303, 9258, 4973, 13173 are always active (affine), 170 and 2114 are identically 0, 7530/3119 are category indicators, so z is affine in every input except age (hinge 4647) and the 2-input gates 392 and 4249.",
 "tool_latex": "y_0 = \\frac{1}{0.247 e^{0.881 x_{age_a} + 1.84 x_{capital-gain_a} + 0.278 x_{capital-loss} + 0.977 x_{education-num} + 0.054 x_{education:0} - 0.0876 x_{education:12} + 0.135 x_{education:4} + 0.114 x_{education:5} - 0.0837 x_{education:7} - 0.123 x_{education:9} + 0.0864 x_{fnlwgt} + 0.287 x_{hours-per-week} + 0.056 x_{marital-status:1} + 0.703 x_{marital-status:2} + 0.0287 x_{marital-status:3} + 0.0122 x_{native-country:10} + 0.026 x_{native-country:11} + 0.00715 x_{native-country:13} + 0.833 x_{native-country:15} + 0.0168 x_{native-country:16} + 0.0443 x_{native-country:21} + 0.0445 x_{native-country:2} + 0.0565 x_{native-country:30} - 0.00425 x_{native-country:33} - 0.0102 x_{native-country:34} - 0.0344 x_{native-country:38} + 0.144 x_{native-country:39} + 0.0353 x_{native-country:5} + 0.0431 x_{native-country:9} + 0.0296 x_{occupation:11} + 0.0309 x_{occupation:13} - 0.0876 x_{occupation:14} - 0.0955 x_{occupation:3} + 0.139 x_{occupation:4} - 0.211 x_{occupation:5} - 0.18 x_{occupation:6_a} - 0.166 x_{occupation:7} - 0.215 x_{occupation:8_a} - 0.0305 x_{race:0} + 0.00223 x_{race:3} + 0.835 x_{relationship:0_a} + 0.216 x_{relationship:4_a} + 0.568 x_{relationship:5_a} + 0.246 x_{sex} + 0.0215 x_{workclass:0} + 0.0781 x_{workclass:1} - 0.0477 x_{workclass:2} - 0.149 x_{workclass:6} - 1.22 \\operatorname{age}_{late career hinge}{\\left(x_{age_b},x_{native-country:4} \\right)} + 1.6 \\operatorname{capital}_{gain affine relay}{\\left(x_{capital-gain_b} \\right)} + 0.0752 \\operatorname{country}_{32 affine relay}{\\left(x_{native-country:32} \\right)} - 0.00757 \\operatorname{dead}_{relationship chain}{\\left(x_{relationship:5_b},x_{native-country:19},x_{relationship:4_b},x_{relationship:0_b} \\right)} - 0.334 \\operatorname{occupation}_{0 country0 gate}{\\left(x_{native-country:0},x_{occupation:0} \\right)} + 0.00473 \\operatorname{occupation}_{1 relays 0}{\\left(x_{occupation:1} \\right)} - 0.2 \\operatorname{occupation}_{1 relays 1}{\\left(x_{occupation:1} \\right)} - 0.0186 \\operatorname{occupation}_{6 relationship1 gate}{\\left(x_{occupation:6_b},x_{relationship:1_a} \\right)} - 1.1 \\operatorname{occupation}_{8 relays 0}{\\left(x_{occupation:8_b},x_{occ
… (truncated; full payload in the DB evidence record)
```

```json
// ablation_study
{
 "n": 39073,
 "split": "train",
 "method": "local forward passes on in-memory copy; ablation = replace hidden readout by its train mean",
 "baseline": {
  "acc": 0.8558,
  "auc": 0.9099
 },
 "logit_sd": {
  "total": 3.852,
  "direct_linear": 3.365,
  "hidden_readouts": 1.323
 },
 "per_unit": {
  "392": [
   0.8547,
   0.909
  ],
  "1303": [
   0.8554,
   0.9097
  ],
  "1870": [
   0.8553,
   0.9093
  ],
  "4647": [
   0.8436,
   0.9009
  ],
  "4973": [
   0.8552,
   0.9095
  ],
  "7530": [
   0.8533,
   0.9073
  ],
  "others(9258,3119,4249,13173,2114,170)": "no change at 3-4 d.p."
 },
 "only_age_hinge_kept": {
  "acc": 0.8501,
  "auc": 0.9044
 },
 "all_hidden_mean_ablated": {
  "acc": 0.8379,
  "auc": 0.8949
 },
 "all_direct_edges_mean_ablated": {
  "acc": 0.7689,
  "auc": 0.5176
 }
}
```

```json
// shap_importance
{
 "note": "compute_shap(max_samples=500) was started but killed after ~20 min (KernelExplainer, 107 features, machine load ~90); replaced by this exact logit-space attribution, which is exact for the additive part and uses 2,000 background rows for the 2-input gates (additivity error 0.0014).",
 "method": "exact interventional Shapley in logit space, full train split (39,073 rows); 2-input gates use 2,000-row background",
 "mean_abs_phi_top20": [
  [
   0.8193748729500451,
   "relationship:0"
  ],
  [
   0.7234563527808638,
   "education-num"
  ],
  [
   0.7001835559734808,
   "marital-status:2"
  ],
  [
   0.6783843567193849,
   "capital-gain"
  ],
  [
   0.5111661456369536,
   "age"
  ],
  [
   0.39474807563521563,
   "relationship:1"
  ],
  [
   0.2442211751608533,
   "relationship:5"
  ],
  [
   0.23157518106951835,
   "sex"
  ],
  [
   0.21114153095384267,
   "occupation:8"
  ],
  [
   0.17564103151907595,
   "hours-per-week"
  ],
  [
   0.13209415737345723,
   "relationship:4"
  ],
  [
   0.11498770588784692,
   "capital-loss"
  ],
  [
   0.1054164299540196,
   "occupation:0"
  ],
  [
   0.09147905351932276,
   "occupation:4"
  ],
  [
   0.09130214852330759,
   "education:9"
  ],
  [
   0.08667475248859065,
   "native-country:39"
  ],
  [
   0.08003538098696854,
   "occupation:6"
  ],
  [
   0.07974670045281516,
   "occupation:7"
  ],
  [
   0.07952329023829538,
   "workclass:6"
  ],
  [
   0.07254660771137336,
   "occupation:1"
  ]
 ]
}
```

```json
// performance_metrics
{
 "n": 39073,
 "split": "train",
 "auc_roc": 0.9099,
 "accuracy": 0.8558,
 "class1_rate": 0.7612
}
```

</details>

### A.1 Leaf: `capital_gain_affine_relay`

|  |  |
|---|---|
| Annotation id | `ann_115` |
| Entry nodes | -63\_b (capital-gain\_b) |
| Exit nodes | 1870 |
| Internal / exit nodes in its own subgraph | 1870 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0039 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.439 x_{capital-gain\_b} + 0.85 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: affine relay. 1870 = ReLU(0.439\*capital-gain + 0.850) never clips on train, so it is a linear rescaling of capital-gain that adds -0.702\*cg - 1.359 to the logit and steepens the direct capital-gain term (-1.841) to a total slope of -2.54 per SD. No nonlinearity is used on the data domain (the knot sits at cg z = -1.94, below the data minimum -0.147). EVIDENCE: Always active on train (pre-activation in \[0.786, 6.636\], 39,073/39,073 rows), so on the data domain 1870 = 0.439\*cg + 0.850 exactly and its logit contribution is -0.702\*cg - 1.359 (range \[-10.61, -1.26\], SD 0.70). Combined with the direct edge (-1.841), the total capital-gain slope is -2.54 logit per SD, pushing toward class 0 (the 24% minority, presumably \>50K). line\_plot on train (39,073 rows): exit vs capital-gain correlation 1.000 (exactly linear over the observed range); 91.6% of rows sit at the zero-gain floor (cg z = -0.147). Mean-ablation of 1870 (replace by train mean): acc 0.8558-\>0.8553, AUC 0.9099-\>0.9093 (small, because the direct edge carries the same signal).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 1870->0 = -1.599",
 "analysis": "Always active on train (pre-activation in [0.786, 6.636], 39,073/39,073 rows), so on the data domain 1870 = 0.439*cg + 0.850 exactly and its logit contribution is -0.702*cg - 1.359 (range [-10.61, -1.26], SD 0.70). Combined with the direct edge (-1.841), the total capital-gain slope is -2.54 logit per SD, pushing toward class 0 (the 24% minority, presumably >50K)."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 3.7107067614222853,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   0.493399775874042,
   6.928013746970529
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "1870"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-63_b"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.14552350385541227,
   13.187777170087646
  ],
  "scatter_y_range": [
   0.7858822291057005,
   6.63553129373887
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): exit vs capital-gain correlation 1.000 (exactly linear over the observed range); 91.6% of rows sit at the zero-gain floor (cg z = -0.147). Mean-ablation of 1870 (replace by train mean): acc 0.8558->0.8553, AUC 0.9099->0.9093 (small, because the direct edge carries the same signal)."
}
```

</details>

### A.2 Leaf: `age_late_career_hinge`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -1\_b (age\_b), -100 (native-country:4) |
| Exit nodes | 4647 |
| Internal / exit nodes in its own subgraph | 4647 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00955 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.25 x_{age\_b} \\
&\quad + 0.711 x_{native-country:4} \\
&\quad -0.968 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: a single-knot age hinge giving a U-shaped (in class-1 logit) age effect. Below age z\~0.80 the direct age term dominates (class-1 logit falls with age, i.e. the high-income share rises); above the knot 4647 reverses the slope to +0.65 per SD, so older workers drift back toward class 1, matching the empirical rate (minimum P(y=1) \~0.60 near z 0.8). native-country:4 (64 rows, standardized value 24.7) rides on the same unit as a large class-1 offset (memorisation of a rare category). EVIDENCE: Knot (for rows not in native-country:4, where nc4 = -0.041) at age z = 0.798. Active for 21.9% of those rows. Net age effect on logit = -0.881\*age + 1.224\*ReLU(1.249\*age + 0.711\*nc4 - 0.968): slope -0.881 below the knot and +0.648 above it. Logit contribution range \[0, 24.36\], SD 1.00 (the largest hidden-node contribution). The 64 native-country:4 rows (value 24.69) have 4647 \~ 16.6 (+20 logit), i.e. they are forced to class 1. line\_plot on train (39,073 rows): exit 0 below z\~0.8 then linear; scatter exit range \[0, 19.9\] (the top from nc4 rows). Local binned net age effect (non-nc4 rows): z=-1.5: +1.31, z=-1: +0.88, z=0: -0.02, z=0.5: -0.44, z=0.8: -0.67 (minimum), z=1.5: -0.26, z=2: +0.07, z=3: +0.73. Empirical P(y=1) follows the same U: 0.999, 0.944, 0.664, 0.606, 0.597, 0.662, 0.764, 0.816. Mean-ablation of 4647: acc 0.8558-\>0.8436, AUC 0.9099-\>0.9009 (largest hidden-node effect).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 4647->0 = +1.224",
 "analysis": "Knot (for rows not in native-country:4, where nc4 = -0.041) at age z = 0.798. Active for 21.9% of those rows. Net age effect on logit = -0.881*age + 1.224*ReLU(1.249*age + 0.711*nc4 - 0.968): slope -0.881 below the knot and +0.648 above it. Logit contribution range [0, 24.36], SD 1.00 (the largest hidden-node contribution). The 64 native-country:4 rows (value 24.69) have 4647 ~ 16.6 (+20 logit), i.e. they are forced to class 1."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 1.1027898725094278,
  "x_range": [
   -1.8403933369383092,
   4.00486082593868
  ],
  "y_range": [
   0.0,
   4.003919277082176
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "4647"
  ],
  "correlation": 0.43028093025374226,
  "entry_names": [
   "-1_b",
   "-100"
  ],
  "dimensionality": [
   2,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -1.574699965898446,
   3.739167454898817
  ],
  "scatter_y_range": [
   0.0,
   19.898282730212053
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): exit 0 below z~0.8 then linear; scatter exit range [0, 19.9] (the top from nc4 rows). Local binned net age effect (non-nc4 rows): z=-1.5: +1.31, z=-1: +0.88, z=0: -0.02, z=0.5: -0.44, z=0.8: -0.67 (minimum), z=1.5: -0.26, z=2: +0.07, z=3: +0.73. Empirical P(y=1) follows the same U: 0.999, 0.944, 0.664, 0.606, 0.597, 0.662, 0.764, 0.816. Mean-ablation of 4647: acc 0.8558->0.8436, AUC 0.9099->0.9009 (largest hidden-node effect)."
}
```

</details>

### A.3 Leaf: `occupation8_relays`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -49\_b (occupation:8\_b), -49\_c (occupation:8\_c) |
| Exit nodes | 1303, 2114, 9258 |
| Internal / exit nodes in its own subgraph | 1303, 2114, 9258 |
| Dimensionality (in → out, incl. descendants) | 2 → 3 |
| Live closed-form error (3 s.f. display) | 0.00186 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.123 x_{occupation:8\_b} + 1.75 \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.224 \operatorname{relu}{\left(0.123 x_{occupation:8\_b} + 1.75 \right)} \\
&\quad -0.886 \bigr)
\end{aligned} \\ y_2 = \operatorname{relu}{\left(0.871 x_{occupation:8\_c} + 0.53 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: a category offset for occupation:8 implemented by redundant relays. 1303 and 9258 never clip (affine); 2114 is dead on the whole input domain. Net effect: +0.45 logit for occupation:8 rows (via 1303) on top of the direct occupation:8 term; 9258 and 2114 are inert (excess structure). EVIDENCE: occupation:8 is binary (-0.334 / 2.996). 1303 takes only {1.711, 2.120} and 9258 only {0.240, 3.140} (always active, affine). 2114 pre-activation is 0.224\*1303 - 0.886 in \[-0.503, -0.412\] so 2114 is dead on every possible input value of occupation:8, not just on train. Net logit: constant +1.878 plus +0.449 for occupation:8 rows (via 1303), +0.004 via 9258. line\_plot on train (39,073 rows): exit-1303 vs occupation:8 correlation 1.000. Mean-ablation of 1303: acc 0.8558-\>0.8554, AUC 0.9099-\>0.9097; 2114: no change; 9258: max logit change 0.004.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readouts: 1303->0 = +1.098, 2114->0 = +0.542, 9258->0 = +0.0015; also 1303->2114 = 0.224",
 "analysis": "occupation:8 is binary (-0.334 / 2.996). 1303 takes only {1.711, 2.120} and 9258 only {0.240, 3.140} (always active, affine). 2114 pre-activation is 0.224*1303 - 0.886 in [-0.503, -0.412] so 2114 is dead on every possible input value of occupation:8, not just on train. Net logit: constant +1.878 plus +0.449 for occupation:8 rows (via 1303), +0.004 via 9258."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 1.915370307999924,
  "x_range": [
   -0.5002475375559577,
   3.162796977605309
  ],
  "y_range": [
   1.6903561077874645,
   2.140384508212384
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "1303",
   "2114",
   "9258"
  ],
  "correlation": 0.9999999999999981,
  "entry_names": [
   "-49_b",
   "-49_c"
  ],
  "dimensionality": [
   2,
   3
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.3337455141395365,
   2.9962949541888877
  ],
  "scatter_y_range": [
   1.7108119441704153,
   2.119928671829433
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): exit-1303 vs occupation:8 correlation 1.000. Mean-ablation of 1303: acc 0.8558->0.8554, AUC 0.9099->0.9097; 2114: no change; 9258: max logit change 0.004."
}
```

</details>

### A.4 Leaf: `occupation1_relays`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -37 (occupation:1) |
| Exit nodes | 3119, 4973 |
| Internal / exit nodes in its own subgraph | 3119, 4973 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.00269 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.04 x_{occupation:1} - 0.543 \right)} \\ y_1 = \operatorname{relu}{\left(0.588 x_{occupation:1} + 0.766 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: occupation:1 receives a small positive class-1 offset (+0.37 logit) via the affine relay 4973; the parallel indicator 3119 is nearly inert (readout -0.0047). occupation:1 has no direct edge to the output, so this leaf is its only route. EVIDENCE: occupation:1 binary (-0.360 / 2.778). 4973 in {0.555, 2.400}, always active (affine). 3119 is 0 for non-occ1 rows and 2.349 for occ1 rows (a true indicator) but its readout is -0.0047 (max logit change 0.011). Net logit: +0.111 baseline and +0.368 for occupation:1 rows. line\_plot on train (39,073 rows): correlation 1.000 (two-level step). Mean-ablation: 4973 acc 0.8558-\>0.8552, AUC 0.9099-\>0.9095; 3119 negligible (AUC unchanged).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readouts: 3119->0 = -0.0047, 4973->0 = +0.200",
 "analysis": "occupation:1 binary (-0.360 / 2.778). 4973 in {0.555, 2.400}, always active (affine). 3119 is 0 for non-occ1 rows and 2.349 for occ1 rows (a true indicator) but its readout is -0.0047 (max logit change 0.011). Net logit: +0.111 baseline and +0.368 for occupation:1 rows."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 0.8801769907802853,
  "x_range": [
   -0.516832268654424,
   2.9353533135369525
  ],
  "y_range": [
   0.0,
   2.512634479651465
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "3119",
   "4973"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-37"
  ],
  "dimensionality": [
   1,
   2
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.35991474219117964,
   2.778435787073708
  ],
  "scatter_y_range": [
   0.0,
   2.349264619004756
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): correlation 1.000 (two-level step). Mean-ablation: 4973 acc 0.8558->0.8552, AUC 0.9099->0.9095; 3119 negligible (AUC unchanged)."
}
```

</details>

### A.5 Leaf: `occupation6_relationship1_gate`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -47\_b (occupation:6\_b), -52\_a (relationship:1\_a) |
| Exit nodes | 4249 |
| Internal / exit nodes in its own subgraph | 4249 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00343 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.05 x_{occupation:6\_b} \\
&\quad - 0.475 x_{relationship:1\_a} \\
&\quad + 0.0173 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED but near-inert: formally an occupation:6 x relationship:1 interaction gate (off only for relationship:1 rows outside occupation:6), but readout 0.0186 makes it worth at most +0.10 logit. It is a vestigial interaction, not a meaningful part of the decision. EVIDENCE: Four train levels: -1.011 (relationship:1 rows outside occupation:6, 9,693 rows, gate off), 0.074 (neither, 27,697), 4.164 (occ6 and rel1, 374), 5.249 (occ6 not rel1, 1,309). Logit contribution range \[0, 0.098\], SD 0.019. line\_plot on train (39,073 rows): exit vs first entry correlation 0.995. Mean-ablation: acc 0.8558-\>0.8557, AUC unchanged at 4 d.p. (max logit change 0.093).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 4249->0 = +0.0186",
 "analysis": "Four train levels: -1.011 (relationship:1 rows outside occupation:6, 9,693 rows, gate off), 0.074 (neither, 27,697), 4.164 (occ6 and rel1, 374), 5.249 (occ6 not rel1, 1,309). Logit contribution range [0, 0.098], SD 0.019."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 2.665041497700073,
  "x_range": [
   -0.4584391672307476,
   4.959693713120043
  ],
  "y_range": [
   0.0,
   5.50779875775229
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "4249"
  ],
  "correlation": 0.9951976969959367,
  "entry_names": [
   "-47_b",
   "-52_a"
  ],
  "dimensionality": [
   2,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.21216039994207536,
   4.713414945831371
  ],
  "scatter_y_range": [
   0.0,
   5.2490518604751255
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): exit vs first entry correlation 0.995. Mean-ablation: acc 0.8558->0.8557, AUC unchanged at 4 d.p. (max logit change 0.093)."
}
```

</details>

### A.6 Leaf: `relationship1_indicator`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -52\_b (relationship:1\_b) |
| Exit nodes | 7530 |
| Internal / exit nodes in its own subgraph | 7530 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00169 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.56 x_{relationship:1\_b} - 0.18 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: a category indicator for relationship:1 (-1.03 logit). Because the other relationship one-hots enter linearly and standardized one-hots shift every row, this indicator places relationship:1 between relationship:0 and relationship:4 on the class-1 logit scale. It is a lookup, not an interaction. EVIDENCE: relationship:1 binary (-0.589 / 1.697). 7530 = 0 for the 29,006 non-rel1 rows and 2.466 for the 10,067 rel1 rows: a pure indicator, -1.030 logit for relationship:1. relationship:1 has no direct edge to the output, so the net relationship effect for rel1 rows (sum of all relationship terms) is -0.14, vs -0.81 for relationship:0, +0.90 for 2 and 3, +0.18 for 4, -1.75 for 5. line\_plot on train (39,073 rows): two-level step, correlation 1.000. Mean-ablation: acc 0.8558-\>0.8533, AUC 0.9099-\>0.9073 (second-largest hidden effect).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 7530->0 = -0.418",
 "analysis": "relationship:1 binary (-0.589 / 1.697). 7530 = 0 for the 29,006 non-rel1 rows and 2.466 for the 10,067 rel1 rows: a pure indicator, -1.030 logit for relationship:1. relationship:1 has no direct edge to the output, so the net relationship effect for rel1 rows (sum of all relationship terms) is -0.14, vs -0.81 for relationship:0, +0.90 for 2 and 3, +0.18 for 4, -1.75 for 5."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 0.8940023974354666,
  "x_range": [
   -0.7034512351893046,
   1.811765922269474
  ],
  "y_range": [
   0.0,
   2.6445369212806034
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "7530"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-52_b"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.5891231825775419,
   1.6974378696577113
  ],
  "scatter_y_range": [
   0.0,
   2.4663159605676435
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): two-level step, correlation 1.000. Mean-ablation: acc 0.8558->0.8533, AUC 0.9099->0.9073 (second-largest hidden effect)."
}
```

</details>

### A.7 Leaf: `occupation0_country0_gate`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -66 (native-country:0), -36 (occupation:0) |
| Exit nodes | 392 |
| Internal / exit nodes in its own subgraph | 392 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00129 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.58 x_{native-country:0} \\
&\quad + 0.741 x_{occupation:0} \\
&\quad -0.471 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: an AND-NOT gate: +0.92 class-1 logit for occupation:0 workers, cancelled when native-country:0 is set (34 rows). occupation:0 has no direct edge, so this is its only route; the native-country:0 veto touches very few rows and is probably fitted noise. EVIDENCE: Train cells: occupation:0 & not native-country:0: 2,197 rows, 392 = 2.749 (+0.919 logit); occupation:0 & native-country:0: 34 rows, 0; not occupation:0: 0 regardless (36,842 rows). P(y=1) in those cells: 0.907, 0.853, \~0.75. line\_plot on train (39,073 rows) along native-country:0 is near-flat (correlation -0.03) because the gate depends on occupation:0; the cell table above is the operative evidence. Mean-ablation: acc 0.8558-\>0.8547, AUC 0.9099-\>0.9090.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 392->0 = +0.334",
 "analysis": "Train cells: occupation:0 & not native-country:0: 2,197 rows, 392 = 2.749 (+0.919 logit); occupation:0 & native-country:0: 34 rows, 0; not occupation:0: 0 regardless (36,842 rows). P(y=1) in those cells: 0.907, 0.853, ~0.75."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 0.0013990343055103607,
  "x_range": [
   -0.5159715365364947,
   7.914116628669635
  ],
  "y_range": [
   0.0,
   0.16002671899161125
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "392"
  ],
  "correlation": -0.03241115204287548,
  "entry_names": [
   "-66",
   "-36"
  ],
  "dimensionality": [
   2,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.13278571084530705,
   7.530930802978448
  ],
  "scatter_y_range": [
   0.0,
   2.7487160194190703
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows) along native-country:0 is near-flat (correlation -0.03) because the gate depends on occupation:0; the cell table above is the operative evidence. Mean-ablation: acc 0.8558->0.8547, AUC 0.9099->0.9090."
}
```

</details>

### A.8 Leaf: `country32_affine_relay`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -92 (native-country:32) |
| Exit nodes | 13173 |
| Internal / exit nodes in its own subgraph | 13173 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0121 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.932 x_{native-country:32} + 0.915 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: affine relay giving a country-specific offset (-1.98 logit) to the 49 native-country:32 rows; no nonlinearity used. Rare-category memorisation with negligible global effect. EVIDENCE: native-country:32 binary (-0.035 / 28.229). 13173 in {0.882, 27.229}, always active (affine). The 49 native-country:32 rows get -1.98 logit; everyone else a constant -0.066. line\_plot on train (39,073 rows): correlation 1.000. Mean-ablation: no change in acc/AUC at 4 d.p. (max logit change 1.98 on 49 rows).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 13173->0 = -0.075",
 "analysis": "native-country:32 binary (-0.035 / 28.229). 13173 in {0.882, 27.229}, always active (affine). The 49 native-country:32 rows get -1.98 logit; everyone else a constant -0.066."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 14.059674013988616,
  "x_range": [
   -1.4482419509771045,
   29.633511488506198
  ],
  "y_range": [
   0.0,
   28.54619542495967
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "13173"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-92"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.035434976455136,
   28.22070451398423
  ],
  "scatter_y_range": [
   0.8818071913279031,
   27.22884360431054
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): correlation 1.000. Mean-ablation: no change in acc/AUC at 4 d.p. (max logit change 1.98 on 49 rows)."
}
```

</details>

### A.9 Leaf: `dead_relationship_chain`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -56\_b (relationship:5\_b), -77 (native-country:19), -55\_b (relationship:4\_b), -51\_b (relationship:0\_b) |
| Exit nodes | 170 |
| Internal / exit nodes in its own subgraph | 613, 2245, 8678, 170 |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.783 x_{relationship:0\_b} \\
&\quad + 1.43 \operatorname{relu}{\left(0.259 x_{relationship:4\_b} + 0.971 \operatorname{relu}{\left(0.273 - 0.992 \operatorname{relu}{\left(0.493 x_{native-country:19} - 2.66 x_{relationship:5\_b} + 0.309 \right)} \right)} - 0.45 \right)} \\
&\quad -1.08 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> SUPPORTED: dead chain. relationship:5 and native-country:19 -\> 613 -\> 2245 -\> 8678 (+relationship:4) -\> 170 (+relationship:0) never activates for any valid one-hot input, so the 4-deep structure contributes exactly 0 to the logit. Excess structure left over from evolution. EVIDENCE: 170 pre-activation takes only {-1.720, -1.275, -0.123} on train. Analytically, with one-hot relationship (exactly one of rel0/rel4/rel5 can be set): if relationship:0 is set, 613\>=0.88 so 2245 = 0, 8678 = 0 and 170 pre = -0.123; otherwise 0.783\*rel0 \<= -0.644 and 8678 \<= 0.576, so 170 pre \< 0. Hence 170 = 0 on the entire valid input domain, not only on train. line\_plot on train (39,073 rows): exit identically 0 (y range \[0, 0\]). Mean-ablation: no change.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "n": 39073,
 "split": "train",
 "readout": "readout 170->0 = +0.0076",
 "analysis": "170 pre-activation takes only {-1.720, -1.275, -0.123} on train. Analytically, with one-hot relationship (exactly one of rel0/rel4/rel5 can be set): if relationship:0 is set, 613>=0.88 so 2245 = 0, 8678 = 0 and 170 pre = -0.123; otherwise 0.783*rel0 <= -0.644 and 8678 <= 0.576, so 170 pre < 0. Hence 170 = 0 on the entire valid input domain, not only on train."
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 0.0,
  "x_range": [
   -0.4585313675789313,
   4.659517724899673
  ],
  "y_range": [
   0.0,
   0.0
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "170"
  ],
  "correlation": null,
  "entry_names": [
   "-56_b",
   "-77",
   "-55_b",
   "-51_b"
  ],
  "dimensionality": [
   4,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.22589277246626743,
   4.426879129787009
  ],
  "scatter_y_range": [
   0.0,
   0.0
  ]
 },
 "viz_type": "line_plot",
 "max_samples": 100000,
 "local_analysis": "line_plot on train (39,073 rows): exit identically 0 (y range [0, 0]). Mean-ablation: no change."
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 10 |
| rename\_node | 107 |
| split\_node | 8 |

8 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` split -63 into -63\_a, -63\_b. Final copies: -63\_a → income\_logit\_root (entry) \| -63\_b → capital\_gain\_affine\_relay (entry)
- `#108` split -1 into -1\_a, -1\_b. Final copies: -1\_a → income\_logit\_root (entry) \| -1\_b → age\_late\_career\_hinge (entry)
- `#109` split -49 into -49\_a, -49\_b, -49\_c. Final copies: -49\_a → income\_logit\_root (entry) \| -49\_b → occupation8\_relays (entry) \| -49\_c → occupation8\_relays (entry)
- `#110` split -47 into -47\_a, -47\_b. Final copies: -47\_a → income\_logit\_root (entry) \| -47\_b → occupation6\_relationship1\_gate (entry)
- `#111` split -52 into -52\_a, -52\_b. Final copies: -52\_a → occupation6\_relationship1\_gate (entry) \| -52\_b → relationship1\_indicator (entry)
- `#112` split -51 into -51\_a, -51\_b. Final copies: -51\_a → income\_logit\_root (entry) \| -51\_b → dead\_relationship\_chain (entry)
- `#113` split -55 into -55\_a, -55\_b. Final copies: -55\_a → income\_logit\_root (entry) \| -55\_b → dead\_relationship\_chain (entry)
- `#114` split -56 into -56\_a, -56\_b. Final copies: -56\_a → income\_logit\_root (entry) \| -56\_b → dead\_relationship\_chain (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T12:25:06+13:00 (commit `77513b9`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 6.66e-16 | 6.66e-16 | 6.66e-16 | agree |
| Closed-form max abs error (all leaves) | 0.01215 | 0.01215 | 0.01215 | agree |
| Numeric-function max abs error | n/a | 3.55e-15 | 3.55e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit\_root | income\_logit\_root | agree |
| Leaf / composition counts | 9 / 1 | n/a | 9 / 1 | agree |
| Operation counts | 8 splits, 0 identity | {"annotate": 10, "rename\_node": 107, "split\_node": 8} | {"annotate": 10, "rename\_node": 107, "split\_node": 8} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `country32_affine_relay` | 0.01215 | 0 |
| `age_late_career_hinge` | 0.009553 | 3.55e-15 |
| `capital_gain_affine_relay` | 0.003903 | 0 |
| `occupation6_relationship1_gate` | 0.003428 | 0 |
| `occupation1_relays` | 0.002691 | 0 |
| `occupation8_relays` | 0.001863 | 0 |
| `relationship1_indicator` | 0.001687 | 0 |
| `occupation0_country0_gate` | 0.001288 | 0 |
| `dead_relationship_chain` | 0 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Most hidden structure trivial on the domain: five units never clip (affine relays); 2114 and the 4-deep chain 613-\>2245-\>8678-\>170 are identically 0 for every valid one-hot input (shown analytically); three readouts \<0.02. Only the age hinge 4647 is consequential. native-country:4 and :32 are memorised rare categories. No semantic category names claimed.

**Tool notes (record):** compute\_shap (500) ran \~20 min at load \~90 with no result; killed and replaced by an exact local logit-space interventional Shapley on full train (additivity error 0.0014). get\_coverage needs child-exit-\>output readout edges in the root's subgraph\_connections (redone).

## What to look at

- Largest closed-form error: `country32_affine_relay` (0.0121). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `occupation0_country0_gate`: *“SUPPORTED: an AND-NOT gate: +0.92 class-1 logit for occupation:0 workers, cancelled when native-country:0 is set (34 rows).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `capital_gain_affine_relay`: *“EVIDENCE: Always active on train (pre-activation in \[0.786, 6.636\], 39,073/39,073 rows), so on the data domain 1870 = 0.439\*cg + 0.850 exactly and its logit contribution is -0.702\*cg - 1.359 (range \[-10.61, -1.26\], SD 0.70).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 8 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s4.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s4', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
