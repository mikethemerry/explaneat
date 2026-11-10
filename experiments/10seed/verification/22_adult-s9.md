# Verification packet 22 · adult-s9

Pre-registered verification order: 22 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/491150f3-2e88-4fb9-941d-77c48787c013?experiment=4e9e9b86-7e19-4e66-b53b-b79442260b68&name=10seed-adult-s9), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 9 |
| Genome id | `491150f3-2e88-4fb9-941d-77c48787c013` |
| Experiment | 10seed-adult-s9 (4e9e9b86-7e19-4e66-b53b-b79442260b68); best at generation 87 of 100, NEAT key 20287 |
| Test AUC (runs csv) | 0.906 \[95% CI 0.899, 0.912\] |
| Train AUC | 0.907 |
| Size | 61 of 107 inputs connected; 17 hidden nodes; 91 enabled connections; depth 5; 16 skip connections + 54 direct input→output edges |
| Annotations | 10 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 17 of 17 |
| Claude's time | 23.1 min (2026-11-05T11:57:22+13:00 → 2026-11-05T12:20:31+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> P(income\<=50K) = sigmoid(1.382 + 10 independent additive logit terms). Largest: sparse linear backbone over 47 inputs (removing it: AUC 0.907 -\> 0.714). Two real nonlinearities: concave piecewise-linear capital-gain penalty (kinks \~4.4k and 4.7k USD; ablation AUC 0.890) and convex hours term (kinks \~19 h and 50 h). Remaining hidden units are single-category offsets or saturating switches. Train acc 0.851, AUC 0.907.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `full_model`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -4 (workclass:2), -7 (workclass:5), -9 (workclass:7), -13 (education:1), -14 (education:10), -15 (education:11), -16 (education:12), -20 (education:2), -21 (education:3), -24 (education:6), -29 (marital-status:0), -30 (marital-status:1), -32 (marital-status:3), -34 (marital-status:5), -35 (marital-status:6), -36 (occupation:0), -37 (occupation:1), -42 (occupation:14), -43 (occupation:2), -44 (occupation:3), -50 (occupation:9), -53 (relationship:2), -54 (relationship:3), -57 (race:0), -59 (race:2), -67 (native-country:1), -68 (native-country:10), -69 (native-country:11), -70 (native-country:12), -71 (native-country:13), -72 (native-country:14), -73 (native-country:15), -76 (native-country:18), -80 (native-country:21), -85 (native-country:26), -86 (native-country:27), -87 (native-country:28), -88 (native-country:29), -91 (native-country:31), -92 (native-country:32), … (16 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | linear\_backbone, capital\_gain\_hinge, hours\_chain\_thailand, preschool\_protective\_gate, bachelors\_offset, withoutpay\_otherservice\_iran, unmarried\_offset, honduras\_offset, never\_married\_offset, prof\_school\_offset |
| Dimensionality (in → out, incl. descendants) | 56 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.251 e^{- 1.0 \operatorname{bachelors\_offset}{\left(-27 \right)} - 1.0 \operatorname{capital\_gain\_hinge}{\left(-63,-6 \right)} + 0.0758 \operatorname{honduras\_offset}{\left(-74 \right)} - 1.0 \operatorname{hours\_chain\_thailand}{\left(-65,-97 \right)} - 1.0 \operatorname{linear\_backbone}{\left(-1,-2,-3,-5,-8,-11,-12,-19,-22,-23,-25,-26,-28,-31,-38,-40,-41,-45,-46,-47,-48,-51,-52,-56,-58,-60,-61,-62,-64,-66,-75,-77,-78,-81,-82,-83,-84,-89,-90,-93,-100,-101,-102,-103,-104,-105,-107 \right)} - 0.329 \operatorname{never\_married\_offset}{\left(-33 \right)} - 1.0 \operatorname{preschool\_protective\_gate}{\left(-17,-39 \right)} + 0.024 \operatorname{prof\_school\_offset}{\left(-18 \right)} + 0.451 \operatorname{unmarried\_offset}{\left(-55 \right)} - 1.0 \operatorname{withoutpay\_otherservice\_iran}{\left(-10,-49,-79 \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Root composition. P(income\<=50K) = sigmoid(1.382 + linear\_backbone + capital\_gain\_hinge + hours\_chain\_thailand + preschool\_protective\_gate + bachelors\_offset + withoutpay\_otherservice\_iran - 0.451 unmarried\_unit - 0.076 honduras\_unit + 0.329 never\_married\_unit - 0.024 prof\_school\_unit). Hypothesis (supported): purely additive composition in logit space of one sparse linear backbone and nine independent univariate or small-group terms; no child feeds another, so there are no cross-child interactions. The 46 unconnected inputs are zero-effect entries (no edges; SHAP \<= 9.4e-4 = noise). Evidence: closed form via get\_formula; train acc 0.851, AUC 0.907; whole-model SHAP (train, 200 samples) ranks Husband, Married-civ-spouse, education-num, Not-in-family, capital-gain, hours, age, Wife, sex. Mean-ablation on train: backbone AUC 0.907-\>0.714, capital gain -\>0.890, hours -\>0.902, Other-service group -\>0.906, Unmarried -\>0.906, others negligible. Function preservation vs original phenotype \~7e-16.

**Evidence (4 records, DB):**

- `analytical_formula`: Root = sigmoid(1.382 + sum of child logit contributions); 0.251 = exp(-1.382).
- `shap_importance`: Whole-model KernelSHAP on train, max\_samples=200. Top: relationship:0 Husband, Married-civ-spouse, education-num, Not-in-family, capital-gain, hours, age, Wife, sex. Unconnected inputs \<=9.4e-4 (sampling noise; they have no edges).
- `performance_metrics`: compute\_performance on full train split: acc 0.8512, AUC 0.9074.
- `ablation_study`: Additivity is exact by construction (identity exits feed the sigmoid output with weight 1 or the unit's output weight). Backbone dominates; capital gain is the most important nonlinear child.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// shap_importance
{
 "scale": "probability",
 "split": "train",
 "top15": [
  [
   "relationship:0",
   0.11191640801317407
  ],
  [
   "marital-status:2",
   0.07801590075475485
  ],
  [
   "education-num",
   0.07538420358375554
  ],
  [
   "relationship:1",
   0.04886147367697718
  ],
  [
   "capital-gain",
   0.041087417226325966
  ],
  [
   "hours-per-week",
   0.035193311004033535
  ],
  [
   "age",
   0.03483564257726269
  ],
  [
   "relationship:5",
   0.03159224781501147
  ],
  [
   "sex",
   0.03002964782856499
  ],
  [
   "occupation:4",
   0.019830644879865494
  ],
  [
   "relationship:4",
   0.019301826066584066
  ],
  [
   "capital-loss",
   0.01390594989916444
  ],
  [
   "occupation:8",
   0.013692003343363379
  ],
  [
   "occupation:10",
   0.009818132836840803
  ],
  [
   "marital-status:4",
   0.009455113686414721
  ]
 ],
 "base_value": 0.7441869315731376,
 "max_samples": 200,
 "max_abs_shap_unconnected_inputs": 0.00094
}
```

```json
// performance_metrics
{
 "mae": 0.18646973295019792,
 "mse": 0.10370739436711195,
 "rmse": 0.32203632460812853,
 "auc_roc": 0.9073757872167216,
 "accuracy": 0.8512015970107235,
 "n_samples": 39073,
 "correlation": 0.6630661910470431
}
```

```json
// ablation_study
{
 "method": "per-child mean ablation of the logit contribution, full train",
 "ablated_auc": {
  "others": ">=0.907",
  "linear_backbone": 0.714,
  "unmarried_offset": 0.906,
  "capital_gain_hinge": 0.89,
  "hours_chain_thailand": 0.902,
  "withoutpay_otherservice_iran": 0.906
 },
 "child_sd_logit": {
  "honduras_offset": 0.075,
  "linear_backbone": 3.041,
  "bachelors_offset": 0.081,
  "unmarried_offset": 0.508,
  "capital_gain_hinge": 4.825,
  "prof_school_offset": 0.024,
  "hours_chain_thailand": 0.619,
  "never_married_offset": 0.149,
  "preschool_protective_gate": 1.729,
  "withoutpay_otherservice_iran": 0.542
 }
}
```

</details>

### A.1 Leaf: `linear_backbone`

|  |  |
|---|---|
| Annotation id | `ann_113` |
| Entry nodes | -1 (age), -2 (workclass:0), -3 (workclass:1), -5 (workclass:3), -8 (workclass:6), -11 (fnlwgt), -12 (education:0), -19 (education:15), -22 (education:4), -23 (education:5), -25 (education:7), -26 (education:8), -28 (education-num), -31 (marital-status:2), -38 (occupation:10), -40 (occupation:12), -41 (occupation:13), -45 (occupation:4), -46 (occupation:5), -47 (occupation:6), -48 (occupation:7), -51 (relationship:0), -52 (relationship:1), -56 (relationship:5), -58 (race:1), -60 (race:3), -61 (race:4), -62 (sex), -64 (capital-loss), -66 (native-country:0), -75 (native-country:17), -77 (native-country:19), -78 (native-country:2), -81 (native-country:22), -82 (native-country:23), -83 (native-country:24), -84 (native-country:25), -89 (native-country:3), -90 (native-country:30), -93 (native-country:33), … (7 more) |
| Exit nodes | identity\_linear |
| Internal / exit nodes in its own subgraph | identity\_linear |
| Dimensionality (in → out, incl. descendants) | 47 → 1 |
| Live closed-form error (3 s.f. display) | 0.0195 |
| Live numeric-function error | 7.11e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.585 x_{age} \\
&\quad - 0.312 x_{capital-loss} \\
&\quad - 1.09 x_{education-num} \\
&\quad - 0.0478 x_{education:0} \\
&\quad - 0.0323 x_{education:15} \\
&\quad - 0.102 x_{education:4} \\
&\quad - 0.0347 x_{education:5} \\
&\quad + 0.071 x_{education:7} \\
&\quad + 0.0207 x_{education:8} \\
&\quad - 0.0978 x_{fnlwgt} \\
&\quad - 0.891 x_{marital-status:2} \\
&\quad + 0.00708 x_{native-country:0} \\
&\quad - 0.000293 x_{native-country:17} \\
&\quad + 0.00593 x_{native-country:19} \\
&\quad - 0.0358 x_{native-country:22} \\
&\quad - 0.00819 x_{native-country:23} \\
&\quad + 0.0047 x_{native-country:24} \\
&\quad + 0.0665 x_{native-country:25} \\
&\quad - 0.0183 x_{native-country:2} \\
&\quad - 0.124 x_{native-country:30} \\
&\quad + 0.0327 x_{native-country:33} \\
&\quad + 0.0496 x_{native-country:3} \\
&\quad + 0.05 x_{native-country:40} \\
&\quad - 0.0351 x_{native-country:41} \\
&\quad + 0.324 x_{native-country:4} \\
&\quad - 0.0159 x_{native-country:5} \\
&\quad + 0.0436 x_{native-country:6} \\
&\quad + 0.0154 x_{native-country:7} \\
&\quad - 0.0223 x_{native-country:9} \\
&\quad - 0.157 x_{occupation:10} \\
&\quad - 0.0742 x_{occupation:12} \\
&\quad - 0.118 x_{occupation:13} \\
&\quad - 0.276 x_{occupation:4} \\
&\quad + 0.195 x_{occupation:5} \\
&\quad + 0.148 x_{occupation:6} \\
&\quad + 0.0682 x_{occupation:7} \\
&\quad - 0.0621 x_{race:1} \\
&\quad - 0.0319 x_{race:3} \\
&\quad - 0.0884 x_{race:4} \\
&\quad - 1.26 x_{relationship:0} \\
&\quad - 0.884 x_{relationship:1} \\
&\quad - 0.974 x_{relationship:5} \\
&\quad - 0.517 x_{sex} \\
&\quad + 0.19 x_{workclass:0} \\
&\quad - 0.101 x_{workclass:1} \\
&\quad + 0.309 x_{workclass:3} \\
&\quad + 0.182 x_{workclass:6}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear backbone (47 standardized inputs, direct edges only; logit of P(income\<=50K)). Hypothesis: a sparse logistic-regression term. Evidence: closed form is exactly linear, y=sum w\_i x\_i (no bias; bias sits at the output). Largest regular effects (logit, 1 vs 0 or per SD): Wife -4.53, Husband -2.56, Not-in-family -2.02, Married-civ-spouse -1.79, education-num -1.09/SD (-0.43 per year), male -1.10, age -0.585/SD (-0.043/yr), Exec-managerial -0.84, Prof-specialty -0.47, Tech-support -0.69, capital-loss -0.31/SD, Farming-fishing +1.14, Handlers-cleaners +0.73, Self-emp-not-inc +0.68, unknown workclass +0.82. Rare categories act as switches because standardized one-hots are large when 1: Never-worked +20.4 (9 rows), Columbia +8.0 (64), Laos +3.2 (17), Philippines -1.6, Yugoslavia -1.5, Vietnam +1.3. Train mean-ablation: acc 0.851-\>0.799, AUC 0.907-\>0.714 (the dominant part of the model). Sensitivity (get\_viz\_summary) equals \|w\_i\|, consistent with linearity. Supported.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.7992, AUC 0.9074-\>0.7142.
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.7992,
 "ablated_auc": 0.7142,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_linear"
  ],
  "entry_names": [
   "-1",
   "-2",
   "-3",
   "-5",
   "-8",
   "-11",
   "-12",
   "-19",
   "-22",
   "-23",
   "-25",
   "-26",
   "-28",
   "-31",
   "-38",
   "-40",
   "-41",
   "-45",
   "-46",
   "-47",
   "-48",
   "-51",
   "-52",
   "-56",
   "-58",
   "-60",
   "-61",
   "-62",
   "-64",
   "-66",
   "-75",
   "-77",
   "-78",
   "-81",
   "-82",
   "-83",
   "-84",
   "-89",
   "-90",
   "-93",
   "-100",
   "-101",
   "-102",
   "-103",
   "-104",
   "-105",
   "-107"
  ],
  "input_labels": [
   "-1",
   "-2",
   "-3",
   "-5",
   "-8",
   "-11",
   "-12",
   "-19",
   "-22",
   "-23",
   "-25",
   "-26",
   "-28",
   "-31",
   "-38",
   "-40",
   "-41",
   "-45",
   "-46",
   "-47",
   "-48",
   "-51",
   "-52",
   "-56",
   "-58",
   "-60",
   "-61",
   "-62",
   "-64",
   "-66",
   "-75",
   "-77",
   "-78",
   "-81",
   "-82",
   "-83",
   "-84",
   "-89",
   "-90",
   "-93",
   "-100",
   "-101",
   "-102",
   "-103",
   "-104",
   "-105",
   "-107"
  ],
  "sensitivities": {
   "identity_linear": [
    0.58521375499771,
    0.18985535562277508,
    0.10132393440685547,
    0.30896556698732364,
    0.18189803806979654,
    0.09775991211016598,
    0.04779826654221171,
    0.03229601864606613,
    0.10204874028403954,
    0.03469884098893991,
    0.07099911925018476,
    0.020690225866247403,
    1.094400553044527,
    0.8913444703279095,
    0.1569746803925105,
    0.07422378397027989,
    0.11796263814920699,
    0.27630182235641854,
    0.19493220759238436,
    0.14770067916303747,
    0.06818222077627359,
    1.2569961702363228,
    0.8844584090481525,
    0.9741869358582114,
    0.06214287743185179,
    0.03190246769143372,
    0.0883833738377116,
    0.5172406062308134,
    0.31215664489352196,
    0.007081671259844945,
    0.00029260432379999573,
    0.0059305943559350895,
    0.0182552596338942,
    0.03576747204832128,
    0.008186762065261045,
    0.0047002130636017376,
    0.06654849496334306,
    0.049572896536132546,
    0.12417915303643504,
    0.032672133216087675,
    0.3241952257365725,
    0.04997754199561456,
    0.03508763923295502,
    0.015927611559967945,
    0.043640318795051,
    0.015395620470686704,
    0.022312065313671237
   ]
  },
  "dimensionality": [
   47,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

</details>

### A.2 Leaf: `capital_gain_hinge`

|  |  |
|---|---|
| Annotation id | `ann_114` |
| Entry nodes | -63 (capital-gain), -6 (workclass:4) |
| Exit nodes | identity\_capgain |
| Internal / exit nodes in its own subgraph | 3460, 663, identity\_capgain |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0406 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 3.38 x_{capital-gain} \\
&\quad - 0.95 \operatorname{relu}{\left(0.98 x_{capital-gain} - 0.00808 x_{workclass:4} + 0.977 \operatorname{relu}{\left(0.687 x_{capital-gain} - 0.334 \right)} - 0.437 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Capital-gain term (entries capital-gain, workclass:4 Private). Closed form: -3.38 cg - 0.95 relu(0.98 cg - 0.0081 wc4 + 0.977 relu(0.687 cg - 0.334) - 0.437) (standardized). Hypothesis (supported): a monotone decreasing, piecewise-linear, concave function of capital gain that lowers P(\<=50K); slope -3.38/SD (-0.00045/USD) below \~4.4k USD, -4.31/SD between kinks at \~4.4k and \~4.7k USD (z=0.446, 0.486), -4.95/SD above. Values on train: cg=0 (91.6% of rows) +0.49; 3k -0.89; 5k -1.89; 7k -3.33; 15k -8.4; 99,999 -64.5 (P(\<=50K)\~0). Workclass Private weight -0.0081 is negligible (sensitivity 4e-4; identical contribution for Private vs not at cg=0). Line plot correlation with cg -0.999. Mean-ablation: acc 0.851-\>0.829, AUC 0.907-\>0.890; the most important nonlinear leaf.

**Evidence (4 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8289, AUC 0.9074-\>0.8899.
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split (39,073 rows).
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8289,
 "ablated_auc": 0.8899,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_capgain"
  ],
  "entry_names": [
   "-63",
   "-6"
  ],
  "input_labels": [
   "-63",
   "-6"
  ],
  "sensitivities": {
   "identity_capgain": [
    3.4619314443466083,
    0.00042093399705259415
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": -31.612497525636563,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   -67.78771292817147,
   2.743162700588924
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_capgain"
  ],
  "correlation": -0.9991312189925797,
  "entry_names": [
   "-63",
   "-6"
  ],
  "dimensionality": [
   2,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.14552350385541227,
   13.187777170087646
  ],
  "scatter_y_range": [
   -64.50738115704645,
   0.49150490234459804
  ]
 },
 "viz_type": "line_plot"
}
```

</details>

### A.3 Leaf: `hours_chain_thailand`

|  |  |
|---|---|
| Annotation id | `ann_115` |
| Entry nodes | -65 (hours-per-week), -97 (native-country:37) |
| Exit nodes | identity\_hours |
| Internal / exit nodes in its own subgraph | 4445, 5856, 10780, 11637, identity\_hours |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00394 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.21 x_{hours-per-week} \\
&\quad + 0.0296 x_{native-country:37} \\
&\quad + 0.882 \operatorname{relu}{\left(0.398 - 0.516 x_{hours-per-week} \right)} \\
&\quad + 0.406 \operatorname{relu}{\left(0.352 \operatorname{relu}{\left(0.398 - 0.516 x_{hours-per-week} \right)} + 1.37 \right)} \\
&\quad + 0.819 \operatorname{relu}{\left(- 1.55 x_{native-country:37} + 0.855 \operatorname{relu}{\left(0.902 \operatorname{relu}{\left(0.352 \operatorname{relu}{\left(0.398 - 0.516 x_{hours-per-week} \right)} + 1.37 \right)} - 1.64 \right)} - 0.0794 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hours-per-week term with a Thailand switch (entries hours-per-week, native-country:37 Thailand). Serial ReLU chain 4445-\>5856-\>10780-\>11637 plus direct hours and Thailand edges. Hypothesis (supported): a monotone decreasing, convex piecewise-linear function of hours that raises P(\<=50K) for short hours: 4445 is active below \~50 h/week, 10780 below \~19 h; 5856 is always on (bias 1.37) so it is linear in 4445. Contribution on train (non-Thai): 1h +3.45, 10h +2.83, 20h +2.18, 30h +1.59, 40h +0.99, 50h +0.39, 60h +0.23, 80h -0.11, 99h -0.44; slope \~-0.06 logit/h up to 50 h, \~-0.017/h above. Thailand (26 rows) adds a direct +1.15 offset and its -1.55 weight shuts unit 11637, which only matters below \~19 h, so the 'interaction' is practically inert. Sensitivity hours 0.64 vs Thailand 0.076; line plot corr -0.96. Mean-ablation: acc 0.851-\>0.849, AUC 0.907-\>0.902.

**Evidence (4 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8486, AUC 0.9074-\>0.9017.
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8486,
 "ablated_auc": 0.9017,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 0.9888074056600732,
  "x_range": [
   -3.5745149488954633,
   5.123585616693855
  ],
  "y_range": [
   -0.5174631179981987,
   3.7880173042960776
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_hours"
  ],
  "correlation": -0.9591962793187597,
  "entry_names": [
   "-65",
   "-97"
  ],
  "dimensionality": [
   2,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -3.179146741368676,
   4.728217409167068
  ],
  "scatter_y_range": [
   -0.43456613711447895,
   3.926076954809504
  ]
 },
 "viz_type": "line_plot"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_hours"
  ],
  "entry_names": [
   "-65",
   "-97"
  ],
  "input_labels": [
   "-65",
   "-97"
  ],
  "sensitivities": {
   "identity_hours": [
    0.6391008458439631,
    0.07562695878363444
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

</details>

### A.4 Leaf: `preschool_protective_gate`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -17 (education:13), -39 (occupation:11) |
| Exit nodes | identity\_preschool |
| Internal / exit nodes in its own subgraph | 6874, 6544, 1988, 12047, identity\_preschool |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00446 |
| Live numeric-function error | 5.55e-17 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.129 x_{occupation:11} \\
&\quad - 0.405 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(- 0.492 x_{occupation:11} + 0.109 \operatorname{relu}{\left(0.991 x_{education:13} + 0.917 \operatorname{relu}{\left(0.985 x_{education:13} - 0.0492 \right)} + 0.869 \right)} - 0.91 \right)} + 0.87 \right)} \\
&\quad + 0.963 \operatorname{relu}{\left(0.991 x_{education:13} + 0.917 \operatorname{relu}{\left(0.985 x_{education:13} - 0.0492 \right)} + 0.869 \right)} \\
&\quad - 0.151 \operatorname{relu}{\left(- 0.492 x_{occupation:11} + 0.109 \operatorname{relu}{\left(0.991 x_{education:13} + 0.917 \operatorname{relu}{\left(0.985 x_{education:13} - 0.0492 \right)} + 0.869 \right)} - 0.91 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Preschool switch (entries education:13 Preschool, occupation:11 Protective-serv). Chain 6874-\>6544-\>1988-\>12047 plus direct Protective-serv edge. Hypothesis (supported): Preschool is a saturating override, Protective-serv a plain offset; there is no real interaction on data. Train values (3 distinct): neither +0.465; Protective-serv -0.463 (777 rows); Preschool +43.1 (64 rows, P(\<=50K)=1 regardless of the rest); no Preschool+Protective-serv rows. The large value comes from the standardized one-hot (\~+25 when 1) passing through two stacked positive ReLUs. Sensitivity education:13 0.96, occupation:11 0.13. Mean-ablation changes acc by +0.0008 and AUC by +0.0001 (the override only moves 64 already-confident rows).

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.852, AUC 0.9074-\>0.9075.
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.852,
 "ablated_auc": 0.9075,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_preschool"
  ],
  "entry_names": [
   "-17",
   "-39"
  ],
  "input_labels": [
   "-17",
   "-39"
  ],
  "sensitivities": {
   "identity_preschool": [
    0.9558714609174562,
    0.12949713676989133
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

</details>

### A.5 Leaf: `bachelors_offset`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -27 (education:9) |
| Exit nodes | identity\_bachelors |
| Internal / exit nodes in its own subgraph | 5995, identity\_bachelors |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 9.36e-05 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.0816 x_{education:9} \\
&\quad + 0.0276 \operatorname{relu}{\left(- 0.697 x_{education:9} - 0.199 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Bachelors offset (education:9). Closed form 0.0816 x + 0.0276 relu(-0.697 x - 0.199): ReLU is on only for non-Bachelors (z=-0.44), off for Bachelors, so the leaf is an exact two-valued offset: Bachelors +0.184, others -0.033 (difference +0.22 logit, i.e. slightly higher P(\<=50K) than the education-num term alone predicts). Line plot correlation 1.0. Mean-ablation negligible (acc -0.0004, AUC \~0). Supported.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8508, AUC 0.9074-\>0.9074.
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8508,
 "ablated_auc": 0.9074,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 0.07424193220436898,
  "x_range": [
   -0.5783762309842643,
   2.3899608243640755
  ],
  "y_range": [
   -0.04157264465283022,
   0.19511998546490528
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_bachelors"
  ],
  "correlation": 0.9999999999999991,
  "entry_names": [
   "-27"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.4434518193775216,
   2.255036412757333
  ],
  "scatter_y_range": [
   -0.03315654329499092,
   0.1841045541811839
  ]
 },
 "viz_type": "line_plot"
}
```

</details>

### A.6 Leaf: `withoutpay_otherservice_iran`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -10 (workclass:8), -49 (occupation:8), -79 (native-country:20) |
| Exit nodes | identity\_443 |
| Internal / exit nodes in its own subgraph | 443, identity\_443 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00544 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.539 x_{occupation:8} \\
&\quad - 0.0559 x_{workclass:8} \\
&\quad + 0.262 \operatorname{relu}{\left(0.211 x_{native-country:20} + 0.069 x_{occupation:8} + 0.275 x_{workclass:8} - 0.288 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Other-service / Without-pay / Iran offsets (entries workclass:8, occupation:8, native-country:20). Closed form 0.539 occ8 - 0.0559 wc8 + 0.262 relu(0.211 Iran + 0.069 occ8 + 0.275 wc8 - 0.288). Hypothesis (supported): additive offsets raising P(\<=50K), joined by one ReLU that is off for the reference group (bias -0.288) and on when any of the three is 1, so it is OR-like with near-additive behaviour on data. Train values: none -0.18; Iran +1.28 (45 rows); Other-service +1.62 (3909); Other-service+Iran +3.14 (5); Without-pay +0.51 (15); Without-pay+Other-service +2.36 (2). Sensitivity is dominated by Other-service (0.54). Mean-ablation: acc 0.851-\>0.851, AUC 0.907-\>0.906.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8508, AUC 0.9074-\>0.9065.
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8508,
 "ablated_auc": 0.9065,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_443"
  ],
  "entry_names": [
   "-10",
   "-49",
   "-79"
  ],
  "input_labels": [
   "-10",
   "-49",
   "-79"
  ],
  "sensitivities": {
   "identity_443": [
    0.05583718118750559,
    0.5389486960454908,
    9.469822737823052e-05
   ]
  },
  "dimensionality": [
   3,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

</details>

### A.7 Leaf: `unmarried_offset`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -55 (relationship:4) |
| Exit nodes | 11222 |
| Internal / exit nodes in its own subgraph | 11222 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00801 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.13 x_{relationship:4} + 0.82 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Unmarried offset (relationship:4). relu(1.13 x + 0.82) is always on (linear on data), times output weight -0.451: Unmarried -1.86 vs others -0.20, a -1.67 logit offset. Read together with the backbone's relationship terms (Husband -2.56, Not-in-family -2.02, Wife -4.53), the reference categories Own-child and Other-relative have the highest P(\<=50K). Line plot correlation 1.0. Mean-ablation: acc 0.851-\>0.848, AUC 0.907-\>0.906. Supported.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8482, AUC 0.9074-\>0.906.
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8482,
 "ablated_auc": 0.906,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 2.2834590422058927,
  "x_range": [
   -0.5043540756402628,
   3.100926370958356
  ],
  "y_range": [
   0.25132177831600133,
   4.3155963060957845
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "11222"
  ],
  "correlation": 0.9999999999999984,
  "entry_names": [
   "-55"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.3404776917039619,
   2.937049987022055
  ],
  "scatter_y_range": [
   0.4360615295787188,
   4.130856554833067
  ]
 },
 "viz_type": "line_plot"
}
```

</details>

### A.8 Leaf: `honduras_offset`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -74 (native-country:16) |
| Exit nodes | 13264 |
| Internal / exit nodes in its own subgraph | 13264 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0124 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.992 x_{native-country:16} + 0.156 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Honduras offset (native-country:16). relu(0.992 x + 0.156) is always on; times -0.076 gives Honduras -3.99 vs others -0.01 (14 train rows). A large offset fitted to very few rows (standardized one-hot \~52 when 1), lowering P(\<=50K); likely overfit, no effect on aggregate metrics (ablation changes nothing).

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8512, AUC 0.9074-\>0.9074.
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8512,
 "ablated_auc": 0.9074,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 26.410867310544354,
  "x_range": [
   -2.6608676404466323,
   55.461709772552815
  ],
  "y_range": [
   0.0,
   55.18703718920201
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "13264"
  ],
  "correlation": 0.9999999999999856,
  "entry_names": [
   "-74"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.01893230349211132,
   52.8197744355983
  ],
  "scatter_y_range": [
   0.1368527435520781,
   52.56559983464725
  ]
 },
 "viz_type": "line_plot"
}
```

</details>

### A.9 Leaf: `never_married_offset`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -33 (marital-status:4) |
| Exit nodes | 7990 |
| Internal / exit nodes in its own subgraph | 13461, 7990 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 7.38e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.598 \operatorname{relu}{\left(0.926 x_{marital-status:4} + 0.29 \right)} \\
&\quad + 0.0113 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Never-married offset (marital-status:4). relu(0.598 relu(0.926 x + 0.29) + 0.0113): the inner ReLU is off for non-Never-married rows (z=-0.70) and on for Never-married (z=1.42), so the leaf is an exact two-valued switch; times output weight +0.329: Never-married +0.32 vs others +0.004, a +0.32 logit offset raising P(\<=50K) on top of the backbone Married-civ-spouse -1.79. Line plot correlation 1.0 (two values); mean-ablation small (AUC -0.0003). Supported.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8515, AUC 0.9074-\>0.9071.
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8515,
 "ablated_auc": 0.9071,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 0.413491790228803,
  "x_range": [
   -0.8089121689767578,
   1.5295399533338792
  ],
  "y_range": [
   0.011270474631377885,
   1.0309296102418497
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "7990"
  ],
  "correlation": 0.9999999999999963,
  "entry_names": [
   "-33"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.7026188906899106,
   1.423246675047032
  ],
  "scatter_y_range": [
   0.011270474631377885,
   0.972102230566481
  ]
 },
 "viz_type": "line_plot"
}
```

</details>

### A.10 Leaf: `prof_school_offset`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -18 (education:14) |
| Exit nodes | 13622 |
| Internal / exit nodes in its own subgraph | 13622 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00299 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{education:14} + 0.0229 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Prof-school offset (education:14). relu(x + 0.023) is \~0 for non-Prof-school (z=-0.13) and 7.7 for Prof-school; times -0.024: Prof-school -0.185 logit (652 rows), others 0. A small offset lowering P(\<=50K) beyond education-num. Mean-ablation negligible. Supported.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true); exit is the leaf's logit contribution (identity exits) or the hidden activation multiplied by its output weight.
- `ablation_study`: Mean-ablation on train: acc 0.8512-\>0.8511, AUC 0.9074-\>0.9073.
- `empirical_viz`: get\_viz\_summary line\_plot on the full train split (39,073 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "standardized (z-scored on train)"
}
```

```json
// ablation_study
{
 "method": "replace leaf logit contribution by its train mean; local forward pass on full train split (n=39073)",
 "ablated_acc": 0.8511,
 "ablated_auc": 0.9073,
 "baseline_acc": 0.8512,
 "baseline_auc": 0.9074
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 3.8131029045507545,
  "x_range": [
   -0.5206045378958774,
   8.066787943874957
  ],
  "y_range": [
   0.0,
   8.09283044136017
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "13622"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-18"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.130268515997203,
   7.676451921976283
  ],
  "scatter_y_range": [
   0.0,
   7.702343683969615
  ]
 },
 "viz_type": "line_plot"
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 6 |
| annotate | 11 |
| rename\_node | 107 |

0 splits, 6 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node identity\_capgain intercepts 2 connection(s) into 0 (from -63 (capital-gain), 663); sits in capital\_gain\_hinge (exit); full\_model (entry)
- `#108` identity node identity\_hours intercepts 5 connection(s) into 0 (from -65 (hours-per-week), 4445, 5856, 11637, -97 (native-country:37)); sits in hours\_chain\_thailand (exit); full\_model (entry)
- `#109` identity node identity\_preschool intercepts 4 connection(s) into 0 (from 6544, 1988, 12047, -39 (occupation:11)); sits in preschool\_protective\_gate (exit); full\_model (entry)
- `#110` identity node identity\_bachelors intercepts 2 connection(s) into 0 (from -27 (education:9), 5995); sits in bachelors\_offset (exit); full\_model (entry)
- `#111` identity node identity\_443 intercepts 3 connection(s) into 0 (from -10 (workclass:8), -49 (occupation:8), 443); sits in withoutpay\_otherservice\_iran (exit); full\_model (entry)
- `#112` identity node identity\_linear intercepts 47 connection(s) into 0 (from -1 (age), -2 (workclass:0), -3 (workclass:1), -5 (workclass:3), -8 (workclass:6), -11 (fnlwgt), -12 (education:0), -19 (education:15), … (39 more)); sits in linear\_backbone (exit); full\_model (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 7.22e-16 | 7.22e-16 | 7.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.04062 | 0.04062 | 0.04062 | agree |
| Numeric-function max abs error | n/a | 7.11e-15 | 7.11e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | full\_model | full\_model | agree |
| Leaf / composition counts | 10 / 1 | n/a | 10 / 1 | agree |
| Operation counts | 0 splits, 6 identity | {"add\_identity\_node": 6, "annotate": 11, "rename\_node": 107} | {"add\_identity\_node": 6, "annotate": 11, "rename\_node": 107} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capital_gain_hinge` | 0.04062 | 0 |
| `linear_backbone` | 0.01945 | 7.11e-15 |
| `honduras_offset` | 0.01238 | 0 |
| `unmarried_offset` | 0.00801 | 0 |
| `withoutpay_otherservice_iran` | 0.005438 | 0 |
| `preschool_protective_gate` | 0.004464 | 5.55e-17 |
| `hours_chain_thailand` | 0.003939 | 4.44e-16 |
| `prof_school_offset` | 0.002992 | 0 |
| `never_married_offset` | 7.38e-04 | 0 |
| `bachelors_offset` | 9.36e-05 | 0 |

## What resisted (from the record)

> No structural problems (no splits needed). Thailand-by-hours interaction inert in practice. Rare categories act as huge switches fitted to very few rows (Preschool +43 logit/64 rows, Never-worked +20/9, Columbia +8/64, Honduras -4/14): probably overfitting. Category names inferred.

**Tool notes (record):** None (agent-side zsh wrapper issue; stray np.save files moved out of repo). SHAP \~9e-4 on unconnected inputs (noise).

## What to look at

- Largest closed-form error: `capital_gain_hinge` (0.0406), then `linear_backbone` (0.0195). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `hours_chain_thailand`: *“Thailand (26 rows) adds a direct +1.15 offset and its -1.55 weight shuts unit 11637, which only matters below \~19 h, so the 'interaction' is practically inert.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `preschool_protective_gate`: *“Hypothesis (supported): Preschool is a saturating override, Protective-serv a plain offset;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- No splits: every node sits in one annotation as evolved.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s9.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s9', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
