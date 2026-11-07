# Verification packet 01 · adult-s0

Pre-registered verification order: 1 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/6e0e4be0-96c4-45f7-8ac1-74af3c960799?experiment=b0780f6a-8733-4ba1-8947-4188b89e7807&name=10seed-adult-s0), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 0 |
| Genome id | `6e0e4be0-96c4-45f7-8ac1-74af3c960799` |
| Experiment | 10seed-adult-s0 (b0780f6a-8733-4ba1-8947-4188b89e7807); best at generation 99 of 100, NEAT key 9828 |
| Test AUC (runs csv) | 0.903 \[95% CI 0.896, 0.909\] |
| Train AUC | 0.906 |
| Size | 51 of 107 inputs connected; 11 hidden nodes; 70 enabled connections; depth 3; 11 skip connections + 46 direct input→output edges |
| Annotations | 10 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 10 of 11 |
| Claude's time | 48.1 min (2026-11-05T11:08:23+13:00 → 2026-11-05T11:56:32+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> P(income\<=50K) = sigmoid(0.808 + sparse linear backbone over 41 inputs + capital-gain term + female/Priv-house-serv gate + 7 single-category offsets), additive in logit. Hidden units with one one-hot input are exact category offsets (unknown workclass, Without-pay override, Exec-managerial, Wife, Separated, Japan, a dead Private unit). Only real nonlinearities: a convex capital-gain hinge and an OR-like gate (+0.58 women, +2 to +3.6 Priv-house-serv). Train acc 0.852, AUC 0.906; backbone dominates (AUC 0.753 without it).

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `full_model`

|  |  |
|---|---|
| Annotation id | `ann_126` |
| Entry nodes | 816\_a, idn\_o63, 369, 2098, 4158, 5203, idn\_o10, 2058, idn\_o45, 1579, idn\_o56, 6081, idn\_o83, 1009, 6764, lin\_agg, -3 (workclass:1), -5 (workclass:3), -7 (workclass:5), -12 (education:0), -13 (education:1), -14 (education:10), -15 (education:11), -16 (education:12), -17 (education:13), -19 (education:15), -21 (education:3), -22 (education:4), -23 (education:5), -24 (education:6), -26 (education:8), -29 (marital-status:0), -30 (marital-status:1), -36 (occupation:0), -37 (occupation:1), -39 (occupation:11), -42 (occupation:14), -43 (occupation:2), -44 (occupation:3), -47 (occupation:6), … (32 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_hinge, female\_or\_household\_gate, unknown\_workclass\_bump, without\_pay\_override, exec\_managerial\_bump, wife\_bump, japan\_cancelling\_bump, separated\_bump, dead\_private\_unit, linear\_backbone |
| Dimensionality (in → out, incl. descendants) | 72 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.446 e^{0.364 \operatorname{capital\_gain\_hinge\_0}{\left(-63_{ac} \right)} - 1.0 \operatorname{capital\_gain\_hinge\_1}{\left(-63_{ac} \right)} + 0.789 \operatorname{dead\_private\_unit}{\left(-6 \right)} + 0.27 \operatorname{exec\_managerial\_bump\_0}{\left(-45 \right)} - 1.0 \operatorname{exec\_managerial\_bump\_1}{\left(-45 \right)} - 0.863 \operatorname{female\_or\_household\_gate}{\left(-62,-50,-63_{b} \right)} - 0.205 \operatorname{japan\_cancelling\_bump\_0}{\left(-83 \right)} - 1.0 \operatorname{japan\_cancelling\_bump\_1}{\left(-83 \right)} - 1.0 \operatorname{linear\_backbone}{\left(-1,-4,-8,-9,-11,-18,-20,-25,-27,-28,-31,-32,-33,-35,-38,-40,-41,-46,-49,-51,-52,-57,-60,-61,-64,-65,-68,-72,-73,-75,-76,-77,-78,-79,-84,-86,-92,-97,-100,-102,-104 \right)} - 0.201 \operatorname{separated\_bump}{\left(-34 \right)} - 0.45 \operatorname{unknown\_workclass\_bump\_0}{\left(-2 \right)} + 0.368 \operatorname{unknown\_workclass\_bump\_1}{\left(-2 \right)} + 1.04 \operatorname{wife\_bump\_0}{\left(-56 \right)} - 1.0 \operatorname{wife\_bump\_1}{\left(-56 \right)} - 1.1 \operatorname{without\_pay\_override\_0}{\left(-10 \right)} - 1.0 \operatorname{without\_pay\_override\_1}{\left(-10 \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Root: P(income\<=50K) = sigmoid(0.808 + linear\_backbone + capital\_gain\_hinge + female\_or\_household\_gate + 7 single-category offsets). Composition is purely additive in logit space; the children never feed each other. Category offsets (train logit deltas): unknown workclass +0.82, Without-pay +61 (override to \<=50K, 17 rows), Exec-managerial -0.82, Wife -2.90, Japan +0.14 (two large near-cancelling terms), Separated +0.23, dead Private unit 0. Since every bump has a single binary input, each is exactly an affine function of that indicator; the network is therefore a logistic regression plus two genuine nonlinearities: a convex capital-gain hinge (slope -2.66 -\> -3.08 above \~0.43 SD) and an OR-like ReLU gate that adds +0.58 logit for women and +2 to +3.6 logit for Priv-house-serv. 56 unconnected inputs are zero-effect entries. Evidence (train, 39,073 rows): acc 0.852, AUC 0.906. Mean-ablation AUC drops: linear\_backbone 0.906-\>0.753, capital\_gain\_hinge -\>0.888, wife\_bump -\>0.887, exec\_managerial -\>0.903, female gate -\>0.904; others \<0.001. SHAP (500 rows) top: relationship:0 Husband, education-num, capital-gain, Married-civ-spouse, Wife, Exec-managerial, age, hours-per-week, relationship:1, sex.

**Evidence (3 records, DB):**

- `analytical_formula`: Root closed form: sigmoid of bias plus weighted sum of all child exits; additive in logit space.
- `shap_importance`: Whole-model SHAP (train, 500 samples).
- `ablation_study`: Child importance by mean-ablation: linear\_backbone \>\> capital\_gain\_hinge \> wife\_bump \> exec\_managerial\_bump \~ female\_or\_household\_gate \> others; dead\_private\_unit exactly 0.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// shap_importance
{
 "note": "500-row train subsample; unconnected inputs show ~5e-4 mean|SHAP|, which is the estimator noise floor, not a real effect",
 "split": "train",
 "base_value": 0.7358386804443028,
 "max_samples": 500,
 "top15_mean_abs_shap": [
  [
   "relationship:0",
   0.1073346085520229
  ],
  [
   "education-num",
   0.06862116987029213
  ],
  [
   "capital-gain",
   0.04652007866679944
  ],
  [
   "marital-status:2",
   0.03417296391738457
  ],
  [
   "relationship:5",
   0.03117111348003433
  ],
  [
   "occupation:4",
   0.029670288772145788
  ],
  [
   "age",
   0.029539598689926446
  ],
  [
   "hours-per-week",
   0.02529974574243397
  ],
  [
   "relationship:1",
   0.02304629981281469
  ],
  [
   "sex",
   0.022690133228173537
  ],
  [
   "capital-loss",
   0.020536645091342832
  ],
  [
   "marital-status:4",
   0.018810332738419738
  ],
  [
   "occupation:10",
   0.014375871098943909
  ],
  [
   "occupation:8",
   0.009745541269263892
  ],
  [
   "workclass:6",
   0.009236396711915385
  ]
 ]
}
```

```json
// ablation_study
{
 "n": 39073,
 "split": "train",
 "method": "replace one child logit contribution by its train mean, local forward pass",
 "results": {
  "_base": {
   "ll": 0.32108808276786016,
   "acc": 0.8517902387838149,
   "auc": 0.9055934526645549
  },
  "wife_bump": {
   "acc": 0.844956875591841,
   "auc": 0.8867554247771873,
   "logloss": 0.3481842487089354,
   "contrib_std": 0.6235060898943292,
   "contrib_mean": -0.020812750152497606,
   "max_abs_dprob": 0.5980111193217199,
   "mean_abs_dprob": 0.02952560245163202
  },
  "separated_bump": {
   "acc": 0.8519182043866609,
   "auc": 0.9056123362241237,
   "logloss": 0.3210738038942696,
   "contrib_std": 0.04014275731558069,
   "contrib_mean": 0.2552033277013711,
   "max_abs_dprob": 0.05514715318122665,
   "mean_abs_dprob": 0.0009669738929076171
  },
  "linear_backbone": {
   "acc": 0.802344329844138,
   "auc": 0.7528775215569109,
   "logloss": 0.5129581335680878,
   "contrib_std": 1.9929487341829886,
   "contrib_mean": 8.874282882065081e-17,
   "max_abs_dprob": 0.9070993762775107,
   "mean_abs_dprob": 0.1586713843429284
  },
  "dead_private_unit": {
   "acc": 0.8517902387838149,
   "auc": 0.9055934526645549,
   "logloss": 0.32108808276786016,
   "contrib_std": 0.0,
   "contrib_mean": 0.0,
   "max_abs_dprob": 0.0,
   "mean_abs_dprob": 0.0
  },
  "capital_gain_hinge": {
   "acc": 0.8358713177897781,
   "auc": 0.887655045697926,
   "logloss": 0.3509605997582153,
   "contrib_std": 3.049162992114971,
   "contrib_mean": -0.04354547612935339,
   "max_abs_dprob": 0.9974386317853194,
   "mean_abs_dprob": 0.06235557324488968
  },
  "exec_managerial_bump": {
   "acc": 0.8480536431807131,
   "auc": 0.9025393134380195,
   "logloss": 0.3259152234664182,
   "contrib_std": 0.2692306467695333,
   "contrib_mean": -0.03648594014740222,
   "max_abs_dprob": 0.1768516014306411,
   "mean_abs_dprob": 0.020540909122416698
  },
  "without_pay_override": {
   "acc": 0.8517646456632457,
   "auc": 0.9056846097065487,
   "logloss": 0.31946902733360605,
   "contrib_std": 1.2715674996936834,
   "contrib_mean": 0.515936319008584,
   "max_abs_dprob": 0.533061909571296,
   "mean_abs_dprob": 0.0026952973756732396
  },
  "japan_cancelling_bump": {
   "acc": 0.8517390525426766,
   "auc": 0.9055925987428571,
   "logloss": 0.3210889704767551,
   "contrib_std": 0.005695877360173151,
   "contrib_mean": 0.13469557209527142,
   "max_abs_dprob": 0.03405776025969759,
   "mean_abs_dprob": 4.7717517986355154e-05
  },
  "unknown_workclass_bump": {
   "acc": 0.85
… (truncated; full payload in the DB evidence record)
```

</details>

### A.1 Leaf: `capital_gain_hinge`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -63\_ac (capital-gain\_ac) |
| Exit nodes | 816\_a, idn\_o63 |
| Internal / exit nodes in its own subgraph | 816\_a, idn\_o63 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.0233 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.16 x_{capital-gain\_ac} - 0.5 \right)} \\ y_1 = - 2.66 x_{capital-gain\_ac}\end{gathered}
$$

**Hypothesis (DB):**

> Capital-gain (standardised; output is P(income\<=50K)) enters the logit as a convex piecewise-linear decreasing function: a direct linear term -2.658\*cg (via idn\_o63) plus a ReLU hinge -0.364\*relu(1.161\*cg-0.500) that steepens the slope from -2.66 to -3.08 above cg\~0.43 SD (raw \~\$4.3k; \~5.7% of rows). Large capital gains push strongly towards \>50K. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8359 / AUC 0.8877 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0624, max \|dP\| 0.997. Removing only the hinge (keeping the linear term) barely changes performance (AUC 0.9054): the capital-gain effect is mostly linear in the standardised value.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 7.133036610015171,
   "x_range": [
    -0.8121885375525651,
    13.8544422037848
   ],
   "y_range": [
    0.0,
    15.579579841418127
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "816_a",
    "idn_o63"
   ],
   "correlation": 0.9912142942719956,
   "entry_names": [
    "-63_ac"
   ],
   "dimensionality": [
    1,
    2
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.14552350385541227,
    13.187777170087646
   ],
   "scatter_y_range": [
    0.0,
    14.805854381068858
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.8358713177897781,
  "auc": 0.887655045697926,
  "logloss": 0.3509605997582153,
  "contrib_std": 3.049162992114971,
  "contrib_mean": -0.04354547612935339,
  "max_abs_dprob": 0.9974386317853194,
  "mean_abs_dprob": 0.06235557324488968
 },
 "hinge_only_mean_ablation": {
  "acc": 0.85148,
  "auc": 0.90541,
  "max_abs_dprob": 0.1295
 }
}
```

</details>

### A.2 Leaf: `female_or_household_gate`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -62 (sex), -50 (occupation:9), -63\_b (capital-gain\_b) |
| Exit nodes | 369 |
| Internal / exit nodes in its own subgraph | 816\_b, 369 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00458 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.243 x_{occupation:9} \\
&\quad - 0.869 x_{sex} \\
&\quad - 0.00529 \operatorname{relu}{\left(1.16 x_{capital-gain\_b} - 0.5 \right)} \\
&\quad -0.54 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> ReLU gate 369 = relu(-0.869\*sex + 0.243\*occupation:9 - 0.0053\*relu(1.161\*cg-0.5) - 0.540), read out with +0.863. sex=1 (male) closes the gate unless occupation:9 (Priv-house-serv) is set; for women it is open (\~0.68) and adds \~+0.58 logit towards \<=50K, Priv-house-serv adds a further \~+3 logit. The capital-gain path (816\_b) has a negligible weight. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8511 / AUC 0.9039 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0237, max \|dP\| 0.618. Mean 369 by cell: female 0.676, female+Priv-house-serv 4.20, male 0, male+Priv-house-serv 2.36. Dropping the capital-gain path changes 369 by at most 0.078.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 0.14676959195584705,
   "x_range": [
    -1.5257508039692003,
    0.8106481207657674
   ],
   "y_range": [
    0.0,
    0.7686028091521464
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "369"
   ],
   "correlation": -0.8194627707443238,
   "entry_names": [
    "-62",
    "-50",
    "-63_b"
   ],
   "dimensionality": [
    3,
    1
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -1.4195508528448835,
    0.7044481696414507
   ],
   "scatter_y_range": [
    0.0,
    4.20118784595388
   ]
  },
  "sensitivity": {
   "viz_type": "sensitivity",
   "exit_names": [
    "369"
   ],
   "entry_names": [
    "-62",
    "-50",
    "-63_b"
   ],
   "input_labels": [
    "-62",
    "-50",
    "-63_b"
   ],
   "sensitivities": {
    "369": [
     0.2883048597222662,
     0.08075583446383873,
     6.505922505840223e-05
    ]
   },
   "dimensionality": [
    3,
    1
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "gate_table_369": {
  "male,PHS": {
   "n": 12,
   "mean369": 2.3566
  },
  "female,PHS": {
   "n": 175,
   "mean369": 4.2011
  },
  "male,not_PHS": {
   "n": 26102,
   "mean369": 0.0
  },
  "female,not_PHS": {
   "n": 12784,
   "mean369": 0.6761
  }
 },
 "mean_ablation_train": {
  "acc": 0.8510992245284468,
  "auc": 0.9039317534681796,
  "logloss": 0.3244647727518305,
  "contrib_std": 0.35807196429064836,
  "contrib_mean": 0.2076378888450677,
  "max_abs_dprob": 0.618429881198578,
  "mean_abs_dprob": 0.02373799162519538
 },
 "capgain_path_max_change_in_369": 0.0784
}
```

</details>

### A.3 Leaf: `unknown_workclass_bump`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -2 (workclass:0) |
| Exit nodes | 2098, 4158 |
| Internal / exit nodes in its own subgraph | 2098, 4158 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.0158 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.851 x_{workclass:0} + 0.497 \right)} \\ y_1 = \operatorname{relu}{\left(1.06 x_{workclass:0} - 2.04 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> workclass:0 ('?', unknown workclass; 5.7% of rows) feeds two ReLUs: 2098 (always active, +0.45 readout) and 4158 (active only when workclass='?', -0.368 readout). Since the input is binary, together they form a single category offset: +0.82 logit towards \<=50K when workclass is unknown. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8510 / AUC 0.9050 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0072, max \|dP\| 0.191.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 2.1258623883373517,
   "x_range": [
    -0.46145294628212363,
    4.2883192010696805
   ],
   "y_range": [
    0.10437140441000842,
    4.147353372264695
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "2098",
    "4158"
   ],
   "correlation": 0.9999999999999993,
   "entry_names": [
    "-2"
   ],
   "dimensionality": [
    1,
    2
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.2455542123115871,
    4.0724204670991435
   ],
   "scatter_y_range": [
    0.2881433120397669,
    3.9635814646349363
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.85099685204617,
  "auc": 0.9049610569910511,
  "logloss": 0.32204918389620224,
  "contrib_std": 0.19031719130879923,
  "contrib_mean": 0.1764018096284129,
  "max_abs_dprob": 0.19137323383715577,
  "mean_abs_dprob": 0.007166319191802665
 },
 "logit_contribution_by_category": {
  "on": 0.9515,
  "off": 0.1297,
  "n_on": 2222,
  "delta": 0.8218
 }
}
```

</details>

### A.4 Leaf: `without_pay_override`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -10 (workclass:8) |
| Exit nodes | 5203, idn\_o10 |
| Internal / exit nodes in its own subgraph | 5203, idn\_o10 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.00512 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.846 x_{workclass:8} + 0.47 \right)} \\ y_1 = 0.343 x_{workclass:8}\end{gathered}
$$

**Hypothesis (DB):**

> workclass:8 (Without-pay, 17 training rows) passes through an always-active ReLU 5203 (readout +1.098) plus a direct term (idn\_o10). On the rare standardised value (\~48) the sum is \~+61 logit: an override that forces the prediction to \<=50K for unpaid workers; \~0 effect otherwise. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8518 / AUC 0.9057 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0027, max \|dP\| 0.533.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 20.76681828033833,
   "x_range": [
    -2.418472513719065,
    50.328932749214246
   ],
   "y_range": [
    0.0,
    43.04629505261243
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "5203",
    "idn_o10"
   ],
   "correlation": 0.9999999999999969,
   "entry_names": [
    "-10"
   ],
   "dimensionality": [
    1,
    2
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.020863183585732675,
    47.93132341908091
   ],
   "scatter_y_range": [
    0.4523344632409098,
    41.01801121502331
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.8517646456632457,
  "auc": 0.9056846097065487,
  "logloss": 0.31946902733360605,
  "contrib_std": 1.2715674996936834,
  "contrib_mean": 0.515936319008584,
  "max_abs_dprob": 0.533061909571296,
  "mean_abs_dprob": 0.0026952973756732396
 },
 "logit_contribution_by_category": {
  "on": 61.4638,
  "off": 0.4894,
  "n_on": 17,
  "delta": 60.9744
 }
}
```

</details>

### A.5 Leaf: `exec_managerial_bump`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -45 (occupation:4) |
| Exit nodes | 2058, idn\_o45 |
| Internal / exit nodes in its own subgraph | 2058, idn\_o45 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 9.16e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.575 x_{occupation:4} - 0.439 \right)} \\ y_1 = - 0.172 x_{occupation:4}\end{gathered}
$$

**Hypothesis (DB):**

> occupation:4 (Exec-managerial, 12.4% of rows) has a direct term (idn\_o45) plus a ReLU 2058 active only for managers (readout -0.27). Net category offset: -0.82 logit, i.e. towards \>50K for managers. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8481 / AUC 0.9025 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0205, max \|dP\| 0.177.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 0.3612962754129462,
   "x_range": [
    -0.5281876852455317,
    2.807030839015085
   ],
   "y_range": [
    0.0,
    1.1752378452075116
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "2058",
    "idn_o45"
   ],
   "correlation": 0.999999999999998,
   "entry_names": [
    "-45"
   ],
   "dimensionality": [
    1,
    2
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.3765868432336855,
    2.655429997003239
   ],
   "scatter_y_range": [
    0.0,
    1.0880802006132049
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.8480536431807131,
  "auc": 0.9025393134380195,
  "logloss": 0.3259152234664182,
  "contrib_std": 0.2692306467695333,
  "contrib_mean": -0.03648594014740222,
  "max_abs_dprob": 0.1768516014306411,
  "mean_abs_dprob": 0.020540909122416698
 },
 "logit_contribution_by_category": {
  "on": -0.7514,
  "off": 0.0649,
  "n_on": 4853,
  "delta": -0.8163
 }
}
```

</details>

### A.6 Leaf: `wife_bump`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -56 (relationship:5) |
| Exit nodes | 1579, idn\_o56 |
| Internal / exit nodes in its own subgraph | 5009, 1579, idn\_o56 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.00164 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.139 x_{relationship:5} \\
&\quad + 0.129 \operatorname{relu}{\left(0.714 x_{relationship:5} + 1.02 \right)} \\
&\quad -0.742 \bigr)
\end{aligned} \\ y_1 = - 0.531 x_{relationship:5}\end{gathered}
$$

**Hypothesis (DB):**

> relationship:5 (Wife, 4.9% of rows) acts via a direct term (idn\_o56, -0.531), and a two-ReLU serial chain 5009 -\> 1579 where 1579 opens only for wives (readout -1.043). Net offset: -2.90 logit towards \>50K for wives (complementing the direct Husband term in the linear backbone). Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8450 / AUC 0.8868 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0295, max \|dP\| 0.598.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 0.09213563933436143,
   "x_range": [
    -0.4585313675789313,
    4.659517724899673
   ],
   "y_range": [
    0.0,
    0.46491855971009044
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "1579",
    "idn_o56"
   ],
   "correlation": 0.9999999999999948,
   "entry_names": [
    "-56"
   ],
   "dimensionality": [
    1,
    2
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.22589277246626743,
    4.426879129787009
   ],
   "scatter_y_range": [
    0.0,
    0.41118768378883175
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.844956875591841,
  "auc": 0.8867554247771873,
  "logloss": 0.3481842487089354,
  "contrib_std": 0.6235060898943292,
  "contrib_mean": -0.020812750152497606,
  "max_abs_dprob": 0.5980111193217199,
  "mean_abs_dprob": 0.02952560245163202
 },
 "logit_contribution_by_category": {
  "on": -2.781,
  "off": 0.12,
  "n_on": 1897,
  "delta": -2.901
 }
}
```

</details>

### A.7 Leaf: `japan_cancelling_bump`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -83 (native-country:24) |
| Exit nodes | 6081, idn\_o83 |
| Internal / exit nodes in its own subgraph | 6081, idn\_o83 |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.0069 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.957 x_{native-country:24} + 0.658 \right)} \\ y_1 = - 0.19 x_{native-country:24}\end{gathered}
$$

**Hypothesis (DB):**

> native-country:24 (Japan, 68 rows) feeds an always-active ReLU 6081 (readout +0.205, \~+4.7 logit on the rare value) and a direct term (idn\_o83, \~-4.6 logit). The two nearly cancel: net offset +0.14 logit for Japan-born. Effectively a small category offset. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8517 / AUC 0.9056 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0000, max \|dP\| 0.034.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 12.108501200801014,
   "x_range": [
    -1.241341779775199,
    25.149597352751194
   ],
   "y_range": [
    0.0,
    24.733401440713692
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "6081",
    "idn_o83"
   ],
   "correlation": 1.0,
   "entry_names": [
    "-83"
   ],
   "dimensionality": [
    1,
    2
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.04175363738763561,
    23.950009210363632
   ],
   "scatter_y_range": [
    0.6181316994902665,
    23.585055262560196
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.8517390525426766,
  "auc": 0.9055925987428571,
  "logloss": 0.3210889704767551,
  "contrib_std": 0.005695877360173151,
  "contrib_mean": 0.13469557209527142,
  "max_abs_dprob": 0.03405776025969759,
  "mean_abs_dprob": 4.7717517986355154e-05
 },
 "logit_contribution_by_category": {
  "on": 0.2711,
  "off": 0.1345,
  "n_on": 68,
  "delta": 0.1366
 }
}
```

</details>

### A.8 Leaf: `separated_bump`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -34 (marital-status:5) |
| Exit nodes | 1009 |
| Internal / exit nodes in its own subgraph | 1009 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00265 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.199 x_{marital-status:5} + 1.27 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> marital-status:5 (Separated, 3.2% of rows) passes through an always-active ReLU 1009 (readout +0.201); linear in the binary input. Net offset +0.23 logit towards \<=50K for separated people. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8519 / AUC 0.9056 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0010, max \|dP\| 0.055.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 1.7978462710928387,
   "x_range": [
    -0.4659555972050157,
    5.787275743887747
   ],
   "y_range": [
    1.1745217457539376,
    2.4211707964317397
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "1009"
   ],
   "correlation": 0.9999999999999982,
   "entry_names": [
    "-34"
   ],
   "dimensionality": [
    1,
    1
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -0.18171780897352652,
    5.503037955656258
   ],
   "scatter_y_range": [
    1.2311876116938378,
    2.3645049304918393
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.8519182043866609,
  "auc": 0.9056123362241237,
  "logloss": 0.3210738038942696,
  "contrib_std": 0.04014275731558069,
  "contrib_mean": 0.2552033277013711,
  "max_abs_dprob": 0.05514715318122665,
  "mean_abs_dprob": 0.0009669738929076171
 },
 "logit_contribution_by_category": {
  "on": 0.4761,
  "off": 0.2479,
  "n_on": 1249,
  "delta": 0.2282
 }
}
```

</details>

### A.9 Leaf: `dead_private_unit`

|  |  |
|---|---|
| Annotation id | `ann_124` |
| Entry nodes | -6 (workclass:4) |
| Exit nodes | 6764 |
| Internal / exit nodes in its own subgraph | 6764 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{workclass:4} - 0.683 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> workclass:4 (Private) feeds ReLU 6764 = relu(1.0\*wc4 - 0.683). The standardised input never exceeds 0.662, so the unit is always 0 on the training data (dead unit): this subgraph contributes nothing. Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8518 / AUC 0.9056 (baseline 0.8518 / 0.9056), mean \|dP\| 0.0000, max \|dP\| 0.000.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "line": {
   "y_mean": 0.001816306680820831,
   "x_range": [
    -1.6201265544624839,
    0.7702603867408157
   ],
   "y_range": [
    0.0,
    0.08744964839846803
   ],
   "viz_type": "line",
   "x_points": 200,
   "exit_names": [
    "6764"
   ],
   "correlation": null,
   "entry_names": [
    "-6"
   ],
   "dimensionality": [
    1,
    1
   ],
   "scatter_points": 39073,
   "scatter_x_range": [
    -1.5114726025896066,
    0.6616064348679384
   ],
   "scatter_y_range": [
    0.0,
    0.0
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549,
  "logloss": 0.32108808276786016,
  "contrib_std": 0.0,
  "contrib_mean": 0.0,
  "max_abs_dprob": 0.0,
  "mean_abs_dprob": 0.0
 },
 "logit_contribution_by_category": {
  "on": 0.0,
  "off": 0.0,
  "n_on": 27177,
  "delta": 0.0
 }
}
```

</details>

### A.10 Leaf: `linear_backbone`

|  |  |
|---|---|
| Annotation id | `ann_125` |
| Entry nodes | -1 (age), -4 (workclass:2), -8 (workclass:6), -9 (workclass:7), -11 (fnlwgt), -18 (education:14), -20 (education:2), -25 (education:7), -27 (education:9), -28 (education-num), -31 (marital-status:2), -32 (marital-status:3), -33 (marital-status:4), -35 (marital-status:6), -38 (occupation:10), -40 (occupation:12), -41 (occupation:13), -46 (occupation:5), -49 (occupation:8), -51 (relationship:0), -52 (relationship:1), -57 (race:0), -60 (race:3), -61 (race:4), -64 (capital-loss), -65 (hours-per-week), -68 (native-country:10), -72 (native-country:14), -73 (native-country:15), -75 (native-country:17), -76 (native-country:18), -77 (native-country:19), -78 (native-country:2), -79 (native-country:20), -84 (native-country:25), -86 (native-country:27), -92 (native-country:32), -97 (native-country:37), -100 (native-country:4), -102 (native-country:41), … (1 more) |
| Exit nodes | lin\_agg |
| Internal / exit nodes in its own subgraph | lin\_agg |
| Dimensionality (in → out, incl. descendants) | 41 → 1 |
| Live closed-form error (3 s.f. display) | 0.0129 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.355 x_{age} \\
&\quad - 0.272 x_{capital-loss} \\
&\quad - 0.776 x_{education-num} \\
&\quad - 0.0269 x_{education:14} \\
&\quad + 0.00124 x_{education:2} \\
&\quad + 0.0446 x_{education:7} \\
&\quad + 0.0312 x_{education:9} \\
&\quad - 0.0755 x_{fnlwgt} \\
&\quad - 0.357 x_{hours-per-week} \\
&\quad - 0.305 x_{marital-status:2} \\
&\quad - 0.00811 x_{marital-status:3} \\
&\quad + 0.235 x_{marital-status:4} \\
&\quad - 0.0367 x_{marital-status:6} \\
&\quad - 0.00431 x_{native-country:10} \\
&\quad + 0.0124 x_{native-country:14} \\
&\quad - 0.345 x_{native-country:15} \\
&\quad + 0.000695 x_{native-country:17} \\
&\quad - 0.00203 x_{native-country:18} \\
&\quad - 0.000263 x_{native-country:19} \\
&\quad + 0.00577 x_{native-country:20} \\
&\quad + 0.0283 x_{native-country:25} \\
&\quad + 0.0183 x_{native-country:27} \\
&\quad - 0.016 x_{native-country:2} \\
&\quad - 0.0318 x_{native-country:32} \\
&\quad + 0.0175 x_{native-country:37} \\
&\quad - 0.0186 x_{native-country:41} \\
&\quad + 0.161 x_{native-country:4} \\
&\quad + 0.0414 x_{native-country:6} \\
&\quad - 0.177 x_{occupation:10} \\
&\quad - 0.0946 x_{occupation:12} \\
&\quad - 0.103 x_{occupation:13} \\
&\quad + 0.158 x_{occupation:5} \\
&\quad + 0.248 x_{occupation:8} \\
&\quad + 0.0249 x_{race:0} \\
&\quad + 0.000716 x_{race:3} \\
&\quad - 0.049 x_{race:4} \\
&\quad - 0.926 x_{relationship:0} \\
&\quad - 0.291 x_{relationship:1} \\
&\quad + 0.0481 x_{workclass:2} \\
&\quad + 0.182 x_{workclass:6} \\
&\quad + 0.0675 x_{workclass:7}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Sparse linear backbone: 41 inputs each add a single weighted term, summed at identity node lin\_agg (std 1.99 logit on train). Dominant terms (output = P(\<=50K)): relationship:0 Husband -0.93, education-num -0.78, hours-per-week -0.36, age -0.36, marital-status:2 Married-civ-spouse -0.31, relationship:1 -0.29, capital-loss -0.27, occupation:8 Other-service +0.25, marital-status:4 Never-married +0.24. native-country:15 is constant on train (acts as a bias). Evidence (train, 39,073 rows): closed form verified by get\_formula; viz summaries confirm the exits; mean-ablating this leaf's logit contribution gives acc 0.8023 / AUC 0.7529 (baseline 0.8518 / 0.9056), mean \|dP\| 0.1587, max \|dP\| 0.907.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the exits (3 s.f.). Exits are read out by the output node with the weights given in the hypothesis.
- `empirical_viz`: Train split, all 39,073 rows: get\_viz\_summary results plus a local mean-ablation (replace this leaf's logit contribution with its train mean).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)",
 "tractable": true
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_samples": 39073,
 "viz_summaries": {
  "sensitivity": {
   "viz_type": "sensitivity",
   "exit_names": [
    "lin_agg"
   ],
   "entry_names": [
    "-1",
    "-4",
    "-8",
    "-9",
    "-11",
    "-18",
    "-20",
    "-25",
    "-27",
    "-28",
    "-31",
    "-32",
    "-33",
    "-35",
    "-38",
    "-40",
    "-41",
    "-46",
    "-49",
    "-51",
    "-52",
    "-57",
    "-60",
    "-61",
    "-64",
    "-65",
    "-68",
    "-72",
    "-73",
    "-75",
    "-76",
    "-77",
    "-78",
    "-79",
    "-84",
    "-86",
    "-92",
    "-97",
    "-100",
    "-102",
    "-104"
   ],
   "input_labels": [
    "-1",
    "-4",
    "-8",
    "-9",
    "-11",
    "-18",
    "-20",
    "-25",
    "-27",
    "-28",
    "-31",
    "-32",
    "-33",
    "-35",
    "-38",
    "-40",
    "-41",
    "-46",
    "-49",
    "-51",
    "-52",
    "-57",
    "-60",
    "-61",
    "-64",
    "-65",
    "-68",
    "-72",
    "-73",
    "-75",
    "-76",
    "-77",
    "-78",
    "-79",
    "-84",
    "-86",
    "-92",
    "-97",
    "-100",
    "-102",
    "-104"
   ],
   "sensitivities": {
    "lin_agg": [
     0.3552732017990543,
     0.048072258827224254,
     0.1819151178863023,
     0.067479377696681,
     0.0754807521467505,
     0.026853446452315285,
     0.0012393333195565859,
     0.04460081834509667,
     0.031243565821053206,
     0.7754905680028152,
     0.30505044056190067,
     0.008109909113841142,
     0.23531572677012455,
     0.03672442435106244,
     0.17678156768847114,
     0.0945950171978767,
     0.10324760702655124,
     0.15768534757400443,
     0.24807028338934223,
     0.926374872392665,
     0.29102084897247843,
     0.02486245150057444,
     0.0007158871128753382,
     0.04899168772649386,
     0.2720889123010229,
     0.35684458783729084,
     0.004312104705858934,
     0.012353701055730054,
     0.344742431350331,
     0.0006953890422630112,
     0.002033312093855972,
     0.0002629484910013694,
     0.015954839364591127,
     0.005774573671018727,
     0.028258939840262266,
     0.018264208343746542,
     0.03180380809658373,
     0.017511557235157814,
     0.1614373805216011,
     0.01859592292387139,
     0.04138381895184965
    ]
   },
   "dimensionality": [
    41,
    1
   ]
  }
 },
 "baseline_train": {
  "ll": 0.32108808276786016,
  "acc": 0.8517902387838149,
  "auc": 0.9055934526645549
 },
 "mean_ablation_train": {
  "acc": 0.802344329844138,
  "auc": 0.7528775215569109,
  "logloss": 0.5129581335680878,
… (truncated; full payload in the DB evidence record)
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 6 |
| annotate | 11 |
| consolidate\_node | 1 |
| rename\_node | 107 |
| split\_node | 2 |

2 splits, 6 identity nodes, 1 consolidations (1 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node idn\_o63 intercepts 1 connection(s) into 0 (from -63); sits in capital\_gain\_hinge (exit); full\_model (entry)
- `#108` identity node idn\_o10 intercepts 1 connection(s) into 0 (from -10 (workclass:8)); sits in without\_pay\_override (exit); full\_model (entry)
- `#109` identity node idn\_o45 intercepts 1 connection(s) into 0 (from -45 (occupation:4)); sits in exec\_managerial\_bump (exit); full\_model (entry)
- `#110` identity node idn\_o56 intercepts 1 connection(s) into 0 (from -56 (relationship:5)); sits in wife\_bump (exit); full\_model (entry)
- `#111` identity node idn\_o83 intercepts 1 connection(s) into 0 (from -83 (native-country:24)); sits in japan\_cancelling\_bump (exit); full\_model (entry)
- `#112` identity node lin\_agg intercepts 41 connection(s) into 0 (from -1 (age), -4 (workclass:2), -8 (workclass:6), -9 (workclass:7), -11 (fnlwgt), -18 (education:14), -20 (education:2), -25 (education:7), … (33 more)); sits in linear\_backbone (exit); full\_model (entry)
- `#113` split 816 into 816\_a, 816\_b. Final copies: 816\_a → capital\_gain\_hinge (exit); full\_model (entry) \| 816\_b → female\_or\_household\_gate (internal)
- `#114` split -63 into -63\_a, -63\_b, -63\_c. Final copies: -63\_ac → capital\_gain\_hinge (entry) \| -63\_b → female\_or\_household\_gate (entry) \| -63\_ac → capital\_gain\_hinge (entry)
- `#115` consolidate -63\_a, -63\_c into -63\_ac

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 5.55e-16 | 5.55e-16 | 5.55e-16 | agree |
| Closed-form max abs error (all leaves) | 0.02328 | 0.02328 | 0.02328 | agree |
| Numeric-function max abs error | n/a | 1.78e-15 | 1.78e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | full\_model | full\_model | agree |
| Leaf / composition counts | 10 / 1 | n/a | 10 / 1 | agree |
| Operation counts | 2 splits, 6 identity | {"add\_identity\_node": 6, "annotate": 11, "consolidate\_node": 1, "rename\_node": 107, "split\_node": 2} | {"add\_identity\_node": 6, "annotate": 11, "consolidate\_node": 1, "rename\_node": 107, "split\_node": 2} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capital_gain_hinge` | 0.02328 | 0 |
| `unknown_workclass_bump` | 0.0158 | 0 |
| `linear_backbone` | 0.01291 | 1.78e-15 |
| `japan_cancelling_bump` | 0.006896 | 0 |
| `without_pay_override` | 0.005119 | 0 |
| `female_or_household_gate` | 0.004577 | 8.88e-16 |
| `separated_bump` | 0.002651 | 0 |
| `wife_bump` | 0.00164 | 0 |
| `exec_managerial_bump` | 9.16e-04 | 0 |
| `dead_private_unit` | 0 | 0 |

## What resisted (from the record)

> Nothing structural. Capital-gain hinge 816 also feeds sex gate 369 (weight -0.0053): split 816 and capital-gain, consolidated two copies. Japan's net +0.14 is two near-cancelling large terms. Without-pay (17 rows) gets a +61 logit override, fitted to very few rows.

**Tool notes (record):** Root rejected if child exits listed in subgraph\_nodes (paper style: entries only). A child exit with other outputs cannot be a sibling leaf entry. SHAP \~5e-4 on unconnected inputs (noise).

## What to look at

- Largest closed-form error: `capital_gain_hinge` (0.0233), then `unknown_workclass_bump` (0.0158). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `without_pay_override`: its evidence includes a category or regime active on only 17 training rows. *“On the rare standardised value (\~48) the sum is \~+61 logit: an override that forces the prediction to \<=50K for unpaid workers;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `japan_cancelling_bump`: its evidence includes a category or regime active on only 68 training rows. *“native-country:24 (Japan, 68 rows) feeds an always-active ReLU 6081 (readout +0.205, \~+4.7 logit on the rare value) and a direct term (idn\_o83, \~-4.6 logit).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 1 hidden-node split(s), which decide how a unit's roles are divided: 816 → 816\_a → capital\_gain\_hinge (exit); full\_model (entry) \| 816\_b → female\_or\_household\_gate (internal). Do the copies really play the different roles the annotations give them?
- 1 consolidation(s): copies merged back after a split. Check that the merged copies carry the same role.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s0.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s0', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
