# Verification packet 28 · adult-s8

Pre-registered verification order: 28 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/2c47861c-efda-467a-b0b6-9dbd11dfac15?experiment=059121e2-f85b-4a3f-ae1a-8d115ab0e33f&name=10seed-adult-s8), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 8 |
| Genome id | `2c47861c-efda-467a-b0b6-9dbd11dfac15` |
| Experiment | 10seed-adult-s8 (059121e2-f85b-4a3f-ae1a-8d115ab0e33f); best at generation 99 of 100, NEAT key 23756 |
| Test AUC (runs csv) | 0.909 \[95% CI 0.903, 0.915\] |
| Train AUC | 0.911 |
| Size | 63 of 107 inputs connected; 15 hidden nodes; 89 enabled connections; depth 5; 15 skip connections + 54 direct input→output edges |
| Annotations | 11 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 15 of 15 |
| Claude's time | 17.6 min (2026-11-05T12:22:16+13:00 → 2026-11-05T12:39:49+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> P(income\<=50K) = sigmoid(-1.468 + 11 additive logit terms). Sparse linear backbone dominates (ablation AUC 0.911 -\> 0.611). Capital gain acts twice: a convex hinge and a gate cancelling a +4.25 (never-married +2.34) offset towards \<=50K once gains pass \~0.5 SD. An OR gate (youth outside Other-service, or Other-service) pushes towards \<=50K; exactly linear Widowed term; rare-category overrides. Train acc 0.854, AUC 0.911; 44 of 107 inputs unconnected.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `full_model`

|  |  |
|---|---|
| Annotation id | `ann_127` |
| Entry nodes | idn\_cg, idn\_gate, idn\_youth, idn\_widow, idn\_nic, idn\_japan, 574, 6334, 7843, 16365, lin\_agg, -5 (workclass:3), -7 (workclass:5), -9 (workclass:7), -10 (workclass:8), -11 (fnlwgt), -15 (education:11), -19 (education:15), -20 (education:2), -22 (education:4), -23 (education:5), -24 (education:6), -26 (education:8), -36 (occupation:0), -37 (occupation:1), -40 (occupation:12), -42 (occupation:14), -44 (occupation:3), -52 (relationship:1), -60 (race:3), -66 (native-country:0), -68 (native-country:10), -69 (native-country:11), -70 (native-country:12), -73 (native-country:15), -75 (native-country:17), -76 (native-country:18), -79 (native-country:20), -84 (native-country:25), -85 (native-country:26), … (15 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_hinge, capgain\_marital\_gate, age\_youth\_gate, widowed\_offset, nicaragua\_ownchild, japan\_offset, selfemp\_guatemala\_gate, armed\_forces\_offset, tenth\_grade\_offset, italy\_offset, linear\_backbone |
| Dimensionality (in → out, incl. descendants) | 55 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{4.34 e^{- 1.0 \operatorname{age\_youth\_gate}{\left(-1,-49 \right)} + 1.4 \operatorname{armed\_forces\_offset}{\left(-43 \right)} - 1.0 \operatorname{capgain\_marital\_gate}{\left(-63_{b},-33,-74,-67 \right)} - 1.0 \operatorname{capital\_gain\_hinge}{\left(-63_{ac} \right)} + 0.0271 \operatorname{italy\_offset}{\left(-81 \right)} - 1.0 \operatorname{japan\_offset}{\left(-83 \right)} - 1.0 \operatorname{linear\_backbone}{\left(-2,-3,-4,-6,-13,-14,-16,-17,-18,-21,-25,-27,-28,-29,-30,-31,-32,-34,-38,-39,-41,-45,-46,-47,-48,-50,-51,-53,-55,-56,-57,-58,-59,-61,-62,-64,-65,-72,-77,-78,-80,-82,-88,-90,-94,-95,-107 \right)} - 1.0 \operatorname{nicaragua\_ownchild}{\left(-54,-86,-89 \right)} - 0.218 \operatorname{selfemp\_guatemala\_gate}{\left(-8,-71 \right)} - 0.0834 \operatorname{tenth\_grade\_offset}{\left(-12 \right)} - 1.0 \operatorname{widowed\_offset}{\left(-35 \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{1}{4.34 e^{0.0271 x_{16365} - 0.218 x_{574} + 1.4 x_{6334} - 0.0834 x_{7843} - 1.0 x_{idn\_cg} - 1.0 x_{idn\_gate} - 1.0 x_{idn\_japan} - 1.0 x_{idn\_nic} - 1.0 x_{idn\_widow} - 1.0 x_{idn\_youth} - 1.0 x_{lin\_agg}} + 1}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> P(income\<=50K) = sigmoid(-1.468 + linear\_backbone + capital\_gain\_hinge + capgain\_marital\_gate + age\_youth\_gate + widowed\_offset + nicaragua\_ownchild + selfemp\_guatemala\_gate + armed\_forces\_offset + tenth\_grade\_offset + italy\_offset + japan\_offset): purely additive in the logit (output node 0 sees only the 11 child exits; weights 1 for identity exits, 0.218/-1.398/0.083/-0.027 for the four single-unit leaves). The 44 unconnected inputs (incl. fnlwgt, race:3, most countries) are zero-effect entries. Structure: a sparse linear model (marital status, education-num, relationship, sex, hours, capital-loss) plus a capital-gain effect split between a convex hinge and a gate that cancels a +4.25/+2.34 logit baseline offset (not never-married / never-married); a youth-or-Other-service OR gate; and single-category overrides for rare categories (Armed-Forces -80, Honduras +183, Guatemala +4.6, Nicaragua +3.1) fitted to tens of rows. Evidence (train, 39,073 rows): acc 0.8542, AUC 0.9108; mean-ablating the backbone drops AUC to 0.611, every other child to \>=0.902. SHAP (200 train rows, mean \|dP\|): Married-civ 0.245, Never-married 0.066, education-num 0.052, age 0.050, Husband 0.045, capital-gain 0.036. Function preservation error 7e-16.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `shap`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// shap
{
 "note": "unconnected inputs have zero effect by construction",
 "split": "train",
 "base_value": 0.7497,
 "max_samples": 200,
 "top_mean_abs_shap_on_P": [
  [
   "marital-status:2",
   0.2454
  ],
  [
   "marital-status:4",
   0.0662
  ],
  [
   "education-num",
   0.0523
  ],
  [
   "age",
   0.0495
  ],
  [
   "relationship:0",
   0.0454
  ],
  [
   "capital-gain",
   0.0359
  ],
  [
   "marital-status:0",
   0.0302
  ],
  [
   "sex",
   0.0242
  ],
  [
   "hours-per-week",
   0.0226
  ],
  [
   "relationship:3",
   0.0171
  ],
  [
   "occupation:4",
   0.017
  ],
  [
   "capital-loss",
   0.0135
  ]
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "compute_performance": {
  "auc": 0.9108,
  "accuracy": 0.8542
 },
 "local_mean_ablation_per_child": "backbone AUC 0.6107; gate 0.9055; cg 0.9047; youth 0.9024; widowed 0.9091; nic 0.9105; selfemp 0.9100; others >=0.9108",
 "function_preservation_max_abs_err": 7.2e-16
}
```

</details>

### A.1 Leaf: `capital_gain_hinge`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -63\_ac (capital-gain\_ac) |
| Exit nodes | idn\_cg |
| Internal / exit nodes in its own subgraph | 9424, idn\_cg |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0386 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 2.01 x_{capital-gain\_ac} \\
&\quad - 2.61 \operatorname{relu}{\left(0.77 x_{capital-gain\_ac} - 0.409 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Capital-gain (standardised) enters the logit as a convex decreasing piecewise-linear function y = -2.01\*cg - 2.61\*relu(0.770\*cg - 0.409) (exit idn\_cg feeds output with weight 1). Slope is -2.0 below cg = 0.531 SD and -4.0 above; 4.8% of rows are above the kink. Large capital gains drive P(\<=50K) to \~0. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): closed form from get\_formula; line\_plot summary: train contribution ranges -51.9 to +0.29, correlation with cg -0.997 (monotone). Local mean-ablation of this leaf: acc 0.8445, AUC 0.9047, mean\|dP\| 0.057, max\|dP\| 0.998.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": -25.256753806635608,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   -54.58262964620923,
   1.6320782300543728
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "idn_cg"
  ],
  "correlation": -0.9973779965736724,
  "entry_names": [
   "-63_ac"
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
   -51.90485238075245,
   0.2924268585707299
  ]
 },
 "viz_type": "line_plot",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8445 AUC 0.9047 mean|dP| 0.0573 max|dP| 0.998"
}
```

</details>

### A.2 Leaf: `capgain_marital_gate`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -63\_b (capital-gain\_b), -33 (marital-status:4), -74 (native-country:16), -67 (native-country:1) |
| Exit nodes | idn\_gate |
| Internal / exit nodes in its own subgraph | 1594, 1013, 1172, 9617, 3583, idn\_gate |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.0915 |
| Live numeric-function error | 2.84e-14 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.88 x_{native-country:16} \\
&\quad + 0.254 \operatorname{relu}{\left(0.845 x_{marital-status:4} + 0.176 \right)} \\
&\quad + 0.0385 \operatorname{relu}{\left(0.988 \operatorname{relu}{\left(- 0.426 x_{native-country:1} + 0.512 \operatorname{relu}{\left(- 0.438 x_{marital-status:4} + 0.619 x_{native-country:16} - 1.17 \operatorname{relu}{\left(0.682 x_{capital-gain\_b} - 0.35 \right)} + 1.47 \right)} - 0.267 \right)} - 0.272 \right)} \\
&\quad + 2.41 \operatorname{relu}{\left(- 0.438 x_{marital-status:4} + 0.619 x_{native-country:16} - 1.17 \operatorname{relu}{\left(0.682 x_{capital-gain\_b} - 0.35 \right)} + 1.47 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Gate leaf (exit idn\_gate). Main term 2.413\*h1013 with h1013 = relu(1.473 - 0.438\*NeverMarried + 0.619\*Honduras - 1.17\*relu(0.682\*cg - 0.35)); plus 0.254\*relu(0.176 + 0.845\*NeverMarried) (unit 3583), plus Honduras' direct +1.88, plus a negligible chain 0.039\*relu(-0.272 + 0.988\*relu(-0.267 - 0.426\*Cambodia + 0.512\*h1013)). Behaviour: a constant offset towards \<=50K of +4.25 logit for non-never-married rows and +2.34 for never-married rows (1013 shrinks more than 3583 grows), which a second capital-gain hinge (unit 1594, kink at cg 0.51 SD) removes as capital gain rises; the offset is gone above cg 2.74 SD (0.7% of rows). So capital gain also acts by cancelling a baseline offset, and its effect is larger for non-never-married people. Honduras (14 rows) gets +183 logit (forced to \<=50K, fitted to 14 rows). Combined with the linear backbone's marital weights, the net marital effect is: Married-civ-spouse \~3 logit lower than Divorced/Never-married/Separated/Spouse-absent; Widowed in between. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): sensitivity summary: \|dy/dx\| cg 0.08, NeverMarried 0.98, Honduras 3.37, Cambodia 0.011 (Cambodia path is negligible). Local mean-ablation: acc 0.8402, AUC 0.9055, mean\|dP\| 0.068.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_gate"
  ],
  "entry_names": [
   "-63_b",
   "-33",
   "-74",
   "-67"
  ],
  "input_labels": [
   "-63_b",
   "-33",
   "-74",
   "-67"
  ],
  "sensitivities": {
   "idn_gate": [
    0.08250204711389983,
    0.9842374963556945,
    3.3685399133236196,
    0.010555313318213775
   ]
  },
  "dimensionality": [
   4,
   1
  ]
 },
 "viz_type": "sensitivity",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8402 AUC 0.9055 mean|dP| 0.0677 max|dP| 0.989"
}
```

</details>

### A.3 Leaf: `age_youth_gate`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -1 (age), -49 (occupation:8) |
| Exit nodes | idn\_youth |
| Internal / exit nodes in its own subgraph | 251, idn\_youth |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0168 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0411 x_{age} \\
&\quad + 0.374 x_{occupation:8} \\
&\quad + 1.42 \operatorname{relu}{\left(- 1.1 x_{age} - 2.11 x_{occupation:8} - 0.82 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Exit idn\_youth = 1.416\*relu(-0.820 - 1.096\*age - 2.108\*OtherService) - 0.041\*age + 0.374\*OtherService (age, OtherService standardised). For non-Other-service workers the ReLU fires below about median age (45% of rows active) and the logit towards \<=50K rises linearly as age falls (+2.1 at age -1.5 SD, \~-0.15 above median). For Other-service workers the ReLU is dead and they get a flat +1.1 regardless of age: Other-service and youth are substitutes (OR-like), not additive. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): sensitivity \|dy/dx\| age 0.74, OtherService 1.35. Local mean-ablation: acc 0.8489, AUC 0.9024, mean\|dP\| 0.053.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_youth"
  ],
  "entry_names": [
   "-1",
   "-49"
  ],
  "input_labels": [
   "-1",
   "-49"
  ],
  "sensitivities": {
   "idn_youth": [
    0.737866128775087,
    1.3525287117818754
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8489 AUC 0.9024 mean|dP| 0.0533 max|dP| 0.349"
}
```

</details>

### A.4 Leaf: `widowed_offset`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -35 (marital-status:6) |
| Exit nodes | idn\_widow |
| Internal / exit nodes in its own subgraph | 8342, idn\_widow |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00246 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0759 x_{marital-status:6} \\
&\quad - 0.368 \operatorname{relu}{\left(0.987 x_{marital-status:6} + 1.29 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Exit idn\_widow = -0.368\*relu(1.295 + 0.987\*Widowed) - 0.076\*Widowed. Unit 8342 is always active on the data, so the leaf is exactly linear: -0.40 for non-widowed rows, -2.96 for the 1,188 widowed rows (a -2.56 shift towards \>50K relative to other unmarried categories, partly offsetting the +4.25 gate offset). Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): line\_plot summary: correlation -1.000 (linear), range -2.96 to -0.40. Local mean-ablation: AUC 0.9091, mean\|dP\| 0.0095.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": -1.677843042202045,
  "x_range": [
   -0.46829113246881526,
   5.938308380796323
  ],
  "y_range": [
   -3.084818448944952,
   -0.2708676354591384
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "idn_widow"
  ],
  "correlation": -0.9999999999999947,
  "entry_names": [
   "-35"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.17708206368403626,
   5.647099312011544
  ],
  "scatter_y_range": [
   -2.956911593786506,
   -0.3987744906175845
  ]
 },
 "viz_type": "line_plot",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8537 AUC 0.9091 mean|dP| 0.0095 max|dP| 0.551"
}
```

</details>

### A.5 Leaf: `nicaragua_ownchild`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -54 (relationship:3), -86 (native-country:27), -89 (native-country:3) |
| Exit nodes | idn\_nic |
| Internal / exit nodes in its own subgraph | 6580, 2342, idn\_nic |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00931 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.396 x_{relationship:3} \\
&\quad + 0.035 \operatorname{relu}{\left(0.233 x_{native-country:27} - 0.145 x_{native-country:3} + 1.0 \right)} \\
&\quad + 0.108 \operatorname{relu}{\left(1.0 x_{native-country:27} + 1.0 x_{relationship:3} - 3.89 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Exit idn\_nic = 0.396\*OwnChild + 0.108\*relu(-3.887 + OwnChild + Nicaragua) + 0.035\*relu(1.0 + 0.233\*Nicaragua - 0.145\*China). In practice: a linear Own-child offset (+0.92 logit for own-child vs -0.17 otherwise, towards \<=50K); unit 6580 fires only for Nicaragua (42 rows; \~+2.8 logit, \~+3.1 with unit 2342, +0.25 more with own-child; the own-child x Nicaragua interaction is real but fitted to a handful of rows); unit 2342 is \~constant (0.035) with tiny Nicaragua/China slopes. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): sensitivity \|dy/dx\| OwnChild 0.40, Nicaragua 0.008, China 0.005. Local mean-ablation: AUC 0.9105, mean\|dP\| 0.018.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_nic"
  ],
  "entry_names": [
   "-54",
   "-86",
   "-89"
  ],
  "input_labels": [
   "-54",
   "-86",
   "-89"
  ],
  "sensitivities": {
   "idn_nic": [
    0.39568614166363647,
    0.008255086578077624,
    0.005066065643915805
   ]
  },
  "dimensionality": [
   3,
   1
  ]
 },
 "viz_type": "sensitivity",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8547 AUC 0.9105 mean|dP| 0.0184 max|dP| 0.503"
}
```

</details>

### A.6 Leaf: `japan_offset`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -83 (native-country:24) |
| Exit nodes | idn\_japan |
| Internal / exit nodes in its own subgraph | 10879, idn\_japan |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 6.09e-05 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.00633 x_{native-country:24} \\
&\quad - 0.00417 \operatorname{relu}{\left(0.998 x_{native-country:24} + 1.48 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Exit idn\_japan = -0.004\*relu(1.476 + 0.998\*Japan) + 0.006\*Japan; unit 10879 always active, so linear with net slope \~0.002: effectively zero (max contribution 0.046 logit for the 68 Japan rows). Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): line\_plot correlation 1.000, range -0.006 to 0.046. Local mean-ablation: no change (mean\|dP\| 0.00002).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 0.019789547557411843,
  "x_range": [
   -1.241341779775199,
   25.149597352751194
  ],
  "y_range": [
   -0.008845126155533557,
   0.048424221270357254
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "idn_japan"
  ],
  "correlation": 0.9999999999999942,
  "entry_names": [
   "-83"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.04175363738763561,
   23.950009210363632
  ],
  "scatter_y_range": [
   -0.006241973999811246,
   0.04582106911463494
  ]
 },
 "viz_type": "line_plot",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8542 AUC 0.9108 mean|dP| 0.0000 max|dP| 0.013"
}
```

</details>

### A.7 Leaf: `selfemp_guatemala_gate`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -8 (workclass:6), -71 (native-country:13) |
| Exit nodes | 574 |
| Internal / exit nodes in its own subgraph | 574 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0101 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.958 x_{native-country:13} \\
&\quad + 0.955 x_{workclass:6} \\
&\quad -1.3 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Exit is unit 574 activation a = relu(-1.296 + 0.955\*SelfEmpNotInc + 0.958\*Guatemala), entering the logit with weight +0.218. Zero for 92% of rows; for Self-emp-not-inc (3,034 rows) a \~1.98 (+0.43 logit towards \<=50K); for Guatemala (72 rows) a \~21 (+4.6 logit); both \~24 (+5.3). OR-like gate over two unrelated categories. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): sensitivity summary \|da/dx\| 0.076 for each input. Local mean-ablation: acc 0.8530, AUC 0.9100, mean\|dP\| 0.0075.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "574"
  ],
  "entry_names": [
   "-8",
   "-71"
  ],
  "input_labels": [
   "-8",
   "-71"
  ],
  "sensitivities": {
   "574": [
    0.07581901354502464,
    0.07610231673749583
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8530 AUC 0.9100 mean|dP| 0.0075 max|dP| 0.610"
}
```

</details>

### A.8 Leaf: `armed_forces_offset`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -43 (occupation:2) |
| Exit nodes | 6334 |
| Internal / exit nodes in its own subgraph | 6334 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 2.26e-05 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{occupation:2} - 0.0367 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Exit is unit 6334 activation a = relu(-0.037 + ArmedForces), weight -1.398 into the logit. Exactly zero except for the 12 Armed-Forces rows, where a = 57.0 and the logit drops by 79.7, forcing P(\<=50K) to \~0 (predicts \>50K) although 75% of those rows are \<=50K: an overfit single-category override. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): line\_plot: activation 0 to 57.0, correlation 1.0 with the indicator. Local mean-ablation: acc unchanged 0.8542, AUC 0.9110 (slightly better without it), max\|dP\| 0.993.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 28.55563494988276,
  "x_range": [
   -2.8710707121798156,
   59.90688087162306
  ],
  "y_range": [
   0.0,
   59.87020343011147
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "6334"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-43"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.017527458370593843,
   57.05333761781384
  ],
  "scatter_y_range": [
   0.0,
   57.016660176302246
  ]
 },
 "viz_type": "line_plot",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8542 AUC 0.9110 mean|dP| 0.0026 max|dP| 0.993"
}
```

</details>

### A.9 Leaf: `tenth_grade_offset`

|  |  |
|---|---|
| Annotation id | `ann_124` |
| Entry nodes | -12 (education:0) |
| Exit nodes | 7843 |
| Internal / exit nodes in its own subgraph | 7843 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00183 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.948 x_{education:0} - 0.958 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Exit is unit 7843 activation a = relu(-0.958 + 0.948\*Education10th), weight +0.083. Zero except for 10th-grade education (1,156 rows), a = 4.47, +0.37 logit towards \<=50K. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): line\_plot: activation 0 to 4.47. Local mean-ablation: mean\|dP\| 0.0015, AUC 0.9108.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 1.8357625908572377,
  "x_range": [
   -0.46969464560207264,
   6.022231061911879
  ],
  "y_range": [
   0.0,
   4.7491529918345865
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "7843"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-12"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.1746071134423475,
   5.727143529752154
  ],
  "scatter_y_range": [
   0.0,
   4.469507061671054
  ]
 },
 "viz_type": "line_plot",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8543 AUC 0.9108 mean|dP| 0.0015 max|dP| 0.090"
}
```

</details>

### A.10 Leaf: `italy_offset`

|  |  |
|---|---|
| Annotation id | `ann_125` |
| Entry nodes | -81 (native-country:22) |
| Exit nodes | 16365 |
| Internal / exit nodes in its own subgraph | 16365 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00607 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{native-country:22} - 0.541 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Exit is unit 16365 activation a = relu(-0.541 + Italy), weight -0.027. Zero except for Italy (88 rows), a = 20.5, -0.56 logit (towards \>50K). Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): line\_plot: activation 0 to 20.5. Local mean-ablation: mean\|dP\| 0.0003.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "y_mean": 10.018150374798658,
  "x_range": [
   -1.1022781221507731,
   22.102602427148508
  ],
  "y_range": [
   0.0,
   21.5552248983002
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "16365"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-81"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.04751082445535133,
   21.047835129453087
  ],
  "scatter_y_range": [
   0.0,
   20.500761647274842
  ]
 },
 "viz_type": "line_plot",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.8541 AUC 0.9108 mean|dP| 0.0003 max|dP| 0.138"
}
```

</details>

### A.11 Leaf: `linear_backbone`

|  |  |
|---|---|
| Annotation id | `ann_126` |
| Entry nodes | -2 (workclass:0), -3 (workclass:1), -4 (workclass:2), -6 (workclass:4), -13 (education:1), -14 (education:10), -16 (education:12), -17 (education:13), -18 (education:14), -21 (education:3), -25 (education:7), -27 (education:9), -28 (education-num), -29 (marital-status:0), -30 (marital-status:1), -31 (marital-status:2), -32 (marital-status:3), -34 (marital-status:5), -38 (occupation:10), -39 (occupation:11), -41 (occupation:13), -45 (occupation:4), -46 (occupation:5), -47 (occupation:6), -48 (occupation:7), -50 (occupation:9), -51 (relationship:0), -53 (relationship:2), -55 (relationship:4), -56 (relationship:5), -57 (race:0), -58 (race:1), -59 (race:2), -61 (race:4), -62 (sex), -64 (capital-loss), -65 (hours-per-week), -72 (native-country:14), -77 (native-country:19), -78 (native-country:2), … (7 more) |
| Exit nodes | lin\_agg |
| Internal / exit nodes in its own subgraph | lin\_agg |
| Dimensionality (in → out, incl. descendants) | 47 → 1 |
| Live closed-form error (3 s.f. display) | 0.0121 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.273 x_{capital-loss} \\
&\quad - 0.654 x_{education-num} \\
&\quad - 0.0295 x_{education:10} \\
&\quad - 0.0307 x_{education:12} \\
&\quad + 0.0375 x_{education:13} \\
&\quad - 0.0557 x_{education:14} \\
&\quad + 0.0904 x_{education:1} \\
&\quad - 0.0269 x_{education:3} \\
&\quad + 0.0186 x_{education:7} \\
&\quad - 0.0192 x_{education:9} \\
&\quad - 0.314 x_{hours-per-week} \\
&\quad - 0.685 x_{marital-status:0} \\
&\quad - 0.154 x_{marital-status:1} \\
&\quad - 2.46 x_{marital-status:2} \\
&\quad - 0.241 x_{marital-status:3} \\
&\quad - 0.348 x_{marital-status:5} \\
&\quad + 0.0199 x_{native-country:14} \\
&\quad + 0.00718 x_{native-country:19} \\
&\quad - 0.0362 x_{native-country:21} \\
&\quad - 0.0132 x_{native-country:23} \\
&\quad + 0.0349 x_{native-country:29} \\
&\quad - 0.0154 x_{native-country:2} \\
&\quad - 0.0218 x_{native-country:30} \\
&\quad + 0.027 x_{native-country:34} \\
&\quad + 0.0664 x_{native-country:35} \\
&\quad - 0.0316 x_{native-country:9} \\
&\quad - 0.114 x_{occupation:10} \\
&\quad - 0.0533 x_{occupation:11} \\
&\quad - 0.0822 x_{occupation:13} \\
&\quad - 0.212 x_{occupation:4} \\
&\quad + 0.189 x_{occupation:5} \\
&\quad + 0.149 x_{occupation:6} \\
&\quad + 0.103 x_{occupation:7} \\
&\quad + 0.114 x_{occupation:9} \\
&\quad + 0.0109 x_{race:0} \\
&\quad - 0.0656 x_{race:1} \\
&\quad - 0.036 x_{race:2} \\
&\quad - 0.125 x_{race:4} \\
&\quad + 0.464 x_{relationship:0} \\
&\quad + 0.189 x_{relationship:2} \\
&\quad + 0.0423 x_{relationship:4} \\
&\quad - 0.0459 x_{relationship:5} \\
&\quad - 0.341 x_{sex} \\
&\quad + 0.159 x_{workclass:0} \\
&\quad - 0.0946 x_{workclass:1} \\
&\quad + 0.0412 x_{workclass:2} \\
&\quad - 0.0425 x_{workclass:4}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Sparse linear logit over 47 inputs (identity node lin\_agg). Largest terms in logit units per SD: Married-civ-spouse -2.46, Divorced -0.69, education-num -0.65, Husband +0.46, Separated -0.35, sex(Male) -0.34, hours-per-week -0.31, capital-loss -0.27, Spouse-absent -0.24, Exec-managerial -0.21, Farming-fishing +0.19, Other-relative +0.19, unknown workclass +0.16, Married-AF -0.15, Handlers-cleaners +0.15. Being married-civ, educated, working long hours, having capital loss, managerial occupation push towards \>50K; country terms are tiny. Note the marital terms are relative: never-married has no direct weight here but is handled in capgain\_marital\_gate. Evidence (train split, 39,073 rows; output is P(income\<=50K); baseline acc 0.8542 / AUC 0.9108): sensitivity summary = the weights; contribution range -6.9 to +8.7 logit, SD 2.53. Local mean-ablation: acc 0.7805, AUC 0.6107, mean\|dP\| 0.21: the backbone carries most of the ranking.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "lin_agg"
  ],
  "entry_names": [
   "-2",
   "-3",
   "-4",
   "-6",
   "-13",
   "-14",
   "-16",
   "-17",
   "-18",
   "-21",
   "-25",
   "-27",
   "-28",
   "-29",
   "-30",
   "-31",
   "-32",
   "-34",
   "-38",
   "-39",
   "-41",
   "-45",
   "-46",
   "-47",
   "-48",
   "-50",
   "-51",
   "-53",
   "-55",
   "-56",
   "-57",
   "-58",
   "-59",
   "-61",
   "-62",
   "-64",
   "-65",
   "-72",
   "-77",
   "-78",
   "-80",
   "-82",
   "-88",
   "-90",
   "-94",
   "-95",
   "-107"
  ],
  "input_labels": [
   "-2",
   "-3",
   "-4",
   "-6",
   "-13",
   "-14",
   "-16",
   "-17",
   "-18",
   "-21",
   "-25",
   "-27",
   "-28",
   "-29",
   "-30",
   "-31",
   "-32",
   "-34",
   "-38",
   "-39",
   "-41",
   "-45",
   "-46",
   "-47",
   "-48",
   "-50",
   "-51",
   "-53",
   "-55",
   "-56",
   "-57",
   "-58",
   "-59",
   "-61",
   "-62",
   "-64",
   "-65",
   "-72",
   "-77",
   "-78",
   "-80",
   "-82",
   "-88",
   "-90",
   "-94",
   "-95",
   "-107"
  ],
  "sensitivities": {
   "lin_agg": [
    0.15892342318734368,
    0.0945564016156201,
    0.041190106737814064,
    0.04246751815531145,
    0.09036898438236926,
    0.029481715477644596,
    0.0307303688705341,
    0.03753433249680371,
    0.05568261425149504,
    0.02688146962448984,
    0.018644509888152385,
    0.019207079150133966,
    0.6543732176363949,
    0.6850432065931957,
    0.15381230316832256,
    2.455722445926491,
    0.24144159092853765,
    0.3484317037112482,
    0.11392080968832669,
    0.05325775037007262,
    0.08220973644799978,
    0.2119856083414365,
    0.18899730804111076,
    0.1493890594291165,
    0.10270869737691884,
    0.1142413866681919,
    0.4643790513422235,
    0.18857938055777895,
    0.04227707134294154,
    0.04592602758313266,
    0.010932646303942284,
    0.06557272164320344,
    0.0360187315794365,
    0.1253032647658916,
    0.3413999625726866,
    0.2725458390308226,
    0.31362624636239084,
    0.019887178024857403,
    0.0071846149680465845,
    0.015399763739314024,
    0.03615647552864442,
    0.013197840442206196,
    0.03491741119390486,
    0.021756558781635904,
    0.027046186182104887,
    0.0664013232815402,
    0.0315946598837067
   ]
  },
  "dimensionality": [
   47,
   1
  ]
 },
 "viz_type": "sensitivity",
 "n_samples": 39073,
 "local_mean_ablation_train": "acc 0.7805 AUC 0.6107 mean|dP| 0.2092 max|dP| 0.936"
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 7 |
| annotate | 12 |
| consolidate\_node | 1 |
| rename\_node | 107 |
| split\_node | 1 |

1 splits, 7 identity nodes, 1 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` split -63 into -63\_a, -63\_b, -63\_c. Final copies: -63\_ac → capital\_gain\_hinge (entry) \| -63\_b → capgain\_marital\_gate (entry) \| -63\_ac → capital\_gain\_hinge (entry)
- `#108` consolidate -63\_a, -63\_c into -63\_ac
- `#109` identity node idn\_cg intercepts 2 connection(s) into 0 (from -63\_ac (capital-gain\_ac), 9424); sits in capital\_gain\_hinge (exit); full\_model (entry)
- `#110` identity node idn\_gate intercepts 4 connection(s) into 0 (from -74 (native-country:16), 1013, 9617, 3583); sits in capgain\_marital\_gate (exit); full\_model (entry)
- `#111` identity node idn\_youth intercepts 3 connection(s) into 0 (from -1 (age), -49 (occupation:8), 251); sits in age\_youth\_gate (exit); full\_model (entry)
- `#112` identity node idn\_widow intercepts 2 connection(s) into 0 (from -35 (marital-status:6), 8342); sits in widowed\_offset (exit); full\_model (entry)
- `#113` identity node idn\_nic intercepts 3 connection(s) into 0 (from -54 (relationship:3), 6580, 2342); sits in nicaragua\_ownchild (exit); full\_model (entry)
- `#114` identity node idn\_japan intercepts 2 connection(s) into 0 (from -83 (native-country:24), 10879); sits in japan\_offset (exit); full\_model (entry)
- `#115` identity node lin\_agg intercepts 47 connection(s) into 0 (from -2 (workclass:0), -3 (workclass:1), -4 (workclass:2), -6 (workclass:4), -13 (education:1), -14 (education:10), -16 (education:12), -17 (education:13), … (39 more)); sits in linear\_backbone (exit); full\_model (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 7.22e-16 | 7.22e-16 | 7.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.09155 | 0.09155 | 0.09155 | agree |
| Numeric-function max abs error | n/a | 2.84e-14 | 2.84e-14 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | full\_model | full\_model | agree |
| Leaf / composition counts | 11 / 1 | n/a | 11 / 1 | agree |
| Operation counts | 1 splits, 7 identity | {"add\_identity\_node": 7, "annotate": 12, "consolidate\_node": 1, "rename\_node": 107, "split\_node": 1} | {"add\_identity\_node": 7, "annotate": 12, "consolidate\_node": 1, "rename\_node": 107, "split\_node": 1} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capgain_marital_gate` | 0.09155 | 2.84e-14 |
| `capital_gain_hinge` | 0.03857 | 0 |
| `age_youth_gate` | 0.01677 | 0 |
| `linear_backbone` | 0.01211 | 3.55e-15 |
| `selfemp_guatemala_gate` | 0.01014 | 0 |
| `nicaragua_ownchild` | 0.009314 | 8.88e-16 |
| `italy_offset` | 0.006073 | 0 |
| `widowed_offset` | 0.00246 | 0 |
| `tenth_grade_offset` | 0.001825 | 0 |
| `japan_offset` | 6.09e-05 | 0 |
| `armed_forces_offset` | 2.26e-05 | 0 |

## What resisted (from the record)

> Nothing structural. Single-category overrides fitted to tens of rows (Armed-Forces -80 logit/12 rows, Honduras +183/14, Guatemala +4.6, Nicaragua +3.1); Own-child x Nicaragua interaction rests on a handful of rows. Marital status split across backbone and gate (net effect only readable across both). Cambodia chain and Japan unit effectively zero. Category names inferred.

**Tool notes (record):** get\_viz\_summary for a leaf whose exit is a hidden unit reports pre-weight activation, not logit contribution (narratives give both).

## What to look at

- Largest closed-form error: `capgain_marital_gate` (0.0915), then `capital_gain_hinge` (0.0386). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `widowed_offset`: *“Unit 8342 is always active on the data, so the leaf is exactly linear: -0.40 for non-widowed rows, -2.96 for the 1,188 widowed rows (a -2.56 shift towards \>50K relative to other unmarried categories, partly offsetting the +4.25 ga…”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `age_youth_gate`: *“For Other-service workers the ReLU is dead and they get a flat +1.1 regardless of age: Other-service and youth are substitutes (OR-like), not additive.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 1 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- 1 consolidation(s): copies merged back after a split. Check that the merged copies carry the same role.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s8.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s8', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
