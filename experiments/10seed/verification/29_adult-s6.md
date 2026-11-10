# Verification packet 29 · adult-s6

Pre-registered verification order: 29 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66?experiment=4793995f-ca47-46b9-b087-5f2655a166df&name=10seed-adult-s6), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 6 |
| Genome id | `9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66` |
| Experiment | 10seed-adult-s6 (4793995f-ca47-46b9-b087-5f2655a166df); best at generation 99 of 100, NEAT key 18196 |
| Test AUC (runs csv) | 0.905 \[95% CI 0.899, 0.911\] |
| Train AUC | 0.908 |
| Size | 53 of 107 inputs connected; 15 hidden nodes; 77 enabled connections; depth 4; 14 skip connections + 43 direct input→output edges |
| Annotations | 7 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 15 of 15 |
| Claude's time | 15.2 min (2026-11-05T12:22:26+13:00 → 2026-11-05T12:37:38+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> P(\<=50K) = sigmoid(0.776 + sparse linear backbone + additive child terms); backbone does most of the work (ablation AUC 0.908 -\> 0.748). Two nonlinear terms matter: a concave piecewise-linear capital-gain penalty (kink \~5.2k USD; ablation AUC 0.890) and a youth hinge (+0.16 logit per year below \~35; ablation AUC 0.900). Five tiny categorical gates. Train acc 0.852, AUC 0.908.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | identity\_age, identity\_capgain, identity\_techsupport, identity\_asianpac, 7165, 11966, 10191, -2 (workclass:0), -3 (workclass:1), -4 (workclass:2), -6 (workclass:4), -8 (workclass:6), -9 (workclass:7), -11 (fnlwgt), -20 (education:2), -21 (education:3), -23 (education:5), -28 (education-num), -31 (marital-status:2), -38 (occupation:10), -39 (occupation:11), -45 (occupation:4), -46 (occupation:5), -49 (occupation:8), -51 (relationship:0), -54 (relationship:3), -56 (relationship:5), -59 (race:2), -62 (sex), -64 (capital-loss), -65 (hours-per-week), -68 (native-country:10), -72 (native-country:14), -74 (native-country:16), -75 (native-country:17), -76 (native-country:18), -77 (native-country:19), -80 (native-country:21), -82 (native-country:23), -83 (native-country:24), … (61 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | youth\_hinge, capital\_gain\_penalty, tech\_support\_gate, asian\_pac\_offset, handlers\_doctorate\_relay, priv\_house\_serv\_gate, china\_gate |
| Dimensionality (in → out, incl. descendants) | 101 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.46 e^{0.298 x_{capital-loss} + 1.41 x_{education-num} + 0.0673 x_{education:2} + 0.187 x_{education:3} + 0.261 x_{education:5} + 0.0918 x_{fnlwgt} + 0.331 x_{hours-per-week} + 0.725 x_{marital-status:2} + 0.00311 x_{native-country:10} - 0.0208 x_{native-country:14} + 0.897 x_{native-country:15\_a} + 0.0144 x_{native-country:16} - 0.0117 x_{native-country:17} + 0.00657 x_{native-country:18} - 0.0241 x_{native-country:19} + 0.0448 x_{native-country:21} + 0.00848 x_{native-country:23} - 0.0156 x_{native-country:24} + 0.037 x_{native-country:32\_a} - 0.0296 x_{native-country:34} - 0.0546 x_{native-country:40} + 0.0117 x_{native-country:5} - 0.0463 x_{native-country:6} + 0.0209 x_{native-country:9} + 0.00226 x_{occupation:10} + 0.0745 x_{occupation:11} + 0.181 x_{occupation:4} - 0.181 x_{occupation:5} - 0.294 x_{occupation:8} - 0.0682 x_{race:2} + 0.559 x_{relationship:0} - 0.426 x_{relationship:3} + 0.562 x_{relationship:5} + 0.416 x_{sex} - 0.218 x_{workclass:0} + 0.0623 x_{workclass:1} - 0.0967 x_{workclass:2} - 0.0399 x_{workclass:4} - 0.217 x_{workclass:6} - 0.113 x_{workclass:7} - 1.0 \operatorname{asian\_pac\_offset}{\left(-61,-58 \right)} - 1.0 \operatorname{capital\_gain\_penalty}{\left(-63 \right)} - 0.0621 \operatorname{china\_gate}{\left(-89 \right)} - 0.294 \operatorname{handlers\_doctorate\_relay}{\left(-14,-47,-67 \right)} - 0.1 \operatorname{priv\_house\_serv\_gate}{\left(-19,-92_{b},-73_{b},-50 \right)} - 1.0 \operatorname{tech\_support\_gate}{\left(-41,-71,-66 \right)} - 1.0 \operatorname{youth\_hinge}{\left(-1 \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: the network is an additive logistic model. logit P(\<=50K) = 0.776 + sparse linear backbone over 40 direct inputs + sum of 7 child contributions, then sigmoid. Backbone (direct edges, standardised inputs): education-num -1.41, marital-status:2 (Married-civ-spouse) -0.73, relationship:5 (Wife) -0.56, relationship:0 (Husband) -0.56, relationship:3 (Own-child) +0.43, sex (Male) -0.42, hours-per-week -0.33, capital-loss -0.30, plus small workclass/occupation/education/country offsets. Children: capital\_gain\_penalty (concave piecewise-linear, kink \~5.2k USD), youth\_hinge (+0.16 logit per year below \~35), and five near-inert categorical gates (Handlers-cleaners/Doctorate relay, Tech-support, Priv-house-serv, China, Asian-Pac offset). The children have no multiplicative interactions; they meet only at the output sum. 54 one-hot inputs are unconnected (zero effect); native-country:15 is constant in train; four hidden nodes (9712, 5394, 12643, 2197) never activate. Evidence: closed form (get\_formula); train acc 0.852, AUC 0.908 (n=39073); local mean-ablations: backbone AUC-\>0.748, capital gain -\>0.890, age -\>0.900, each gate costs \<=0.0003 AUC; KernelSHAP (200 train samples) top mean\|SHAP\|: education-num 0.105, marital-status:2 0.068, relationship:0 0.054, identity\_age 0.046, identity\_capgain 0.040, sex 0.026. Supported.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `performance`: (no narrative)
- `shap`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "composition": "additive in the logit: z = 0.776 + linear backbone + sum of 7 child exits (identity_* exits weight 1.0; 7165 x0.294, 11966 x0.1003, 10191 x0.0621); output = sigmoid(z)"
}
```

```json
// performance
{
 "n": 39073,
 "mae": 0.18544741232716244,
 "mse": 0.10290348354022202,
 "rmse": 0.3207857283923679,
 "split": "train",
 "auc_roc": 0.9078326749585034,
 "accuracy": 0.8520461699895068,
 "n_samples": 39073,
 "correlation": 0.6645991166532825,
 "local_ablations": "Each component replaced by its train mean (AUC from 0.9078): direct linear backbone 0.7482 (acc 0.801); capital_gain_penalty 0.8896; youth_hinge 0.9002; handlers_doctorate_relay 0.9076; tech_support_gate 0.9077; china_gate 0.9077; asian_pac_offset 0.9078; priv_house_serv_gate 0.9078. Direct-edge single-input ablations: education-num dAUC -0.030, marital-status:2 -0.010, relationship:5 -0.009, relationship:0 -0.007, all others > -0.003."
}
```

```json
// shap
{
 "split": "train",
 "method": "compute_shap (KernelSHAP) on root entry nodes, output-probability scale",
 "base_value": 0.739943393309731,
 "max_samples": 200,
 "top12_mean_abs_shap": [
  [
   "education-num",
   0.1051
  ],
  [
   "marital-status:2",
   0.0679
  ],
  [
   "relationship:0",
   0.0538
  ],
  [
   "identity_age",
   0.0463
  ],
  [
   "identity_capgain",
   0.0399
  ],
  [
   "sex",
   0.0262
  ],
  [
   "hours-per-week",
   0.0218
  ],
  [
   "relationship:5",
   0.0214
  ],
  [
   "relationship:3",
   0.0143
  ],
  [
   "capital-loss",
   0.014
  ],
  [
   "occupation:4",
   0.0136
  ],
  [
   "workclass:6",
   0.0091
  ]
 ]
}
```

</details>

### A.1 Leaf: `youth_hinge`

|  |  |
|---|---|
| Annotation id | `ann_113` |
| Entry nodes | -1 (age) |
| Exit nodes | identity\_age |
| Internal / exit nodes in its own subgraph | 599, 8375, 3328, identity\_age |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0134 |
| Live numeric-function error | 2.22e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.371 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} \\
&\quad + 1.3 \operatorname{relu}{\left(0.31 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} + 0.812 \operatorname{relu}{\left(0.456 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} + 0.0871 \right)} + 0.285 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: a youth bonus to P(\<=50K). Age enters only via the ReLU hinge 599=relu(-1.763\*age-0.494), active below z=-0.280 (\~35 years, using est. mean 38.6/sd 13.7 y). 8375 and 3328 are always active, so they are linear relays and the contribution collapses to 1.258\*relu(-1.763\*age-0.494)+0.463: flat (0.463) at 35+ years, +0.161 logit per year younger, 3.335 at age 17. Evidence: closed form from get\_formula; line plot on full train (n=39073) shows the hinge, corr -0.78; local ablation AUC 0.9078-\>0.9002. Supported.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 0.9312672711256783,
  "x_range": [
   -1.8403933369383092,
   4.00486082593868
  ],
  "y_range": [
   0.46293134114456314,
   3.9242629286807684
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_age"
  ],
  "correlation": -0.776517485075646,
  "entry_names": [
   "-1"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -1.574699965898446,
   3.739167454898817
  ],
  "scatter_y_range": [
   0.46293134114456314,
   3.3348563081860187
  ]
 },
 "viz_type": "line_plot",
 "local_ablation": "Local ablation (contribution replaced by train mean): AUC 0.9078->0.9002, acc 0.8520->0.8502, mean |dlogit| 0.739. Contribution by age z: 0.463 for z>=-0.264 (age>=35 y, est. mean 38.6 sd 13.7); 0.590 at 34 y ... 3.335 at 17 y (step +0.161/yr). 8375 and 3328 active on 100% of rows (pure linear relays); 599 active on 43.05%."
}
```

</details>

### A.2 Leaf: `capital_gain_penalty`

|  |  |
|---|---|
| Annotation id | `ann_114` |
| Entry nodes | -63 (capital-gain) |
| Exit nodes | identity\_capgain |
| Internal / exit nodes in its own subgraph | 9352, identity\_capgain |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0491 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 2.79 x_{capital-gain} \\
&\quad - 3.58 \operatorname{relu}{\left(0.818 x_{capital-gain} - 0.453 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: capital gain's whole effect, a concave piecewise-linear penalty on P(\<=50K). Contribution = -2.787\*cg - 3.581\*relu(0.8175\*cg-0.453): slope -2.79 per sd below z=0.554 (\~5.2k USD), -5.71 per sd above it. +0.41 logit at cg=0 (95% of rows), -9.0 at the 99th pct, -73.7 at 99,999 USD (output saturated to \~0). Evidence: closed form; line plot on full train, corr -0.997; ablation AUC 0.9078-\>0.8896, acc 0.852-\>0.832. The largest non-linear effect in the network. Supported.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": -35.839906914546845,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   -77.55245426279427,
   2.2637693420226443
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_capgain"
  ],
  "correlation": -0.9971060747010466,
  "entry_names": [
   "-63"
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
   -73.74263050272937,
   0.40560982006013035
  ]
 },
 "viz_type": "line_plot",
 "local_ablation": "Local ablation: AUC 0.9078->0.8896, acc 0.8520->0.8319, mean |dlogit| 1.264. 9352 active on 4.50% of rows. Contribution +0.406 at cg=0 (95% of rows), -1.46 at the 95th pct, -8.99 at the 99th pct, -73.7 at the max (99,999 USD). Kink at z=0.554 (~5.2k USD, from max z 13.19 => sd ~7.5k USD)."
}
```

</details>

### A.3 Leaf: `tech_support_gate`

|  |  |
|---|---|
| Annotation id | `ann_115` |
| Entry nodes | -41 (occupation:13), -71 (native-country:13), -66 (native-country:0) |
| Exit nodes | identity\_techsupport |
| Internal / exit nodes in its own subgraph | 2892, 8892, 12120, identity\_techsupport |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 8.90e-04 |
| Live numeric-function error | 5.55e-17 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.015 x_{occupation:13} \\
&\quad - 0.179 \operatorname{relu}{\left(1.02 x_{occupation:13} - 0.197 \right)} \\
&\quad + 0.0205 \operatorname{relu}{\left(1.05 x_{native-country:0} + 1.44 \operatorname{relu}{\left(0.0943 x_{native-country:13} + 0.963 x_{occupation:13} - 0.334 \right)} - 1.25 \right)} \\
&\quad + 0.0781 \operatorname{relu}{\left(0.0943 x_{native-country:13} + 0.963 x_{occupation:13} - 0.334 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: small per-category offsets for occupation:13 (inferred Tech-support) and two rare countries. Three ReLU gates (2892, 8892, 12120) plus the direct occupation:13 edge, all off unless one of the one-hot inputs is set. Train table: baseline -0.003; Tech-support -0.39 (or -0.16 with nc13); native-country:0 ('?') alone +0.13; native-country:13 alone +0.15. Evidence: closed form; sensitivity on full train (occ13 dominant); ablation leaves AUC unchanged (mean \|dlogit\| 0.023). Supported but practically negligible.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
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
   "identity_techsupport"
  ],
  "entry_names": [
   "-41",
   "-71",
   "-66"
  ],
  "input_labels": [
   "-41",
   "-71",
   "-66"
  ],
  "sensitivities": {
   "identity_techsupport": [
    0.016673414610251125,
    0.00032250540442979196,
    0.0010422056166394793
   ]
  },
  "dimensionality": [
   3,
   1
  ]
 },
 "viz_type": "sensitivity",
 "local_ablation": "Local ablation: AUC unchanged to 4 d.p. (0.9077), acc 0.8515, mean |dlogit| 0.023. Exhaustive table on train: neither (n=37244) -0.003 or +0.151 (nc13); nc0 only (656) +0.133; occ13 only (1152) -0.392 or -0.156; both (21) -0.228. 2892 active 3.18%, 8892 4.86%, 12120 3.00%."
}
```

</details>

### A.4 Leaf: `asian_pac_offset`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -61 (race:4), -58 (race:1) |
| Exit nodes | identity\_asianpac |
| Internal / exit nodes in its own subgraph | 9712, 5394, identity\_asianpac |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 1.18e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0256 x_{race:1} \\
&\quad + 0.0413 \operatorname{relu}{\left(- 1.65 x_{race:1} + 1.0 \operatorname{relu}{\left(0.882 x_{race:4} - 1.06 \right)} - 0.866 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: a small negative offset for race:1 (inferred Asian-Pac-Islander), -0.149 logit relative to others, via the direct edge. The race:4 (White) gate 9712 and the gate 5394 are dead on train (0% active; bias outweighs every reachable input), contributing a constant 0. Evidence: closed form; sensitivity zero for race:4; ablation AUC unchanged. Supported; the gates are dead code.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
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
   "identity_asianpac"
  ],
  "entry_names": [
   "-61",
   "-58"
  ],
  "input_labels": [
   "-61",
   "-58"
  ],
  "sensitivities": {
   "identity_asianpac": [
    0.0,
    0.02557906485877372
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity",
 "local_ablation": "Local ablation: AUC 0.9078 unchanged, mean |dlogit| 0.009. Values: -0.144 for race:1 (1201 rows), +0.005 otherwise. 9712 and 5394 are active on 0.00% of train rows (dead)."
}
```

</details>

### A.5 Leaf: `handlers_doctorate_relay`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -14 (education:10), -47 (occupation:6), -67 (native-country:1) |
| Exit nodes | 7165 |
| Internal / exit nodes in its own subgraph | 5993, 7165 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00125 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.737 \operatorname{relu}{\left(- 0.453 x_{education:10} - 0.729 x_{native-country:1} + 0.604 x_{occupation:6} + 1.86 \right)} \\
&\quad + 0.999 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: a three-level categorical offset. Gate 5993=relu(1.861-0.453\*edu10+0.605\*occ6-0.729\*nc1) is relayed linearly by 7165 (always active) into the output with weight 0.294. Contribution: baseline 0.683; occupation:6 (inferred Handlers-cleaners) 1.328 (+0.645, more likely \<=50K); education:10 (Doctorate) or native-country:1 (Cambodia) clamp 5993 to 0, giving 0.294 (-0.389). Evidence: closed form; exhaustive value table on train (36898/1683/492 rows); ablation AUC 0.9078-\>0.9076. Supported.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
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
   "7165"
  ],
  "entry_names": [
   "-14",
   "-47",
   "-67"
  ],
  "input_labels": [
   "-14",
   "-47",
   "-67"
  ],
  "sensitivities": {
   "7165": [
    0.32952692911314896,
    0.439803126432612,
    0.530221758200542
   ]
  },
  "dimensionality": [
   3,
   1
  ]
 },
 "viz_type": "sensitivity",
 "local_ablation": "Local ablation: AUC 0.9078->0.9076, acc 0.8512, mean |dlogit| 0.054. Contribution (0.294*7165) takes 3 values: 0.683 (n=36898, baseline), 1.328 (n=1683, occupation:6), 0.294 (n=492, education:10 or native-country:1 clamps 5993 to 0). 5993 active 98.74%, 7165 active 100% (linear relay)."
}
```

</details>

### A.6 Leaf: `priv_house_serv_gate`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -19 (education:15), -92\_b (native-country:32\_b), -73\_b (native-country:15\_b), -50 (occupation:9) |
| Exit nodes | 11966 |
| Internal / exit nodes in its own subgraph | 12643, 2197, 11966 |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.065 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 x_{occupation:9} \\
&\quad + 0.968 \operatorname{relu}{\left(0.907 x_{education:15} - 0.0292 x_{native-country:32\_b} - 0.108 \operatorname{relu}{\left(1.0 x_{native-country:15\_b} - 0.878 \right)} - 1.72 \right)} \\
&\quad -0.516 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: occupation:9 (inferred Priv-house-serv) switch. 11966=relu(1.0045\*occ9-0.516+0.968\*2197) is 13.97 for the 187 occupation:9 rows (+1.40 logit, more likely \<=50K), else 0. The upstream chain (2197 \<- education:15, native-country:32, 12643 \<- native-country:15) never activates on train: native-country:15 is constant and 0.907\*max(edu15)=1.70 \< bias 1.72. So the leaf reduces to relu(1.0045\*occ9-0.516). Evidence: closed form; sensitivity only on occ9; ablation AUC unchanged. Supported; dead branch noted.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
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
   "11966"
  ],
  "entry_names": [
   "-19",
   "-92_b",
   "-73_b",
   "-50"
  ],
  "input_labels": [
   "-19",
   "-92_b",
   "-73_b",
   "-50"
  ],
  "sensitivities": {
   "11966": [
    0.0,
    0.0,
    0.0,
    0.00480744551397886
   ]
  },
  "dimensionality": [
   4,
   1
  ]
 },
 "viz_type": "sensitivity",
 "local_ablation": "Local ablation: AUC unchanged, mean |dlogit| 0.013. 11966 is 0 for 38886 rows and 13.97 (contribution +1.401 logit) for the 187 occupation:9 rows. 12643 and 2197 active 0.00% of rows (dead): native-country:15 is constant in train and 0.907*max(edu15)=1.70 < 1.72."
}
```

</details>

### A.7 Leaf: `china_gate`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -89 (native-country:3) |
| Exit nodes | 10191 |
| Internal / exit nodes in its own subgraph | 10191 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0635 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.01 x_{native-country:3} - 0.419 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: a native-country:3 (inferred China) switch. 10191=relu(1.0132\*nc3-0.419) is 20.0 for the 96 China rows (+1.24 logit after weight 0.0621), else 0. Evidence: closed form; line plot corr 1.0 over the live range; ablation AUC 0.9078-\>0.9077. Supported.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true",
 "tractable": true
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "y_mean": 9.816047758460194,
  "x_range": [
   -1.0595949647094565,
   21.159666597850904
  ],
  "y_range": [
   0.0,
   21.01899939985969
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "10191"
  ],
  "correlation": 0.9999999999999976,
  "entry_names": [
   "-89"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.0496285300476217,
   20.14970016318907
  ],
  "scatter_y_range": [
   0.0,
   19.99572777427598
  ]
 },
 "viz_type": "line_plot",
 "local_ablation": "Local ablation: AUC 0.9077, mean |dlogit| 0.006. 10191 is 0 for 38977 rows and 20.0 (contribution 0.0621*20.0 = +1.242 logit) for the 96 native-country:3 rows."
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 4 |
| annotate | 8 |
| rename\_node | 107 |
| split\_node | 2 |

2 splits, 4 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node identity\_age intercepts 2 connection(s) into 0 (from 599, 3328); sits in youth\_hinge (exit); income\_logit (entry)
- `#108` identity node identity\_capgain intercepts 2 connection(s) into 0 (from -63 (capital-gain), 9352); sits in capital\_gain\_penalty (exit); income\_logit (entry)
- `#109` identity node identity\_techsupport intercepts 4 connection(s) into 0 (from -41 (occupation:13), 2892, 8892, 12120); sits in tech\_support\_gate (exit); income\_logit (entry)
- `#110` identity node identity\_asianpac intercepts 2 connection(s) into 0 (from -58 (race:1), 5394); sits in asian\_pac\_offset (exit); income\_logit (entry)
- `#111` split -92 into -92\_a, -92\_b. Final copies: -92\_a → income\_logit (entry) \| -92\_b → priv\_house\_serv\_gate (entry)
- `#112` split -73 into -73\_a, -73\_b. Final copies: -73\_a → income\_logit (entry) \| -73\_b → priv\_house\_serv\_gate (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 4.44e-16 | 4.44e-16 | 4.44e-16 | agree |
| Closed-form max abs error (all leaves) | 0.06501 | 0.06501 | 0.06501 | agree |
| Numeric-function max abs error | n/a | 2.22e-16 | 2.22e-16 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit | income\_logit | agree |
| Leaf / composition counts | 7 / 1 | n/a | 7 / 1 | agree |
| Operation counts | 2 splits, 4 identity | {"add\_identity\_node": 4, "annotate": 8, "rename\_node": 107, "split\_node": 2} | {"add\_identity\_node": 4, "annotate": 8, "rename\_node": 107, "split\_node": 2} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `priv_house_serv_gate` | 0.06501 | 0 |
| `china_gate` | 0.06353 | 0 |
| `capital_gain_penalty` | 0.04914 | 0 |
| `youth_hinge` | 0.01338 | 2.22e-16 |
| `handlers_doctorate_relay` | 0.001246 | 0 |
| `tech_support_gate` | 8.90e-04 | 5.55e-17 |
| `asian_pac_offset` | 1.18e-04 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Four hidden nodes never activate on train (race gate 9712/5394; education:15/Portugal/Holand-Netherlands chain 12643/2197), documented as dead code. Five categorical gates real but negligible (AUC change \<=0.0003 each). Category names and USD/year scales inferred.

**Tool notes (record):** None. KernelSHAP (200) \~3 min.

## What to look at

- Largest closed-form error: `priv_house_serv_gate` (0.065), then `china_gate` (0.0635). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `asian_pac_offset`: *“The race:4 (White) gate 9712 and the gate 5394 are dead on train (0% active;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `handlers_doctorate_relay`: *“exhaustive value table on train (36898/1683/492 rows);”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 2 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s6.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s6', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
