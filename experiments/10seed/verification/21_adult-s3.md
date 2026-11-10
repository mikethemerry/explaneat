# Verification packet 21 · adult-s3

Pre-registered verification order: 21 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/fe61877a-be94-4cdb-8f82-9a8ebe7eb975?experiment=90c7e61f-c427-4fbc-bc08-a84a01bde35e&name=10seed-adult-s3), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 3 |
| Genome id | `fe61877a-be94-4cdb-8f82-9a8ebe7eb975` |
| Experiment | 10seed-adult-s3 (90c7e61f-c427-4fbc-bc08-a84a01bde35e); best at generation 99 of 100, NEAT key 22471 |
| Test AUC (runs csv) | 0.909 \[95% CI 0.902, 0.915\] |
| Train AUC | 0.913 |
| Size | 63 of 107 inputs connected; 15 hidden nodes; 85 enabled connections; depth 4; 14 skip connections + 54 direct input→output edges |
| Annotations | 12 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 15 of 15 |
| Claude's time | 26.2 min (2026-11-05T11:54:23+13:00 → 2026-11-05T12:20:38+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> P(income\<=50K) = sigmoid(2.095 + linear backbone + capital-gain penalty + youth hinge + categorical offsets), additive. Backbone a sparse logistic regression (education-num, married-civ-spouse, husband, wife, male, hours, capital-loss; Without-pay switch). Capital-gain chain is a concave piecewise-linear penalty with kinks \~3.9k and 4.9k USD (ablation AUC 0.913 -\> 0.891). Age adds up to +2.15 logit only below the mean. Nine small one-hot gates add per-category offsets. Train acc 0.858, AUC 0.913.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit`

|  |  |
|---|---|
| Annotation id | `ann_128` |
| Entry nodes | cg\_out, age\_out, cat\_agg, lin\_agg, -2 (workclass:0), -4 (workclass:2), -5 (workclass:3), -6 (workclass:4), -12 (education:0), -13 (education:1), -14 (education:10), -15 (education:11), -17 (education:13), -18 (education:14), -19 (education:15), -20 (education:2), -21 (education:3), -24 (education:6), -29 (marital-status:0), -33 (marital-status:4), -34 (marital-status:5), -37 (occupation:1), -39 (occupation:11), -41 (occupation:13), -43 (occupation:2), -44 (occupation:3), -48 (occupation:7), -50 (occupation:9), -54 (relationship:3), -59 (race:2), -60 (race:3), -67 (native-country:1), -68 (native-country:10), -69 (native-country:11), -70 (native-country:12), -72 (native-country:14), -73 (native-country:15), -75 (native-country:17), -76 (native-country:18), -78 (native-country:2), … (8 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_penalty, youth\_hinge, category\_offsets, linear\_backbone |
| Dimensionality (in → out, incl. descendants) | 48 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.0 \operatorname{capital\_gain\_penalty}{\left(-63,-52 \right)} + 1.0 \operatorname{category\_offsets}{\left(us_{out},scot_{out},rel_{2 out},elsal_{out},5934,7067,8156,10197,12550 \right)} + 1.0 \operatorname{linear\_backbone}{\left(-7,-8,-9,-10,-11,-16,-22,-23,-25,-26,-27,-28,-30,-31,-35,-36,-38,-40,-45,-46,-49,-51,-56,-57,-61,-62,-64,-65,-71,-74,-77,-79,-82,-83,-84,-87,-88,-89,-90,-92,-95,-96,-98,-100,-101,-102,-103,-104,-107 \right)} + 1.0 \operatorname{youth\_hinge}{\left(-1 \right)}}}{e^{1.0 \operatorname{capital\_gain\_penalty}{\left(-63,-52 \right)} + 1.0 \operatorname{category\_offsets}{\left(us_{out},scot_{out},rel_{2 out},elsal_{out},5934,7067,8156,10197,12550 \right)} + 1.0 \operatorname{linear\_backbone}{\left(-7,-8,-9,-10,-11,-16,-22,-23,-25,-26,-27,-28,-30,-31,-35,-36,-38,-40,-45,-46,-49,-51,-56,-57,-61,-62,-64,-65,-71,-74,-77,-79,-82,-83,-84,-87,-88,-89,-90,-92,-95,-96,-98,-100,-101,-102,-103,-104,-107 \right)} + 1.0 \operatorname{youth\_hinge}{\left(-1 \right)}} + 0.123}\end{gathered}
$$

**Hypothesis (DB):**

> Root: P(income\<=50K) = sigmoid(2.095 + lin\_agg + cg\_out + age\_out + cat\_agg), an additive (GAM-like) logit with no interactions between branches: a linear backbone (education-num, married-civ-spouse, husband/wife, sex, hours, capital-loss), a capital-gain hinge penalty, a youth hinge on age, and a table of categorical offsets. The 44 input features with no edges (listed as entries) have zero effect. EVIDENCE (train, 39,073 rows): acc 0.858, AUC 0.913, mean prediction 0.765 (base rate 0.761). Mean-ablation per branch: linear backbone acc 0.801 / AUC 0.681; capital gain 0.820 / 0.891; age 0.852 / 0.905; categories 0.857 / 0.912. SHAP (200 samples): marital-status:2 0.074, education-num 0.074, relationship:0 0.069, age 0.051, capital-gain 0.035, sex 0.025; unconnected inputs are at the KernelSHAP noise floor (\<= 0.00045).

**Evidence (3 records, DB):**

- `shap_importance`: (no narrative)
- `performance_metrics`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// shap_importance
{
 "top": [
  [
   "marital-status:2",
   0.07370187278504264
  ],
  [
   "education-num",
   0.07369130525452397
  ],
  [
   "relationship:0",
   0.06868626602433192
  ],
  [
   "age",
   0.05050801348690496
  ],
  [
   "capital-gain",
   0.035257029319459364
  ],
  [
   "sex",
   0.025118764163284703
  ],
  [
   "hours-per-week",
   0.022673140535954414
  ],
  [
   "relationship:5",
   0.02170198364590723
  ],
  [
   "relationship:1",
   0.021555325415165215
  ],
  [
   "occupation:4",
   0.019660323059199358
  ],
  [
   "capital-loss",
   0.0149007053213581
  ],
  [
   "education:9",
   0.010909667539961608
  ],
  [
   "occupation:8",
   0.009418447800483423
  ],
  [
   "occupation:10",
   0.009315299124500277
  ],
  [
   "workclass:6",
   0.008227729495126406
  ]
 ],
 "note": "KernelSHAP noise floor ~0.0004 (unconnected inputs show <=0.00045)",
 "split": "train",
 "max_samples": 200
}
```

```json
// performance_metrics
{
 "mae": 0.1943236084900537,
 "mse": 0.09780801598738123,
 "rmse": 0.3127427313102276,
 "auc_roc": 0.9125898709350319,
 "accuracy": 0.8577790289970056,
 "n_samples": 39073,
 "correlation": 0.6797436700962227
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "max": 1.0,
  "min": 2.3175621887868083e-45,
  "std": 0.28889317126950925,
  "mean": 0.7646226428164707,
  "count": 39073,
  "median": 0.9106229397365295
 }
}
```

</details>

### A.1 Leaf: `capital_gain_penalty`

|  |  |
|---|---|
| Annotation id | `ann_115` |
| Entry nodes | -63 (capital-gain), -52 (relationship:1) |
| Exit nodes | cg\_out |
| Internal / exit nodes in its own subgraph | 3273, 6463, 7310, cg\_out |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.574 |
| Live numeric-function error | 1.42e-14 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 2.29 \operatorname{relu}{\left(1.14 \operatorname{relu}{\left(1.47 x_{capital-gain} + 1.13 \right)} - 1.91 \right)} \\
&\quad - 2.37 \operatorname{relu}{\left(0.221 x_{relationship:1} + 1.01 \operatorname{relu}{\left(1.14 \operatorname{relu}{\left(1.47 x_{capital-gain} + 1.13 \right)} - 1.91 \right)} - 0.0918 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Capital-gain hinge penalty with a relationship:1 (Not-in-family) offset. 3273 = 1.467\*cg + 1.129 is always active (a linear rescaling). 6463 = ReLU(1.671\*cg - 0.626) switches on above cg = 0.375 SD (about 3.9k USD). 7310 = ReLU(0.222\*rel1 + 1.006\*6463 - 0.092) is a constant 0.284 for Not-in-family rows and, for other rows, passes 6463 through once 6463 \> 0.22 (cg about 0.51 SD, about 4.9k USD). Output cg\_out = -2.29\*6463 - 2.37\*7310 (logit of P(\<=50K)): -0.67 for Not-in-family, otherwise 0 below the first kink, then a concave piecewise-linear decline (slope about -3.8 then -7.8 logit per SD of capital gain) reaching -100 at the maximum. Large capital gains force P(\<=50K) towards 0. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). Empirical: 5.98% of train rows lie above the first kink; observed P(\<=50K) there is 0.196 and the model's mean prediction 0.200 (vs 0.761 overall). Heatmap (train, all rows) confirms cg\_out depends on capital-gain and only shifts by the relationship:1 offset. Kinks are almost the same as in adult-s1 (about 3.8k and 4.8k USD). Mean-ablation of the contribution: acc 0.858 -\> 0.820, AUC 0.913 -\> 0.891.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary heatmap on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  2,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "heatmap",
  "correlation": -0.025950371948527645,
  "scatter_y_range": [
   -0.5891231825775419,
   1.6974378696577113
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8197,
 "auc": 0.8906,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

### A.2 Leaf: `youth_hinge`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -1 (age) |
| Exit nodes | age\_out |
| Internal / exit nodes in its own subgraph | 3460, age\_out |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00576 |
| Live numeric-function error | 2.22e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.000242 x_{age} \\
&\quad + 1.62 \operatorname{relu}{\left(0.0238 - 0.826 x_{age} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Age hinge: age\_out = 1.625\*ReLU(0.024 - 0.826\*age) + 0.0002\*age. Below the mean age (hinge at age = +0.03 SD, about 39 years) each SD younger adds 1.34 logit to P(\<=50K), up to +2.15 for the youngest; above the hinge the contribution is flat (about 0). The direct age edge is negligible. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). Empirical (train): mean age\_out by age band: \<-1 SD 1.77 (obs P(\<=50K) 0.989), -1 to -0.5 SD 1.07 (0.863), -0.5 to 0 SD 0.39 (0.726), above the mean 0.00 (0.63-0.69). Line plot correlation -0.84 with age; flat beyond the hinge, so the model does not reproduce the slight rise in P(\<=50K) for older people. SHAP rank 4 (mean \|SHAP\| 0.051, 200 samples). Mean-ablation of the contribution: acc 0.858 -\> 0.852, AUC 0.913 -\> 0.905.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -1.8403933369383092,
   4.00486082593868
  ],
  "y_range": [
   9.545210290471345e-06,
   2.5074811102164545
  ],
  "viz_type": "line",
  "correlation": -0.8418198944191102,
  "scatter_y_range": [
   2.4059728997579565e-05,
   2.151076402717793
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8523,
 "auc": 0.9052,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

### A.3 Composition: `category_offsets`

|  |  |
|---|---|
| Annotation id | `ann_127` |
| Entry nodes | us\_out, scot\_out, rel2\_out, elsal\_out, 5934, 7067, 8156, 10197, 12550 |
| Exit nodes | cat\_agg |
| Internal / exit nodes in its own subgraph | cat\_agg |
| Children | us\_born\_offset, scotland\_switch, other\_relative\_offset, el\_salvador\_offset, handlers\_asian\_gate, federal\_gov\_offset, transport\_moving\_offset, unmarried\_offset, spouse\_absent\_unknown\_country |
| Dimensionality (in → out, incl. descendants) | 9 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.0 \operatorname{el\_salvador\_offset}{\left(-106 \right)} \\
&\quad - 0.0959 \operatorname{federal\_gov\_offset}{\left(-3 \right)} \\
&\quad + 0.358 \operatorname{handlers\_asian\_gate}{\left(-47,-58 \right)} \\
&\quad + 1.0 \operatorname{other\_relative\_offset}{\left(-53 \right)} \\
&\quad + 1.0 \operatorname{scotland\_switch}{\left(-94 \right)} \\
&\quad - 0.0082 \operatorname{spouse\_absent\_unknown\_country}{\left(-32,-66 \right)} \\
&\quad + 0.0216 \operatorname{transport\_moving\_offset}{\left(-42 \right)} \\
&\quad - 0.167 \operatorname{unmarried\_offset}{\left(-55 \right)} \\
&\quad + 1.0 \operatorname{us\_born\_offset}{\left(-99 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Additive sum of nine sparse categorical offsets: cat\_agg = us\_out + scot\_out + rel2\_out + elsal\_out + 0.358\*5934 - 0.096\*7067 + 0.022\*8156 - 0.167\*10197 - 0.008\*12550. Each child is an indicator-like term for one or two one-hot categories, so cat\_agg is a per-category lookup table added to the logit. Apart from the Scotland switch (+83 logit, 17 rows), effects are within +/-0.8 logit. EVIDENCE (train): sensitivity viz returns exactly these coefficients (additive, no interaction between children). Mean-ablating all nine together: acc 0.858 -\> 0.857, AUC 0.913 -\> 0.912, so these offsets are minor refinements.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  9,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "sensitivities": {
   "cat_agg": [
    1.000000000000001,
    1.0000000000000004,
    1.000000000000001,
    1.0000000000000004,
    0.35824596202548703,
    0.09591318976528677,
    0.021649183950052126,
    0.1668253647211558,
    0.008196480695487118
   ]
  }
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8565,
 "auc": 0.9115,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.1 Leaf: `us_born_offset`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -99 (native-country:39) |
| Exit nodes | us\_out |
| Internal / exit nodes in its own subgraph | 4018, 4607, us\_out |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 3.21e-05 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0406 x_{native-country:39} \\
&\quad - 0.215 \operatorname{relu}{\left(0.942 x_{native-country:39} + 0.471 \right)} \\
&\quad - 0.268 \operatorname{relu}{\left(1.0 x_{native-country:39} - 0.569 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> native-country:39 (United-States) offset. 4018 = ReLU(0.942\*x + 0.471) is 0.787 for US-born and 0 otherwise; 4607 = ReLU(x - 0.570) is dead on the data (x max 0.335). us\_out = -0.215\*4018 - 0.268\*4607 - 0.041\*x: -0.183 logit for US-born, +0.121 for others; a 0.30 logit lower P(\<=50K) for US-born. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.912.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -3.1477927996031707,
   0.5012175307471751
  ],
  "y_range": [
   -0.22313187483723498,
   0.12776653390788373
  ],
  "viz_type": "line",
  "correlation": -0.9999999999999982,
  "scatter_y_range": [
   -0.18279847304038016,
   0.12103423503597546
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8575,
 "auc": 0.9125,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.2 Leaf: `scotland_switch`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -94 (native-country:34) |
| Exit nodes | scot\_out |
| Internal / exit nodes in its own subgraph | 9997, scot\_out |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00707 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.891 x_{native-country:34} \\
&\quad + 0.807 \operatorname{relu}{\left(1.0 x_{native-country:34} - 0.564 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> native-country:34 (Scotland, 17 train rows, standardized value 49.4) saturating switch. scot\_out = 0.807\*ReLU(x - 0.564) + 0.891\*x: +83 logit when Scotland (P(\<=50K) = 1), -0.02 otherwise. The gate 9997 is redundant with the direct edge (the output is already saturated). EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.913.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -2.491607492683581,
   51.87847730298475
  ],
  "y_range": [
   -2.2195615380114977,
   87.64193627948356
  ],
  "viz_type": "line",
  "correlation": 0.9999999999999912,
  "scatter_y_range": [
   -0.0180300990572863,
   83.44519709190789
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8577,
 "auc": 0.9127,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.3 Leaf: `other_relative_offset`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -53 (relationship:2) |
| Exit nodes | rel2\_out |
| Internal / exit nodes in its own subgraph | 14966, rel2\_out |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 5.28e-04 |
| Live numeric-function error | 5.55e-17 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.00812 x_{relationship:2} \\
&\quad - 0.0655 \operatorname{relu}{\left(0.998 x_{relationship:2} - 1.14 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> relationship:2 (Other-relative) offset. 14966 = ReLU(x - 1.144) is 4.51 for Other-relative rows and 0 otherwise. rel2\_out = -0.065\*14966 - 0.094\*x: -0.34 logit for Other-relative, +0.001 otherwise. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.912.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -0.4686298947145074,
   5.958971612728233
  ],
  "y_range": [
   -0.3627056944277682,
   0.0038067115919808913
  ],
  "viz_type": "line",
  "correlation": -1.0,
  "scatter_y_range": [
   -0.34125038906874544,
   0.001433446517172451
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8579,
 "auc": 0.9125,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.4 Leaf: `el_salvador_offset`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -106 (native-country:8) |
| Exit nodes | elsal\_out |
| Internal / exit nodes in its own subgraph | 13897, elsal\_out |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 1.45e-04 |
| Live numeric-function error | 1.39e-17 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0204 x_{native-country:8} \\
&\quad + 0.0303 \operatorname{relu}{\left(0.984 x_{native-country:8} - 1.73 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> native-country:8 (El-Salvador) offset. 13897 = ReLU(0.984\*x - 1.732) is 16.1 for El-Salvador rows and 0 otherwise. elsal\_out = 0.030\*13897 - 0.020\*x: +0.12 logit for El-Salvador, about 0 otherwise. Small. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.913.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -0.9626678323935918,
   19.00006212038487
  ],
  "y_range": [
   -0.03556970216387739,
   0.126317066291451
  ],
  "viz_type": "line",
  "correlation": 1.0,
  "scatter_y_range": [
   0.0011260882506605428,
   0.11778179989626027
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8578,
 "auc": 0.9126,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.5 Leaf: `handlers_asian_gate`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -47 (occupation:6), -58 (race:1) |
| Exit nodes | 5934 |
| Internal / exit nodes in its own subgraph | 5934 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00208 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.432 x_{occupation:6} \\
&\quad - 0.301 x_{race:1} \\
&\quad + 0.531 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> 5934 = ReLU(0.432\*occupation:6 - 0.301\*race:1 + 0.531): occupation:6 (Handlers-cleaners) raises it, race:1 (Asian-Pac-Islander) lowers it. Four values: 0.49 baseline, 2.62 Handlers-cleaners, 0 for Asian-Pac-Islander non-handlers (the ReLU clips), 0.88 for both. Downstream weight +0.358, so Handlers-cleaners add about +0.76 logit to P(\<=50K) and Asian-Pac-Islander about -0.18. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.857, AUC 0.913 -\> 0.912.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary heatmap on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  2,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "heatmap",
  "correlation": -0.015140678146033862,
  "scatter_y_range": [
   -0.17807886681260568,
   5.615489462054123
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8572,
 "auc": 0.9121,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.6 Leaf: `federal_gov_offset`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -3 (workclass:1) |
| Exit nodes | 7067 |
| Internal / exit nodes in its own subgraph | 7067 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.026 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{workclass:1} - 0.0662 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> 7067 = ReLU(1.004\*workclass:1 - 0.066): 5.75 for workclass:1 (Federal-gov) and 0 otherwise. Downstream weight -0.096, so Federal-gov lowers P(\<=50K) by 0.55 logit. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.912.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -0.47083535134857263,
   6.087473157901361
  ],
  "y_range": [
   0.0,
   6.048626995961948
  ],
  "viz_type": "line",
  "correlation": 0.999999999999995,
  "scatter_y_range": [
   0.0,
   5.749181802446674
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8575,
 "auc": 0.9123,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.7 Leaf: `transport_moving_offset`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -42 (occupation:14) |
| Exit nodes | 8156 |
| Internal / exit nodes in its own subgraph | 8156 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 3.52e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.977 x_{occupation:14} + 0.609 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> 8156 = ReLU(0.977\*occupation:14 + 0.609): 4.89 for occupation:14 (Transport-moving), 0.39 otherwise; linear on the data. Downstream weight +0.022, so Transport-moving adds +0.10 logit. Very small. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.913.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -0.45871592892162133,
   4.612663500767256
  ],
  "y_range": [
   0.1606758945008614,
   5.115948568282394
  ],
  "viz_type": "line",
  "correlation": 1.0,
  "scatter_y_range": [
   0.3859155614909311,
   4.890708901292324
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8575,
 "auc": 0.9126,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.8 Leaf: `unmarried_offset`

|  |  |
|---|---|
| Annotation id | `ann_124` |
| Entry nodes | -55 (relationship:4) |
| Exit nodes | 10197 |
| Internal / exit nodes in its own subgraph | 10197 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 8.42e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.764 x_{relationship:4} - 0.317 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> 10197 = ReLU(0.764\*relationship:4 - 0.317): 1.93 for relationship:4 (Unmarried), 0 otherwise. Downstream weight -0.167, so Unmarried lowers P(\<=50K) by 0.32 logit (conditional on the other terms). EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.912.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary line on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  1,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "x_range": [
   -0.5043540756402628,
   3.100926370958356
  ],
  "y_range": [
   0.0,
   2.0529943783781768
  ],
  "viz_type": "line",
  "correlation": 0.9999999999999998,
  "scatter_y_range": [
   0.0,
   1.9277481905998912
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8581,
 "auc": 0.9123,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

#### A.3.9 Leaf: `spouse_absent_unknown_country`

|  |  |
|---|---|
| Annotation id | `ann_125` |
| Entry nodes | -32 (marital-status:3), -66 (native-country:0) |
| Exit nodes | 12550 |
| Internal / exit nodes in its own subgraph | 15301, 12550 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0186 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.976 x_{marital-status:3} \\
&\quad + 1.45 \operatorname{relu}{\left(0.997 x_{native-country:0} + 0.309 \right)} \\
&\quad + 0.0795 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> 12550 = ReLU(0.976\*marital-status:3 + 1.448\*ReLU(0.997\*native-country:0 + 0.309) + 0.080); both ReLUs are always active, so it is linear: marital-status:3 (Married-spouse-absent) and native-country:0 (unknown country) both raise it. Downstream weight -0.008, so the effect is at most -0.16 logit. Practically inert. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). get\_viz\_summary line/heatmap confirms the stated values. Mean-ablation of the contribution: acc 0.858 -\> 0.858, AUC 0.913 -\> 0.913.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary heatmap on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  2,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "heatmap",
  "correlation": 0.027512163838115812,
  "scatter_y_range": [
   -0.13278571084530705,
   7.530930802978448
  ]
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8579,
 "auc": 0.9126,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

### A.4 Leaf: `linear_backbone`

|  |  |
|---|---|
| Annotation id | `ann_126` |
| Entry nodes | -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -10 (workclass:8), -11 (fnlwgt), -16 (education:12), -22 (education:4), -23 (education:5), -25 (education:7), -26 (education:8), -27 (education:9), -28 (education-num), -30 (marital-status:1), -31 (marital-status:2), -35 (marital-status:6), -36 (occupation:0), -38 (occupation:10), -40 (occupation:12), -45 (occupation:4), -46 (occupation:5), -49 (occupation:8), -51 (relationship:0), -56 (relationship:5), -57 (race:0), -61 (race:4), -62 (sex), -64 (capital-loss), -65 (hours-per-week), -71 (native-country:13), -74 (native-country:16), -77 (native-country:19), -79 (native-country:20), -82 (native-country:23), -83 (native-country:24), -84 (native-country:25), -87 (native-country:28), -88 (native-country:29), -89 (native-country:3), -90 (native-country:30), -92 (native-country:32), … (9 more) |
| Exit nodes | lin\_agg |
| Internal / exit nodes in its own subgraph | lin\_agg |
| Dimensionality (in → out, incl. descendants) | 49 → 1 |
| Live closed-form error (3 s.f. display) | 0.0184 |
| Live numeric-function error | 1.42e-14 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.264 x_{capital-loss} \\
&\quad - 0.879 x_{education-num} \\
&\quad + 0.067 x_{education:12} \\
&\quad - 0.0945 x_{education:4} \\
&\quad - 0.0792 x_{education:5} \\
&\quad + 0.0739 x_{education:7} \\
&\quad + 0.0344 x_{education:8} \\
&\quad + 0.102 x_{education:9} \\
&\quad - 0.0874 x_{fnlwgt} \\
&\quad - 0.301 x_{hours-per-week} \\
&\quad - 0.0516 x_{marital-status:1} \\
&\quad - 0.691 x_{marital-status:2} \\
&\quad - 0.0945 x_{marital-status:6} \\
&\quad + 0.00269 x_{native-country:13} \\
&\quad - 0.014 x_{native-country:16} \\
&\quad - 0.00582 x_{native-country:19} \\
&\quad + 0.00841 x_{native-country:20} \\
&\quad - 0.0145 x_{native-country:23} \\
&\quad + 0.00293 x_{native-country:24} \\
&\quad + 0.0284 x_{native-country:25} \\
&\quad + 0.00869 x_{native-country:28} \\
&\quad + 0.0295 x_{native-country:29} \\
&\quad - 0.0397 x_{native-country:30} \\
&\quad - 0.0404 x_{native-country:32} \\
&\quad + 0.0362 x_{native-country:35} \\
&\quad + 0.0024 x_{native-country:36} \\
&\quad + 0.0619 x_{native-country:38} \\
&\quad + 0.0312 x_{native-country:3} \\
&\quad + 0.0368 x_{native-country:40} \\
&\quad - 0.023 x_{native-country:41} \\
&\quad + 0.163 x_{native-country:4} \\
&\quad - 0.014 x_{native-country:5} \\
&\quad + 0.0316 x_{native-country:6} \\
&\quad - 0.0336 x_{native-country:9} \\
&\quad + 0.155 x_{occupation:0} \\
&\quad - 0.121 x_{occupation:10} \\
&\quad - 0.0659 x_{occupation:12} \\
&\quad - 0.223 x_{occupation:4} \\
&\quad + 0.169 x_{occupation:5} \\
&\quad + 0.268 x_{occupation:8} \\
&\quad + 0.023 x_{race:0} \\
&\quad - 0.0782 x_{race:4} \\
&\quad - 0.622 x_{relationship:0} \\
&\quad - 0.516 x_{relationship:5} \\
&\quad - 0.342 x_{sex} \\
&\quad - 0.024 x_{workclass:5} \\
&\quad + 0.173 x_{workclass:6} \\
&\quad + 0.0527 x_{workclass:7} \\
&\quad + 1.05 x_{workclass:8}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear backbone: 49 direct input edges summed at lin\_agg (logit of P(\<=50K)). Dominant terms per SD: education-num -0.88, marital-status:2 (Married-civ-spouse) -0.69, relationship:0 (Husband) -0.62, relationship:5 (Wife) -0.52, sex (male) -0.34, hours-per-week -0.30, capital-loss -0.26 (linear in capital loss), occupation:8 (Other-service) +0.27, occupation:4 (Exec-managerial) -0.22. workclass:8 (Without-pay, 17 rows, value 47.9) is a saturating switch (+50 logit). Remaining one-hot terms (mostly native countries) are small. EVIDENCE (train split, 39,073 rows): closed form from get\_formula matches the forward pass (check\_explanation). Empirical: sensitivity viz returns exactly the edge weights (linear). Alone, sigmoid(2.095 + lin\_agg) gives train acc 0.833, AUC 0.879. SHAP (model output, 200 samples) top features are all backbone terms: marital-status:2 0.074, education-num 0.074, relationship:0 0.069, sex 0.025, hours-per-week 0.023. Mean-ablation of the contribution: acc 0.858 -\> 0.801, AUC 0.913 -\> 0.681.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (3 s.f.).
- `empirical_viz`: get\_viz\_summary sensitivity on the full train split.
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "tractable": true,
 "dimensionality": [
  49,
  1
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "sensitivities": {
   "lin_agg": [
    0.023953217824896476,
    0.17340052301504869,
    0.05273146679061569,
    1.049674409072578,
    0.08740811178353353,
    0.06698149492263517,
    0.09446969177321073,
    0.0792174489320216,
    0.07392143995957175,
    0.03439693912384369,
    0.10228820273193864,
    0.879348697296242,
    0.051552421145672434,
    0.6912546059121509,
    0.09448857823607969,
    0.15525622197199795,
    0.1209486152588448,
    0.06590534254446098,
    0.2234928954858057,
    0.16874244686749293,
    0.2682972736344297,
    0.6223323217382727,
    0.5158119140990569,
    0.023037342297887356,
    0.07822170829045047,
    0.34213573960190197,
    0.26367128032809767,
    0.30107127918021087,
    0.002688009047794558,
    0.013949780377742773,
    0.005820947767591979,
    0.008412098283574948,
    0.014495765465550087,
    0.0029347078868508475,
    0.02836225343238147,
    0.00869291993940603,
    0.029497507948035306,
    0.031216257761422717,
    0.03971912986264044,
    0.0404238450817065,
    0.036222507332560584,
    0.002395395641707005,
    0.06193330356581944,
    0.16276478394946456,
    0.036840460457867964,
    0.022987239256400845,
    0.014007050826804547,
    0.03155612980977491,
    0.03355626663399187
   ]
  }
 }
}
```

```json
// ablation_study
{
 "n": 39073,
 "acc": 0.8007,
 "auc": 0.6812,
 "split": "train",
 "method": "replace the annotation's logit contribution by its train mean (local forward pass)",
 "baseline_acc": 0.8578,
 "baseline_auc": 0.9126
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 8 |
| annotate | 14 |
| rename\_node | 107 |

0 splits, 8 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node cg\_out intercepts 2 connection(s) into 0 (from 6463, 7310); sits in capital\_gain\_penalty (exit); income\_logit (entry)
- `#108` identity node age\_out intercepts 2 connection(s) into 0 (from 3460, -1 (age)); sits in youth\_hinge (exit); income\_logit (entry)
- `#109` identity node us\_out intercepts 3 connection(s) into 0 (from 4018, 4607, -99 (native-country:39)); sits in us\_born\_offset (exit); category\_offsets (entry)
- `#110` identity node scot\_out intercepts 2 connection(s) into 0 (from 9997, -94 (native-country:34)); sits in scotland\_switch (exit); category\_offsets (entry)
- `#111` identity node rel2\_out intercepts 2 connection(s) into 0 (from 14966, -53 (relationship:2)); sits in other\_relative\_offset (exit); category\_offsets (entry)
- `#112` identity node elsal\_out intercepts 2 connection(s) into 0 (from 13897, -106 (native-country:8)); sits in el\_salvador\_offset (exit); category\_offsets (entry)
- `#113` identity node cat\_agg intercepts 9 connection(s) into 0 (from us\_out, scot\_out, rel2\_out, elsal\_out, 5934, 7067, 8156, 10197, … (1 more)); sits in category\_offsets (exit, composition); income\_logit (entry)
- `#114` identity node lin\_agg intercepts 49 connection(s) into 0 (from -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -10 (workclass:8), -11 (fnlwgt), -16 (education:12), -22 (education:4), -23 (education:5), … (41 more)); sits in linear\_backbone (exit); income\_logit (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 5.55e-16 | 5.55e-16 | 5.55e-16 | agree |
| Closed-form max abs error (all leaves) | 0.5745 | 0.5745 | 0.5745 | agree |
| Numeric-function max abs error | n/a | 1.42e-14 | 1.42e-14 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit | income\_logit | agree |
| Leaf / composition counts | 12 / 2 | n/a | 12 / 2 | agree |
| Operation counts | 0 splits, 8 identity | {"add\_identity\_node": 8, "annotate": 14, "rename\_node": 107} | {"add\_identity\_node": 8, "annotate": 14, "rename\_node": 107} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capital_gain_penalty` | 0.5745 | 1.42e-14 |
| `federal_gov_offset` | 0.02601 | 0 |
| `spouse_absent_unknown_country` | 0.01865 | 0 |
| `linear_backbone` | 0.01841 | 1.42e-14 |
| `scotland_switch` | 0.007073 | 0 |
| `youth_hinge` | 0.005764 | 2.22e-16 |
| `handlers_asian_gate` | 0.002084 | 0 |
| `unmarried_offset` | 8.42e-04 | 0 |
| `other_relative_offset` | 5.28e-04 | 5.55e-17 |
| `transport_moving_offset` | 3.52e-04 | 0 |
| `el_salvador_offset` | 1.45e-04 | 1.39e-17 |
| `us_born_offset` | 3.21e-05 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Closed-form error 0.57 logit is 3-s.f. rounding in the capital-gain penalty at extreme gains (logit \~-100, output saturated); other leaves \<=0.026. Dead/redundant nodes: 4607 never active; Scotland gate masked by its saturating direct edge; 12550/15301 linear and near-inert. Age effect flat above the mean. Category names and USD kinks inferred.

**Tool notes (record):** Composition subgraph\_nodes must contain only its own nodes (child exits in entry\_nodes). KernelSHAP (200) \~9 min; noise on unconnected inputs.

## What to look at

- Largest closed-form error: `capital_gain_penalty` (0.574), then `federal_gov_offset` (0.026). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. It is large enough that a reader using the displayed coefficients would get visibly different exit values: check which coefficient drives it.
- Strong claim in `scotland_switch`: *“native-country:34 (Scotland, 17 train rows, standardized value 49.4) saturating switch.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `linear_backbone`: *“workclass:8 (Without-pay, 17 rows, value 47.9) is a saturating switch (+50 logit).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- No splits: every node sits in one annotation as evolved.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s3.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s3', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
