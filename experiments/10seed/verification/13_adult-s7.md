# Verification packet 13 · adult-s7

Pre-registered verification order: 13 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/bec72bab-2859-4988-8862-3cfd3e478abd?experiment=cc40bb89-41b1-478b-97e3-f15f1eebb3de&name=10seed-adult-s7), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 7 |
| Genome id | `bec72bab-2859-4988-8862-3cfd3e478abd` |
| Experiment | 10seed-adult-s7 (cc40bb89-41b1-478b-97e3-f15f1eebb3de); best at generation 99 of 100, NEAT key 18777 |
| Test AUC (runs csv) | 0.908 \[95% CI 0.901, 0.914\] |
| Train AUC | 0.91 |
| Size | 65 of 107 inputs connected; 15 hidden nodes; 95 enabled connections; depth 4; 18 skip connections + 57 direct input→output edges |
| Annotations | 8 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 15 of 15 |
| Claude's time | 53.0 min (2026-11-05T11:26:33+13:00 → 2026-11-05T12:19:31+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> logit P(income\<=50K): bias + eight disjoint additive terms. Mostly linear: household block (marital, relationship, sex) largest (ablation AUC 0.910 -\> 0.853), then numeric terms and category offsets. Three nonlinear pockets: a one-knot capital-gain ReLU; a youth gate (below-median age, threshold moved by Handlers-cleaners); hub 350, an off-switch shared by Exec-managerial, Farming-fishing, Doctorate, Without-pay and Armed-Forces. Train acc 0.854, AUC 0.910.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit_additive`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -1 (age), -2 (workclass:0), -3 (workclass:1), -4 (workclass:2), -5 (workclass:3), -6 (workclass:4), -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -10 (workclass:8), -11 (fnlwgt), -12 (education:0), -13 (education:1), -14 (education:10), -15 (education:11), -16 (education:12), -17 (education:13), -18 (education:14), -19 (education:15), -20 (education:2), -21 (education:3), -22 (education:4), -23 (education:5), -24 (education:6), -25 (education:7), -26 (education:8), -27 (education:9), -28 (education-num), -29 (marital-status:0), -30 (marital-status:1), -31 (marital-status:2), -32 (marital-status:3), -33 (marital-status:4), -34 (marital-status:5), -35 (marital-status:6), -36 (occupation:0), -37 (occupation:1), -38 (occupation:10), -39 (occupation:11), -40 (occupation:12), … (67 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_effect, age\_youth\_gate, occupation\_hub\_350, race\_white\_effect, gated\_category\_offsets, household\_linear, numeric\_linear, category\_linear |
| Dimensionality (in → out, incl. descendants) | 107 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.324 e^{- 1.0 \operatorname{age\_youth\_gate}{\left(x_{age},x_{occupation:6} \right)} - 1.0 \operatorname{capital\_gain\_effect}{\left(x_{capital-gain},x_{native-country:32} \right)} - 1.0 \operatorname{category\_linear}{\left(x_{workclass:0},x_{workclass:1},x_{workclass:3},x_{workclass:5},x_{workclass:6},x_{education:12},x_{education:13},x_{education:14},x_{education:15},x_{education:6},x_{education:7},x_{education:9},x_{occupation:0},x_{occupation:10},x_{occupation:14},x_{occupation:9},x_{race:0},x_{race:3},x_{native-country:12},x_{native-country:13},x_{native-country:14},x_{native-country:26},x_{native-country:27},x_{native-country:28},x_{native-country:29},x_{native-country:3},x_{native-country:35},x_{native-country:36},x_{native-country:38},x_{native-country:40},x_{native-country:6},x_{native-country:7},x_{native-country:8} \right)} - 1.0 \operatorname{gated\_category\_offsets\_0}{\left(x_{race:1},x_{occupation:8},x_{education:5},x_{education:1},x_{native-country:18},x_{education:0},x_{education:8},x_{occupation:7} \right)} + 0.115 \operatorname{gated\_category\_offsets\_1}{\left(x_{race:1},x_{occupation:8},x_{education:5},x_{education:1},x_{native-country:18},x_{education:0},x_{education:8},x_{occupation:7} \right)} + 0.00556 \operatorname{gated\_category\_offsets\_2}{\left(x_{race:1},x_{occupation:8},x_{education:5},x_{education:1},x_{native-country:18},x_{education:0},x_{education:8},x_{occupation:7} \right)} - 1.0 \operatorname{household\_linear}{\left(x_{marital-status:1},x_{marital-status:2},x_{relationship:0},x_{relationship:1},x_{relationship:2},x_{relationship:4},x_{relationship:5},x_{sex} \right)} - 1.0 \operatorname{numeric\_linear}{\left(x_{education-num},x_{hours-per-week},x_{capital-loss},x_{fnlwgt} \right)} - 1.0 \operatorname{occupation\_hub\_350}{\left(x_{workclass:8},x_{education:10},x_{occupation:12},x_{occupation:2},x_{occupation:4},x_{occupation:5},x_{native-country:4} \right)} - 1.0 \operatorname{race\_white\_effect}{\left(x_{race:4} \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Root composition: P(income\<=50K) = sigmoid(1.127 + capital\_gain\_effect + age\_youth\_gate + occupation\_hub\_350 + race\_white\_effect + gated\_category\_offsets\[idn\_gatedcats\] - 0.115\*gated\_category\_offsets\[10878\] - 0.0056\*gated\_category\_offsets\[7734\] + household\_linear + numeric\_linear + category\_linear). The composition is purely additive in logit space. Children share no nodes and none feeds another, so the network is a generalised additive model with a few interaction pockets, all inside children: the capital-gain knot (with a native-country:32 knot shift), the age x occupation:6 youth gate, the shared off-switch of hub 350, and the occupation:7 switch in 3845. Ablation ranking on train (term replaced by its mean; baseline acc 0.854 / AUC 0.910): household\_linear (AUC 0.853, 13.3% of predictions flip) \> capital\_gain\_effect (0.892, 8.1%) \> numeric\_linear (0.893, 7.0%) \> age\_youth\_gate (0.903, 5.2%) \> category\_linear (0.905, 3.2%) \> occupation\_hub\_350 (0.906, 2.5%) \> gated\_category\_offsets (0.909, 1.2%) \> race\_white\_effect (0.910, 0.7%). SHAP on the full model (train, max\_samples=500) agrees: top features are marital-status:2 0.122, relationship:0 0.081, relationship:1 0.061, age 0.049, education-num 0.047, capital-gain 0.045, relationship:5 0.027, relationship:4 0.026. 42 of 107 inputs have no enabled edge and are zero-effect entries of this root: -4, -6, -9, -15, -20, -21, -22, -29, -32, -33, -34, -35, -37, -39, -41, -44, -54, -59, -66, -67, -68, -69, -73, -74, -75, -77, -78, -79, -80, -81, -82, -83, -84, -90, -91, -93, -94, -97, -99, -102, -103, -107. Output class 1 = \<=50K; category names are inferred from PMLB alphabetical label encoding.

**Evidence (4 records, DB):**

- `analytical_formula`: (no narrative)
- `shap_importance`: (no narrative)
- `ablation_study`: (no narrative)
- `performance_metrics`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "output_bias": 1.1273749610694883
}
```

```json
// shap_importance
{
 "top": [
  [
   "marital-status:2",
   0.1223278875410322
  ],
  [
   "relationship:0",
   0.08100022073869054
  ],
  [
   "relationship:1",
   0.061241889429684296
  ],
  [
   "age",
   0.04926820372172896
  ],
  [
   "education-num",
   0.04685427587807796
  ],
  [
   "capital-gain",
   0.044515757931614434
  ],
  [
   "relationship:5",
   0.02695483958159609
  ],
  [
   "relationship:4",
   0.02557554803305623
  ],
  [
   "sex",
   0.024497156154345733
  ],
  [
   "occupation:4",
   0.02341427554497008
  ],
  [
   "hours-per-week",
   0.020503363955048422
  ],
  [
   "capital-loss",
   0.019773095674624697
  ]
 ],
 "split": "train",
 "max_samples": 500
}
```

```json
// ablation_study
{
 "split": "train",
 "n_rows": 39073,
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 },
 "per_child": {
  "L1": {
   "acc": 0.8383794436055588,
   "auc": 0.8922164269803341,
   "std": 3.7231519494773293,
   "flip": 0.0810278197220587,
   "mean": -0.1345593357213411
  },
  "L2": {
   "acc": 0.8493077060886034,
   "auc": 0.902670912860166,
   "std": 0.8641102897986561,
   "flip": 0.052235559081718834,
   "mean": 0.5456442050957111
  },
  "L3": {
   "acc": 0.8504082102730786,
   "auc": 0.9063658464584221,
   "std": 0.5024193711432141,
   "flip": 0.025081258157807183,
   "mean": 0.265976062681495
  },
  "L4": {
   "acc": 0.8541703989967496,
   "auc": 0.9096455146923317,
   "std": 0.11003632561882694,
   "flip": 0.006577431986282087,
   "mean": 0.6943872299658277
  },
  "L5": {
   "acc": 0.8530954879328436,
   "auc": 0.9086131864130667,
   "std": 0.29366400118351565,
   "flip": 0.012105546029227345,
   "mean": -0.23674619644047462
  },
  "L6": {
   "acc": 0.814552248355642,
   "auc": 0.8532555108971722,
   "std": 1.6159678111435953,
   "flip": 0.13326337880377753,
   "mean": -2.2112967181539218e-16
  },
  "L7a": {
   "acc": 0.8385585954495431,
   "auc": 0.8929365658105635,
   "std": 0.7500587630552874,
   "flip": 0.07030430220356768,
   "mean": -1.091100354352264e-18
  },
  "L7b": {
   "acc": 0.8494868579325877,
   "auc": 0.9049456593776536,
   "std": 0.5964238613615097,
   "flip": 0.0321449594349039,
   "mean": 4.7281015355264775e-17
  }
 }
}
```

```json
// performance_metrics
{
 "mae": 0.1963766533816903,
 "mse": 0.09972734376822265,
 "rmse": 0.3157963643999447,
 "auc_roc": 0.9099367272125733,
 "accuracy": 0.8541959921173189,
 "n_samples": 39073,
 "correlation": 0.6719696790109818
}
```

</details>

### A.1 Leaf: `capital_gain_effect`

|  |  |
|---|---|
| Annotation id | `ann_115` |
| Entry nodes | -63 (capital-gain), -92 (native-country:32) |
| Exit nodes | idn\_capgain |
| Internal / exit nodes in its own subgraph | 1690, idn\_capgain |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0177 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 2.49 x_{capital-gain} \\
&\quad - 2.32 \operatorname{relu}{\left(0.576 x_{capital-gain} - 0.0522 x_{native-country:32} - 0.274 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Capital-gain term (exit idn\_capgain, added to the output logit): f = -2.49\*cg - 2.32\*relu(0.576\*cg - 0.0522\*nc32 - 0.274), inputs z-scored. For the 91.6% of train rows with zero capital gain (cg\_z=-0.146) the gate is off and f=+0.36. The ReLU 1690 opens at cg\_z=0.476 (5.2% of rows), steepening the slope from -2.49 to -3.83 per SD: a monotone one-knot piecewise-linear decrease in logit P(\<=50K). Any substantial gain forces a \>50K prediction (f=-5.7 at cg\_z=1.66, -50 at the max). native-country:32 (Portugal, 51 rows) shifts the knot to cg\_z\~3.0 for those rows only. Evidence: line plot on the full train split (39,073 rows) gives corr(cg, f)=-0.999; ablation (term replaced by its train mean) drops train accuracy 0.854-\>0.838 and AUC 0.910-\>0.892 and flips 8.1% of predictions, the largest nonlinear term. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "y_mean": -24.408538982295052,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   -52.398762308829234,
   2.0240880774847714
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "idn_capgain"
  ],
  "correlation": -0.9989119971010222,
  "entry_names": [
   "-63",
   "-92"
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
   -49.8470030508121,
   0.36266504084771767
  ]
 },
 "viz_type": "line_plot"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8383794436055588,
  "auc": 0.8922164269803341,
  "std": 3.7231519494773293,
  "flip": 0.0810278197220587,
  "mean": -0.1345593357213411
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.2 Leaf: `age_youth_gate`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -1 (age), -47 (occupation:6) |
| Exit nodes | idn\_age |
| Internal / exit nodes in its own subgraph | 2682, idn\_age |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00599 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0624 x_{age} \\
&\quad + 0.669 \operatorname{relu}{\left(- 2.95 x_{age} + 0.395 x_{occupation:6} - 0.861 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Youth gate (exit idn\_age): f = -0.0624\*age + 0.669\*relu(-2.95\*age + 0.395\*occ6 - 0.861). ReLU 2682 is active when age\_z \< -0.32 (43% of train rows, younger than about the median age), where it raises logit P(\<=50K) linearly as age falls, reaching +2.58 at the youngest. Above the threshold only the small direct term (-0.062 per SD) remains. occupation:6 (Handlers-cleaners, 4.3% of rows) moves the threshold to age\_z \< 0.34: an age x occupation interaction in which handlers-cleaners count as 'young' up to an older age. Evidence: heatmap of age x occupation:6 on full train (39,073 rows); ablation flips 5.2% of predictions, accuracy 0.854-\>0.849, AUC 0.910-\>0.903. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "viz_type": "heatmap",
  "exit_names": [
   "idn_age"
  ],
  "correlation": -0.09402753568222359,
  "entry_names": [
   "-1",
   "-47"
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
   -0.21216039994207536,
   4.713414945831371
  ]
 },
 "viz_type": "heatmap"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8493077060886034,
  "auc": 0.902670912860166,
  "std": 0.8641102897986561,
  "flip": 0.052235559081718834,
  "mean": 0.5456442050957111
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.3 Leaf: `occupation_hub_350`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -10 (workclass:8), -14 (education:10), -40 (occupation:12), -43 (occupation:2), -45 (occupation:4), -46 (occupation:5), -100 (native-country:4) |
| Exit nodes | idn\_hub350 |
| Internal / exit nodes in its own subgraph | 7974, 10971, 3077, 350, idn\_hub350 |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |
| Live closed-form error (3 s.f. display) | 0.0115 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0919 x_{occupation:12} \\
&\quad + 0.00525 x_{occupation:2} \\
&\quad - 0.128 x_{occupation:4} \\
&\quad + 0.22 x_{occupation:5} \\
&\quad + 0.0206 x_{workclass:8} \\
&\quad + 0.238 \operatorname{relu}{\left(- 0.422 x_{education:10} + 0.956 x_{native-country:4} + 0.105 x_{occupation:12} - 0.211 x_{occupation:2} - 0.494 x_{occupation:4} - 0.703 x_{occupation:5} - 0.221 x_{workclass:8} - 0.981 \operatorname{relu}{\left(1.01 x_{occupation:4} + 0.246 \right)} + 0.982 \operatorname{relu}{\left(0.786 \operatorname{relu}{\left(0.979 x_{native-country:4} + 0.684 \right)} + 0.327 \right)} + 0.105 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hub ReLU 350 plus direct occupation/workclass terms (exit idn\_hub350). Every input is a one-hot, so on the data the subgraph is a lookup table over 16 observed category combinations. 7974 (occupation:4 only) and the 10971-\>3077 chain (native-country:4 only) are single-input relays. With none of the seven categories set, 350=1.21 and f=+0.33. Each of occupation:4 (Exec-managerial), occupation:5 (Farming-fishing), occupation:2 (Armed-Forces), education:10 (Doctorate) and workclass:8 (Without-pay) turns 350 off; the direct weights then set the category value: Exec-managerial f=-0.35, Doctorate +0.04, Farming-fishing +1.33, Without-pay +1.03, Armed-Forces +0.34, Sales (occupation:12) +0.12 with 350 still on. native-country:4 (Columbia, 64 rows, 98% \<=50K) drives 350 to \~43.5 and f to +10.4, a hard-coded 'Columbia =\> \<=50K' rule. So the hub works as an off-switch shared by several high- or low-income categories, plus one memorised country rule. Evidence: per-combination enumeration on full train; sensitivity on train (largest: native-country:4 0.34, occupation:4 0.23); ablation flips 2.5% of predictions, AUC 0.910-\>0.906. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_hub350"
  ],
  "entry_names": [
   "-10",
   "-14",
   "-40",
   "-43",
   "-45",
   "-46",
   "-100"
  ],
  "input_labels": [
   "-10",
   "-14",
   "-40",
   "-43",
   "-45",
   "-46",
   "-100"
  ],
  "sensitivities": {
   "idn_hub350": [
    0.03014181300619171,
    0.08377931285605673,
    0.07105755610498089,
    0.03840027625146286,
    0.22657970366922034,
    0.08063446646223864,
    0.3397141501597166
   ]
  },
  "dimensionality": [
   7,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8504082102730786,
  "auc": 0.9063658464584221,
  "std": 0.5024193711432141,
  "flip": 0.025081258157807183,
  "mean": 0.265976062681495
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.4 Leaf: `race_white_effect`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -61 (race:4) |
| Exit nodes | idn\_white |
| Internal / exit nodes in its own subgraph | 7193, 771, idn\_white |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00206 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.395 x_{race:4} \\
&\quad + 0.761 \operatorname{relu}{\left(1.41 x_{race:4} + 1.4 \operatorname{relu}{\left(0.617 x_{race:4} - 0.185 \right)} + 0.388 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> race:4 (White, 85.5% of rows) term (exit idn\_white): f = -0.395\*r4 + 0.761\*relu(1.41\*r4 + 1.40\*relu(0.617\*r4 - 0.185) + 0.388). As a function of one binary input it takes two values: +0.649 for White and +0.962 for non-White, so being White lowers logit P(\<=50K) by 0.31. The two-ReLU chain (7193-\>771) is excess structure for a single offset. Evidence: line plot on full train (two observed points, corr -1.0); ablation is negligible (AUC 0.9099-\>0.9096, 0.66% of predictions flip). Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "y_mean": 0.5264837816735867,
  "x_range": [
   -2.574887179225688,
   0.5532543250226041
  ],
  "y_range": [
   0.11068120977439083,
   1.0183044306948514
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "idn_white"
  ],
  "correlation": -1.0,
  "entry_names": [
   "-61"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -2.432698929032584,
   0.4110660748294999
  ],
  "scatter_y_range": [
   0.6491550295050212,
   0.9620724814535154
  ]
 },
 "viz_type": "line_plot"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8541703989967496,
  "auc": 0.9096455146923317,
  "std": 0.11003632561882694,
  "flip": 0.006577431986282087,
  "mean": 0.6943872299658277
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.5 Leaf: `gated_category_offsets`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -58 (race:1), -49 (occupation:8), -23 (education:5), -13 (education:1), -76 (native-country:18), -12 (education:0), -26 (education:8), -48 (occupation:7) |
| Exit nodes | idn\_gatedcats, 10878, 7734 |
| Internal / exit nodes in its own subgraph | 12248, 10878, 1339, 5330, 10607, 7734, 3845, idn\_gatedcats |
| Dimensionality (in → out, incl. descendants) | 8 → 3 |
| Live closed-form error (3 s.f. display) | 0.0112 |
| Live numeric-function error | 1.67e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.0175 x_{education:1} \\
&\quad + 0.0944 x_{education:5} \\
&\quad - 0.0136 x_{education:8} \\
&\quad + 0.302 x_{occupation:8} \\
&\quad + 0.0486 \operatorname{relu}{\left(0.988 x_{education:1} + 0.796 \right)} \\
&\quad - 0.092 \operatorname{relu}{\left(0.948 x_{education:5} + 1.95 \right)} \\
&\quad - 0.0645 \operatorname{relu}{\left(0.686 x_{occupation:8} - 0.454 \right)} \\
&\quad - 0.0383 \operatorname{relu}{\left(0.705 x_{education:0} + 0.114 x_{education:8} - 0.94 x_{occupation:7} + 1.75 \right)}
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 1.01 \operatorname{relu}{\left(0.977 x_{race:1} - 0.818 \right)} \\
&\quad -0.684 \bigr)
\end{aligned} \\ y_2 = \operatorname{relu}{\left(1.0 x_{native-country:18} - 0.645 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Single-category ReLU units on one-hot inputs, with their direct edges. Exits: idn\_gatedcats (summed term), 10878 (race:1 chain) and 7734 (native-country:18); all three feed the output logit with weights 1, -0.115 and -0.0056. Each unit sees one binary input, so on the data it is two-valued and exactly affine in that input. 5330 and 10607 are always active (linear); 1339 is active only for occupation:8; 12248-\>10878 only for race:1; 7734 only for native-country:18. The one real interaction is 3845 (education:0, education:8, occupation:7): it switches off for occupation:7 (Machine-op-inspct), which removes the tiny education:0/8 effects for machine operators. On-vs-off logit effects on train: occupation:8 (Other-service) +0.91, education:1 (11th) +0.46, race:1 (Asian-Pac-Islander) -0.44, native-country:18 (Hungary, 16 rows) -0.31, education:5 (7th-8th) +0.12, education:8 (Assoc-voc) -0.12, education:0 (10th) -0.06, occupation:7 -0.02. Evidence: sensitivity on full train (occupation:8 dominates at 0.30); ablation flips 1.2% of predictions, AUC 0.910-\>0.909. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_gatedcats",
   "10878",
   "7734"
  ],
  "entry_names": [
   "-58",
   "-49",
   "-23",
   "-13",
   "-76",
   "-12",
   "-26",
   "-48"
  ],
  "input_labels": [
   "-58",
   "-49",
   "-23",
   "-13",
   "-76",
   "-12",
   "-26",
   "-48"
  ],
  "sensitivities": {
   "7734": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.00040946789168903636,
    0.0,
    0.0,
    0.0
   ],
   "10878": [
    0.030257701337527528,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ],
   "idn_gatedcats": [
    0.0,
    0.2977101398447447,
    0.007205518594226755,
    0.06547106179842119,
    0.0,
    0.02541453208566583,
    0.017757779750433265,
    0.03390212345143487
   ]
  },
  "dimensionality": [
   8,
   3
  ]
 },
 "viz_type": "sensitivity"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8530954879328436,
  "auc": 0.9086131864130667,
  "std": 0.29366400118351565,
  "flip": 0.012105546029227345,
  "mean": -0.23674619644047462
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.6 Leaf: `household_linear`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -30 (marital-status:1), -31 (marital-status:2), -51 (relationship:0), -52 (relationship:1), -53 (relationship:2), -55 (relationship:4), -56 (relationship:5), -62 (sex) |
| Exit nodes | idn\_household |
| Internal / exit nodes in its own subgraph | idn\_household |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |
| Live closed-form error (3 s.f. display) | 0.00389 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0857 x_{marital-status:1} \\
&\quad - 1.19 x_{marital-status:2} \\
&\quad - 0.748 x_{relationship:0} \\
&\quad - 0.852 x_{relationship:1} \\
&\quad - 0.212 x_{relationship:2} \\
&\quad - 0.543 x_{relationship:4} \\
&\quad - 0.556 x_{relationship:5} \\
&\quad - 0.314 x_{sex}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Purely linear household block (exit idn\_household): marital-status, relationship and sex one-hots. On-vs-off logit effects: marital-status:1 (Married-AF-spouse) -3.15, relationship:5 (Wife) -2.59, marital-status:2 (Married-civ-spouse) -2.39, relationship:1 (Not-in-family) -1.95, relationship:4 (Unmarried) -1.78, relationship:0 (Husband) -1.52, relationship:2 (Other-relative) -1.24, sex=Male -0.67. relationship:3 (Own-child) has no edge, so it is the reference category with the highest P(\<=50K). A married husband gets -1.52-2.39-0.67 relative to an unmarried female own-child. Evidence: sensitivity on full train equals \|weights\|; ablation is the largest of all children: accuracy 0.854-\>0.815, AUC 0.910-\>0.853, 13.3% of predictions flip. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_household"
  ],
  "entry_names": [
   "-30",
   "-31",
   "-51",
   "-52",
   "-53",
   "-55",
   "-56",
   "-62"
  ],
  "input_labels": [
   "-30",
   "-31",
   "-51",
   "-52",
   "-53",
   "-55",
   "-56",
   "-62"
  ],
  "sensitivities": {
   "idn_household": [
    0.08574128103289438,
    1.1882261423187046,
    0.7477683560943067,
    0.8521349670569047,
    0.21185435244225723,
    0.5431562641501545,
    0.5556539124719789,
    0.31435716496255317
   ]
  },
  "dimensionality": [
   8,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.814552248355642,
  "auc": 0.8532555108971722,
  "std": 1.6159678111435953,
  "flip": 0.13326337880377753,
  "mean": -2.2112967181539218e-16
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.7 Leaf: `numeric_linear`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -28 (education-num), -65 (hours-per-week), -64 (capital-loss), -11 (fnlwgt) |
| Exit nodes | idn\_numeric |
| Internal / exit nodes in its own subgraph | idn\_numeric |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.00493 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.273 x_{capital-loss} \\
&\quad - 0.557 x_{education-num} \\
&\quad - 0.078 x_{fnlwgt} \\
&\quad - 0.31 x_{hours-per-week}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear numeric block (exit idn\_numeric), logit change per SD: education-num -0.557, hours-per-week -0.310, capital-loss -0.273, fnlwgt -0.078. More schooling, longer hours and capital losses (a proxy for owning assets) all push toward \>50K. There are no hidden units. Evidence: sensitivity on full train matches the weights; ablation: accuracy 0.854-\>0.839, AUC 0.910-\>0.893, 7.0% of predictions flip. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_numeric"
  ],
  "entry_names": [
   "-28",
   "-65",
   "-64",
   "-11"
  ],
  "input_labels": [
   "-28",
   "-65",
   "-64",
   "-11"
  ],
  "sensitivities": {
   "idn_numeric": [
    0.5566056472521663,
    0.3096179590118003,
    0.27258994886345905,
    0.07800103350616687
   ]
  },
  "dimensionality": [
   4,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8385585954495431,
  "auc": 0.8929365658105635,
  "std": 0.7500587630552874,
  "flip": 0.07030430220356768,
  "mean": -1.091100354352264e-18
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

### A.8 Leaf: `category_linear`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -2 (workclass:0), -3 (workclass:1), -5 (workclass:3), -7 (workclass:5), -8 (workclass:6), -16 (education:12), -17 (education:13), -18 (education:14), -19 (education:15), -24 (education:6), -25 (education:7), -27 (education:9), -36 (occupation:0), -38 (occupation:10), -42 (occupation:14), -50 (occupation:9), -57 (race:0), -60 (race:3), -70 (native-country:12), -71 (native-country:13), -72 (native-country:14), -85 (native-country:26), -86 (native-country:27), -87 (native-country:28), -88 (native-country:29), -89 (native-country:3), -95 (native-country:35), -96 (native-country:36), -98 (native-country:38), -101 (native-country:40), -104 (native-country:6), -105 (native-country:7), -106 (native-country:8) |
| Exit nodes | idn\_catlinear |
| Internal / exit nodes in its own subgraph | idn\_catlinear |
| Dimensionality (in → out, incl. descendants) | 33 → 1 |
| Live closed-form error (3 s.f. display) | 0.006 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0929 x_{education:12} \\
&\quad + 0.0471 x_{education:13} \\
&\quad - 0.0917 x_{education:14} \\
&\quad - 0.0735 x_{education:15} \\
&\quad + 0.0278 x_{education:6} \\
&\quad + 0.00371 x_{education:7} \\
&\quad - 0.0996 x_{education:9} \\
&\quad + 0.0109 x_{native-country:12} \\
&\quad + 0.0162 x_{native-country:13} \\
&\quad + 0.0154 x_{native-country:14} \\
&\quad + 0.109 x_{native-country:26} \\
&\quad + 0.0246 x_{native-country:27} \\
&\quad + 0.0151 x_{native-country:28} \\
&\quad + 0.0359 x_{native-country:29} \\
&\quad + 0.0641 x_{native-country:35} \\
&\quad + 0.0068 x_{native-country:36} \\
&\quad + 0.0767 x_{native-country:38} \\
&\quad + 0.0491 x_{native-country:3} \\
&\quad + 0.0488 x_{native-country:40} \\
&\quad + 0.0461 x_{native-country:6} \\
&\quad + 0.0169 x_{native-country:7} \\
&\quad + 0.0298 x_{native-country:8} \\
&\quad + 0.246 x_{occupation:0} \\
&\quad - 0.132 x_{occupation:10} \\
&\quad + 0.0284 x_{occupation:14} \\
&\quad + 0.433 x_{occupation:9} \\
&\quad + 0.0143 x_{race:0} \\
&\quad - 0.0196 x_{race:3} \\
&\quad - 0.108 x_{workclass:0} \\
&\quad - 0.085 x_{workclass:1} \\
&\quad + 0.0066 x_{workclass:3} \\
&\quad - 0.0345 x_{workclass:5} \\
&\quad + 0.15 x_{workclass:6}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear one-hot offsets (exit idn\_catlinear) for the remaining workclass, education, occupation, race and native-country categories. Inputs are z-scored one-hots, so a rare category gets a large on-value and a small weight becomes a large effect. On-vs-off logit effects: occupation:9 (Priv-house-serv) +6.27, native-country:38 (Trinadad&Tobago) +3.23, native-country:35 (South) +1.32, native-country:40 (Vietnam) +1.23, education:13 (Preschool) +1.16, native-country:29 (Peru) +1.15, occupation:0 (unknown) +1.06, education:14 (Prof-school) -0.72. Most are rare low-income categories, memorised as offsets. Evidence: sensitivity on full train equals \|weights\|; ablation flips 3.2% of predictions, AUC 0.910-\>0.905. Category names (in parentheses) are inferred from PMLB's alphabetical label encoding and checked against category frequency and P(\<=50K) on train; the network sees only the index. Output class 1 is income \<=50K (train base rate 0.761), so negative terms push toward \>50K.

**Evidence (3 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)
- `ablation_study`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "closed_form_note": "3 s.f. display; numeric max abs err <1e-14 on train"
}
```

```json
// empirical_viz
{
 "split": "train",
 "n_rows": 39073,
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "idn_catlinear"
  ],
  "entry_names": [
   "-2",
   "-3",
   "-5",
   "-7",
   "-8",
   "-16",
   "-17",
   "-18",
   "-19",
   "-24",
   "-25",
   "-27",
   "-36",
   "-38",
   "-42",
   "-50",
   "-57",
   "-60",
   "-70",
   "-71",
   "-72",
   "-85",
   "-86",
   "-87",
   "-88",
   "-89",
   "-95",
   "-96",
   "-98",
   "-101",
   "-104",
   "-105",
   "-106"
  ],
  "input_labels": [
   "-2",
   "-3",
   "-5",
   "-7",
   "-8",
   "-16",
   "-17",
   "-18",
   "-19",
   "-24",
   "-25",
   "-27",
   "-36",
   "-38",
   "-42",
   "-50",
   "-57",
   "-60",
   "-70",
   "-71",
   "-72",
   "-85",
   "-86",
   "-87",
   "-88",
   "-89",
   "-95",
   "-96",
   "-98",
   "-101",
   "-104",
   "-105",
   "-106"
  ],
  "sensitivities": {
   "idn_catlinear": [
    0.10841644636914492,
    0.0849664849905087,
    0.006599501596652227,
    0.03451891798799803,
    0.14961007123568057,
    0.09293578006121647,
    0.04708619246764778,
    0.09169391401979997,
    0.07354396765684976,
    0.027787766249599687,
    0.0037147687323593405,
    0.09963118037251999,
    0.24593241441942287,
    0.13181275556181235,
    0.028354818639032463,
    0.4326698551611571,
    0.014280304222533645,
    0.019557048970507473,
    0.010902036602382198,
    0.0161512443544197,
    0.015373551077646859,
    0.10887147815425088,
    0.02456982774925105,
    0.01506471170412076,
    0.03589827831321322,
    0.04905378934573559,
    0.06410135642451383,
    0.006797760636186438,
    0.07666701472333637,
    0.04876016169623222,
    0.04608820954702055,
    0.016882034844484733,
    0.02975701792411636
   ]
  },
  "dimensionality": [
   33,
   1
  ]
 },
 "viz_type": "sensitivity"
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace term with its train mean, local forward pass on in-memory copy",
 "n_rows": 39073,
 "ablated": {
  "acc": 0.8494868579325877,
  "auc": 0.9049456593776536,
  "std": 0.5964238613615097,
  "flip": 0.0321449594349039,
  "mean": 4.7281015355264775e-17
 },
 "baseline": {
  "auc": 0.9099,
  "accuracy": 0.8542
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 8 |
| annotate | 9 |
| rename\_node | 107 |

0 splits, 8 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node idn\_capgain intercepts 2 connection(s) into 0 (from -63 (capital-gain), 1690); sits in capital\_gain\_effect (exit)
- `#108` identity node idn\_age intercepts 2 connection(s) into 0 (from -1 (age), 2682); sits in age\_youth\_gate (exit)
- `#109` identity node idn\_hub350 intercepts 6 connection(s) into 0 (from -10 (workclass:8), -40 (occupation:12), -43 (occupation:2), -45 (occupation:4), -46 (occupation:5), 350); sits in occupation\_hub\_350 (exit)
- `#110` identity node idn\_white intercepts 2 connection(s) into 0 (from -61 (race:4), 771); sits in race\_white\_effect (exit)
- `#111` identity node idn\_gatedcats intercepts 8 connection(s) into 0 (from -49 (occupation:8), 1339, -23 (education:5), 5330, -13 (education:1), 10607, -26 (education:8), 3845); sits in gated\_category\_offsets (exit)
- `#112` identity node idn\_household intercepts 8 connection(s) into 0 (from -30 (marital-status:1), -31 (marital-status:2), -51 (relationship:0), -52 (relationship:1), -53 (relationship:2), -55 (relationship:4), -56 (relationship:5), -62 (sex)); sits in household\_linear (exit)
- `#113` identity node idn\_numeric intercepts 4 connection(s) into 0 (from -28 (education-num), -65 (hours-per-week), -64 (capital-loss), -11 (fnlwgt)); sits in numeric\_linear (exit)
- `#114` identity node idn\_catlinear intercepts 33 connection(s) into 0 (from -2 (workclass:0), -3 (workclass:1), -5 (workclass:3), -7 (workclass:5), -8 (workclass:6), -16 (education:12), -17 (education:13), -18 (education:14), … (25 more)); sits in category\_linear (exit)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T12:25:06+13:00 (commit `77513b9`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 3.33e-16 | 3.33e-16 | 3.33e-16 | agree |
| Closed-form max abs error (all leaves) | 0.0177 | 0.01774 | 0.01774 | agree |
| Numeric-function max abs error | n/a | 3.55e-15 | 3.55e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit\_additive | income\_logit\_additive | agree |
| Leaf / composition counts | 8 / 1 | n/a | 8 / 1 | agree |
| Operation counts | 0 splits, 8 identity | {"add\_identity\_node": 8, "annotate": 9, "rename\_node": 107} | {"add\_identity\_node": 8, "annotate": 9, "rename\_node": 107} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capital_gain_effect` | 0.01774 | 0 |
| `occupation_hub_350` | 0.0115 | 3.55e-15 |
| `gated_category_offsets` | 0.01121 | 1.67e-16 |
| `category_linear` | 0.005995 | 1.78e-15 |
| `age_youth_gate` | 0.005995 | 0 |
| `numeric_linear` | 0.004932 | 8.88e-16 |
| `household_linear` | 0.003885 | 4.44e-16 |
| `race_white_effect` | 0.002064 | 0 |

## What resisted (from the record)

> Nothing structural. Much hidden structure is excess: most ReLU units see one z-scored one-hot input, so each is a two-valued offset (race:4 chain, race:1 chain, 10971-\>3077 relay). Hub 350 is a 16-entry lookup table including a memorised rule (native-country:4, 64 rows, +10.4 logit). Category names inferred from PMLB alphabetical coding.

**Tool notes (record):** None. Compositions list only their own nodes in subgraph\_nodes. compute\_shap (500) \~17 min.

## What to look at

- Largest closed-form error: `capital_gain_effect` (0.0177), then `occupation_hub_350` (0.0115). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `capital_gain_effect`: *“native-country:32 (Portugal, 51 rows) shifts the knot to cg\_z\~3.0 for those rows only.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `age_youth_gate`: *“Above the threshold only the small direct term (-0.062 per SD) remains.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- No splits: every node sits in one annotation as evolved.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s7.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s7', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
