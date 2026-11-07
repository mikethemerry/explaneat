# Verification packet 10 · adult-s2

Pre-registered verification order: 10 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/b6600773-0fcc-4667-b053-776782913641?experiment=7ac82a92-3161-4e89-ac6d-496d07343c00&name=10seed-adult-s2), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 2 |
| Genome id | `b6600773-0fcc-4667-b053-776782913641` |
| Experiment | 10seed-adult-s2 (7ac82a92-3161-4e89-ac6d-496d07343c00); best at generation 99 of 100, NEAT key 23772 |
| Test AUC (runs csv) | 0.903 \[95% CI 0.897, 0.91\] |
| Train AUC | 0.906 |
| Size | 60 of 107 inputs connected; 16 hidden nodes; 91 enabled connections; depth 3; 9 skip connections + 54 direct input→output edges |
| Annotations | 14 leaves, 5 compositions |
| Hidden nodes in exactly one leaf, unsplit | 16 of 16 |
| Claude's time | 71.3 min (2026-11-05T11:08:23+13:00 → 2026-11-05T12:19:41+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Generalised additive logistic model of P(income\<=50K): logit = 2.407 + capital + household + education + work + background, no cross-group interactions. Convex piecewise-linear capital-gain penalty (kink \~\$4-4.7k); married-civ-spouse -3.52 partly offset by a Husband ReLU (+0.74); education years as a hinge steepening from HS-grad up. Train acc 0.850, AUC 0.906.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit`

|  |  |
|---|---|
| Annotation id | `ann_142` |
| Entry nodes | -63 (capital-gain), -64 (capital-loss), -5 (workclass:3), -51 (relationship:0), -77 (native-country:19), -30 (marital-status:1), -31 (marital-status:2), -33 (marital-status:4), -34 (marital-status:5), -53 (relationship:2), -54 (relationship:3), -56 (relationship:5), -28 (education-num), -69 (native-country:11), -19 (education:15), -12 (education:0), -14 (education:10), -16 (education:12), -21 (education:3), -25 (education:7), -36 (occupation:0), -47 (occupation:6), -88 (native-country:29), -6 (workclass:4), -17 (education:13), -48 (occupation:7), -3 (workclass:1), -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -37 (occupation:1), -38 (occupation:10), -39 (occupation:11), -40 (occupation:12), -45 (occupation:4), -46 (occupation:5), -49 (occupation:8), -65 (hours-per-week), -1 (age), -58 (race:1), … (67 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_hinge, household\_effect, education\_effect, work\_effect, background\_effect |
| Dimensionality (in → out, incl. descendants) | 107 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.0 \operatorname{background\_effect}{\left(x_{age},x_{race:1},x_{race:3},x_{race:4},x_{sex},x_{native-country:12},x_{native-country:15},x_{native-country:16},x_{native-country:20},x_{native-country:22},x_{native-country:25},x_{native-country:28},x_{native-country:3},x_{native-country:30},x_{native-country:37},x_{native-country:4},x_{native-country:40},x_{native-country:7},x_{native-country:8},x_{native-country:9},x_{native-country:36},x_{fnlwgt} \right)} + 1.0 \operatorname{capital\_gain\_hinge}{\left(x_{capital-gain},x_{capital-loss} \right)} + 1.0 \operatorname{education\_effect}{\left(x_{education-num},x_{native-country:11},x_{education:15},x_{education:0},x_{education:10},x_{education:12},x_{education:3},x_{education:7} \right)} + 1.0 \operatorname{household\_effect}{\left(x_{workclass:3},x_{relationship:0},x_{native-country:19},x_{marital-status:1},x_{marital-status:2},x_{marital-status:4},x_{marital-status:5},x_{relationship:2},x_{relationship:3},x_{relationship:5} \right)} + 1.0 \operatorname{work\_effect}{\left(x_{occupation:0},x_{occupation:6},x_{native-country:29},x_{workclass:4},x_{education:13},x_{occupation:7},x_{workclass:1},x_{workclass:5},x_{workclass:6},x_{workclass:7},x_{occupation:1},x_{occupation:10},x_{occupation:11},x_{occupation:12},x_{occupation:4},x_{occupation:5},x_{occupation:8},x_{hours-per-week} \right)}}}{e^{1.0 \operatorname{background\_effect}{\left(x_{age},x_{race:1},x_{race:3},x_{race:4},x_{sex},x_{native-country:12},x_{native-country:15},x_{native-country:16},x_{native-country:20},x_{native-country:22},x_{native-country:25},x_{native-country:28},x_{native-country:3},x_{native-country:30},x_{native-country:37},x_{native-country:4},x_{native-country:40},x_{native-country:7},x_{native-country:8},x_{native-country:9},x_{native-country:36},x_{fnlwgt} \right)} + 1.0 \operatorname{capital\_gain\_hinge}{\left(x_{capital-gain},x_{capital-loss} \right)} + 1.0 \operatorname{education\_effect}{\left(x_{education-num},x_{native-country:11},x_{education:15},x_{education:0},x_{education:10},x_{education:12},x_{education:3},x_{education:7} \right)} + 1.0 \operatorname{household\_effect}{\left(x_{workclass:3},x_{relationship:0},x_{native-country:19},x_{marital-status:1},x_{marital-status:2},x_{marital-status:4},x_{marital-status:5},x_{relationship:2},x_{relationship:3},x_{relationship:5} \right)} + 1.0 \operatorname{work\_effect}{\left(x_{occupation:0},x_{occupation:6},x_{native-country:29},x_{workclass:4},x_{education:13},x_{occupation:7},x_{workclass:1},x_{workclass:5},x_{workclass:6},x_{workclass:7},x_{occupation:1},x_{occupation:10},x_{occupation:11},x_{occupation:12},x_{occupation:4},x_{occupation:5},x_{occupation:8},x_{hours-per-week} \right)}} + 0.0901}\end{gathered}
$$

**Hypothesis (DB):**

> Output = logit P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs z-scored; category names inferred from alphabetical codes, consistent with train frequencies. ROOT. HYPOTHESIS: a generalised additive logistic model, logit = 2.407 + capital + household + education + work + background, with no cross-group interactions (every hidden relu sees inputs from one feature group). Material nonlinearity: capital-gain hinge (dominant), education-years hinge, Husband indicator gate; everything else is linear or exactly affine on train (8774 and 4066 are always off, 2255/4651/5909/14057/2821/6000 always on). 47 inputs have no edges and are zero-effect entries. EVIDENCE (train n=39,073): acc 0.850, AUC 0.9064; best all-affine fit of the logit has AUC 0.9008 (residual sd 0.59 of logit sd 7.53), so \~0.0055 AUC is due to nonlinearity, mostly capital gain. Block mean-ablation AUCs: capital 0.884, household 0.856, education 0.884, work 0.891, background 0.898. SHAP (compute\_shap, train, max\_samples=500): marital-status:2 0.168, education-num 0.080, capital-gain 0.042, sex 0.042, age 0.037, relationship:0 0.036, hours-per-week 0.028. Caveat: z-scoring makes small weights on rare one-hots large (Outlying-US +17 logit, n=20).

**Evidence (3 records, DB):**

- `shap`: compute\_shap on the train split, max\_samples=500 (subsample because the train split has 39,073 rows).
- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// shap
{
 "split": "train",
 "output": "model output (probability scale per tool)",
 "base_value": 0.7148248578825098,
 "sample_size": 500,
 "top12_mean_abs_shap": {
  "age": 0.03661,
  "sex": 0.04195,
  "capital-gain": 0.04218,
  "capital-loss": 0.0182,
  "occupation:4": 0.01551,
  "occupation:8": 0.01782,
  "education-num": 0.08044,
  "hours-per-week": 0.028,
  "relationship:0": 0.03602,
  "relationship:3": 0.02391,
  "relationship:5": 0.01432,
  "marital-status:2": 0.16806
 }
}
```

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  107,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "logit_sd": 7.5328,
 "acc_mean_ablated": 0.7612,
 "auc_mean_ablated": 0.42613,
 "auc_affine_replaced": 0.90084,
 "contribution_logit_sd": 7.5328,
 "affine_fit_residual_sd": 0.5948,
 "affine_fit_max_abs_resid": 2.7636,
 "all_affine_logit_fit_auc": 0.90079
}
```

</details>

### A.1 Leaf: `capital_gain_hinge`

|  |  |
|---|---|
| Annotation id | `ann_124` |
| Entry nodes | -63 (capital-gain), -64 (capital-loss) |
| Exit nodes | identity\_capital |
| Internal / exit nodes in its own subgraph | 4780, 2450, 7922, identity\_capital |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0466 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 1.99 x_{capital-gain} \\
&\quad - 0.297 x_{capital-loss} \\
&\quad + 0.0542 \operatorname{relu}{\left(1.7 - 2.29 \operatorname{relu}{\left(0.748 x_{capital-gain} - 0.291 \right)} \right)} \\
&\quad - 2.66 \operatorname{relu}{\left(0.901 x_{capital-gain} + 1.13 \operatorname{relu}{\left(0.748 x_{capital-gain} - 0.291 \right)} - 0.519 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output = logit P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs z-scored; category names inferred from alphabetical codes, consistent with train frequencies. HYPOTHESIS: convex piecewise-linear penalty on capital gain plus linear capital loss. Slope -1.99/sd (-0.27 logit per \$1000) up to \~\$4.0k (4780 on), 2450 on at \~\$4.7k, slope then -6.63/sd (-0.88 per \$1000); 7922 is a +0.054-weight bump vanishing by \~\$11.5k; capital-loss -0.30/sd. EVIDENCE (train n=39,073): closed form from get\_formula; line plot over capital gain monotone decreasing (corr -0.99, range +0.45 to -85 logit); contribution sd 6.29 logit (largest block); mean-ablation AUC 0.9064-\>0.8838, acc 0.850-\>0.816; replacing the hinge with its best affine fit gives AUC 0.9013 (most of the network's total nonlinearity benefit: full-affine model 0.9008). Data agree: rows with gains \$7k-10k have P(\<=50K)=0.011. SHAP (500 rows) ranks capital-gain 3rd.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8156,
 "auc_mean_ablated": 0.8838,
 "line_plot_summary": {
  "y_mean": -41.19574567726309,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   -89.55550537121022,
   1.7761859497858623
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_capital"
  ],
  "correlation": -0.9939907731060694,
  "entry_names": [
   "-63",
   "-64"
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
   -85.13467482345726,
   0.44685339210633196
  ]
 },
 "auc_affine_replaced": 0.9013,
 "contribution_logit_sd": 6.2855,
 "affine_fit_residual_sd": 0.6037,
 "affine_fit_max_abs_resid": 2.5191
}
```

</details>

### A.2 Composition: `household_effect`

|  |  |
|---|---|
| Annotation id | `ann_138` |
| Entry nodes | -5 (workclass:3), -51 (relationship:0), -77 (native-country:19), -30 (marital-status:1), -31 (marital-status:2), -33 (marital-status:4), -34 (marital-status:5), -53 (relationship:2), -54 (relationship:3), -56 (relationship:5) |
| Exit nodes | identity\_household |
| Internal / exit nodes in its own subgraph | identity\_household |
| Children | husband\_gate, family\_linear |
| Dimensionality (in → out, incl. descendants) | 10 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.0 \operatorname{family\_linear}{\left(x_{marital-status:1},x_{marital-status:2},x_{marital-status:4},x_{marital-status:5},x_{relationship:2},x_{relationship:3},x_{relationship:5} \right)} \\
&\quad + 1.0 \operatorname{husband\_gate}{\left(x_{workclass:3},x_{relationship:0},x_{native-country:19} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output = logit P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs z-scored; category names inferred from alphabetical codes, consistent with train frequencies. COMPOSITION (additive): husband\_gate + family\_linear. Married-civ-spouse -3.52 logit; Husband +0.74 back (net \~-2.8), Wife a further -1.33 (net \~-4.85); Own-child +2.42; Other-relative +1.11; Separated -0.80. EVIDENCE: contribution sd 2.31 logit (largest after capital); mean-ablation AUC 0.9064-\>0.8563; affine residual sd 0.016. SHAP top feature is marital-status:2.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  10,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8235,
 "auc_mean_ablated": 0.85632,
 "auc_affine_replaced": 0.90637,
 "contribution_logit_sd": 2.3144,
 "affine_fit_residual_sd": 0.0163,
 "affine_fit_max_abs_resid": 0.3354
}
```

</details>

#### A.2.1 Leaf: `husband_gate`

|  |  |
|---|---|
| Annotation id | `ann_132` |
| Entry nodes | -5 (workclass:3), -51 (relationship:0), -77 (native-country:19) |
| Exit nodes | identity\_husband |
| Internal / exit nodes in its own subgraph | 406, identity\_husband |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.0408 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.679 x_{workclass:3} \\
&\quad + 0.387 \operatorname{relu}{\left(0.294 x_{native-country:19} + 1.69 x_{relationship:0} + 0.994 x_{workclass:3} - 0.121 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output = logit P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs z-scored; category names inferred from alphabetical codes, consistent with train frequencies. HYPOTHESIS: relu 406 is a Husband indicator: +0.74 logit when relationship=Husband (it is on for exactly the 40.3% Husband rows plus India-born rows), partly cancelling the -3.52 married-civ-spouse term. Rare side inputs: India +1.46 logit; Never-worked (9 rows) +69 logit total. EVIDENCE: closed form; activation takes 5 values on train; affine fit residual sd 0.016; mean-ablation AUC -\>0.9047, acc -\>0.842; affine replacement AUC unchanged (0.9064), so the relu is effectively an indicator, not an interaction.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  3,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8416,
 "auc_mean_ablated": 0.9047,
 "auc_affine_replaced": 0.90637,
 "contribution_logit_sd": 1.1153,
 "affine_fit_residual_sd": 0.0163,
 "affine_fit_max_abs_resid": 0.3347
}
```

</details>

#### A.2.2 Leaf: `family_linear`

|  |  |
|---|---|
| Annotation id | `ann_133` |
| Entry nodes | -30 (marital-status:1), -31 (marital-status:2), -33 (marital-status:4), -34 (marital-status:5), -53 (relationship:2), -54 (relationship:3), -56 (relationship:5) |
| Exit nodes | identity\_family\_linear |
| Internal / exit nodes in its own subgraph | identity\_family\_linear |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |
| Live closed-form error (3 s.f. display) | 0.00474 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.105 x_{marital-status:1} \\
&\quad - 1.75 x_{marital-status:2} \\
&\quad - 0.0227 x_{marital-status:4} \\
&\quad - 0.141 x_{marital-status:5} \\
&\quad + 0.19 x_{relationship:2} \\
&\quad + 0.878 x_{relationship:3} \\
&\quad - 0.286 x_{relationship:5}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Linear marital/relationship terms. Dominant: marital-status:2 (Married-civ-spouse) -3.52 logit when on (the largest single effect after capital gain). relationship:3 (Own-child) +2.42, relationship:5 (Wife) -1.33, relationship:2 (Other-relative) +1.11, marital-status:5 (Separated) -0.80, marital-status:1 (Married-AF, 29 rows) -3.86.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  7,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8123,
 "auc_mean_ablated": 0.82779,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 2.3132,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

### A.3 Composition: `education_effect`

|  |  |
|---|---|
| Annotation id | `ann_139` |
| Entry nodes | -28 (education-num), -69 (native-country:11), -19 (education:15), -12 (education:0), -14 (education:10), -16 (education:12), -21 (education:3), -25 (education:7) |
| Exit nodes | identity\_education |
| Internal / exit nodes in its own subgraph | identity\_education |
| Children | education\_years\_hinge, some\_college\_relay, education\_category\_linear |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.0 \operatorname{education\_category\_linear}{\left(x_{education:0},x_{education:10},x_{education:12},x_{education:3},x_{education:7} \right)} \\
&\quad + 1.0 \operatorname{education\_years\_hinge}{\left(x_{education-num},x_{native-country:11} \right)} \\
&\quad + 1.0 \operatorname{some\_college\_relay}{\left(x_{education:15} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K); positive = toward low income. Additive: education = education-years hinge (dominant, convex: steeper decline from HS-grad up) + affine Some-college term + small per-degree linear corrections.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  8,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8288,
 "auc_mean_ablated": 0.88444,
 "auc_affine_replaced": 0.90629,
 "contribution_logit_sd": 0.8971,
 "affine_fit_residual_sd": 0.1182,
 "affine_fit_max_abs_resid": 0.695
}
```

</details>

#### A.3.1 Leaf: `education_years_hinge`

|  |  |
|---|---|
| Annotation id | `ann_126` |
| Entry nodes | -28 (education-num), -69 (native-country:11) |
| Exit nodes | identity\_edunum |
| Internal / exit nodes in its own subgraph | 13337, 6000, identity\_edunum |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00768 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.35 x_{education-num} \\
&\quad + 0.0474 x_{native-country:11} \\
&\quad - 1.58 \operatorname{relu}{\left(0.193 x_{education-num} + 0.358 \operatorname{relu}{\left(0.75 x_{education-num} + 0.14 x_{native-country:11} + 0.437 \right)} + 1.03 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output = logit P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs z-scored; category names inferred from alphabetical codes, consistent with train frequencies. HYPOTHESIS: concave-down hinge on education years: -0.255 logit per level for levels 1-8, 13337 switches on at \~8.5 and the slope steepens to -0.42 per level from HS-grad (9) to Doctorate (16); total +0.68 to -4.37 logit. native-country:11 (Germany) is a negligible side input. EVIDENCE: closed form; line plot corr -0.99; contribution sd 0.95; mean-ablation AUC -\>0.8801; affine replacement AUC 0.9061 (hinge worth only \~0.0003 AUC). SHAP ranks education-num 2nd.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8244,
 "auc_mean_ablated": 0.88012,
 "line_plot_summary": {
  "y_mean": -1.5675442851390045,
  "x_range": [
   -3.8311006135397268,
   2.6010574854593473
  ],
  "y_range": [
   -4.685041521027119,
   0.8708392091710471
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_edunum"
  ],
  "correlation": -0.988747863729363,
  "entry_names": [
   "-28",
   "-69"
  ],
  "dimensionality": [
   2,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -3.5387297908579507,
   2.3086866627775713
  ],
  "scatter_y_range": [
   -4.8583234431513125,
   0.6794957652650402
  ]
 },
 "auc_affine_replaced": 0.90607,
 "contribution_logit_sd": 0.954,
 "affine_fit_residual_sd": 0.1395,
 "affine_fit_max_abs_resid": 0.7128
}
```

</details>

#### A.3.2 Leaf: `some_college_relay`

|  |  |
|---|---|
| Annotation id | `ann_128` |
| Entry nodes | -19 (education:15) |
| Exit nodes | identity\_somecollege |
| Internal / exit nodes in its own subgraph | 5909, 4066, identity\_somecollege |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 7.87e-05 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0587 x_{education:15} \\
&\quad + 0.515 \operatorname{relu}{\left(1.6 - 0.0132 x_{education:15} \right)} \\
&\quad - 0.403 \operatorname{relu}{\left(- 0.00902 \operatorname{relu}{\left(1.6 - 0.0132 x_{education:15} \right)} - 1.1 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). education:15 (Some-college) direct term plus relay 5909 (always active) and 4066 (always off). Exactly affine on train: -0.065/sd, i.e. -0.15 logit for Some-college. Negligible.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8502,
 "auc_mean_ablated": 0.90634,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.0655,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

#### A.3.3 Leaf: `education_category_linear`

|  |  |
|---|---|
| Annotation id | `ann_137` |
| Entry nodes | -12 (education:0), -14 (education:10), -16 (education:12), -21 (education:3), -25 (education:7) |
| Exit nodes | identity\_edcat\_linear |
| Internal / exit nodes in its own subgraph | identity\_edcat\_linear |
| Dimensionality (in → out, incl. descendants) | 5 → 1 |
| Live closed-form error (3 s.f. display) | 6.29e-04 |
| Live numeric-function error | 1.11e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0119 x_{education:0} \\
&\quad + 0.0588 x_{education:10} \\
&\quad + 0.0604 x_{education:12} \\
&\quad - 0.0495 x_{education:3} \\
&\quad + 0.0718 x_{education:7}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Small linear corrections from five education one-hots on top of the education-years curve: Doctorate +0.54, Masters +0.27, Assoc-acdm +0.41, 1st-4th -0.71, 10th -0.07 logit. Contribution sd 0.12.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  5,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8496,
 "auc_mean_ablated": 0.90671,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.1203,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

### A.4 Composition: `work_effect`

|  |  |
|---|---|
| Annotation id | `ann_140` |
| Entry nodes | -36 (occupation:0), -47 (occupation:6), -88 (native-country:29), -6 (workclass:4), -17 (education:13), -48 (occupation:7), -3 (workclass:1), -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -37 (occupation:1), -38 (occupation:10), -39 (occupation:11), -40 (occupation:12), -45 (occupation:4), -46 (occupation:5), -49 (occupation:8), -65 (hours-per-week) |
| Exit nodes | identity\_work |
| Internal / exit nodes in its own subgraph | identity\_work |
| Children | occupation\_unknown\_affine, private\_sector\_gate, machine\_op\_gate, work\_linear |
| Dimensionality (in → out, incl. descendants) | 18 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.166 \operatorname{machine\_op\_gate}{\left(x_{occupation:7} \right)} \\
&\quad + 1.0 \operatorname{occupation\_unknown\_affine}{\left(x_{occupation:0},x_{occupation:6},x_{native-country:29} \right)} \\
&\quad + 1.0 \operatorname{private\_sector\_gate}{\left(x_{workclass:4},x_{education:13} \right)} \\
&\quad + 1.0 \operatorname{work\_linear}{\left(x_{workclass:1},x_{workclass:5},x_{workclass:6},x_{workclass:7},x_{occupation:1},x_{occupation:10},x_{occupation:11},x_{occupation:12},x_{occupation:4},x_{occupation:5},x_{occupation:8},x_{hours-per-week} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K); positive = toward low income. Additive: work = linear occupation/workclass/hours terms + affine unknown-occupation/Handlers-cleaners term + Machine-op indicator gate + negligible Private gate. All children are affine or indicator-valued on train; no interactions between work features.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  18,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8416,
 "auc_mean_ablated": 0.89112,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.9338,
 "affine_fit_residual_sd": 0.0002,
 "affine_fit_max_abs_resid": 0.0076
}
```

</details>

#### A.4.1 Leaf: `occupation_unknown_affine`

|  |  |
|---|---|
| Annotation id | `ann_127` |
| Entry nodes | -36 (occupation:0), -47 (occupation:6), -88 (native-country:29) |
| Exit nodes | identity\_occ\_unknown |
| Internal / exit nodes in its own subgraph | 2255, 8774, 4651, identity\_occ\_unknown |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00252 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.152 x_{occupation:6} \\
&\quad - 0.635 \operatorname{relu}{\left(- 0.826 x_{native-country:29} - 1.33 \right)} \\
&\quad + 0.639 \operatorname{relu}{\left(0.289 x_{occupation:0} + 0.0296 x_{occupation:6} + 1.07 \right)} \\
&\quad + 0.452 \operatorname{relu}{\left(- 1.18 \operatorname{relu}{\left(- 0.826 x_{native-country:29} - 1.33 \right)} + 0.993 \operatorname{relu}{\left(0.289 x_{occupation:0} + 0.0296 x_{occupation:6} + 1.07 \right)} - 0.447 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Hidden nodes 2255 and 4651 are always active and 8774 is always off on train (native-country:29 only pushes it further negative), so this subgraph is exactly affine: +0.314/sd for occupation:0 ('?' unknown occupation, +1.3 logit when on) and +0.184/sd for occupation:6 (Handlers-cleaners). Both raise P(\<=50K).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  3,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8494,
 "auc_mean_ablated": 0.9045,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.3556,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

#### A.4.2 Leaf: `private_sector_gate`

|  |  |
|---|---|
| Annotation id | `ann_129` |
| Entry nodes | -6 (workclass:4), -17 (education:13) |
| Exit nodes | identity\_private |
| Internal / exit nodes in its own subgraph | 13603, identity\_private |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 1.16e-05 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.00184 x_{workclass:4} \\
&\quad + 0.0585 \operatorname{relu}{\left(0.00684 x_{education:13} + 1.02 x_{workclass:4} - 0.363 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). workclass:4 (Private) direct term plus gate 13603, which opens only for Private workers; education:13 (Preschool) adds 0.007. Essentially affine (+0.01/sd, \~+0.02 logit for Private). Negligible.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8499,
 "auc_mean_ablated": 0.90634,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.0102,
 "affine_fit_residual_sd": 0.0002,
 "affine_fit_max_abs_resid": 0.0076
}
```

</details>

#### A.4.3 Leaf: `machine_op_gate`

|  |  |
|---|---|
| Annotation id | `ann_131` |
| Entry nodes | -48 (occupation:7) |
| Exit nodes | 16502 |
| Internal / exit nodes in its own subgraph | 16502 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0122 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.01 x_{occupation:7} - 0.961 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). occupation:7 (Machine-op-inspct) through relu 16502, which is on only when the occupation is Machine-op-inspct: an indicator worth +0.50 logit toward \<=50K.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8494,
 "auc_mean_ablated": 0.90579,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.1191,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

#### A.4.4 Leaf: `work_linear`

|  |  |
|---|---|
| Annotation id | `ann_134` |
| Entry nodes | -3 (workclass:1), -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -37 (occupation:1), -38 (occupation:10), -39 (occupation:11), -40 (occupation:12), -45 (occupation:4), -46 (occupation:5), -49 (occupation:8), -65 (hours-per-week) |
| Exit nodes | identity\_work\_linear |
| Internal / exit nodes in its own subgraph | identity\_work\_linear |
| Dimensionality (in → out, incl. descendants) | 12 → 1 |
| Live closed-form error (3 s.f. display) | 0.00459 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.461 x_{hours-per-week} \\
&\quad - 0.0674 x_{occupation:10} \\
&\quad - 0.0213 x_{occupation:11} \\
&\quad + 0.00664 x_{occupation:12} \\
&\quad + 0.0989 x_{occupation:1} \\
&\quad - 0.17 x_{occupation:4} \\
&\quad + 0.23 x_{occupation:5} \\
&\quad + 0.538 x_{occupation:8} \\
&\quad - 0.0803 x_{workclass:1} \\
&\quad + 0.0071 x_{workclass:5} \\
&\quad + 0.212 x_{workclass:6} \\
&\quad + 0.0696 x_{workclass:7}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Linear work terms. hours-per-week -0.46/sd (more hours -\> \>50K); occupation:8 (Other-service) +1.79, occupation:5 (Farming-fishing) +1.35, occupation:4 (Exec-managerial) -0.52, workclass:6 (Self-emp-not-inc) +0.79, workclass:1 (Federal-gov) -0.48, other occupations/workclasses within +-0.4 logit.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  12,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8426,
 "auc_mean_ablated": 0.89413,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.8661,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

### A.5 Composition: `background_effect`

|  |  |
|---|---|
| Annotation id | `ann_141` |
| Entry nodes | -1 (age), -58 (race:1), -60 (race:3), -61 (race:4), -62 (sex), -70 (native-country:12), -73 (native-country:15), -74 (native-country:16), -79 (native-country:20), -81 (native-country:22), -84 (native-country:25), -87 (native-country:28), -89 (native-country:3), -90 (native-country:30), -97 (native-country:37), -100 (native-country:4), -101 (native-country:40), -105 (native-country:7), -106 (native-country:8), -107 (native-country:9), -96 (native-country:36), -11 (fnlwgt) |
| Exit nodes | identity\_background |
| Internal / exit nodes in its own subgraph | identity\_background |
| Children | demographics\_linear, country\_linear, taiwan\_relay, fnlwgt\_hinge |
| Dimensionality (in → out, incl. descendants) | 22 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.0 \operatorname{country\_linear}{\left(x_{native-country:12},x_{native-country:15},x_{native-country:16},x_{native-country:20},x_{native-country:22},x_{native-country:25},x_{native-country:28},x_{native-country:3},x_{native-country:30},x_{native-country:37},x_{native-country:4},x_{native-country:40},x_{native-country:7},x_{native-country:8},x_{native-country:9} \right)} \\
&\quad + 1.0 \operatorname{demographics\_linear}{\left(x_{age},x_{race:1},x_{race:3},x_{race:4},x_{sex} \right)} \\
&\quad + 1.0 \operatorname{fnlwgt\_hinge}{\left(x_{fnlwgt} \right)} \\
&\quad + 1.0 \operatorname{taiwan\_relay}{\left(x_{native-country:36} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K); positive = toward low income. Additive: background = linear age/sex/race + linear native-country one-hots + affine Taiwan relay + small fnlwgt hinge.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  22,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.846,
 "auc_mean_ablated": 0.89831,
 "auc_affine_replaced": 0.90642,
 "contribution_logit_sd": 0.9539,
 "affine_fit_residual_sd": 0.0209,
 "affine_fit_max_abs_resid": 0.2161
}
```

</details>

#### A.5.1 Leaf: `demographics_linear`

|  |  |
|---|---|
| Annotation id | `ann_135` |
| Entry nodes | -1 (age), -58 (race:1), -60 (race:3), -61 (race:4), -62 (sex) |
| Exit nodes | identity\_demo\_linear |
| Internal / exit nodes in its own subgraph | identity\_demo\_linear |
| Dimensionality (in → out, incl. descendants) | 5 → 1 |
| Live closed-form error (3 s.f. display) | 0.00227 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.512 x_{age} \\
&\quad - 0.122 x_{race:1} \\
&\quad - 0.0309 x_{race:3} \\
&\quad - 0.112 x_{race:4} \\
&\quad - 0.601 x_{sex}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Linear demographic terms: age -0.51/sd (older -\> \>50K, no nonlinearity), sex (Male) -1.28 logit, race:1 (Asian-Pac-Islander) -0.70, race:4 (White) -0.32, race:3 (Other) -0.33.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  5,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8467,
 "auc_mean_ablated": 0.89858,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.8415,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

#### A.5.2 Leaf: `country_linear`

|  |  |
|---|---|
| Annotation id | `ann_136` |
| Entry nodes | -70 (native-country:12), -73 (native-country:15), -74 (native-country:16), -79 (native-country:20), -81 (native-country:22), -84 (native-country:25), -87 (native-country:28), -89 (native-country:3), -90 (native-country:30), -97 (native-country:37), -100 (native-country:4), -101 (native-country:40), -105 (native-country:7), -106 (native-country:8), -107 (native-country:9) |
| Exit nodes | identity\_country\_linear |
| Internal / exit nodes in its own subgraph | identity\_country\_linear |
| Dimensionality (in → out, incl. descendants) | 15 → 1 |
| Live closed-form error (3 s.f. display) | 0.00884 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.0116 x_{native-country:12} \\
&\quad - 0.865 x_{native-country:15} \\
&\quad - 0.0165 x_{native-country:16} \\
&\quad + 0.0188 x_{native-country:20} \\
&\quad - 0.0287 x_{native-country:22} \\
&\quad + 0.0452 x_{native-country:25} \\
&\quad + 0.389 x_{native-country:28} \\
&\quad - 0.0196 x_{native-country:30} \\
&\quad + 0.0315 x_{native-country:37} \\
&\quad + 0.0601 x_{native-country:3} \\
&\quad + 0.059 x_{native-country:40} \\
&\quad + 0.199 x_{native-country:4} \\
&\quad + 0.0124 x_{native-country:7} \\
&\quad - 0.00552 x_{native-country:8} \\
&\quad - 0.0241 x_{native-country:9}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Linear native-country one-hots, all rare (\<=235 rows each). Because inputs are z-scored, small weights on rare categories give large swings: native-country:28 (Outlying-US, 20 rows) +17.2 logit, :4 (Columbia) +4.9, :25 (Laos) +2.2. native-country:15 (Holand-Netherlands) is constant on train, so its -0.865 weight is a bias shift. Contribution sd 0.45.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
 "tractable": true,
 "dimensionality": [
  15,
  1
 ]
}
```

```json
// empirical_viz
{
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8496,
 "auc_mean_ablated": 0.90595,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.4511,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

#### A.5.3 Leaf: `taiwan_relay`

|  |  |
|---|---|
| Annotation id | `ann_130` |
| Entry nodes | -96 (native-country:36) |
| Exit nodes | identity\_taiwan |
| Internal / exit nodes in its own subgraph | 14057, identity\_taiwan |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00425 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0914 x_{native-country:36} \\
&\quad + 0.115 \operatorname{relu}{\left(0.959 x_{native-country:36} + 0.795 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). native-country:36 (Taiwan) direct term plus always-active relay 14057: exactly affine, +0.019/sd (+0.54 logit for the 49 Taiwan rows). Negligible overall.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.85,
 "auc_mean_ablated": 0.90635,
 "auc_affine_replaced": 0.90635,
 "contribution_logit_sd": 0.0187,
 "affine_fit_residual_sd": 0.0,
 "affine_fit_max_abs_resid": 0.0
}
```

</details>

#### A.5.4 Leaf: `fnlwgt_hinge`

|  |  |
|---|---|
| Annotation id | `ann_125` |
| Entry nodes | -11 (fnlwgt) |
| Exit nodes | identity\_fnlwgt |
| Internal / exit nodes in its own subgraph | 830, 2821, identity\_fnlwgt |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00275 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0413 x_{fnlwgt} \\
&\quad - 0.11 \operatorname{relu}{\left(0.726 \operatorname{relu}{\left(1.05 x_{fnlwgt} + 0.385 \right)} + 0.464 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Output logit is for P(income \<= 50K) (train target y=1 is \<=50K, base rate 0.761); positive = toward low income. Inputs are z-scored; category names inferred from alphabetical codes (consistent with train frequencies). Census sampling weight through a two-relu chain (830 -\> 2821). 2821 is always active, so this is a single hinge: slope -0.04/sd below fnlwgt z=-0.37 (37% of rows), -0.12/sd above. Small effect (contribution sd 0.11 logit); likely a spurious fit to the sampling weight.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form extracted by get\_formula.
- `empirical_viz`: Train-split (n=39,073) contribution statistics and ablations.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)",
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
 "split": "train",
 "method": "local forward pass on in-memory copy of the phenotype (read-only); contribution = this annotation's additive share of the output logit; mean-ablation replaces it with its train mean; affine-replaced substitutes the least-squares affine fit on its entry inputs",
 "n_train": 39073,
 "acc_full": 0.85,
 "auc_full": 0.90635,
 "acc_mean_ablated": 0.8498,
 "auc_mean_ablated": 0.90625,
 "auc_affine_replaced": 0.90642,
 "contribution_logit_sd": 0.1081,
 "affine_fit_residual_sd": 0.0209,
 "affine_fit_max_abs_resid": 0.2157
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 17 |
| annotate | 19 |
| rename\_node | 107 |

0 splits, 17 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node identity\_capital intercepts 4 connection(s) into 0 (from -63 (capital-gain), 2450, 7922, -64 (capital-loss)); sits in capital\_gain\_hinge (exit)
- `#108` identity node identity\_fnlwgt intercepts 2 connection(s) into 0 (from -11 (fnlwgt), 2821); sits in fnlwgt\_hinge (exit)
- `#109` identity node identity\_edunum intercepts 3 connection(s) into 0 (from -28 (education-num), 6000, -69 (native-country:11)); sits in education\_years\_hinge (exit)
- `#110` identity node identity\_occ\_unknown intercepts 4 connection(s) into 0 (from -47 (occupation:6), 2255, 8774, 4651); sits in occupation\_unknown\_affine (exit)
- `#111` identity node identity\_somecollege intercepts 3 connection(s) into 0 (from -19 (education:15), 5909, 4066); sits in some\_college\_relay (exit)
- `#112` identity node identity\_private intercepts 2 connection(s) into 0 (from -6 (workclass:4), 13603); sits in private\_sector\_gate (exit)
- `#113` identity node identity\_taiwan intercepts 2 connection(s) into 0 (from -96 (native-country:36), 14057); sits in taiwan\_relay (exit)
- `#114` identity node identity\_husband intercepts 2 connection(s) into 0 (from -5 (workclass:3), 406); sits in husband\_gate (exit)
- `#115` identity node identity\_family\_linear intercepts 7 connection(s) into 0 (from -30 (marital-status:1), -31 (marital-status:2), -33 (marital-status:4), -34 (marital-status:5), -53 (relationship:2), -54 (relationship:3), -56 (relationship:5)); sits in family\_linear (exit)
- `#116` identity node identity\_work\_linear intercepts 12 connection(s) into 0 (from -3 (workclass:1), -7 (workclass:5), -8 (workclass:6), -9 (workclass:7), -37 (occupation:1), -38 (occupation:10), -39 (occupation:11), -40 (occupation:12), … (4 more)); sits in work\_linear (exit)
- `#117` identity node identity\_demo\_linear intercepts 5 connection(s) into 0 (from -1 (age), -58 (race:1), -60 (race:3), -61 (race:4), -62 (sex)); sits in demographics\_linear (exit)
- `#118` identity node identity\_country\_linear intercepts 15 connection(s) into 0 (from -70 (native-country:12), -73 (native-country:15), -74 (native-country:16), -79 (native-country:20), -81 (native-country:22), -84 (native-country:25), -87 (native-country:28), -89 (native-country:3), … (7 more)); sits in country\_linear (exit)
- `#119` identity node identity\_edcat\_linear intercepts 5 connection(s) into 0 (from -12 (education:0), -14 (education:10), -16 (education:12), -21 (education:3), -25 (education:7)); sits in education\_category\_linear (exit)
- `#120` identity node identity\_household intercepts 2 connection(s) into 0 (from identity\_husband, identity\_family\_linear); sits in household\_effect (exit, composition)
- `#121` identity node identity\_education intercepts 3 connection(s) into 0 (from identity\_edunum, identity\_somecollege, identity\_edcat\_linear); sits in education\_effect (exit, composition)
- `#122` identity node identity\_work intercepts 4 connection(s) into 0 (from identity\_occ\_unknown, identity\_private, 16502, identity\_work\_linear); sits in work\_effect (exit, composition)
- `#123` identity node identity\_background intercepts 4 connection(s) into 0 (from identity\_demo\_linear, identity\_country\_linear, identity\_taiwan, identity\_fnlwgt); sits in background\_effect (exit, composition)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 4.44e-16 | 4.44e-16 | 4.44e-16 | agree |
| Closed-form max abs error (all leaves) | 0.04663 | 0.04663 | 0.04663 | agree |
| Numeric-function max abs error | n/a | 3.55e-15 | 3.55e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit | income\_logit | agree |
| Leaf / composition counts | 14 / 5 | n/a | 14 / 5 | agree |
| Operation counts | 0 splits, 17 identity | {"add\_identity\_node": 17, "annotate": 19, "rename\_node": 107} | {"add\_identity\_node": 17, "annotate": 19, "rename\_node": 107} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capital_gain_hinge` | 0.04663 | 1.78e-15 |
| `husband_gate` | 0.04078 | 0 |
| `machine_op_gate` | 0.01223 | 0 |
| `country_linear` | 0.008844 | 3.55e-15 |
| `education_years_hinge` | 0.007683 | 8.88e-16 |
| `family_linear` | 0.004737 | 4.44e-16 |
| `work_linear` | 0.004588 | 4.44e-16 |
| `taiwan_relay` | 0.004254 | 0 |
| `fnlwgt_hinge` | 0.002754 | 0 |
| `occupation_unknown_affine` | 0.002517 | 4.44e-16 |
| `demographics_linear` | 0.002269 | 4.44e-16 |
| `education_category_linear` | 6.29e-04 | 1.11e-16 |
| `some_college_relay` | 7.87e-05 | 0 |
| `private_sector_gate` | 1.16e-05 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Every hidden ReLU takes inputs from one feature group (no splits). Most hidden nodes do nothing nonlinear on train (two always off, six always on, two indicators); only the capital-gain hinge matters: all-affine fit AUC 0.9008 vs 0.9064 full. z-scoring side effects: rare one-hots give large swings (native-country:28, n=20, +17 logit); native-country:15 constant on train acts as bias.

**Tool notes (record):** Annotate validator accepted compositions with empty subgraph\_connections (coverage then 0.907; undone and redone with child-exit-\>identity edges). compute\_shap (500) \~28 min at \~440% CPU.

## What to look at

- Largest closed-form error: `capital_gain_hinge` (0.0466), then `husband_gate` (0.0408). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `occupation_unknown_affine`: *“Hidden nodes 2255 and 4651 are always active and 8774 is always off on train (native-country:29 only pushes it further negative), so this subgraph is exactly affine: +0.314/sd for occupation:0 ('?' unknown occupation, +1.3 logit w…”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `taiwan_relay`: *“native-country:36 (Taiwan) direct term plus always-active relay 14057: exactly affine, +0.019/sd (+0.54 logit for the 49 Taiwan rows).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- No splits: every node sits in one annotation as evolved.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s2.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s2', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
