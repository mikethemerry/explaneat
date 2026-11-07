# Verification packet 18 · heart-s9

Pre-registered verification order: 18 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/b19b9536-00dd-4e95-9f5a-def6dd8d6889?experiment=f93130a2-6035-4c5f-aef2-53c78fdba6fa&name=10seed-heart-s9), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 9 |
| Genome id | `b19b9536-00dd-4e95-9f5a-def6dd8d6889` |
| Experiment | 10seed-heart-s9 (f93130a2-6035-4c5f-aef2-53c78fdba6fa); best at generation 39 of 40, NEAT key 8956 |
| Test AUC (runs csv) | 0.872 \[95% CI 0.774, 0.949\] |
| Train AUC | 0.944 |
| Size | 20 of 26 inputs connected; 5 hidden nodes; 29 enabled connections; depth 2; 0 skip connections + 15 direct input→output edges |
| Annotations | 5 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 5 of 5 |
| Claude's time | 13.6 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:37:34+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Essentially logistic regression on 15 inputs (logit SD 4.96); ca effective weight 2.60 via direct edge plus affine 148. Four small additive one-sided corrections (combined logit SD 1.41): oldpeak hinge above +0.39 SD; protective no-exercise-angina gate (stronger for women); -3.19 correction for fixed thal defect with normal fbs (n=10); near-inert cp:2 offset. Readouts fixed at means: train AUC 0.9437 -\> 0.9295.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_35` |
| Entry nodes | -1 (age), -2\_a (sex:0\_a), -3\_a (sex:1\_a), -4 (cp:1), -5\_a (cp:2\_a), -6 (cp:3), -7 (cp:4), -8 (trestbps), -9 (chol), -11 (fbs:1), -15 (thalach), -19 (slope:1), -20 (slope:2), -22\_a (ca\_a), -26 (thal:7), -12 (restecg:0), -13 (restecg:1), -14 (restecg:2), -17 (exang:1), -21 (slope:3), -23 (thal:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | vessel\_count\_linear\_boost, sex\_exang\_protective\_gate, fbs\_normal\_fixed\_defect\_gate, oldpeak\_hinge, cp2\_constant\_offset |
| Dimensionality (in → out, incl. descendants) | 30 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{2.6 e^{- 0.205 x_{age} - 2.06 x_{ca\_a} - 1.07 x_{chol} + 0.916 x_{cp:1} + 0.0574 x_{cp:2\_a} + 0.972 x_{cp:3} - 0.619 x_{cp:4} + 0.667 x_{fbs:1} + 0.0699 x_{sex:0\_a} - 1.66 x_{sex:1\_a} + 0.499 x_{slope:1} - 1.3 x_{slope:2} - 0.461 x_{thal:7} + 0.419 x_{thalach} - 0.616 x_{trestbps} + 0.154 \operatorname{cp\_2\_constant\_offset}{\left(x_{cp:2\_b} \right)} + 0.942 \operatorname{fbs\_normal\_fixed\_defect\_gate}{\left(x_{fbs:0},x_{thal:6} \right)} - 1.11 \operatorname{oldpeak\_hinge}{\left(x_{oldpeak} \right)} + 0.452 \operatorname{sex\_exang\_protective\_gate}{\left(x_{sex:0\_b},x_{sex:1\_b},x_{exang:0} \right)} - 1.89 \operatorname{vessel\_count\_linear\_boost}{\left(x_{ca\_b},x_{thal:4} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.205 x_{age} \\
&\quad + 2.06 x_{ca\_a} \\
&\quad + 1.07 x_{chol} \\
&\quad - 0.916 x_{cp:1} \\
&\quad - 0.0574 x_{cp:2\_a} \\
&\quad - 0.972 x_{cp:3} \\
&\quad + 0.619 x_{cp:4} \\
&\quad - 0.667 x_{fbs:1} \\
&\quad - 0.0699 x_{sex:0\_a} \\
&\quad + 1.66 x_{sex:1\_a} \\
&\quad - 0.499 x_{slope:1} \\
&\quad + 1.3 x_{slope:2} \\
&\quad + 0.461 x_{thal:7} \\
&\quad - 0.419 x_{thalach} \\
&\quad + 0.616 x_{trestbps} \\
&\quad - 0.154 \operatorname{cp\_2\_constant\_offset}{\left(-5_{b} \right)} \\
&\quad - 0.942 \operatorname{fbs\_normal\_fixed\_defect\_gate}{\left(-10,-25 \right)} \\
&\quad + 1.11 \operatorname{oldpeak\_hinge}{\left(-18 \right)} \\
&\quad - 0.452 \operatorname{sex\_exang\_protective\_gate}{\left(-2_{b},-3_{b},-16 \right)} \\
&\quad + 1.89 \operatorname{vessel\_count\_linear\_boost}{\left(-22_{b},-24 \right)} \\
&\quad -0.954 \bigr)
\end{aligned}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: logit z = -0.954 + LINEAR CORE + 1.893\*vessel\_count\_linear\_boost - 0.452\*sex\_exang\_protective\_gate - 0.942\*fbs\_normal\_fixed\_defect\_gate + 1.108\*oldpeak\_hinge - 0.154\*cp2\_constant\_offset; prediction = sigmoid(z) (get\_formula shows the output node as ReLU, but the evaluator overrides it with sigmoid for binary classification; compute\_performance on train, AUC 0.9437 and acc 0.888, matches sigmoid(z)). LINEAR CORE (standardised inputs): +2.065 ca, +1.665 sex:1, -0.070 sex:0, +1.298 slope:2, -0.499 slope:1, +1.075 chol, -0.972 cp:3, -0.916 cp:1, +0.618 cp:4, -0.057 cp:2, -0.667 fbs:1, +0.616 trestbps, +0.461 thal:7, -0.419 thalach, +0.205 age. Inputs restecg:0/1/2, exang:1, slope:3 and thal:3 are disconnected in the phenotype and enter the root with zero effect. CHILDREN (all additive, read out directly by the output; no serial composition): vessel\_count\_linear\_boost is affine (never clips), raising the effective ca weight to 2.60; sex\_exang\_protective\_gate lowers the logit by 0.43 (men) or 1.67 (women) only when there is no exercise angina; oldpeak\_hinge adds 1.11 per SD of ST depression above +0.39 SD (the only oldpeak pathway); fbs\_normal\_fixed\_defect\_gate subtracts 3.19 for the 10 training patients with a fixed thallium defect and normal fasting sugar (likely an overfitted correction); cp2\_constant\_offset is a near-inert -0.044 offset for cp:2. EVIDENCE (train, n=242): the linear core dominates (logit sd 4.96 vs 1.41 for all readouts); with every readout fixed at its mean, train AUC is 0.9295 and accuracy 0.860, vs 0.9437 and 0.888 for the full model. SHAP (probability scale) ranks ca 0.158, sex:1 0.104, slope:2 0.092, cp:3 0.053, chol 0.044, cp:4 0.041. So the network is a logistic regression (more vessels, male sex, flat ST slope, high cholesterol, asymptomatic chest pain mean more risk) with two clinically plausible one-sided terms (marked ST depression adds risk; no exertional angina lowers it, more for women) and one small-sample correction.

**Evidence (4 records, DB):**

- `analytical_formula`: Root closed form: affine core in 15 inputs plus five child readouts; sigmoid at output.
- `shap_importance`: Train-split SHAP (probability scale): ca 0.158, sex:1 0.104, slope:2 0.092, cp:3 0.053, chol 0.044, cp:4 0.041 dominate; restecg, exang:1, slope:3, thal:3 are exactly 0 (disconnected).
- `ablation_study`: Linear core carries most of the logit variance (sd 4.96 vs 1.41 for all readouts). Readouts add +0.014 train AUC and +2.9 points accuracy; the largest single effects are the fixed-defect gate and the oldpeak hinge.
- `performance_metrics`: Train performance of the explained network (sigmoid output).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "get_formula(force=true) latex_collapsed. Output shown as ReLU, but the evaluator overrides the output activation with sigmoid for binary classification, so prediction = sigmoid(z). The stored ReLU would clip 59.5% of train logits (z<0) and lower AUC to 0.893; compute_performance (sigmoid) gives AUC 0.9437, matching sigmoid(z).",
 "effective_weights": {
  "ca": "2.065 direct + 1.893*0.282 via 148 = 2.598 per SD",
  "sex": "-0.070 sex:0 + 1.665 sex:1 = 1.735*sex:1 (sex:1 = -sex:0 on standardised one-hot), i.e. +3.66 logit male vs female before 791"
 }
}
```

```json
// shap_importance
{
 "n": 242,
 "space": "probability",
 "split": "train",
 "base_value": 0.43139138099781016,
 "top10_mean_abs_shap": {
  "ca": 0.15774656782093274,
  "chol": 0.044299814091270026,
  "cp:1": 0.026864618940939798,
  "cp:3": 0.05342417482654845,
  "cp:4": 0.04066155531899442,
  "sex:1": 0.10422713056986248,
  "thal:7": 0.025402690147866506,
  "slope:1": 0.02760611866840299,
  "slope:2": 0.0915971602371253,
  "trestbps": 0.0239080415796241
 }
}
```

```json
// ablation_study
{
 "split": "train",
 "ladder": [
  {
   "acc": 0.8884,
   "auc": 0.9437,
   "config": "full"
  },
  {
   "acc": 0.8595,
   "auc": 0.9295,
   "config": "linear core only (all 5 readouts at mean)"
  },
  {
   "acc": 0.8719,
   "auc": 0.9388,
   "config": "ablate fbs_normal_fixed_defect_gate"
  },
  {
   "acc": 0.876,
   "auc": 0.94,
   "config": "ablate oldpeak_hinge"
  },
  {
   "acc": 0.8884,
   "auc": 0.9408,
   "config": "ablate vessel_count_linear_boost"
  },
  {
   "acc": 0.8884,
   "auc": 0.9418,
   "config": "ablate sex_exang_protective_gate"
  },
  {
   "acc": 0.8884,
   "auc": 0.9434,
   "config": "ablate cp2_constant_offset"
  }
 ],
 "method": "local logit decomposition z = linear core + sum(readout weight * node activation), verified equal to the network pre-activation to 1.8e-15; each readout replaced by its train mean, no refit",
 "logit_sd": {
  "148": 0.624,
  "791": 0.661,
  "2034": 0.634,
  "2985": 0.668,
  "3229": 0.017,
  "total": 5.861,
  "linear_core": 4.955,
  "all_readouts": 1.41
 }
}
```

```json
// performance_metrics
{
 "n": 242,
 "mse": 0.0898,
 "split": "train",
 "source": "compute_performance(split=train, max_samples=100000)",
 "auc_roc": 0.9437,
 "accuracy": 0.8884,
 "coverage": {
  "structural": 1.0,
  "compositional": 1.0
 },
 "operations": {
  "splits": 4,
  "identity_nodes": 0,
  "function_changing": 0
 }
}
```

</details>

### A.1 Leaf: `vessel_count_linear_boost`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -22\_b (ca\_b), -24 (thal:4) |
| Exit nodes | 148 |
| Internal / exit nodes in its own subgraph | 148 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00142 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.282 x_{ca\_b} \\
&\quad + 0.19 x_{thal:4} \\
&\quad + 0.552 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}\bigl(0.282 x_{ca} + 0.19 x_{thal:4} + 0.552\bigr)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Node 148 = ReLU(0.552 + 0.282\*ca + 0.190\*thal:4), read out with +1.893. Not a gate: it is active for 100% of training patients (min 0.330), so it is an affine extra channel. Its effect is to add +0.534 logit per SD of ca (number of fluoroscopy-coloured major vessels) on top of the direct ca edge (+2.065), giving ca an effective weight of 2.60, the largest in the network, plus +0.36 per SD of the rare thal:4 code (n=2 on train). Readout contribution sd 0.62 logits; replacing it by its mean drops train AUC 0.9437 -\> 0.9408.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Never clips on train (activation rate 100%, min 0.330), so the ReLU is inert and 148 is exactly affine in ca and thal:4. Logit contribution +0.534 per SD of ca, on top of the direct ca edge (+2.065), i.e. effective ca weight 2.598.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "148 = relu(0.282 ca + 0.190 thal:4 + 0.552); readout 148 -> 0 weight +1.893"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "by_group": "ca=-0.728 (n=139): 0.330; ca=0.394 (n=53): 0.646; ca=1.515 (n=33): 0.963; ca=2.636 (n=12): 1.280; thal:4 present (n=2): 2.428",
 "activation_profile": {
  "max": 2.427796563415656,
  "min": 0.3295769963302928,
  "std": 0.32970765062249896,
  "mean": 0.552368810100331,
  "count": 242,
  "median": 0.3295769963302928,
  "zero_fraction": 0.0,
  "activation_rate": 1.0
 },
 "sensitivity_mean_abs_grad": {
  "ca": 0.282,
  "thal:4": 0.19
 }
}
```

</details>

### A.2 Leaf: `sex_exang_protective_gate`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | -2\_b (sex:0\_b), -3\_b (sex:1\_b), -16 (exang:0) |
| Exit nodes | 791 |
| Internal / exit nodes in its own subgraph | 791 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00515 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.2 x_{exang:0} \\
&\quad + 1.08 x_{sex:0\_b} \\
&\quad - 0.218 x_{sex:1\_b} \\
&\quad + 0.325 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}\bigl(2.2 x_{exang:0} + 1.08 x_{sex:0} - 0.218 x_{sex:1} + 0.325\bigr)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Node 791 = ReLU(0.325 + 1.084\*sex:0 - 0.218\*sex:1 + 2.200\*exang:0), read out with -0.452. Since sex and exang are one-hot binaries it takes only three values on train: 0 for all patients with exercise-induced angina (n=82, disease rate 0.73), 0.959 for men without it (n=96) and 3.702 for women without it (n=64). It is therefore a protective interaction: absence of exertional angina lowers the logit by 0.43 for men but by 1.67 for women, and has no effect if angina is present. This is the network's only exang pathway (no direct exang edge). Readout sd 0.66 logits; mean-ablation drops train AUC 0.9437 -\> 0.9418, accuracy unchanged.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Three-valued gate on train: exactly 0 for every patient with exercise-induced angina; 0.959 for men and 3.702 for women without it. Logit contribution 0, -0.43 or -1.67.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "791 = relu(2.200 exang:0 + 1.084 sex:0 - 0.218 sex:1 + 0.325); readout 791 -> 0 weight -0.452"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "truth_table": {
  "male,exang (n=63)": 0.0,
  "female,exang (n=19)": 0.0,
  "male,no exang (n=96)": 0.959,
  "female,no exang (n=64)": 3.702
 },
 "disease_rate": "active 0.29 (n=160) vs inactive 0.73 (n=82)",
 "activation_profile": {
  "max": 3.7016351299639894,
  "min": 0.0,
  "std": 1.4631030954621602,
  "mean": 1.3593231089802071,
  "count": 242,
  "median": 0.9588702505782787,
  "zero_fraction": 0.33884297520661155,
  "activation_rate": 0.6611570247933884
 }
}
```

</details>

### A.3 Leaf: `fbs_normal_fixed_defect_gate`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | -10 (fbs:0), -25 (thal:6) |
| Exit nodes | 2034 |
| Internal / exit nodes in its own subgraph | 2034 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0129 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.16 x_{fbs:0} + 1.02 x_{thal:6} - 1.03 \right)}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}\left(1.16 x_{fbs:0} + 1.02 x_{thal:6} - 1.03\right)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Node 2034 = ReLU(-1.027 + 1.156\*fbs:0 + 1.016\*thal:6), read out with -0.942. An AND gate: it is 3.386 only for patients with a fixed thallium defect (thal:6) AND normal fasting blood sugar (n=10 on train), 0.018 for fixed-defect patients with high sugar (n=5), and 0 for everyone else (93.8%). It subtracts 3.19 logits for that small group. Evidence suggests it is a training-set correction rather than a clinical rule: among the 10, the linear core over-predicted 4 of 5 negatives as positive; the gate moves all 5 negatives below 0 while the 5 positives stay above 0. This is the only thal:6 pathway and the only fbs:0 pathway. Mean-ablation drops train AUC 0.9437 -\> 0.9388 and accuracy 0.888 -\> 0.872 (largest single-gate effect), but it rests on 10 patients, so treat it as likely overfitted.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: AND gate: 3.386 only for fixed-defect patients with normal fasting sugar (n=10), \~0 otherwise. Subtracts 3.19 logits for those 10, which flips 4 false positives to correct negatives while all 5 positives stay positive.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "2034 = relu(1.156 fbs:0 + 1.016 thal:6 - 1.027); readout 2034 -> 0 weight -0.942"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "truth_table": {
  "no fixed defect (n=227)": 0.0,
  "fixed defect & high fbs (n=5)": 0.018,
  "fixed defect & normal fbs (n=10)": 3.386
 },
 "activation_profile": {
  "max": 3.3859644115100975,
  "min": 0.0,
  "std": 0.6738529286610648,
  "mean": 0.14028153093765972,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.9380165289256198,
  "activation_rate": 0.06198347107438017
 },
 "fixed_defect_normal_fbs_labels": "y=[1,0,0,1,0,0,0,1,1,1]; logit without gate [5.82,1.61,1.57,7.81,-4.64,0.15,0.60,4.99,8.95,11.92]; with gate [2.63,-1.58,-1.62,4.62,-7.82,-3.04,-2.59,1.81,5.76,8.73]"
}
```

</details>

### A.4 Leaf: `oldpeak_hinge`

|  |  |
|---|---|
| Annotation id | `ann_33` |
| Entry nodes | -18 (oldpeak) |
| Exit nodes | 2985 |
| Internal / exit nodes in its own subgraph | 2985 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00771 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{oldpeak} - 0.386 \right)}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}\left(1.0 x_{oldpeak} - 0.386\right)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Node 2985 = ReLU(1.002\*oldpeak - 0.386), read out with +1.108. A hinge on exercise-induced ST depression: zero for patients with oldpeak below +0.385 SD (71% of train), then +1.11 logit per SD above it (max +4.41). Disease rate 0.71 where active vs 0.33 where inactive. It is the network's only oldpeak pathway, so the network treats small ST depression as uninformative and only marked depression as risk. Mean-ablation drops train AUC 0.9437 -\> 0.9400 and accuracy 0.888 -\> 0.876.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Hinge: zero for 71% of train patients (oldpeak below +0.385 SD), linear above with slope 1.11 logit per SD of ST depression; max contribution +4.41.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "2985 = relu(1.002 oldpeak - 0.386); readout 2985 -> 0 weight +1.108"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "disease_rate": "active 0.71 (n=70) vs inactive 0.33 (n=172)",
 "threshold_sd": 0.385,
 "activation_profile": {
  "max": 3.9784243772919563,
  "min": 0.0,
  "std": 0.6032071870239945,
  "mean": 0.2651822416687229,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.7107438016528925,
  "activation_rate": 0.2892561983471074
 }
}
```

</details>

### A.5 Leaf: `cp2_constant_offset`

|  |  |
|---|---|
| Annotation id | `ann_34` |
| Entry nodes | -5\_b (cp:2\_b) |
| Exit nodes | 3229 |
| Internal / exit nodes in its own subgraph | 3229 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00179 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.947 x_{cp:2\_b} - 1.78 \right)}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}\left(0.947 x_{cp:2} - 1.78\right)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Node 3229 = ReLU(0.947\*cp:2 - 1.779), read out with -0.154. Because cp:2 is a standardised one-hot, the node is exactly an indicator of atypical angina (cp:2, n=42) with value 0.288, and 0 otherwise. Net effect: -0.044 logit for cp:2 patients, which simply adds to the direct cp:2 edge (-0.057 per SD). Practically inert (mean-ablation changes train AUC 0.9437 -\> 0.9434).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Binary indicator of cp:2 with value 0.288; logit contribution -0.044 for cp:2 patients. Mean-ablation changes train AUC by 0.0003.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "3229 = relu(0.947 cp:2 - 1.779); readout 3229 -> 0 weight -0.154"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "truth_table": {
  "cp:2 (n=42)": 0.288,
  "other (n=200)": 0.0
 },
 "activation_profile": {
  "max": 0.28830968997848694,
  "min": 0.0,
  "std": 0.10919016346410475,
  "mean": 0.050037218921886166,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.8264462809917356,
  "activation_rate": 0.17355371900826447
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 6 |
| rename\_node | 26 |
| split\_node | 4 |

4 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -2 into -2\_a, -2\_b. Final copies: -2\_a → heart\_disease\_logit (entry) \| -2\_b → sex\_exang\_protective\_gate (entry)
- `#27` split -3 into -3\_a, -3\_b. Final copies: -3\_a → heart\_disease\_logit (entry) \| -3\_b → sex\_exang\_protective\_gate (entry)
- `#28` split -5 into -5\_a, -5\_b. Final copies: -5\_a → heart\_disease\_logit (entry) \| -5\_b → cp2\_constant\_offset (entry)
- `#29` split -22 into -22\_a, -22\_b. Final copies: -22\_a → heart\_disease\_logit (entry) \| -22\_b → vessel\_count\_linear\_boost (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.78e-15 | 1.78e-15 | 3.33e-16 | agree |
| Closed-form max abs error (all leaves) | 0.01293 | 0.01293 | 0.01293 | agree |
| Numeric-function max abs error | n/a | 0 | 0 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 5 / 1 | n/a | 5 / 1 | agree |
| Operation counts | 4 splits, 0 identity | {"annotate": 6, "rename\_node": 26, "split\_node": 4} | {"annotate": 6, "rename\_node": 26, "split\_node": 4} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `fbs_normal_fixed_defect_gate` | 0.01293 | 0 |
| `oldpeak_hinge` | 0.007714 | 0 |
| `sex_exang_protective_gate` | 0.005146 | 0 |
| `cp2_constant_offset` | 0.001786 | 0 |
| `vessel_count_linear_boost` | 0.001422 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Single hidden layer: 5 hidden nodes each read out directly, flat root. 148 never clips (affine); 3229 is a constant -0.044 offset for cp:2. 2034 (fixed thal defect AND normal fbs) acts on 10 training patients, looks like an overfitted correction (flips 4 FP to negatives). 6 disconnected inputs (restecg:0/1/2, exang:1, slope:3, thal:3) added to root as zero-effect entries.

**Tool notes (record):** activation\_profile needs node\_id. get\_coverage lists disconnected inputs as uncovered until annotated. get\_formula output shown as ReLU; compute\_performance matches sigmoid(z). One transient empty begin\_explanation output. compute\_shap \~3.7 min.

## What to look at

- Largest closed-form error: `fbs_normal_fixed_defect_gate` (0.0129). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `fbs_normal_fixed_defect_gate`: *“An AND gate: it is 3.386 only for patients with a fixed thallium defect (thal:6) AND normal fasting blood sugar (n=10 on train), 0.018 for fixed-defect patients with high sugar (n=5), and 0 for everyone else (93.8%).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `oldpeak_hinge`: *“It is the network's only oldpeak pathway, so the network treats small ST depression as uninformative and only marked depression as risk.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 4 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s9.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s9', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
