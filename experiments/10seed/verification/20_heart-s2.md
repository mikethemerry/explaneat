# Verification packet 20 · heart-s2

Pre-registered verification order: 20 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/0057df7a-62a2-43c3-b107-9afe3735a7c3?experiment=122b75e3-2110-4c57-8c07-c048361cdc44&name=10seed-heart-s2), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 2 |
| Genome id | `0057df7a-62a2-43c3-b107-9afe3735a7c3` |
| Experiment | 10seed-heart-s2 (122b75e3-2110-4c57-8c07-c048361cdc44); best at generation 38 of 40, NEAT key 9083 |
| Test AUC (runs csv) | 0.897 \[95% CI 0.807, 0.966\] |
| Train AUC | 0.943 |
| Size | 18 of 26 inputs connected; 6 hidden nodes; 31 enabled connections; depth 3; 6 skip connections + 16 direct input→output edges |
| Annotations | 3 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 6 of 6 |
| Claude's time | 12.2 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:36:08+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Mostly logistic regression: 16 direct edges (logit SD 3.28, train AUC 0.930 alone) plus three additive nonlinear corrections (logit SD 0.91) to 0.943: a protective heart-rate-reserve gate ReLU(-1.52 - 1.49 age + 1.02 thalach - 1.54 thal:7) read out -0.58 (largest effect); a flat-ST-slope gate scaled by oldpeak and down by chol; a convex two-kink oldpeak dose-response. Age and oldpeak reach the output only via these gates.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_34` |
| Entry nodes | -2 (sex:0), -3 (sex:1), -4 (cp:1), -5 (cp:2), -7 (cp:4), -8 (trestbps), -9\_a (chol\_a), -11 (fbs:1), -13 (restecg:1), -15\_a (thalach\_a), -17 (exang:1), -19 (slope:1), -20\_a (slope:2\_a), -22 (ca), -25 (thal:6), -26\_a (thal:7\_a), -6 (cp:3), -10 (fbs:0), -12 (restecg:0), -14 (restecg:2), -16 (exang:0), -21 (slope:3), -23 (thal:3), -24 (thal:4), -1 (age), -15\_b (thalach\_b), -26\_b (thal:7\_b), -26\_c (thal:7\_c), -9\_b (chol\_b), -18\_a (oldpeak\_a), -20\_b (slope:2\_b), -18\_b (oldpeak\_b) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | age\_hr\_reserve\_gate, st\_slope\_chol\_gate, oldpeak\_hinges |
| Dimensionality (in → out, incl. descendants) | 32 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{1.23 e^{- 1.68 x_{ca} - 0.613 x_{chol\_a} + 0.179 x_{cp:1} - 0.416 x_{cp:2} - 0.993 x_{cp:4} - 0.282 x_{exang:1} + 0.306 x_{fbs:1} - 0.0703 x_{restecg:1} + 0.745 x_{sex:0} - 0.45 x_{sex:1} + 0.473 x_{slope:1} - 0.494 x_{slope:2\_a} + 0.279 x_{thal:6} - 0.288 x_{thal:7\_a} + 0.0959 x_{thalach\_a} - 0.469 x_{trestbps} + 0.582 \operatorname{age\_hr\_reserve\_gate}{\left(x_{age},x_{thalach\_b},x_{thal:7\_b},x_{thal:7\_c} \right)} - 0.377 \operatorname{oldpeak\_hinges\_0}{\left(x_{oldpeak\_b} \right)} - 0.411 \operatorname{oldpeak\_hinges\_1}{\left(x_{oldpeak\_b} \right)} - 0.225 \operatorname{st\_slope\_chol\_gate}{\left(x_{chol\_b},x_{oldpeak\_a},x_{slope:2\_b} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}z = -0.21 + 1.68 x_{ca} + 0.993 x_{cp:4} - 0.745 x_{sex:0} + 0.613 x_{chol\_a} + 0.494 x_{slope:2\_a} - 0.473 x_{slope:1} + 0.469 x_{trestbps} + 0.45 x_{sex:1} + 0.416 x_{cp:2} - 0.306 x_{fbs:1} + 0.288 x_{thal:7\_a} + 0.282 x_{exang:1} - 0.279 x_{thal:6} - 0.179 x_{cp:1} - 0.0959 x_{thalach\_a} + 0.0703 x_{restecg:1} - 0.582\,\mathrm{age\_hr\_reserve\_gate} + 0.225\,\mathrm{st\_slope\_chol\_gate} + 0.377\,\mathrm{oldpeak\_hinges}_0 + 0.411\,\mathrm{oldpeak\_hinges}_1\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: logit z = -0.210 + LINEAR CORE - 0.582\*age\_hr\_reserve\_gate + 0.225\*st\_slope\_chol\_gate + 0.377\*oldpeak\_hinges\[1946\] + 0.411\*oldpeak\_hinges\[3284\]; prediction = sigmoid(z) (the genome output node is shown as ReLU, but for binary classification the evaluator applies a sigmoid, so z is a logistic score). LINEAR CORE (16 direct input-\>output edges, z-scored inputs): ca +1.68, cp:4 +0.99, sex:0 -0.74, chol +0.61, slope:2 +0.49, slope:1 -0.47, trestbps +0.47, sex:1 +0.45, cp:2 +0.42, fbs:1 -0.31, thal:7 +0.29, exang:1 +0.28, thal:6 -0.28, cp:1 -0.18, thalach -0.10, restecg:1 +0.07. The signs mostly match cardiology: more fluoroscopy-coloured vessels (ca), asymptomatic chest pain (cp:4), male sex, higher cholesterol and resting BP, flat ST slope and exertional angina raise risk, and upsloping ST (slope:1) lowers it. Exceptions are cp:2 +0.42 and fbs:1 -0.31, which run against the usual clinical direction and are left unexplained. Eight inputs are unused (cp:3, fbs:0, restecg:0, restecg:2, exang:0, slope:3, thal:3, thal:4). Age and oldpeak have no direct edge: they act only through the gates. COMPOSITION: purely additive in the logit. The three children add three nonlinear corrections to the linear core: (1) age\_hr\_reserve\_gate, a protective ReLU that is on when max heart rate is high for the patient's age and there is no reversible thallium defect; (2) st\_slope\_chol\_gate, a flat-ST-slope detector scaled up by oldpeak and down by cholesterol; (3) oldpeak\_hinges, a convex two-kink dose-response in ST depression. The children share no hidden nodes. They are coupled only through shared input features (thal:7, thalach, chol and slope:2 also have direct edges, and oldpeak feeds both (2) and (3)). EVIDENCE (train, n=242): full model AUC 0.943, acc 0.884 (sigmoid(z) reproduces the evaluator exactly). Linear core alone gives AUC 0.930, so the nonlinear children add +0.013. Logit SD is 3.28 for the linear core vs 0.91 for all nonlinear readouts. Removing one readout at a time: 1718 -0.0057, 1476 -0.0017, 1946 -0.0007, 3284 -0.0004 AUC; all oldpeak-driven terms together -0.0064. Top SHAP features: ca, cp:4, sex:0, slope:2, slope:1 (all with direct edges). Interpretation: close to logistic regression, plus a protective heart-rate-reserve-for-age gate (blocked by a reversible defect) and two oldpeak/flat-slope ischaemia corrections.

**Evidence (4 records, DB):**

- `analytical_formula`: Root closed form: additive logit of linear core plus three child readouts.
- `performance_metrics`: Full coverage with 5 input splits; the explained network computes the same function.
- `ablation_study`: Linear core carries most of the signal; the nonlinear children add +0.013 train AUC, mostly via the heart-rate-reserve gate.
- `shap_importance`: Top drivers are direct-edge features; oldpeak (gate-only) ranks 10th and age (gate-only) 0.013.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "from get_formula force=true latex_collapsed; the tool renders the output as relu(z), but for binary classification the evaluator applies sigmoid: p = sigmoid(z). Confirmed locally: sigmoid(z) reproduces compute_performance train AUC 0.9434 / acc 0.884 exactly."
}
```

```json
// performance_metrics
{
 "n": 242,
 "auc": 0.9434,
 "split": "train",
 "accuracy": 0.884,
 "coverage": {
  "structural": 1.0,
  "compositional": 1.0,
  "hierarchy_complete": true
 },
 "operations": {
  "node_splits": 5,
  "identity_nodes": 0,
  "function_changing": 0
 },
 "function_preservation_max_abs_err": 1.78e-15
}
```

```json
// ablation_study
{
 "split": "train",
 "ladder": [
  {
   "auc": 0.9434,
   "config": "full"
  },
  {
   "auc": 0.9302,
   "config": "linear core only"
  },
  {
   "auc": 0.9378,
   "config": "remove age_hr_reserve_gate (1718)"
  },
  {
   "auc": 0.9418,
   "config": "remove st_slope_chol_gate (1476)"
  },
  {
   "auc": 0.9427,
   "config": "remove hinge 1946"
  },
  {
   "auc": 0.943,
   "config": "remove hinge 3284"
  },
  {
   "auc": 0.9414,
   "config": "remove both hinges"
  },
  {
   "auc": 0.937,
   "config": "remove all oldpeak-driven terms (1476,1946,3284)"
  }
 ],
 "method": "local logit decomposition z = linear core + sum(readout weight * node activation), readouts removed without re-fitting",
 "logit_sd": {
  "1476": 0.18,
  "1718": 0.64,
  "1946": 0.23,
  "3284": 0.18,
  "linear_core": 3.28,
  "all_nonlinear": 0.91
 },
 "corr_nonlinear_vs_linear": 0.54
}
```

```json
// shap_importance
{
 "n": 242,
 "scale": "probability",
 "split": "train",
 "base_value": 0.448,
 "top10_mean_abs_shap": {
  "ca": 0.149,
  "chol": 0.034,
  "cp:4": 0.106,
  "sex:0": 0.064,
  "sex:1": 0.032,
  "thal:7": 0.041,
  "oldpeak": 0.026,
  "slope:1": 0.045,
  "slope:2": 0.059,
  "trestbps": 0.028
 }
}
```

</details>

### A.1 Leaf: `age_hr_reserve_gate`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | -1 (age), -15\_b (thalach\_b), -26\_b (thal:7\_b), -26\_c (thal:7\_c) |
| Exit nodes | 1718 (hr\_reserve\_gate) |
| Internal / exit nodes in its own subgraph | 3195 (thal7\_relay), 1718 (hr\_reserve\_gate) |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.011 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.49 x_{age} \\
&\quad - 0.762 x_{thal:7\_b} \\
&\quad + 1.02 x_{thalach\_b} \\
&\quad - 0.72 \operatorname{relu}{\left(1.07 x_{thal:7\_c} + 0.981 \right)} \\
&\quad -0.812 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}(-1.49 x_{age} - 0.762 x_{thal:7\_b} + 1.02 x_{thalach\_b} - 0.72\,\operatorname{relu}(1.07 x_{thal:7\_c} + 0.981) - 0.812)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Heart-rate-reserve gate 1718 = ReLU(-0.812 - 1.490\*age + 1.024\*thalach - 0.762\*thal:7 - 0.720\*R3195), with relay R3195 = ReLU(0.981 + 1.073\*thal:7). Hypothesis: 3195 is an always-on affine relay of thal:7 (thal:7 is one-hot, z-scored to {-0.769, 1.300}, so 3195 in {0.155, 2.376} \> 0), which makes the gate's net thal:7 coefficient -1.535 and bias -1.518. The gate fires for patients whose achieved max heart rate is high for their age (thalach up, age down) AND who do NOT have a reversible thallium defect; its readout into the output is -0.582, so it is a protective (risk-lowering) term. EVIDENCE (train, n=242): 3195 is active for 100% of patients, so the gate reduces exactly to 1718 = ReLU(-1.518 - 1.490\*age + 1.024\*thalach - 1.535\*thal:7). Active 33% overall: 51% of patients without a reversible defect vs 3% with one. Disease rate 11% when active vs 60% when inactive. Mean \|sensitivity\|: age 0.49 \> thalach 0.34 \> thal:7 (two copies) 0.25 each. Removing its readout drops train AUC 0.943 -\> 0.938, the largest single nonlinear effect (readout logit SD 0.64). Supported.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form; relay 3195 is affine on the data so the gate is a single ReLU of age, thalach and thal:7.
- `empirical_viz`: Gate is a protective 'good heart-rate reserve for age, no reversible defect' detector.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "logit += -0.582*1718",
 "simplified": "3195 always active on data (thal:7 in {-0.769,1.300} => 3195 in {0.155,2.376}); exact affine reduction: 1718 = relu(-1.518 - 1.490 age + 1.024 thalach - 1.535 thal:7). Pre-activation bias is -0.337 without reversible defect and -3.514 with it."
}
```

```json
// empirical_viz
{
 "n": 242,
 "viz": "sensitivity (max_samples 100000)",
 "split": "train",
 "outcome": "disease rate 11.3% when gate active vs 60.5% when inactive",
 "ablation": "removing the 1718 readout lowers train AUC 0.9434 -> 0.9378 (largest single nonlinear term); readout logit SD 0.64",
 "activity": "1718 active 33.1% overall; 50.7% of the 152 patients without reversible defect (thal:7=0) vs 3.3% of the 90 with it",
 "mean_abs_sensitivity": {
  "age": 0.492,
  "thalach": 0.339,
  "thal:7_b": 0.252,
  "thal:7_c(via 3195)": 0.256
 }
}
```

</details>

### A.2 Leaf: `st_slope_chol_gate`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | -9\_b (chol\_b), -18\_a (oldpeak\_a), -20\_b (slope:2\_b) |
| Exit nodes | 1476 (flat\_slope\_gate) |
| Internal / exit nodes in its own subgraph | 3464 (chol\_relay), 1476 (flat\_slope\_gate) |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00607 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.77 x_{oldpeak\_a} \\
&\quad + 1.15 x_{slope:2\_b} \\
&\quad - 0.374 \operatorname{relu}{\left(1.01 x_{chol\_b} + 1.3 \right)} \\
&\quad + 0.257 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}(0.77 x_{oldpeak\_a} + 1.15 x_{slope:2\_b} - 0.374\,\operatorname{relu}(1.01 x_{chol\_b} + 1.3) + 0.257)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Flat-ST-slope gate 1476 = ReLU(0.257 + 1.145\*slope:2 + 0.770\*oldpeak - 0.374\*C3464), with C3464 = ReLU(1.300 + 1.011\*chol). Hypothesis: a one-sided risk detector that opens for patients with a flat peak-exercise ST segment (slope:2), its height scaled by ST depression (oldpeak) and reduced by cholesterol via an almost-always-active affine chol relay (3464 is zero only when z-chol \< -1.29). Readout into output +0.225 (raises risk). EVIDENCE (train, n=242): 3464 is active for 93% (off only when z-chol \< -1.29), so 1476 is nearly ReLU(-0.229 + 1.145\*slope:2 + 0.770\*oldpeak - 0.378\*chol). Active for 98% of the 110 flat-slope patients vs 7% of the others, so it acts as a flat-slope indicator whose height rises with oldpeak (corr 0.69) and falls with cholesterol. Disease rate 67% when active vs 23% when inactive. Mean \|sensitivity\|: slope:2 0.55 \> oldpeak 0.37 \> chol 0.17. Removing its readout drops train AUC 0.943 -\> 0.942 (logit SD 0.18). Supported. The negative chol term partly offsets chol's direct +0.613 edge for flat-slope patients; no clinical reading is offered for that.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form; the chol relay is near-affine.
- `empirical_viz`: Gate is essentially a flat-ST-slope indicator whose magnitude grows with ST depression and shrinks with cholesterol.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "logit += 0.225*1476",
 "simplified": "3464 active 92.6% (zero only for z-chol < -1.29); where active, 1476 = relu(-0.229 + 0.770 oldpeak + 1.145 slope:2 - 0.378 chol). slope:2 in {-0.913, 1.095} so slope:2 shifts the pre-activation by 2.30 between categories."
}
```

```json
// empirical_viz
{
 "n": 242,
 "viz": "sensitivity (max_samples 100000)",
 "split": "train",
 "outcome": "disease rate 66.7% when active vs 23.2% when inactive; corr(1476, oldpeak)=0.69",
 "ablation": "removing the 1476 readout lowers train AUC 0.9434 -> 0.9418; readout logit SD 0.18",
 "activity": "1476 active 48.3%; 98.2% of the 110 flat-slope (slope:2) patients vs 6.8% of the others",
 "mean_abs_sensitivity": {
  "chol": 0.169,
  "oldpeak": 0.372,
  "slope:2": 0.554
 }
}
```

</details>

### A.3 Leaf: `oldpeak_hinges`

|  |  |
|---|---|
| Annotation id | `ann_33` |
| Entry nodes | -18\_b (oldpeak\_b) |
| Exit nodes | 1946 (oldpeak\_hinge1), 3284 (oldpeak\_hinge2) |
| Internal / exit nodes in its own subgraph | 1946 (oldpeak\_hinge1), 3284 (oldpeak\_hinge2) |
| Dimensionality (in → out, incl. descendants) | 1 → 2 |
| Live closed-form error (3 s.f. display) | 0.00857 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.767 x_{oldpeak\_b} + 0.167 \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 1.0 \operatorname{relu}{\left(0.767 x_{oldpeak\_b} + 0.167 \right)} \\
&\quad -0.582 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}(0.767 x_{oldpeak\_b} + 0.167), \quad y_1 = \operatorname{relu}(1.0\,\operatorname{relu}(0.767 x_{oldpeak\_b} + 0.167) - 0.582)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Oldpeak hinge pair: 1946 = ReLU(0.167 + 0.767\*oldpeak) and 3284 = ReLU(-0.582 + 1.003\*1946). Hypothesis: a convex piecewise-linear function of ST depression alone, with kinks at z-oldpeak = -0.218 (1946 opens) and 0.539 (3284 opens). Both exits read out positively (0.377 and 0.411), so beyond the second kink the logit slope in oldpeak roughly doubles: mild ST depression adds a little risk, marked ST depression adds much more. EVIDENCE (train, n=242): 1946 active 50% (disease rate 64%), 3284 active 24% (disease rate 79%). Contribution to the logit 0.377\*1946 + 0.411\*3284 has slope 0 below z-oldpeak -0.218, 0.289 between the kinks and 0.605 above 0.539. Mean \|sensitivity\| to oldpeak: 0.38 for 1946, 0.18 for 3284. Removing either readout changes train AUC by \<= 0.001. Supported: a small convex ST-depression dose-response.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the hinge pair; a convex piecewise-linear function of oldpeak.
- `empirical_viz`: Hinges confirmed: oldpeak raises risk only above about the median, and more steeply at high values.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "logit += 0.377*1946 + 0.411*3284: slope 0 below -0.218, 0.289 between kinks, 0.605 above 0.539 (per z-unit oldpeak)",
 "simplified": "kinks at z-oldpeak = -0.218 (1946) and 0.539 (3284); 3284 = relu(0.769 oldpeak - 0.415)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "viz": "sensitivity (max_samples 100000)",
 "split": "train",
 "ablation": "removing 1946 readout: AUC 0.9434 -> 0.9427; removing 3284 readout: -> 0.9430",
 "activity": "1946 active 50% (121 patients, disease rate 63.6%); 3284 active 24% (58 patients, disease rate 79.3%)",
 "mean_abs_sensitivity": {
  "1946 wrt oldpeak": 0.383,
  "3284 wrt oldpeak": 0.184
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 4 |
| rename\_node | 32 |
| split\_node | 5 |

5 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -26 into -26\_a, -26\_b, -26\_c. Final copies: -26\_a → heart\_disease\_logit (entry) \| -26\_b → age\_hr\_reserve\_gate (entry); heart\_disease\_logit (entry) \| -26\_c → age\_hr\_reserve\_gate (entry); heart\_disease\_logit (entry)
- `#27` split -15 into -15\_a, -15\_b. Final copies: -15\_a → heart\_disease\_logit (entry) \| -15\_b → age\_hr\_reserve\_gate (entry); heart\_disease\_logit (entry)
- `#28` split -9 into -9\_a, -9\_b. Final copies: -9\_a → heart\_disease\_logit (entry) \| -9\_b → st\_slope\_chol\_gate (entry); heart\_disease\_logit (entry)
- `#29` split -20 into -20\_a, -20\_b. Final copies: -20\_a → heart\_disease\_logit (entry) \| -20\_b → st\_slope\_chol\_gate (entry); heart\_disease\_logit (entry)
- `#30` split -18 into -18\_a, -18\_b. Final copies: -18\_a → st\_slope\_chol\_gate (entry); heart\_disease\_logit (entry) \| -18\_b → oldpeak\_hinges (entry); heart\_disease\_logit (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.78e-15 | 1.78e-15 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.01105 | 0.01105 | 0.01105 | agree |
| Numeric-function max abs error | n/a | 8.88e-16 | 8.88e-16 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 3 / 1 | n/a | 3 / 1 | agree |
| Operation counts | 5 splits, 0 identity | {"annotate": 4, "rename\_node": 32, "split\_node": 5} | {"annotate": 4, "rename\_node": 32, "split\_node": 5} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `age_hr_reserve_gate` | 0.01105 | 8.88e-16 |
| `oldpeak_hinges` | 0.008572 | 0 |
| `st_slope_chol_gate` | 0.00607 | 4.44e-16 |

## What resisted (from the record)

> Nothing resisted coverage. Clinically counter-directional signs reported, not explained: cp:2 +0.42, fbs:1 -0.31 in the linear core, negative chol inside the flat-slope gate. Relays 3195 (thal:7) and 3464 (chol) are affine on the data (100%/93% active).

**Tool notes (record):** get\_formula shows output as ReLU (sigmoid(z) reproduces train AUC 0.9434). compute\_shap \>120s. Input fan-out splits (-26,-15,-9,-20,-18) found by hand.

## What to look at

- Largest closed-form error: `age_hr_reserve_gate` (0.011). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `age_hr_reserve_gate`: *“The gate fires for patients whose achieved max heart rate is high for their age (thalach up, age down) AND who do NOT have a reversible thallium defect;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `st_slope_chol_gate`: *“…scaled by ST depression (oldpeak) and reduced by cholesterol via an almost-always-active affine chol relay (3464 is zero only when z-chol \< -1.29).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 5 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s2.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s2', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
