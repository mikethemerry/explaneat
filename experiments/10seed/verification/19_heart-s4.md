# Verification packet 19 · heart-s4

Pre-registered verification order: 19 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/15783a91-457e-49a9-9209-cc3aa77df0cb?experiment=c04ee4d5-d015-4a8c-9c62-3df243d3b89f&name=10seed-heart-s4), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 4 |
| Genome id | `15783a91-457e-49a9-9209-cc3aa77df0cb` |
| Experiment | 10seed-heart-s4 (c04ee4d5-d015-4a8c-9c62-3df243d3b89f); best at generation 38 of 40, NEAT key 9650 |
| Test AUC (runs csv) | 0.874 \[95% CI 0.774, 0.952\] |
| Train AUC | 0.942 |
| Size | 20 of 26 inputs connected; 4 hidden nodes; 27 enabled connections; depth 4; 4 skip connections + 17 direct input→output edges |
| Annotations | 3 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 4 of 4 |
| Claude's time | 12.8 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:36:48+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Essentially logistic regression: 17 direct edges (logit SD 4.44; ca +2.16, slope:2 +1.08, sex:1 +0.85). One meaningful nonlinearity: discrete gate 2065 = ReLU(0.795 thal:6 - 0.885 fbs:1 - 0.719), firing only for fixed-defect patients with normal fasting sugar and lowering their logit by \~3.6, cancelling the linear core's over-calling in that group. cp:2 indicator (3497) and floored thalach relay (707-\>3677) near-inert. Train AUC 0.942 vs 0.934 linear core.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | -1 (age), -2 (sex:0), -3 (sex:1), -4 (cp:1), -6 (cp:3), -7 (cp:4), -8 (trestbps), -9 (chol), -10 (fbs:0), -11\_a (fbs:1\_a), -15\_a (thalach\_a), -17 (exang:1), -18 (oldpeak), -19 (slope:1), -20 (slope:2), -22 (ca), -23 (thal:3), -12 (restecg:0), -14 (restecg:2), -16 (exang:0), -21 (slope:3), -24 (thal:4), -26 (thal:7) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | thal6\_fixed\_defect\_gate, cp2\_atypical\_angina\_indicator, thalach\_relu\_relay |
| Dimensionality (in → out, incl. descendants) | 28 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.948 e^{- 0.0956 x_{age} - 2.16 x_{ca} - 0.677 x_{chol} + 0.658 x_{cp:1} + 0.755 x_{cp:3} - 0.295 x_{cp:4} - 0.342 x_{exang:1} - 0.00701 x_{fbs:0} + 0.526 x_{fbs:1\_a} - 0.529 x_{oldpeak} + 0.484 x_{sex:0} - 0.849 x_{sex:1} + 0.199 x_{slope:1} - 1.08 x_{slope:2} + 0.539 x_{thal:3} + 0.196 x_{thalach\_a} - 0.566 x_{trestbps} + 0.12 \operatorname{cp}_{2 \mathcal{atypi} angina indicator}{\left(x_{cp:2} \right)} + 1.12 \operatorname{thal\_6\_fixed\_defect\_gate}{\left(x_{fbs:1\_b},x_{thal:6} \right)} + 0.0891 \operatorname{thalach\_relu\_relay}{\left(2.18 \operatorname{thal\_6\_fixed\_defect\_gate}{\left(x_{fbs:1\_b},x_{thal:6} \right)},x_{thalach\_b},x_{restecg:1} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.0956 x_{age} \\
&\quad + 2.16 x_{ca} \\
&\quad + 0.677 x_{chol} \\
&\quad - 0.658 x_{cp:1} \\
&\quad - 0.755 x_{cp:3} \\
&\quad + 0.295 x_{cp:4} \\
&\quad + 0.342 x_{exang:1} \\
&\quad + 0.00701 x_{fbs:0} \\
&\quad - 0.526 x_{fbs:1\_a} \\
&\quad + 0.529 x_{oldpeak} \\
&\quad - 0.484 x_{sex:0} \\
&\quad + 0.849 x_{sex:1} \\
&\quad - 0.199 x_{slope:1} \\
&\quad + 1.08 x_{slope:2} \\
&\quad - 0.539 x_{thal:3} \\
&\quad - 0.196 x_{thalach\_a} \\
&\quad + 0.566 x_{trestbps} \\
&\quad - 0.12 \operatorname{cp}_{2 \mathcal{atypi} angina indicator}{\left(-5 \right)} \\
&\quad - 1.12 \operatorname{thal\_6\_fixed\_defect\_gate}{\left(-11_{b},-25 \right)} \\
&\quad - 0.0891 \operatorname{thalach\_relu\_relay}{\left(identity_{707},-15_{b},-13 \right)} \\
&\quad + 0.0532 \bigr)
\end{aligned}
\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: heart-disease logit z = 0.053 + LINEAR CORE - 1.117\*thal6\_fixed\_defect\_gate - 0.120\*cp2\_atypical\_angina\_indicator - 0.089\*thalach\_relu\_relay; prediction = sigmoid(z) (the genome output node is ReLU but the evaluator overrides it with sigmoid for binary classification, so z is a logistic score). LINEAR CORE (17 direct input-\>output edges on z-scored inputs): ca +2.16, slope:2 +1.08, sex:1 +0.85, sex:0 -0.48, cp:3 -0.76, chol +0.68, cp:1 -0.66, trestbps +0.57, thal:3 -0.54, oldpeak +0.53, fbs:1 -0.53, exang:1 +0.34, cp:4 +0.30, slope:1 -0.20, thalach -0.20, age +0.10, fbs:0 +0.01; inputs restecg:0, restecg:2, exang:0, slope:3, thal:4, thal:7 are unused (listed as entries only for coverage). COMPOSITION: additive readout of three leaves plus one serial link: thal6\_fixed\_defect\_gate feeds thalach\_relu\_relay (2065 -\> identity\_707 weight 2.178 -\> 707), so the fixed-defect correction totals about -3.57 logit for thal:6 patients with normal FBS. Interpretation: a logistic regression (number of major vessels, flat ST slope, male sex, non-anginal/typical chest pain protective, cholesterol, resting BP, ST depression) plus one discrete subgroup correction that cancels the linear core's over-calling of disease in fixed-defect patients with normal fasting sugar (n=10 on train); the cp:2 and thalach hidden paths are near-inert refinements. EVIDENCE: TRAIN n=242 (compute\_performance): accuracy 0.884, AUC 0.942. Logit SD 4.44. Local mean-ablation of the hidden readouts on TRAIN (forward pass on in-memory copy): full AUC 0.9424/acc 0.884; thal6\_fixed\_defect\_gate (2065 direct) ablated: AUC 0.9363/acc 0.872, max \|dlogit\| 2.92; removing the whole thal:6 path (direct + via 707): AUC 0.9347; cp2 indicator ablated: AUC 0.9422, max \|dlogit\| 0.09; thalach relay ablated: AUC 0.9414, max \|dlogit\| 0.63; all three hidden readouts ablated (linear core only): AUC 0.9342/acc 0.868. Logit contribution SDs: 2065 0.61, 3677 0.11, 3497 0.04, vs linear core direct terms ca 2.16, slope:2 1.08, sex:1 0.85, cp:3 0.76, chol 0.68, cp:1 0.66, trestbps 0.57, oldpeak 0.53, thal:3 0.54, fbs:1 0.53, sex:0 0.48, exang:1 0.34, cp:4 0.30, slope:1 0.20, thalach 0.20, age 0.10, fbs:0 0.01. SHAP: compute\_shap (train, n=242, probability scale, base 0.433) mean \|SHAP\|: ca 0.168, slope:2 0.095, sex:1 0.063, cp:3 0.054, thal:3 0.046, chol 0.033, oldpeak 0.030, trestbps 0.030, sex:0 0.028, cp:1 0.024, exang:1 0.022, cp:4 0.019, thal:6 0.014, fbs:1 0.012, thalach 0.010, slope:1 0.004, age 0.0001; cp:2 reported 0.000 although it acts via 3497 (tool estimate; the logit effect is only -0.113). Unused inputs 0.

**Evidence (4 records, DB):**

- `analytical_formula`: Collapsed closed form of the root (get\_formula force=true, latex\_collapsed).
- `performance_metrics`: TRAIN n=242 (compute\_performance): accuracy 0.884, AUC 0.942. Logit SD 4.44.
- `ablation_study`: Local mean-ablation of the hidden readouts on TRAIN (forward pass on in-memory copy): full AUC 0.9424/acc 0.884; thal6\_fixed\_defect\_gate (2065 direct) ablated: AUC 0.9363/acc 0.872, max \|dlogit\| 2.92; removing the whole thal:6 path (direct + via 707): AUC 0.9347; cp2 indicator ablated: AUC 0.9422, max \|dlogit\| 0.09; thalach relay ablated: AUC 0.9414, max \|dlogit\| 0.63; all three hidden readouts ablated (linear core only): AUC 0.9342/acc 0.868. Logit contribution SDs: 2065 0.61, 3677 0.11, 3497 0.04, vs linear core direct terms ca 2.16, slope:2 1.08, sex:1 0.85, cp:3 0.76, chol 0.68, cp:1 0.66, trestbps 0.57, oldpeak 0.53, thal:3 0.54, fbs:1 0.53, sex:0 0.48, exang:1 0.34, cp:4 0.30, slope:1 0.20, thalach 0.20, age 0.10, fbs:0 0.01.
- `shap_importance`: compute\_shap (train, n=242, probability scale, base 0.433) mean \|SHAP\|: ca 0.168, slope:2 0.095, sex:1 0.063, cp:3 0.054, thal:3 0.046, chol 0.033, oldpeak 0.030, trestbps 0.030, sex:0 0.028, cp:1 0.024, exang:1 0.022, cp:4 0.019, thal:6 0.014, fbs:1 0.012, thalach 0.010, slope:1 0.004, age 0.0001; cp:2 reported 0.000 although it acts via 3497 (tool estimate; the logit effect is only -0.113). Unused inputs 0.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "Output node shown as relu; evaluator applies sigmoid to the pre-activation for binary classification."
}
```

```json
// performance_metrics
{
 "n": 242,
 "split": "train",
 "auc_roc": 0.9424,
 "accuracy": 0.8843
}
```

```json
// ablation_study
{
 "split": "train",
 "summary": "Local mean-ablation of the hidden readouts on TRAIN (forward pass on in-memory copy): full AUC 0.9424/acc 0.884; thal6_fixed_defect_gate (2065 direct) ablated: AUC 0.9363/acc 0.872, max |dlogit| 2.92; removing the whole thal:6 path (direct + via 707): AUC 0.9347; cp2 indicator ablated: AUC 0.9422, max |dlogit| 0.09; thalach relay ablated: AUC 0.9414, max |dlogit| 0.63; all three hidden readouts ablated (linear core only): AUC 0.9342/acc 0.868. Logit contribution SDs: 2065 0.61, 3677 0.11, 3497 0.04, vs linear core direct terms ca 2.16, slope:2 1.08, sex:1 0.85, cp:3 0.76, chol 0.68, cp:1 0.66, trestbps 0.57, oldpeak 0.53, thal:3 0.54, fbs:1 0.53, sex:0 0.48, exang:1 0.34, cp:4 0.30, slope:1 0.20, thalach 0.20, age 0.10, fbs:0 0.01."
}
```

```json
// shap_importance
{
 "split": "train",
 "summary": "compute_shap (train, n=242, probability scale, base 0.433) mean |SHAP|: ca 0.168, slope:2 0.095, sex:1 0.063, cp:3 0.054, thal:3 0.046, chol 0.033, oldpeak 0.030, trestbps 0.030, sex:0 0.028, cp:1 0.024, exang:1 0.022, cp:4 0.019, thal:6 0.014, fbs:1 0.012, thalach 0.010, slope:1 0.004, age 0.0001; cp:2 reported 0.000 although it acts via 3497 (tool estimate; the logit effect is only -0.113). Unused inputs 0."
}
```

</details>

### A.1 Leaf: `thal6_fixed_defect_gate`

|  |  |
|---|---|
| Annotation id | `ann_29` |
| Entry nodes | -11\_b (fbs:1\_b), -25 (thal:6) |
| Exit nodes | 2065 |
| Internal / exit nodes in its own subgraph | 2065 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00154 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.885 x_{fbs:1\_b} \\
&\quad + 0.795 x_{thal:6} \\
&\quad -0.719 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}(-0.885\,x_{fbs:1} + 0.795\,x_{thal:6} - 0.719)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Fixed-thallium-defect correction gate. 2065 = ReLU(0.795\*thal:6 - 0.885\*fbs:1 - 0.719) on z-scored one-hots is a discrete indicator: 0 unless thal:6=1 (fixed defect), 2.727 if additionally fasting blood sugar is normal, 0.148 if fbs\>120. It enters the logit with weight -1.117 (-3.05) and, via identity\_707, adds 2.178\*2065 to the thalach gate 707 (a further \~-0.52 through 3677). Its role is to cancel the linear core's over-prediction of disease for thal:6 patients with normal FBS (the linear core gives them +0.60 through the thal:3 edge and other terms). EVIDENCE: TRAIN n=242: 2065 is exactly 0 for all thal:6=0 patients (227/242). thal:6=1 & fbs:1=0 (n=10): 2065=2.727; thal:6=1 & fbs:1=1 (n=5): 2065=0.148. Active 6.2%. Sensitivity (mean \|grad\|): fbs:1 0.055, thal:6 0.049. Readout: direct -1.117\*2065 =\> logit -3.05 for fixed-defect/normal-FBS patients; via identity\_707 -\> 707 -\> 3677 a further \~-0.52. Mean-ablating 2065 lowers train AUC 0.9424 -\> 0.9363 and acc 0.884 -\> 0.872. On the 10 thal:6&normal-FBS patients (disease rate 0.5) the linear core alone gives mean logit +3.78 (9/10 predicted diseased; thal:6 patients get +0.60 from the negative thal:3 edge); with the gate the mean logit is +0.25 and all 10 are classified correctly. Thus it is a learned correction for a 10-patient subgroup the linear core over-calls (likely partly fitted to train noise).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: TRAIN n=242: 2065 is exactly 0 for all thal:6=0 patients (227/242). thal:6=1 & fbs:1=0 (n=10): 2065=2.727; thal:6=1 & fbs:1=1 (n=5): 2065=0.148. Active 6.2%. Sensitivity (mean \|grad\|): fbs:1 0.055, thal:6 0.049. Readout: direct -1.117\*2065 =\> logit -3.05 for fixed-defect/normal-FBS patients; via identity\_707 -\> 707 -\> 3677 a further \~-0.52. Mean-ablating 2065 lowers train AUC 0.9424 -\> 0.9363 and acc 0.884 -\> 0.872. On the 10 thal:6&normal-FBS patients (disease rate 0.5) the linear core alone gives mean logit +3.78 (9/10 predicted diseased; thal:6 patients get +0.60 from the negative thal:3 edge); with the gate the mean logit is +0.25 and all 10 are classified correctly. Thus it is a learned correction for a 10-patient subgroup the linear core over-calls (likely partly fitted to train noise).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "summary": "TRAIN n=242: 2065 is exactly 0 for all thal:6=0 patients (227/242). thal:6=1 & fbs:1=0 (n=10): 2065=2.727; thal:6=1 & fbs:1=1 (n=5): 2065=0.148. Active 6.2%. Sensitivity (mean |grad|): fbs:1 0.055, thal:6 0.049. Readout: direct -1.117*2065 => logit -3.05 for fixed-defect/normal-FBS patients; via identity_707 -> 707 -> 3677 a further ~-0.52. Mean-ablating 2065 lowers train AUC 0.9424 -> 0.9363 and acc 0.884 -> 0.872. On the 10 thal:6&normal-FBS patients (disease rate 0.5) the linear core alone gives mean logit +3.78 (9/10 predicted diseased; thal:6 patients get +0.60 from the negative thal:3 edge); with the gate the mean logit is +0.25 and all 10 are classified correctly. Thus it is a learned correction for a 10-patient subgroup the linear core over-calls (likely partly fitted to train noise).",
 "viz_type": "sensitivity + local forward pass"
}
```

</details>

### A.2 Leaf: `cp2_atypical_angina_indicator`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -5 (cp:2) |
| Exit nodes | 3497 |
| Internal / exit nodes in its own subgraph | 3497 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00371 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.984 x_{cp:2} - 1.21 \right)}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}(0.984\,x_{cp:2} - 1.21)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Atypical-angina indicator. 3497 = ReLU(0.984\*cp:2 - 1.206) is 0 for every chest-pain type except cp=2 (atypical angina), where it is 0.941; i.e. it is exactly an affine re-coding of the cp:2 one-hot. Read out with weight -0.120, it lowers the logit by 0.113 for atypical-angina patients (disease rate 0.19 on train vs 0.50 otherwise). Functionally near-inert (ablation changes AUC by 0.0002). EVIDENCE: TRAIN n=242: 3497 takes exactly two values: 0 for the 200 non-cp:2 patients, 0.941 for the 42 atypical-angina (cp:2) patients (disease rate 0.19 vs 0.50 for others). Sensitivity mean \|grad\| 0.171. Logit contribution -0.120\*0.941 = -0.113 for cp:2 patients. Mean-ablating 3497: AUC 0.9424 -\> 0.9422, max \|dlogit\| 0.093. Near-inert: an affine function of cp:2 (no direct cp:2 -\> output edge exists, so this node is the network's only use of cp:2).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: TRAIN n=242: 3497 takes exactly two values: 0 for the 200 non-cp:2 patients, 0.941 for the 42 atypical-angina (cp:2) patients (disease rate 0.19 vs 0.50 for others). Sensitivity mean \|grad\| 0.171. Logit contribution -0.120\*0.941 = -0.113 for cp:2 patients. Mean-ablating 3497: AUC 0.9424 -\> 0.9422, max \|dlogit\| 0.093. Near-inert: an affine function of cp:2 (no direct cp:2 -\> output edge exists, so this node is the network's only use of cp:2).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "summary": "TRAIN n=242: 3497 takes exactly two values: 0 for the 200 non-cp:2 patients, 0.941 for the 42 atypical-angina (cp:2) patients (disease rate 0.19 vs 0.50 for others). Sensitivity mean |grad| 0.171. Logit contribution -0.120*0.941 = -0.113 for cp:2 patients. Mean-ablating 3497: AUC 0.9424 -> 0.9422, max |dlogit| 0.093. Near-inert: an affine function of cp:2 (no direct cp:2 -> output edge exists, so this node is the network's only use of cp:2).",
 "viz_type": "sensitivity + local forward pass"
}
```

</details>

### A.3 Leaf: `thalach_relu_relay`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | identity\_707, -15\_b (thalach\_b), -13 (restecg:1) |
| Exit nodes | 3677 |
| Internal / exit nodes in its own subgraph | 707, 3677 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00185 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.983 \operatorname{relu}{\left(1.0 x_{identity\_707} - 0.429 x_{restecg:1} + 0.944 x_{thalach\_b} + 0.657 \right)} \\
&\quad + 0.438 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}\bigl(0.983\,\operatorname{relu}(1.0\,x_{identity\\_707} - 0.429\,x_{restecg:1} + 0.944\,x_{thalach} + 0.657) + 0.438\bigr)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Max-heart-rate relay with a floor. 707 = ReLU(0.944\*thalach - 0.429\*restecg:1 + identity\_707 + 0.657), where identity\_707 = 2.178\*thal6\_fixed\_defect\_gate; 3677 = 0.983\*707 + 0.4385 (its ReLU never binds on train). Read out with weight -0.089 it adds a small extra protective effect of high maximum heart rate (beyond the direct thalach edge -0.196), clipped to zero for low-thalach patients (bottom \~22%), plus a small extra protective shift for fixed-defect patients. Small effect (logit SD 0.11). EVIDENCE: TRAIN n=242: 3677 is active for 100% of samples (min 0.4385), so the outer ReLU is the identity and 3677 = 0.983\*707 + 0.4385. 707 is zero for 21.5% of patients (22.5% of thal:6=0 patients, i.e. thalach below \~-0.70 SD when identity\_707=0 and restecg:1 normal; only 1/15 thal:6 patients, whose +5.94 from identity\_707 holds the gate open). restecg:1 occurs in only 2 train patients. Sensitivity mean \|grad\|: identity\_707 0.77, thalach 0.73, restecg:1 0.33. Readout weight -0.089 into the logit: contribution range \[-0.77, -0.04\], SD 0.107. Mean-ablating 3677: AUC 0.9424 -\> 0.9414; replacing ReLU(707) by its pre-activation: AUC 0.9423, max \|dlogit\| 0.48.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: TRAIN n=242: 3677 is active for 100% of samples (min 0.4385), so the outer ReLU is the identity and 3677 = 0.983\*707 + 0.4385. 707 is zero for 21.5% of patients (22.5% of thal:6=0 patients, i.e. thalach below \~-0.70 SD when identity\_707=0 and restecg:1 normal; only 1/15 thal:6 patients, whose +5.94 from identity\_707 holds the gate open). restecg:1 occurs in only 2 train patients. Sensitivity mean \|grad\|: identity\_707 0.77, thalach 0.73, restecg:1 0.33. Readout weight -0.089 into the logit: contribution range \[-0.77, -0.04\], SD 0.107. Mean-ablating 3677: AUC 0.9424 -\> 0.9414; replacing ReLU(707) by its pre-activation: AUC 0.9423, max \|dlogit\| 0.48.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "summary": "TRAIN n=242: 3677 is active for 100% of samples (min 0.4385), so the outer ReLU is the identity and 3677 = 0.983*707 + 0.4385. 707 is zero for 21.5% of patients (22.5% of thal:6=0 patients, i.e. thalach below ~-0.70 SD when identity_707=0 and restecg:1 normal; only 1/15 thal:6 patients, whose +5.94 from identity_707 holds the gate open). restecg:1 occurs in only 2 train patients. Sensitivity mean |grad|: identity_707 0.77, thalach 0.73, restecg:1 0.33. Readout weight -0.089 into the logit: contribution range [-0.77, -0.04], SD 0.107. Mean-ablating 3677: AUC 0.9424 -> 0.9414; replacing ReLU(707) by its pre-activation: AUC 0.9423, max |dlogit| 0.48.",
 "viz_type": "sensitivity + local forward pass"
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 4 |
| rename\_node | 26 |
| split\_node | 2 |

2 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -11 into -11\_a, -11\_b. Final copies: -11\_a → heart\_disease\_logit (entry) \| -11\_b → thal6\_fixed\_defect\_gate (entry)
- `#27` split -15 into -15\_a, -15\_b. Final copies: -15\_a → heart\_disease\_logit (entry) \| -15\_b → thalach\_relu\_relay (entry)
- `#28` identity node identity\_707 intercepts 1 connection(s) into 707 (from 2065); sits in thalach\_relu\_relay (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 3.55e-15 | 3.55e-15 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.003711 | 0.003711 | 0.003711 | agree |
| Numeric-function max abs error | n/a | 1.78e-15 | 1.78e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 3 / 1 | n/a | 3 / 1 | agree |
| Operation counts | 2 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 26, "split\_node": 2} | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 26, "split\_node": 2} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `cp2_atypical_angina_indicator` | 0.003711 | 0 |
| `thalach_relu_relay` | 0.001852 | 1.78e-15 |
| `thal6_fixed_defect_gate` | 0.001544 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. 2065 reads out to both output and gate 707; handled with identity node identity\_707 on 2065-\>707 rather than a split, root composes the two leaves serially. 6 unconnected inputs (restecg:0, restecg:2, exang:0, slope:3, thal:4, thal:7) listed as root entries only to reach structural coverage 1.0. thal:6 correction rests on 10 training patients; may be partly noise-fitted.

**Tool notes (record):** compute\_shap mean \|SHAP\| = 0 for cp:2 despite logit effect -0.113 via 3497. compute\_shap \>2 min. get\_formula shows output as ReLU.

## What to look at

- Largest closed-form error: `cp2_atypical_angina_indicator` (0.00371). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `thal6_fixed_defect_gate`: *“EVIDENCE: TRAIN n=242: 2065 is exactly 0 for all thal:6=0 patients (227/242).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `thalach_relu_relay`: *“only 1/15 thal:6 patients, whose +5.94 from identity\_707 holds the gate open).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 2 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s4.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s4', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
