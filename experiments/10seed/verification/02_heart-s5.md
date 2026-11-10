# Verification packet 02 · heart-s5

Pre-registered verification order: 2 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/1f2cf7ea-5c30-452a-9277-0b4043e888d9?experiment=138d99ce-afe5-4b70-88d7-3953be1a6bbd&name=10seed-heart-s5), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 5 |
| Genome id | `1f2cf7ea-5c30-452a-9277-0b4043e888d9` |
| Experiment | 10seed-heart-s5 (138d99ce-afe5-4b70-88d7-3953be1a6bbd); best at generation 37 of 40, NEAT key 9330 |
| Test AUC (runs csv) | 0.893 \[95% CI 0.801, 0.963\] |
| Train AUC | 0.942 |
| Size | 20 of 26 inputs connected; 6 hidden nodes; 36 enabled connections; depth 5; 8 skip connections + 18 direct input→output edges |
| Annotations | 3 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 2 of 6 |
| Claude's time | 17.4 min (2026-11-03T09:03:09+13:00 → 2026-11-03T09:20:34+13:00); within the 3 h box: yes |
| Protocol | pilot (pre-f109cad+1 protocol; evidence tools used default max\_samples=200 subsample of 242 train rows) (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Essentially logistic regression (18 direct edges; logit SD 3.18) plus one small nonlinear ischaemia correction (logit SD 0.78): a ReLU gate on oldpeak, thal:7 and restecg:0 that adds to risk directly and feeds a resting-BP-offset gate read out directly, via a high-threshold amplifier, and via an age term active only without ischaemia evidence. Nonlinear parts add \~+0.010 train AUC (0.932 linear-only vs 0.942).

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_40` |
| Entry nodes | -2 (sex:0), -4 (cp:1), -7 (cp:4), -9 (chol), -10 (fbs:0), -15 (thalach), -16 (exang:0), -17 (exang:1), -19 (slope:1), -21 (slope:3), -22 (ca), -23 (thal:3), -1\_a (age\_a), -6\_a (cp:3\_a), -8\_a (trestbps\_a), -12\_a (restecg:0\_a), -25\_a (thal:6\_a), -26\_a (thal:7\_a), -24 (thal:4), -20 (slope:2), -14 (restecg:2), -13 (restecg:1), -11 (fbs:1), -3 (sex:1) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | st\_ischaemia\_gate\_output, ischaemia\_pathway |
| Dimensionality (in → out, incl. descendants) | 36 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{1.64 e^{- 0.113 x_{age\_a} - 1.51 x_{ca} - 0.533 x_{chol} + 0.53 x_{cp:1} + 0.473 x_{cp:3\_a} - 0.331 x_{cp:4} - 0.151 x_{exang:0} - 0.43 x_{exang:1} - 0.275 x_{fbs:0} + 0.184 x_{restecg:0\_a} + 1.08 x_{sex:0} + 0.9 x_{slope:1} + 0.413 x_{slope:3} + 0.432 x_{thal:3} + 0.383 x_{thal:6\_a} + 0.0654 x_{thal:7\_a} + 0.252 x_{thalach} - 0.492 x_{trestbps\_a} - 0.135 \operatorname{ischaemia\_pathway\_0}{\left(x_{age\_b},x_{cp:2},x_{cp:3\_b},x_{trestbps\_b},x_{thal:6\_b},x_{restecg:0\_b\_d},x_{oldpeak\_b},x_{thal:7\_b\_d},x_{thal:6\_c} \right)} - 0.165 \operatorname{ischaemia\_pathway\_1}{\left(x_{age\_b},x_{cp:2},x_{cp:3\_b},x_{trestbps\_b},x_{thal:6\_b},x_{restecg:0\_b\_d},x_{oldpeak\_b},x_{thal:7\_b\_d},x_{thal:6\_c} \right)} - 0.0971 \operatorname{ischaemia\_pathway\_2}{\left(x_{age\_b},x_{cp:2},x_{cp:3\_b},x_{trestbps\_b},x_{thal:6\_b},x_{restecg:0\_b\_d},x_{oldpeak\_b},x_{thal:7\_b\_d},x_{thal:6\_c} \right)} - 0.117 \operatorname{st\_ischaemia\_gate\_output}{\left(x_{restecg:0\_b\_c},x_{oldpeak\_a},x_{thal:7\_b\_c} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.113 x_{age\_a} \\
&\quad + 1.51 x_{ca} \\
&\quad + 0.533 x_{chol} \\
&\quad - 0.53 x_{cp:1} \\
&\quad - 0.473 x_{cp:3\_a} \\
&\quad + 0.331 x_{cp:4} \\
&\quad + 0.151 x_{exang:0} \\
&\quad + 0.43 x_{exang:1} \\
&\quad + 0.275 x_{fbs:0} \\
&\quad - 0.184 x_{restecg:0\_a} \\
&\quad - 1.08 x_{sex:0} \\
&\quad - 0.9 x_{slope:1} \\
&\quad - 0.413 x_{slope:3} \\
&\quad - 0.432 x_{thal:3} \\
&\quad - 0.383 x_{thal:6\_a} \\
&\quad - 0.0654 x_{thal:7\_a} \\
&\quad - 0.252 x_{thalach} \\
&\quad + 0.492 x_{trestbps\_a} \\
&\quad + 0.135 \operatorname{ischaemia\_pathway\_0}{\left(-1_{b},-5,-6_{b},-8_{b},-25_{b},-12_{d},-18_{b},-26_{d},-25_{c} \right)} \\
&\quad + 0.165 \operatorname{ischaemia\_pathway\_1}{\left(-1_{b},-5,-6_{b},-8_{b},-25_{b},-12_{d},-18_{b},-26_{d},-25_{c} \right)} \\
&\quad + 0.0971 \operatorname{ischaemia\_pathway\_2}{\left(-1_{b},-5,-6_{b},-8_{b},-25_{b},-12_{d},-18_{b},-26_{d},-25_{c} \right)} \\
&\quad + 0.117 \operatorname{st\_ischaemia\_gate\_output}{\left(-12_{c},-18_{a},-26_{c} \right)} \\
&\quad -0.496 \bigr)
\end{aligned}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: logit z = -0.496 + LINEAR CORE + 0.117\*st\_ischaemia\_gate\_output + ischaemia\_pathway readouts (0.135\*290 + 0.165\*1834 + 0.097\*1987); prediction = sigmoid(z) (the genome output node is ReLU but the evaluator overrides it with sigmoid for binary classification, so it is a logistic score). LINEAR CORE (18 direct input-\>output edges, z-scored inputs): ca +1.51, sex:0 -1.08, slope:1 -0.90, chol +0.53, cp:1 -0.53, trestbps +0.49, cp:3 -0.47, thal:3 -0.43, exang:1 +0.43, slope:3 -0.41, thal:6 -0.38, cp:4 +0.33, fbs:0 +0.28, thalach -0.25, restecg:0 -0.18, exang:0 +0.15, age +0.11, thal:7 -0.07. Signs match cardiology: more fluoroscopy-coloured vessels, male sex, high cholesterol/BP, exertional angina and asymptomatic chest pain raise risk; upsloping ST (slope:1), normal thallium (thal:3), typical/non-anginal pain and high max heart rate lower it. 6 inputs are unused (sex:1, fbs:1, restecg:1, restecg:2, slope:2, thal:4 - mostly redundant one-hot complements or rare categories). COMPOSITION: purely additive in the logit - the two children add to the linear core. Child 1 (st\_ischaemia\_gate\_output) adds a ReLU ischaemia score (oldpeak, thal:7, restecg:0); child 2 (ischaemia\_pathway) adds a BP-adjusted, convexly amplified version of the same score, plus an age term that is active only when the score is quiet. oldpeak and thal:7 influence the output ONLY through these gates (oldpeak has no direct edge). EVIDENCE (train, n=242): full model AUC 0.942 / acc 0.876; linear core alone AUC 0.932 (nonlinear parts add +0.010 train AUC); logit SD of linear core 3.18 vs 0.78 for all nonlinear readouts combined. Removing individual readouts: 290 -0.002, 1834 -0.0016, 204\_a -0.0016, 1987 -0.0007, whole pathway -0.0055 AUC. Whole-model SHAP (prob scale): ca 0.139, sex:0 0.105, slope:1 0.099, thal:3 0.045, cp:3 0.041, exang:1 0.039, oldpeak 0.038, chol 0.033, cp:4 0.032. The model is essentially a logistic regression with one small nonlinear ischaemia correction.

**Evidence (4 records, DB):**

- `performance_metrics`: Complete structural and compositional coverage with 10 function-preserving splits; explained network computes the same function as the evolved one.
- `ablation_study`: The linear core carries most of the signal; the nonlinear ischaemia machinery adds \~+0.01 train AUC. Test split not inspected (protocol).
- `shap_importance`: Top drivers ca, sex:0, slope:1 are all direct linear-core edges; oldpeak (gate-only input) ranks 7th.
- `analytical_formula`: Closed-form root: an 18-term linear logit plus four additive nonlinear readouts from the two children.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// performance_metrics
{
 "split": "train",
 "train_n": 242,
 "coverage": {
  "edges": 39,
  "structural": 1.0,
  "compositional": 1.0,
  "candidate_nodes": 43
 },
 "split_id": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
 "train_acc": 0.876,
 "train_auc": 0.942,
 "operations": {
  "node_splits": 10,
  "function_changing": 0,
  "identity_node_insertions": 0
 },
 "annotations": {
  "leaf": 3,
  "composition": 2
 },
 "function_preserved_check": "train AUC/acc/MSE identical before and after all splits"
}
```

```json
// ablation_study
{
 "split": "train",
 "ladder": [
  {
   "config": "full",
   "train_auc": 0.942
  },
  {
   "config": "linear core only",
   "train_auc": 0.9316
  },
  {
   "config": "remove 290 readout",
   "train_auc": 0.9397
  },
  {
   "config": "remove 1834 readout",
   "train_auc": 0.9404
  },
  {
   "config": "remove 204_a readout",
   "train_auc": 0.9404
  },
  {
   "config": "remove 1987 readout",
   "train_auc": 0.9413
  },
  {
   "config": "remove pathway (290,1834,1987)",
   "train_auc": 0.9364
  }
 ],
 "method": "local recomputation of logit decomposition z = linear core + sum(readout weight * node activation); verified sigmoid(z) equals model output to 2e-16; readouts removed without re-fitting",
 "logit_sd": {
  "204_a": 0.26,
  "pathway": 0.53,
  "linear_core": 3.18,
  "all_nonlinear": 0.78
 },
 "corr_nonlinear_vs_linear": 0.39
}
```

```json
// shap_importance
{
 "n": 242,
 "scale": "probability",
 "split": "train",
 "base_value": 0.45169222239986206,
 "top10_mean_abs_shap": {
  "ca": 0.1387,
  "chol": 0.0335,
  "cp:3": 0.0407,
  "cp:4": 0.0325,
  "sex:0": 0.1048,
  "thal:3": 0.0451,
  "exang:1": 0.039,
  "oldpeak": 0.0381,
  "slope:1": 0.0995,
  "trestbps": 0.0268
 }
}
```

```json
// analytical_formula
{
 "note": "formula shows the genome output activation (ReLU); evaluation/training for binary classification overrides it with sigmoid, so the effective model is p = sigmoid(argument)",
 "composition": "additive: linear core + 0.117*st_ischaemia_gate_output + 0.135*290 + 0.165*1834 + 0.097*1987 - 0.496"
}
```

</details>

### A.1 Leaf: `st_ischaemia_gate_output`

|  |  |
|---|---|
| Annotation id | `ann_36` |
| Entry nodes | -12\_c (restecg:0\_b\_c), -18\_a (oldpeak\_a), -26\_c (thal:7\_b\_c) |
| Exit nodes | 204\_a (ischaemia\_gate\_out) |
| Internal / exit nodes in its own subgraph | 204\_a (ischaemia\_gate\_out) |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.0217 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.55 x_{oldpeak\_a} \\
&\quad + 0.937 x_{restecg:0\_b\_c} \\
&\quad + 1.13 x_{thal:7\_b\_c} \\
&\quad + 0.276 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hidden gate 204 = ReLU(2.554\*oldpeak + 1.133\*thal:7 + 0.937\*restecg:0 + 0.276): a one-sided ischaemia-evidence detector that switches on for patients with exercise-induced ST depression and/or a reversible thallium defect, normal resting ECG adding a little. EVIDENCE (train, n=242): hard-zero for 52%, active for 48%; mean \|grad\| oldpeak 1.23 \> thal:7 0.55 \> restecg:0 0.45; oldpeak alone correlates 0.88 with the gate. This is the output-facing copy 204\_a (bit-identical to 204\_b); it reads out directly to the output with +0.117, raising predicted risk.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of a single ReLU gate over three inputs; oldpeak weight is 2.3x/2.7x the others.
- `empirical_viz`: Genuine gate on training data: hard-zero for 52% of patients, active for 48%. Mean \|grad\| ranks oldpeak \>\> thal:7 \> restecg:0; oldpeak alone correlates 0.88 with the gate output.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "204_a = relu(2.554 oldpeak + 1.133 thal:7 + 0.937 restecg:0 + 0.276)",
 "readout": "204_a -> 0 weight +0.117",
 "interpretation": "one-sided ischaemia-evidence detector dominated by ST depression (oldpeak); all weights positive so monotone increasing in each input"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "activation_profile": {
  "max": 12.5197392864441,
  "min": 0.0,
  "std": 2.2446579847663157,
  "mean": 1.3755107515918263,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.5165289256198347,
  "activation_rate": 0.4834710743801653
 },
 "sensitivity_mean_abs_grad": {
  "thal:7": 0.5477218771190896,
  "oldpeak": 1.2348670397124366,
  "restecg:0": 0.45294564747724697
 },
 "line_oldpeak_vs_gate_correlation": 0.8796682565541298
}
```

</details>

### A.2 Composition: `ischaemia_pathway`

|  |  |
|---|---|
| Annotation id | `ann_39` |
| Entry nodes | -1\_b (age\_b), -5 (cp:2), -6\_b (cp:3\_b), -8\_b (trestbps\_b), -25\_b (thal:6\_b), -12\_d (restecg:0\_b\_d), -18\_b (oldpeak\_b), -26\_d (thal:7\_b\_d), -25\_c (thal:6\_c) |
| Exit nodes | 290 (bp\_adjusted\_ischaemia), 1834 (strong\_ischaemia\_amp), 1987 (age\_if\_no\_ischaemia) |
| Internal / exit nodes in its own subgraph | 290 (bp\_adjusted\_ischaemia), 1834 (strong\_ischaemia\_amp), 1987 (age\_if\_no\_ischaemia) |
| Children | st\_ischaemia\_gate\_bp, thal6\_linear\_passthrough |
| Dimensionality (in → out, incl. descendants) | 9 → 3 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.151 x_{cp:2} \\
&\quad + 0.0167 x_{cp:3\_b} \\
&\quad + 0.0459 x_{thal:6\_b} \\
&\quad - 0.817 x_{trestbps\_b} \\
&\quad + 1.09 \operatorname{st\_ischaemia\_gate\_bp}{\left(x_{restecg:0\_b\_d},x_{oldpeak\_b},x_{thal:7\_b\_d} \right)} \\
&\quad - 0.0132 \operatorname{thal\_6\_linear\_passthrough}{\left(x_{thal:6\_c} \right)} \\
&\quad -0.485 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.941 \operatorname{relu}{\left(0.151 x_{cp:2} + 0.0167 x_{cp:3\_b} + 0.0459 x_{thal:6\_b} - 0.817 x_{trestbps\_b} + 1.09 \operatorname{st\_ischaemia\_gate\_bp}{\left(x_{restecg:0\_b\_d},x_{oldpeak\_b},x_{thal:7\_b\_d} \right)} - 0.0132 \operatorname{thal\_6\_linear\_passthrough}{\left(x_{thal:6\_c} \right)} - 0.485 \right)} \\
&\quad -1.67 \bigr)
\end{aligned} \\ \begin{aligned}
y_2 &= \operatorname{relu}\bigl( 0.88 x_{age\_b} \\
&\quad - 1.58 \operatorname{relu}{\left(0.151 x_{cp:2} + 0.0167 x_{cp:3\_b} + 0.0459 x_{thal:6\_b} - 0.817 x_{trestbps\_b} + 1.09 \operatorname{st\_ischaemia\_gate\_bp}{\left(x_{restecg:0\_b\_d},x_{oldpeak\_b},x_{thal:7\_b\_d} \right)} - 0.0132 \operatorname{thal\_6\_linear\_passthrough}{\left(x_{thal:6\_c} \right)} - 0.485 \right)} \\
&\quad + 0.179 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Serial composition of the ischaemia gate into a BP-adjusted gate and its readouts. (1) Gate 290 = ReLU(1.088\*st\_ischaemia\_gate\_bp - 0.817\*trestbps + 0.151\*cp:2 + 0.017\*cp:3 + 0.046\*thal:6 - 0.013\*thal6\_linear\_passthrough - 0.485): it passes on the ischaemia evidence of 204\_b, offset downward by resting blood pressure (high-BP patients need more ST-depression evidence to open it). (2) 1834 = ReLU(0.941\*290 - 1.672): high-threshold amplifier, fires only when 290 \> 1.78, making the pathway convex. (3) 1987 = ReLU(0.880\*age - 1.585\*290 + 0.179): an age term switched OFF by the gate, so age adds risk only for patients in whom 290 is quiet. Output contribution 0.135\*290 + 0.165\*1834 + 0.097\*1987. EVIDENCE (train, n=242): 290 active 56% (100/135 of its active cases have 204 on; 35 open from low BP alone, mean predicted risk 0.18); 1834 active 26%, only when 204 and 290 are both on, and 78% of those patients are diseased; 1987 active in 64% of patients with 290 quiet vs 13% with 290 active. The thal:6 chain is functionally inert. Interpretation: the network has exactly one nonlinear mechanism - an ischaemia pathway that amplifies strong ST-depression/defect evidence and uses age as a fallback signal when that evidence is absent.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the composed pathway; 290 is a serial composition over the child gate, and 1834/1987 are readouts of 290 that respectively amplify strong ischaemia evidence and switch off the age term.
- `empirical_viz`: Empirics agree with the formula: 1834 fires only in the regime where both 204 and 290 are on (26% of patients, 78% of them diseased) - an amplifier for strong ischaemia evidence. 1987 is on for 64% of patients with 290 quiet but only 13% when 290 is active, confirming that age acts as a fallback risk signal that the ischaemia pathway switches off. 290 is opened by 204 in 100/135 of its active cases; 35 cases open it from low resting BP alone and these patients have low predicted risk (mean 0.18).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "get_formula without force returned null; with force the non-collapsed latex omits child terms (tool bug); latex_collapsed is correct",
 "plain": [
  "290 = relu(1.088*G204 - 0.817 trestbps + 0.151 cp:2 + 0.017 cp:3 + 0.046 thal:6 - 0.0132*P2116 - 0.485)  (P2116 = 1.003 thal:6 + 2.165 on data => net thal:6 coef 0.033, bias -0.514)",
  "1834 = relu(0.941*290 - 1.672)  (on iff 290 > 1.777)",
  "1987 = relu(0.880 age - 1.585*290 + 0.179)"
 ],
 "readout": "logit += 0.135*290 + 0.165*1834 + 0.097*1987 (204_a, the other copy of the ischaemia gate, adds 0.117*204 via the root)",
 "slope_of_logit_in_290": "0.135 baseline; +0.155 extra when 1834 on; -0.154 when 1987 on (age gate cancels the direct readout)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "tool_note": "composition-level function evaluation (sensitivity/line by annotation_id) ignores child subgraphs, so node_id evaluations were used",
 "regime_map": {
  "both off": 90,
  "290 on & 204 on": 100,
  "290 off & 204 on": 17,
  "290 on & 204 off": 35,
  "1987 on given 290 on": "17/135 (13%)",
  "1987 on given 290 off": "68/107 (64%)",
  "1834-on regime class balance": "49 positive / 14 negative, mean pred 0.78",
  "1834 on (always with 204&290 on)": 63
 },
 "activation_rates": {
  "204": 0.483,
  "290": 0.558,
  "1834": 0.26,
  "1987": 0.351,
  "2116": 1.0,
  "3671": 1.0
 },
 "sensitivity_mean_abs_grad": {
  "290": {
   "cp:2": 0.084,
   "cp:3": 0.009,
   "thal:7": 0.509,
   "oldpeak": 1.149,
   "trestbps": 0.456,
   "restecg:0": 0.421,
   "thal:6 (direct+chain)": 0.033
  },
  "1834": {
   "cp:2": 0.037,
   "thal:7": 0.302,
   "oldpeak": 0.681,
   "trestbps": 0.2,
   "restecg:0": 0.25
  },
  "1987": {
   "age": 0.309,
   "thal:7": 0.087,
   "oldpeak": 0.188,
   "trestbps": 0.091,
   "restecg:0": 0.073
  }
 }
}
```

</details>

#### A.2.1 Leaf: `st_ischaemia_gate_bp`

|  |  |
|---|---|
| Annotation id | `ann_37` |
| Entry nodes | -12\_d (restecg:0\_b\_d), -18\_b (oldpeak\_b), -26\_d (thal:7\_b\_d) |
| Exit nodes | 204\_b (ischaemia\_gate\_to\_bp) |
| Internal / exit nodes in its own subgraph | 204\_b (ischaemia\_gate\_to\_bp) |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.0217 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.55 x_{oldpeak\_b} \\
&\quad + 0.937 x_{restecg:0\_b\_d} \\
&\quad + 1.13 x_{thal:7\_b\_d} \\
&\quad + 0.276 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hidden gate 204 = ReLU(2.554\*oldpeak + 1.133\*thal:7 + 0.937\*restecg:0 + 0.276): a one-sided ischaemia-evidence detector that switches on for patients with exercise-induced ST depression and/or a reversible thallium defect, normal resting ECG adding a little. EVIDENCE (train, n=242): hard-zero for 52%, active for 48%; mean \|grad\| oldpeak 1.23 \> thal:7 0.55 \> restecg:0 0.45; oldpeak alone correlates 0.88 with the gate. This is copy 204\_b (bit-identical to 204\_a), routed into gate 290 with +1.088, where it is the dominant excitatory input.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of a single ReLU gate over three inputs; oldpeak weight is 2.3x/2.7x the others.
- `empirical_viz`: Genuine gate on training data: hard-zero for 52% of patients, active for 48%. Mean \|grad\| ranks oldpeak \>\> thal:7 \> restecg:0; oldpeak alone correlates 0.88 with the gate output.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "204_b = relu(2.554 oldpeak + 1.133 thal:7 + 0.937 restecg:0 + 0.276)",
 "readout": "204_b -> 290 weight +1.088",
 "interpretation": "one-sided ischaemia-evidence detector dominated by ST depression (oldpeak); all weights positive so monotone increasing in each input"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "activation_profile": {
  "max": 12.5197392864441,
  "min": 0.0,
  "std": 2.2446579847663157,
  "mean": 1.3755107515918263,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.5165289256198347,
  "activation_rate": 0.4834710743801653
 },
 "sensitivity_mean_abs_grad": {
  "thal:7": 0.5477218771190896,
  "oldpeak": 1.2348670397124366,
  "restecg:0": 0.45294564747724697
 },
 "line_oldpeak_vs_gate_correlation": 0.8796682565541298
}
```

</details>

#### A.2.2 Leaf: `thal6_linear_passthrough`

|  |  |
|---|---|
| Annotation id | `ann_38` |
| Entry nodes | -25\_c (thal:6\_c) |
| Exit nodes | 2116 (thal6\_relay2) |
| Internal / exit nodes in its own subgraph | 3671 (thal6\_relay1), 2116 (thal6\_relay2) |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0195 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 \operatorname{relu}{\left(1.0 x_{thal:6\_c} + 2.12 \right)} \\
&\quad + 0.0355 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Serial chain thal:6 -\> 3671 -\> 2116: 3671 = ReLU(1.000\*thal:6 + 2.123), 2116 = ReLU(1.003\*3671 + 0.036). Since thal:6 \>= -0.257 on the data both ReLUs are always active (100% on train), so 2116 = 1.003\*thal:6 + 2.165 exactly (r = 1.000 on train). It feeds gate 290 with weight -0.013, contributing -0.013\*thal:6 - 0.029 to its pre-activation: an always-on affine relay that is functionally a vestige.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form reduces to an affine map over the data domain; the two ReLUs never clip.
- `empirical_viz`: Both nodes are 100% active on training data and 2116 is perfectly linear in thal:6 (r = 1.000). The chain is a linear pass-through whose downstream weight (-0.013) makes it practically inert.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "2116 = relu(1.003*relu(0.9996 thal:6 + 2.123) + 0.036)",
 "readout": "2116 -> 290 weight -0.0132 => net -0.0132 thal:6 - 0.0286 into 290",
 "simplified_on_data": "2116 = 1.003 thal:6 + 2.165 (both ReLUs always active since thal:6 >= -0.257)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "activation_profile_2116": {
  "max": 6.065,
  "min": 1.907,
  "activation_rate": 1.0
 },
 "activation_profile_3671": {
  "max": 6.011,
  "min": 1.866,
  "activation_rate": 1.0
 },
 "line_thal6_vs_2116_correlation": 1.0
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 5 |
| rename\_node | 33 |
| split\_node | 10 |

10 splits, 0 identity nodes, 0 consolidations (1 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -1 into -1\_a, -1\_b. Final copies: -1\_a → heart\_disease\_logit (entry) \| -1\_b → ischaemia\_pathway (entry)
- `#27` split -6 into -6\_a, -6\_b. Final copies: -6\_a → heart\_disease\_logit (entry) \| -6\_b → ischaemia\_pathway (entry)
- `#28` split -8 into -8\_a, -8\_b. Final copies: -8\_a → heart\_disease\_logit (entry) \| -8\_b → ischaemia\_pathway (entry)
- `#29` split -12 into -12\_a, -12\_b. Final copies: -12\_a → heart\_disease\_logit (entry) \| -12\_c → st\_ischaemia\_gate\_output (entry) \| -12\_d → st\_ischaemia\_gate\_bp (entry); ischaemia\_pathway (entry)
- `#30` split -25 into -25\_a, -25\_b, -25\_c. Final copies: -25\_a → heart\_disease\_logit (entry) \| -25\_b → ischaemia\_pathway (entry) \| -25\_c → thal6\_linear\_passthrough (entry); ischaemia\_pathway (entry)
- `#31` split -26 into -26\_a, -26\_b. Final copies: -26\_a → heart\_disease\_logit (entry) \| -26\_c → st\_ischaemia\_gate\_output (entry) \| -26\_d → st\_ischaemia\_gate\_bp (entry); ischaemia\_pathway (entry)
- `#32` split 204 into 204\_a, 204\_b. Final copies: 204\_a → st\_ischaemia\_gate\_output (exit) \| 204\_b → st\_ischaemia\_gate\_bp (exit)
- `#33` split -18 into -18\_a, -18\_b. Final copies: -18\_a → st\_ischaemia\_gate\_output (entry) \| -18\_b → st\_ischaemia\_gate\_bp (entry); ischaemia\_pathway (entry)
- `#34` split -12\_b into -12\_c, -12\_d. Final copies: -12\_c → st\_ischaemia\_gate\_output (entry) \| -12\_d → st\_ischaemia\_gate\_bp (entry); ischaemia\_pathway (entry)
- `#35` split -26\_b into -26\_c, -26\_d. Final copies: -26\_c → st\_ischaemia\_gate\_output (entry) \| -26\_d → st\_ischaemia\_gate\_bp (entry); ischaemia\_pathway (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.78e-15 | 1.78e-15 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.0217 | 0.0217 | 0.0217 | agree |
| Numeric-function max abs error | n/a | 0 | 0 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 3 / 2 | n/a | 3 / 2 | agree |
| Operation counts | 10 splits, 0 identity | {"annotate": 5, "rename\_node": 33, "split\_node": 10} | {"annotate": 5, "rename\_node": 33, "split\_node": 10} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `st_ischaemia_gate_output` | 0.0217 | 0 |
| `st_ischaemia_gate_bp` | 0.0217 | 0 |
| `thal6_linear_passthrough` | 0.01951 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Validator only treats direct-child nodes as internal, forcing a split of gate 204 (and its 3 inputs) because it reads out to both the output and gate 290. Nodes 290/1834/1987 take inputs from several annotations so live only in the ischaemia\_pathway composition. thal:6 chain 3671-\>2116 is affine and near-inert (weight -0.013 into 290).

**Tool notes (record):** get\_formula composition latex drops child terms (latex\_collapsed correct); viz tools with annotation\_id on compositions zero the children (use node\_id); validator ignores grandchildren; detect\_splits misses input fan-out across annotations; compute\_shap tqdm on stderr; get\_formula shows output ReLU but model applies sigmoid; tool compositional\_coverage always 1.0.

## What to look at

- Largest closed-form error: `st_ischaemia_gate_output` (0.0217), then `st_ischaemia_gate_bp` (0.0217). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `thal6_linear_passthrough`: *“Since thal:6 \>= -0.257 on the data both ReLUs are always active (100% on train), so 2116 = 1.003\*thal:6 + 2.165 exactly (r = 1.000 on train).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `st_ischaemia_gate_output`: *“This is the output-facing copy 204\_a (bit-identical to 204\_b);”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 1 hidden-node split(s), which decide how a unit's roles are divided: 204 → 204\_a → st\_ischaemia\_gate\_output (exit) \| 204\_b → st\_ischaemia\_gate\_bp (exit). Do the copies really play the different roles the annotations give them?
- Explained with protocol v2 tooling (the pilot): get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s5.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s5', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
