# Verification packet 25 · heart-s7

Pre-registered verification order: 25 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/7c355ee7-77ad-426c-8a6d-9df81d1c65b3?experiment=1ca745d9-85c4-419d-8e68-328cac99d542&name=10seed-heart-s7), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 7 |
| Genome id | `7c355ee7-77ad-426c-8a6d-9df81d1c65b3` |
| Experiment | 10seed-heart-s7 (1ca745d9-85c4-419d-8e68-328cac99d542); best at generation 39 of 40, NEAT key 9858 |
| Test AUC (runs csv) | 0.886 \[95% CI 0.791, 0.959\] |
| Train AUC | 0.947 |
| Size | 22 of 26 inputs connected; 4 hidden nodes; 32 enabled connections; depth 3; 4 skip connections + 19 direct input→output edges |
| Annotations | 3 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 4 of 4 |
| Claude's time | 10.1 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:34:04+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Essentially logistic regression: 19 direct edges (logit SD 4.90; ca +2.61, female -1.57) plus three one-sided ReLU corrections: an ischaemia gate on oldpeak and thal:7 (SD 0.78); a Boolean AND of female sex and exercise angina adding +2.07 logit, cancelling most of the female protective term (SD 0.56); a protective gate on high thalach mostly with non-anginal pain (SD 0.49). Linear core train AUC 0.930 vs 0.947 full; mean-ablating each gate gives 0.938, 0.940, 0.945.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_35` |
| Entry nodes | -1 (age), -2\_a (sex:0\_a), -3 (sex:1), -4 (cp:1), -6\_a (cp:3\_a), -7 (cp:4), -8 (trestbps), -9 (chol), -10\_a (fbs:0\_a), -12 (restecg:0), -15\_a (thalach\_a), -16 (exang:0), -17\_a (exang:1\_a), -19 (slope:1), -20 (slope:2), -22 (ca), -24 (thal:4), -25 (thal:6), -26\_a (thal:7\_a), -5 (cp:2), -11 (fbs:1), -13 (restecg:1), -14 (restecg:2), -10\_b (fbs:0\_b), -18 (oldpeak), -21 (slope:3), -26\_b (thal:7\_b), -2\_b (sex:0\_b), -17\_b (exang:1\_b), -6\_b (cp:3\_b), -15\_b (thalach\_b), -23 (thal:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | st\_depression\_gate, female\_exang\_and\_gate, thalach\_cp3\_protective\_gate |
| Dimensionality (in → out, incl. descendants) | 32 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{1.97 e^{- 0.11 x_{age} - 2.61 x_{ca} - 0.87 x_{chol} + 0.785 x_{cp:1} + 0.743 x_{cp:3\_a} - 0.312 x_{cp:4} + 0.0506 x_{exang:0} + 0.0636 x_{exang:1\_a} - 0.627 x_{fbs:0\_a} + 0.096 x_{restecg:0} + 1.57 x_{sex:0\_a} - 0.39 x_{sex:1} + 0.833 x_{slope:1} - 0.743 x_{slope:2} - 0.383 x_{thal:4} + 0.351 x_{thal:6} - 0.49 x_{thal:7\_a} + 0.47 x_{thalach\_a} - 0.519 x_{trestbps} - 1.73 \operatorname{female\_exang\_and\_gate}{\left(x_{sex:0\_b},x_{exang:1\_b} \right)} - 0.746 \operatorname{st\_depression\_gate}{\left(x_{fbs:0\_b},x_{oldpeak},x_{slope:3},x_{thal:7\_b} \right)} + 0.918 \operatorname{thalach\_cp3\_protective\_gate}{\left(x_{cp:3\_b},x_{thalach\_b},x_{thal:3} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}z = -0.68 + 0.11 x_{age} + 2.61 x_{ca} + 0.87 x_{chol} - 0.785 x_{cp:1} - 0.743 x_{cp:3} + 0.312 x_{cp:4} - 0.0506 x_{exang:0} - 0.0636 x_{exang:1} + 0.627 x_{fbs:0} - 0.096 x_{restecg:0} - 1.57 x_{sex:0} + 0.39 x_{sex:1} - 0.833 x_{slope:1} + 0.743 x_{slope:2} + 0.383 x_{thal:4} - 0.351 x_{thal:6} + 0.49 x_{thal:7} - 0.47 x_{thalach} + 0.519 x_{trestbps} + 1.73\,\mathrm{female\_exang\_and\_gate} + 0.746\,\mathrm{st\_depression\_gate} - 0.918\,\mathrm{thalach\_cp3\_protective\_gate};\ \hat p = \sigma(z)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root hypothesis: an additive composition that is essentially logistic regression plus three one-sided ReLU corrections. Logit z = -0.680 + LINEAR CORE + 0.746\*st\_depression\_gate + 1.730\*female\_exang\_and\_gate - 0.918\*thalach\_cp3\_protective\_gate, and the prediction is sigmoid(z) (get\_formula renders the output as ReLU, but for this binary task the model applies a sigmoid to the output pre-activation). LINEAR CORE = 19 direct input-\>output edges, dominated by ca (+2.607, number of vessels), sex:0 (-1.573, women lower risk), chol (+0.870), slope:1 (-0.833, upsloping ST protective), cp:1 (-0.785), slope:2 (+0.743), cp:3 (-0.743), fbs:0 (+0.627), trestbps (+0.519), thal:7 (+0.490), thalach (-0.470), sex:1 (+0.390), thal:4 (+0.383), thal:6 (-0.351), cp:4 (+0.312); age, restecg:0, exang:0/1 near zero. cp:2, fbs:1, restecg:1, restecg:2 are unused. Corrections: (a) ischaemia evidence (ST depression, reversible thal defect) adds risk; (b) women with exercise angina lose most of their sex protection (AND gate, +2.07 logit); (c) high max heart rate, especially with non-anginal pain, removes extra risk. Evidence (train, n=242): accuracy 0.876, AUC 0.947. Linear core logit SD 4.90 vs gates 0.78 / 0.56 / 0.49. Linear core alone AUC 0.930; mean-ablating each gate gives AUC 0.938 / 0.940 / 0.945, so together the nonlinear parts add \~+0.017 train AUC. SHAP (probability scale): ca 0.170 \>\> sex:0 0.080 \> slope:1 0.063 \> cp:3 0.058 \> slope:2 0.053 \> thal:7 0.050 \> chol 0.038 \> thalach 0.034 \> oldpeak 0.029. Function preservation of the refactored model 1.8e-15; all leaf closed forms exact up to 3-s.f. display rounding (max 0.027).

**Evidence (4 records, DB):**

- `analytical_formula`: (no narrative)
- `performance_metrics`: (no narrative)
- `ablation_study`: (no narrative)
- `shap_importance`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true latex_collapsed; output shown as relu by tool but model applies sigmoid"
}
```

```json
// performance_metrics
{
 "n": 242,
 "rmse": 0.293,
 "split": "train",
 "auc_roc": 0.9467,
 "accuracy": 0.876
}
```

```json
// ablation_study
{
 "split": "train",
 "method": "replace node by its train mean, local forward pass on in-memory copy",
 "logit_sd": {
  "linear_core": 4.9,
  "st_depression_gate": 0.775,
  "female_exang_and_gate": 0.558,
  "thalach_cp3_protective_gate": 0.487
 },
 "auc_full_logit": 0.9467,
 "auc_linear_core_only": 0.9299,
 "auc_ablate_st_depression_gate": 0.9382,
 "auc_ablate_female_exang_and_gate": 0.9405,
 "auc_ablate_thalach_cp3_protective_gate": 0.9446
}
```

```json
// shap_importance
{
 "n": 242,
 "zero": [
  "cp:2",
  "fbs:1",
  "restecg:0",
  "restecg:1",
  "restecg:2",
  "exang:0",
  "thal:3"
 ],
 "split": "train",
 "output": "probability (base 0.441)",
 "top_mean_abs_shap": {
  "ca": 0.17,
  "chol": 0.038,
  "cp:1": 0.026,
  "cp:3": 0.058,
  "sex:0": 0.08,
  "thal:7": 0.05,
  "oldpeak": 0.029,
  "slope:1": 0.063,
  "slope:2": 0.053,
  "thalach": 0.034,
  "trestbps": 0.021
 }
}
```

</details>

### A.1 Leaf: `st_depression_gate`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | -10\_b (fbs:0\_b), -18 (oldpeak), -21 (slope:3), -26\_b (thal:7\_b) |
| Exit nodes | 557 |
| Internal / exit nodes in its own subgraph | 557 |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.0269 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.417 x_{fbs:0\_b} \\
&\quad + 1.28 x_{oldpeak} \\
&\quad - 1.08 x_{slope:3} \\
&\quad + 0.559 x_{thal:7\_b} \\
&\quad + 0.085 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = \operatorname{relu}(-0.417 x_{fbs:0} + 1.28 x_{oldpeak} - 1.08 x_{slope:3} + 0.559 x_{thal:7} + 0.085)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: hidden gate 557 = ReLU(1.283\*oldpeak + 0.559\*thal:7 - 1.076\*slope:3 - 0.417\*fbs:0 + 0.085) is a one-sided ischaemia-evidence detector. It switches on mainly with exercise-induced ST depression (oldpeak), more readily when there is a reversible thal defect (thal:7), is pushed down by a downsloping ST segment (slope:3, which is clinically counter-intuitive; it is rare, 7%, and the direct linear core has no slope:3 term, so this is probably how the network absorbs slope:3 at all), and slightly by normal fasting sugar (fbs:0). It enters the logit with weight +0.746, so it only ever adds risk. Evidence (train, n=242): closed form exact (single ReLU); active on 47.9% of patients (83% with thal:7, 27% without); max 4.72, logit contribution SD 0.78; sensitivity oldpeak 0.62 \> slope:3 0.52 \> thal:7 0.27 \> fbs:0 0.20; replacing it by its mean drops train AUC of the logit from 0.947 to 0.938, the largest of the three gates.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

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
 "max": 4.72,
 "split": "train",
 "active_frac": 0.479,
 "sensitivity": {
  "fbs:0": 0.2,
  "thal:7": 0.268,
  "oldpeak": 0.615,
  "slope:3": 0.516
 },
 "logit_contrib_sd": 0.775,
 "active_frac_thal7_absent": 0.27,
 "active_frac_thal7_present": 0.833,
 "ablation_to_mean_train_auc_logit": [
  0.9467,
  0.9382
 ]
}
```

</details>

### A.2 Leaf: `female_exang_and_gate`

|  |  |
|---|---|
| Annotation id | `ann_33` |
| Entry nodes | -2\_b (sex:0\_b), -17\_b (exang:1\_b) |
| Exit nodes | 1239 |
| Internal / exit nodes in its own subgraph | 1239 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 6.55e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.903 x_{exang:1\_b} \\
&\quad + 0.571 x_{sex:0\_b} \\
&\quad -0.853 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = \operatorname{relu}(0.903 x_{exang:1} + 0.571 x_{sex:0} - 0.853)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: hidden gate 1239 = ReLU(0.571\*sex:0 + 0.903\*exang:1 - 0.853) is a Boolean AND of female sex (sex:0) and exercise-induced angina (exang:1). Evidence (train, n=242): truth table of the standardised one-hot inputs gives activation 1.199 for women with exang (n=19, positive rate 63%) and exactly 0 for the other three cells (male/no exang n=96 pos 40%; male/exang n=63 pos 76%; female/no exang n=64 pos 14%; the male+exang pre-activation is -0.004, just below the threshold). Weighted +1.730 into the logit it adds +2.07, which cancels most of the direct female protective term (-1.573\*sex:0 contributes -2.18 for women vs +1.14 for men): the network learns that exercise angina erases the female advantage. Active on 7.9% of patients, logit contribution SD 0.56; mean-ablation drops train AUC 0.947 -\> 0.940. Closed form exact.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

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
 "sensitivity": {
  "sex:0": 0.095,
  "exang:1": 0.208
 },
 "truth_table": {
  "male,exang": {
   "n": 63,
   "act": 0.0,
   "pos_rate": 0.762
  },
  "female,exang": {
   "n": 19,
   "act": 1.199,
   "pos_rate": 0.632
  },
  "male,no_exang": {
   "n": 96,
   "act": 0.0,
   "pos_rate": 0.396
  },
  "female,no_exang": {
   "n": 64,
   "act": 0.0,
   "pos_rate": 0.141
  }
 },
 "logit_contrib_sd": 0.558,
 "ablation_to_mean_train_auc_logit": [
  0.9467,
  0.9405
 ]
}
```

</details>

### A.3 Leaf: `thalach_cp3_protective_gate`

|  |  |
|---|---|
| Annotation id | `ann_34` |
| Entry nodes | -6\_b (cp:3\_b), -15\_b (thalach\_b), -23 (thal:3) |
| Exit nodes | 2124 |
| Internal / exit nodes in its own subgraph | 3818, 2124 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00576 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.763 x_{cp:3\_b} \\
&\quad + 1.34 x_{thalach\_b} \\
&\quad - 0.108 \operatorname{relu}{\left(1.0 x_{thal:3} - 0.191 \right)} \\
&\quad -0.932 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = \operatorname{relu}(0.763 x_{cp:3} + 1.34 x_{thalach} - 0.108\,\operatorname{relu}(1.0 x_{thal:3} - 0.191) - 0.932)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: a protective high-exercise-capacity gate. Serial chain thal:3 -\> 3818 -\> 2124. 3818 = ReLU(1.003\*thal:3 - 0.192) is a normal-thallium indicator taking only the values 0 (abnormal thal) and 0.702 (normal thal, 56% of patients). 2124 = ReLU(1.338\*thalach + 0.763\*cp:3 - 0.108\*3818 - 0.932) fires when maximum heart rate is high, much more readily with non-anginal chest pain (cp:3): active on 80% of cp:3 patients but 10.5% of others (30.6% overall, max 2.88). It enters the logit with weight -0.918, so it only ever lowers risk, adding a hinge to the direct linear thalach (-0.470) and cp:3 (-0.743) terms. The thal:3 branch only moves the threshold by -0.076 (sensitivity 0.025 vs thalach 0.41, cp:3 0.23) and is nearly inert. Evidence (train, n=242): closed form exact; logit contribution SD 0.49; mean-ablation drops train AUC only 0.947 -\> 0.945, the weakest of the three gates.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

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
 "max": 2.88,
 "split": "train",
 "sensitivity": {
  "cp:3": 0.233,
  "thal:3": 0.025,
  "thalach": 0.409
 },
 "node3818_values": [
  0.0,
  0.702
 ],
 "logit_contrib_sd": 0.487,
 "node2124_active_frac": 0.306,
 "node3818_active_frac": 0.558,
 "active_frac_cp3_absent": 0.105,
 "active_frac_cp3_present": 0.8,
 "ablation_to_mean_train_auc_logit": [
  0.9467,
  0.9446
 ]
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 4 |
| rename\_node | 26 |
| split\_node | 6 |

6 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -10 into -10\_a, -10\_b. Final copies: -10\_a → heart\_disease\_logit (entry) \| -10\_b → st\_depression\_gate (entry); heart\_disease\_logit (entry)
- `#27` split -26 into -26\_a, -26\_b. Final copies: -26\_a → heart\_disease\_logit (entry) \| -26\_b → st\_depression\_gate (entry); heart\_disease\_logit (entry)
- `#28` split -2 into -2\_a, -2\_b. Final copies: -2\_a → heart\_disease\_logit (entry) \| -2\_b → female\_exang\_and\_gate (entry); heart\_disease\_logit (entry)
- `#29` split -17 into -17\_a, -17\_b. Final copies: -17\_a → heart\_disease\_logit (entry) \| -17\_b → female\_exang\_and\_gate (entry); heart\_disease\_logit (entry)
- `#30` split -6 into -6\_a, -6\_b. Final copies: -6\_a → heart\_disease\_logit (entry) \| -6\_b → thalach\_cp3\_protective\_gate (entry); heart\_disease\_logit (entry)
- `#31` split -15 into -15\_a, -15\_b. Final copies: -15\_a → heart\_disease\_logit (entry) \| -15\_b → thalach\_cp3\_protective\_gate (entry); heart\_disease\_logit (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.78e-15 | 1.78e-15 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.02693 | 0.02693 | 0.02693 | agree |
| Numeric-function max abs error | n/a | 8.88e-16 | 8.88e-16 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 3 / 1 | n/a | 3 / 1 | agree |
| Operation counts | 6 splits, 0 identity | {"annotate": 4, "rename\_node": 26, "split\_node": 6} | {"annotate": 4, "rename\_node": 26, "split\_node": 6} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `st_depression_gate` | 0.02693 | 8.88e-16 |
| `thalach_cp3_protective_gate` | 0.005763 | 4.44e-16 |
| `female_exang_and_gate` | 6.55e-04 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. 4 hidden nodes form 3 independent one-sided ReLU gates feeding the output. 6 input splits (fbs:0, thal:7, sex:0, exang:1, cp:3, thalach). Weak spots: thal:3-\>3818 branch nearly inert; in gate 557 slope:3 suppresses risk (clinically counter-intuitive; rare 7%, no direct edge) - reported as what the network computes. Unconnected inputs (cp:2, fbs:1, restecg:1, restecg:2) added to root entries.

**Tool notes (record):** get\_formula renders output as ReLU. compute\_shap at max\_samples=100000 \>2 min. SHAP gives exactly 0 for restecg:0 and exang:0 despite small nonzero direct weights. child\_annotation\_ids must be names.

## What to look at

- Largest closed-form error: `st_depression_gate` (0.0269). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `thalach_cp3_protective_gate`: *“The thal:3 branch only moves the threshold by -0.076 (sensitivity 0.025 vs thalach 0.41, cp:3 0.23) and is nearly inert.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `female_exang_and_gate`: *“Hypothesis: hidden gate 1239 = ReLU(0.571\*sex:0 + 0.903\*exang:1 - 0.853) is a Boolean AND of female sex (sex:0) and exercise-induced angina (exang:1).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 6 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s7.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s7', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
