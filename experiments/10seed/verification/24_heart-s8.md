# Verification packet 24 · heart-s8

Pre-registered verification order: 24 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/4809878b-8b08-4508-91f1-6f5c7207e787?experiment=331f640f-7299-43f7-ae82-0bc172636bb2&name=10seed-heart-s8), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 8 |
| Genome id | `4809878b-8b08-4508-91f1-6f5c7207e787` |
| Experiment | 10seed-heart-s8 (331f640f-7299-43f7-ae82-0bc172636bb2); best at generation 39 of 40, NEAT key 8971 |
| Test AUC (runs csv) | 0.912 \[95% CI 0.829, 0.974\] |
| Train AUC | 0.947 |
| Size | 18 of 26 inputs connected; 5 hidden nodes; 30 enabled connections; depth 4; 4 skip connections + 16 direct input→output edges |
| Annotations | 2 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 5 of 5 |
| Claude's time | 13.1 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:37:00+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Mostly logistic regression: 16 direct edges (logit SD 3.9) plus two additive nonlinear readouts. Main: an exertional-ischaemia hinge 855 = ReLU(trestbps, oldpeak, exang:1 up; cp:1, slope:1 down) with clipped complement 1501 - flat below threshold, steep above (train AUC 0.947 -\> 0.926 without it). Minor serial chain age -\> 1669 (no exertional angina) -\> 3353 (+cp:4), read out -0.515, tempering cp:4 and making age protective for older patients without exertional angina (AUC -0.002). Linear-only 0.924/0.851 vs full 0.947/0.897.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_34` |
| Entry nodes | -1\_a (age\_a), -2 (sex:0), -3 (sex:1), -5 (cp:2), -7\_a (cp:4\_a), -9 (chol), -10 (fbs:0), -15 (thalach), -16\_a (exang:0\_a), -17\_a (exang:1\_a), -18\_a (oldpeak\_a), -19\_a (slope:1\_a), -20 (slope:2), -22 (ca), -24 (thal:4), -26 (thal:7), -6 (cp:3), -11 (fbs:1), -12 (restecg:0), -13 (restecg:1), -14 (restecg:2), -21 (slope:3), -23 (thal:3), -25 (thal:6) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | exertional\_ischaemia\_hinge, age\_noangina\_cp4\_chain |
| Dimensionality (in → out, incl. descendants) | 32 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{2.22 e^{- 0.236 x_{age\_a} - 2.09 x_{ca} - 0.762 x_{chol} - 0.584 x_{cp:2} - 1.21 x_{cp:4\_a} + 0.147 x_{exang:0\_a} + 0.327 x_{exang:1\_a} - 0.515 x_{fbs:0} - 0.229 x_{oldpeak\_a} - 0.124 x_{sex:0} - 1.42 x_{sex:1} - 0.109 x_{slope:1\_a} - 1.06 x_{slope:2} - 0.419 x_{thal:4} - 0.658 x_{thal:7} + 0.204 x_{thalach} + 0.515 \operatorname{age\_noangina\_cp4\_chain}{\left(x_{age\_b},x_{exang:0\_b},x_{cp:4\_b} \right)} - 1.3 \operatorname{exertional\_ischaemia\_hinge\_0}{\left(x_{cp:1},x_{trestbps},x_{exang:1\_b},x_{oldpeak\_b},x_{slope:1\_b} \right)} - 0.814 \operatorname{exertional\_ischaemia\_hinge\_1}{\left(x_{cp:1},x_{trestbps},x_{exang:1\_b},x_{oldpeak\_b},x_{slope:1\_b} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}z = -0.796 + 2.09ca + 1.42sex:1 + 1.21cp:4 + 1.06slope:2 + 0.762chol + 0.658thal:7 + 0.584cp:2 + 0.515fbs:0 + 0.419thal:4 - 0.327exang:1 + 0.236age + 0.229oldpeak - 0.204thalach - 0.147exang:0 + 0.124sex:0 + 0.109slope:1 + 1.3 H_0 + 0.814 H_1 - 0.515 C;  p = sigmoid(z)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> ROOT: HEART-DISEASE LOGIT. p = sigmoid(z) (get\_formula prints ReLU at the output, but the model applies a sigmoid for this binary task). z = -0.796 + LINEAR CORE + 1.300\*855 + 0.814\*1501 - 0.515\*3353, an additive composition. (1) LINEAR CORE: 16 direct input edges; largest weights ca +2.09, sex:1 +1.42, cp:4 +1.21, slope:2 +1.06, chol +0.76, thal:7 +0.66, cp:2 +0.58, fbs:0 +0.52, thal:4 +0.42; small negative terms exang:1 -0.33, thalach -0.20, exang:0 -0.15. Logit SD 3.94. (2) exertional\_ischaemia\_hinge (855/1501): a hinge on an ischaemia score (trestbps, oldpeak, exang:1 up; cp:1, slope:1 down) that is nearly flat below threshold and adds risk steeply above it (logit SD 1.64). (3) age\_noangina\_cp4\_chain (1958-\>1669-\>3353): a risk-lowering correction that tempers the cp:4 weight and makes age protective for older patients without exertional angina (logit SD 0.34). Eight inputs (cp:3, fbs:1, restecg:0/1/2, slope:3, thal:3, thal:6) are unconnected. EVIDENCE (train, n=242): AUC 0.947, accuracy 0.897. Mean \|SHAP\| (probability): ca 0.162, sex:1 0.106, slope:2 0.094, cp:4 0.088, thal:7 0.056, chol 0.043, oldpeak 0.042. Ablation ladder (child readout replaced by train mean): linear core only AUC 0.924 / acc 0.851; without hinge 0.926 / 0.847; without chain 0.945 / 0.888. So the network is mostly logistic regression plus one meaningful nonlinear ischaemia hinge and a minor damping chain.

**Evidence (3 records, DB):**

- `analytical_formula`: Additive composition: the linear core plus the two child readouts, all summed into the output logit.
- `empirical_viz`: Train SHAP (full split): ca, sex:1, slope:2, cp:4 and thal:7 dominate.
- `ablation`: Nonlinear parts add +0.023 train AUC over the linear core, almost all from the ischaemia hinge.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "get_formula shows relu at the output; for this binary task the model replaces it with a sigmoid (compute_performance overrides the output activation)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "auc": 0.947,
 "split": "train",
 "accuracy": 0.897,
 "logit_sd": {
  "linear_core": 3.94,
  "age_cp4_chain": 0.34,
  "ischaemia_hinge": 1.64
 },
 "mean_abs_shap_prob": {
  "ca": 0.162,
  "chol": 0.043,
  "cp:2": 0.03,
  "cp:4": 0.088,
  "fbs:0": 0.017,
  "sex:1": 0.106,
  "thal:7": 0.056,
  "oldpeak": 0.042,
  "slope:2": 0.094,
  "trestbps": 0.022
 }
}
```

```json
// ablation
{
 "auc": {
  "full": 0.947,
  "without_chain": 0.945,
  "without_hinge": 0.926,
  "linear_core_only": 0.924
 },
 "split": "train",
 "method": "replace child readout(s) by train mean, local forward pass",
 "accuracy": {
  "full": 0.897,
  "without_chain": 0.888,
  "without_hinge": 0.847,
  "linear_core_only": 0.851
 }
}
```

</details>

### A.1 Leaf: `exertional_ischaemia_hinge`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | -4 (cp:1), -8 (trestbps), -17\_b (exang:1\_b), -18\_b (oldpeak\_b), -19\_b (slope:1\_b) |
| Exit nodes | 855, 1501 |
| Internal / exit nodes in its own subgraph | 855, 1501 |
| Dimensionality (in → out, incl. descendants) | 5 → 2 |
| Live closed-form error (3 s.f. display) | 0.019 |
| Live numeric-function error | 9.99e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.36 x_{cp:1} \\
&\quad + 0.85 x_{exang:1\_b} \\
&\quad + 0.896 x_{oldpeak\_b} \\
&\quad - 0.937 x_{slope:1\_b} \\
&\quad + 1.06 x_{trestbps} \\
&\quad -0.802 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.393 \\
&\quad - 1.43 \operatorname{relu}{\left(- 1.36 x_{cp:1} + 0.85 x_{exang:1\_b} + 0.896 x_{oldpeak\_b} - 0.937 x_{slope:1\_b} + 1.06 x_{trestbps} - 0.802 \right)} \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_{855}=\operatorname{relu}(-1.36x_{cp:1}+0.85x_{exang:1}+0.896x_{oldpeak}-0.937x_{slope:1}+1.06x_{trestbps}-0.802), y_{1501}=\operatorname{relu}(0.393-1.43\,y_{855})\end{gathered}
$$

</details>

**Hypothesis (DB):**

> EXERTIONAL-ISCHAEMIA HINGE. Node 855 = ReLU(1.055\*trestbps + 0.896\*oldpeak + 0.850\*exang:1 - 1.358\*cp:1 - 0.937\*slope:1 - 0.802) is a sparse ischaemia score: high resting BP, ST depression and exercise-induced angina push it up; typical angina (cp:1) and upsloping ST (slope:1) push it down. Node 1501 = ReLU(0.393 - 1.428\*855) is its clipped complement. Read out together (+1.300\*855 + 0.814\*1501) they form a hinge: contribution = 0.320 + 0.138\*855 while 855 \< 0.276, and 1.300\*855 above. So below threshold these five features have almost no effect (effective slopes \~0.12-0.19), and above it they act with full weight (trestbps +1.37, oldpeak +1.17, exang:1 +1.11, slope:1 -1.22, cp:1 -1.77 per SD). EVIDENCE (train, n=242): 855 active 36% (mean 0.75, max 7.9); 1501 saturated at 0.393 whenever 855=0; mean 855 is 0.23 without vs 1.78 with exercise angina; positive rate 80% when 855 active vs 24% when zero. Sensitivity of 855: cp:1 0.49, trestbps 0.38, slope:1 0.34, oldpeak 0.32, exang:1 0.31. Ablation (local, readout replaced by train mean): AUC 0.947-\>0.926, accuracy 0.897-\>0.847, the largest nonlinear effect in the network.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula(force). The pair is a hinge: 1501 is the clipped complement of 855, so the combined output contribution is nearly flat (+0.32) until the ischaemia score passes 0.276, then rises with slope 1.30.
- `empirical_viz`: Train split (n=242, max\_samples 100000). 855 is a sparse gate (36% active); when active 80% of cases are positive vs 24% when zero. Exercise-induced angina raises its mean 8x.
- `ablation`: Largest nonlinear contribution: removing it costs 0.021 train AUC and 5 points accuracy.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "output gets 1.300*y855 + 0.814*y1501 = 0.320+0.138*y855 for y855<0.276, and 1.300*y855 above"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "pos_rate": {
  "855_zero": 0.239,
  "855_active": 0.805
 },
 "sensitivity_855": {
  "cp:1": 0.488,
  "exang:1": 0.306,
  "oldpeak": 0.322,
  "slope:1": 0.337,
  "trestbps": 0.379
 },
 "mean_855_by_exang": {
  "no": 0.225,
  "yes": 1.782
 },
 "activation_profile": {
  "855": {
   "max": 7.88,
   "mean": 0.753,
   "active": 0.36
  },
  "1501": {
   "active": 0.653,
   "saturated_at_0.393": "median"
  }
 }
}
```

```json
// ablation
{
 "split": "train",
 "method": "replace combined readout 1.3*855+0.814*1501 by its train mean (local forward pass)",
 "acc_full": 0.897,
 "auc_full": 0.947,
 "acc_without": 0.847,
 "auc_without": 0.926,
 "logit_sd_contribution": 1.64
}
```

</details>

### A.2 Leaf: `age_noangina_cp4_chain`

|  |  |
|---|---|
| Annotation id | `ann_33` |
| Entry nodes | -1\_b (age\_b), -16\_b (exang:0\_b), -7\_b (cp:4\_b) |
| Exit nodes | 3353 |
| Internal / exit nodes in its own subgraph | 1958, 1669, 3353 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00343 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.813 x_{cp:4\_b} \\
&\quad + 1.09 \operatorname{relu}{\left(1.46 x_{exang:0\_b} + 1.1 \operatorname{relu}{\left(1.05 x_{age\_b} + 0.0607 \right)} - 0.916 \right)} \\
&\quad -0.234 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_{3353}=relu(0.813 x_{cp:4} + 1.09 relu(1.46 x_{exang:0} + 1.1 relu(1.05 x_{age} + 0.0607) - 0.916) - 0.234)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> AGE / NO-ANGINA / CP:4 DAMPING CHAIN. Serial ReLU chain: 1958 = ReLU(1.049\*age + 0.061) passes age above about the mean; 1669 = ReLU(1.104\*1958 + 1.460\*exang:0 - 0.916) is active exactly when there is NO exercise-induced angina (100% of exang:0 cases, 0% otherwise) and grows with age; 3353 = ReLU(1.089\*1669 + 0.813\*cp:4 - 0.234) adds asymptomatic chest pain. It is read out with -0.515, so it LOWERS risk. Effect: (1) for cp:4 patients (always active) it trims the direct +1.213 cp:4 weight to a net +0.79; (2) for older patients without exertional angina it subtracts up to \~0.65 per SD of age, reversing the small direct +0.236 age effect in that subgroup, and adds \~-0.82 per SD on exang:0. Reading: an asymptomatic-chest-pain patient who is older but does NOT get angina on exercise is less likely diseased than cp:4 alone suggests. EVIDENCE (train, n=242): 3353 active 54.5% (100% with cp:4, 18% otherwise); mean readout -0.77 for cp:4 & no exertional angina, -0.35 for cp:4 & angina, -0.09 for no cp:4 & no angina, 0 otherwise. Sensitivity of 3353: exang:0 0.45, cp:4 0.44, age 0.26. Ablation (local, readout replaced by train mean): AUC 0.947-\>0.945, accuracy 0.897-\>0.888: a minor correction (logit SD 0.34 vs 3.9 for the linear part).

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula(force): a three-stage serial ReLU chain age -\> (age, no exertional angina) -\> (+ asymptomatic chest pain).
- `empirical_viz`: Train split (n=242). 1669 is active exactly when there is no exercise angina; 3353 is always active for asymptomatic chest pain (cp:4) and otherwise only for older patients without exertional angina.
- `ablation`: Small effect: removing it costs 0.002 train AUC and 0.8 points accuracy.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "output gets -0.515*y3353"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "sensitivity_3353": {
  "age": 0.255,
  "cp:4": 0.443,
  "exang:0": 0.453
 },
 "activation_profile": {
  "1669": {
   "active": 0.661,
   "active_if_exang0": 1.0,
   "active_if_exang1": 0.0
  },
  "1958": {
   "active": 0.537
  },
  "3353": {
   "active": 0.545,
   "active_if_cp4": 1.0,
   "active_if_not_cp4": 0.179
  }
 },
 "mean_readout_by_cp4_exang0": {
  "cp4=0,exang=no": -0.09,
  "cp4=1,exang=no": -0.769,
  "cp4=0,exang=yes": 0.0,
  "cp4=1,exang=yes": -0.346
 }
}
```

```json
// ablation
{
 "split": "train",
 "method": "replace readout -0.515*3353 by its train mean (local forward pass)",
 "acc_full": 0.897,
 "auc_full": 0.947,
 "acc_without": 0.888,
 "auc_without": 0.945,
 "logit_sd_contribution": 0.34
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 3 |
| rename\_node | 26 |
| split\_node | 6 |

6 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -17 into -17\_a, -17\_b. Final copies: -17\_a → heart\_disease\_logit (entry) \| -17\_b → exertional\_ischaemia\_hinge (entry)
- `#27` split -18 into -18\_a, -18\_b. Final copies: -18\_a → heart\_disease\_logit (entry) \| -18\_b → exertional\_ischaemia\_hinge (entry)
- `#28` split -19 into -19\_a, -19\_b. Final copies: -19\_a → heart\_disease\_logit (entry) \| -19\_b → exertional\_ischaemia\_hinge (entry)
- `#29` split -1 into -1\_a, -1\_b. Final copies: -1\_a → heart\_disease\_logit (entry) \| -1\_b → age\_noangina\_cp4\_chain (entry)
- `#30` split -16 into -16\_a, -16\_b. Final copies: -16\_a → heart\_disease\_logit (entry) \| -16\_b → age\_noangina\_cp4\_chain (entry)
- `#31` split -7 into -7\_a, -7\_b. Final copies: -7\_a → heart\_disease\_logit (entry) \| -7\_b → age\_noangina\_cp4\_chain (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.78e-15 | 1.78e-15 | 2.78e-16 | agree |
| Closed-form max abs error (all leaves) | 0.01896 | 0.01896 | 0.01896 | agree |
| Numeric-function max abs error | n/a | 9.99e-16 | 9.99e-16 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 2 / 1 | n/a | 2 / 1 | agree |
| Operation counts | 6 splits, 0 identity | {"annotate": 3, "rename\_node": 26, "split\_node": 6} | {"annotate": 3, "rename\_node": 26, "split\_node": 6} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `exertional_ischaemia_hinge` | 0.01896 | 9.99e-16 |
| `age_noangina_cp4_chain` | 0.003427 | 8.88e-16 |

## What resisted (from the record)

> Nothing resisted coverage. Six inputs (age, exang:0, exang:1, oldpeak, slope:1, cp:4) split by hand (not flagged by tools). The age/no-angina/cp:4 chain is real but small (AUC -0.002, logit SD 0.34); its 'damping correction' hypothesis rests on weight signs and subgroup means. First root (structural 0.784) undone and re-annotated with 8 unconnected inputs as entries.

**Tool notes (record):** get\_viz\_summary rejects 'line\_plot' (valid 'line'). activation\_profile needs node\_id. get\_formula shows output ReLU; compute\_performance applies sigmoid. compute\_shap \>2 min. Tool compositional\_coverage always 1.0.

## What to look at

- Largest closed-form error: `exertional_ischaemia_hinge` (0.019). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `age_noangina_cp4_chain`: *“1669 = ReLU(1.104\*1958 + 1.460\*exang:0 - 0.916) is active exactly when there is NO exercise-induced angina (100% of exang:0 cases, 0% otherwise) and grows with age;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 6 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s8.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s8', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
