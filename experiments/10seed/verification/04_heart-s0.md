# Verification packet 04 · heart-s0

Pre-registered verification order: 4 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/e7221bd4-a46a-4c5b-ad27-ab01bbdeb290?experiment=16ddb6f0-2b52-4b5b-b482-af9d7855463b&name=10seed-heart-s0), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 0 |
| Genome id | `e7221bd4-a46a-4c5b-ad27-ab01bbdeb290` |
| Experiment | 10seed-heart-s0 (16ddb6f0-2b52-4b5b-b482-af9d7855463b); best at generation 39 of 40, NEAT key 9708 |
| Test AUC (runs csv) | 0.875 \[95% CI 0.778, 0.953\] |
| Train AUC | 0.945 |
| Size | 19 of 26 inputs connected; 6 hidden nodes; 30 enabled connections; depth 6; 6 skip connections + 17 direct input→output edges |
| Annotations | 2 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 4 of 6 |
| Claude's time | 10.4 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:34:19+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Essentially logistic regression over 17 inputs (logit SD 4.33) plus one sparse nonlinear correction: gate 665 is an exact AND of female, exercise angina and no LVH, firing on 8/242 train rows (7 positive); a flat-slope ReLU amplifies it super-additively (node 1650 values 0, 0.446, 2.10, 6.09; gate+flat slope 7/7 positive). Removing the amplifier: train AUC 0.945 -\> 0.930.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | -1 (age), -3 (sex:1), -4 (cp:1), -5 (cp:2), -6 (cp:3), -7 (cp:4), -9 (chol), -11 (fbs:1), -15 (thalach), -18 (oldpeak), -19 (slope:1), -21 (slope:3), -22 (ca), -25 (thal:6), -26 (thal:7), -14\_a (restecg:2\_a), -20\_a (slope:2\_a), -24 (thal:4), -23 (thal:3), -17 (exang:1), -13 (restecg:1), -12 (restecg:0), -10 (fbs:0), -8 (trestbps), -20\_b (slope:2\_b), -14\_c (restecg:2\_c) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | flat\_slope\_amplifier, restecg2\_relay\_2447 |
| Dimensionality (in → out, incl. descendants) | 29 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{1.54 e^{- 0.255 x_{age} - 1.93 x_{ca} - 0.716 x_{chol} + 0.196 x_{cp:1} - 0.61 x_{cp:2} + 0.123 x_{cp:3} - 1.21 x_{cp:4} + 0.0843 x_{fbs:1} - 0.707 x_{oldpeak} - 0.224 x_{restecg:2\_a} - 1.67 x_{sex:1} + 0.0862 x_{slope:1} - 1.05 x_{slope:2\_a} - 0.0184 x_{slope:3} + 0.302 x_{thal:6} - 0.526 x_{thal:7} + 0.106 x_{thalach} - 0.772 \operatorname{flat\_slope\_amplifier}{\left(x_{sex:0},x_{exang:0},x_{restecg:2\_b},x_{slope:2\_b} \right)} - 0.0258 \operatorname{restecg\_2\_relay\_2447}{\left(x_{restecg:2\_c} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}z = -0.435 + 1.93\,ca + 1.67\,sex{:}1 + 1.21\,cp{:}4 + 1.05\,slope{:}2 + 0.716\,chol + 0.707\,oldpeak + 0.61\,cp{:}2 + 0.526\,thal{:}7 - 0.302\,thal{:}6 + 0.255\,age + 0.224\,restecg{:}2 - 0.196\,cp{:}1 - 0.123\,cp{:}3 - 0.106\,thalach - 0.086\,slope{:}1 - 0.084\,fbs{:}1 + 0.018\,slope{:}3 + 0.772\,\mathrm{flat\_slope\_amplifier} + 0.0258\,\mathrm{restecg2\_relay};\quad P(\text{disease}) = \sigma(z)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: additive composition. The logit z of node 0 is a linear score over 17 direct standardized inputs (dominated by ca 1.93, sex:1 1.67, cp:4 1.21, slope:2 1.05, chol 0.72, oldpeak 0.71) plus 0.772 x flat\_slope\_amplifier plus 0.026 x restecg2\_relay; 7 inputs (trestbps, fbs:0, restecg:0, restecg:1, exang:1, thal:3, thal:4) are disconnected. get\_formula shows node 0 as ReLU, but for this binary task the model output is sigmoid(z) on the pre-activation (compute\_performance train AUC 0.945 equals AUC of z; sigmoid(relu(z)) would give 0.912). Evidence (train, n=242, local forward passes): direct-linear logit SD 4.33 vs amplifier term SD 0.79 vs relay term SD 0.026. Dropping the amplifier term lowers train AUC 0.945 -\> 0.930 and accuracy 0.888 -\> 0.868; the direct linear part alone gives AUC 0.931. So the network is essentially logistic regression plus one sparse nonlinear correction: a flat-slope bonus (+0.34 logit) and a large boost (+4.7 logit) for the 7 women with exertional angina, no LVH and flat ST slope. Kernel SHAP on the full model (train) ranks ca 0.149, sex:1 0.130, slope:2 0.121, cp:4 0.116, oldpeak 0.049, thal:7 0.046, chol 0.044 (mean \|SHAP\| on probability). Function preservation after the two input splits: max abs err 1.8e-15.

**Evidence (2 records, DB):**

- `analytical_formula`: Root logit: linear score plus amplifier and relay.
- `empirical_viz`: Ablation and SHAP on train.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "output activation is sigmoid on the pre-activation, not ReLU"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "acc_full": 0.888,
 "auc_full": 0.945,
 "logit_sd": {
  "relay": 0.026,
  "direct": 4.33,
  "amplifier": 0.786
 },
 "shap_top": {
  "ca": 0.149,
  "chol": 0.044,
  "cp:4": 0.116,
  "sex:1": 0.13,
  "thal:7": 0.046,
  "oldpeak": 0.049,
  "slope:2": 0.121
 },
 "acc_without_amplifier": 0.868,
 "auc_without_amplifier": 0.93
}
```

</details>

### A.1 Composition: `flat_slope_amplifier`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -2 (sex:0), -16 (exang:0), -14\_b (restecg:2\_b), -20\_b (slope:2\_b) |
| Exit nodes | 1650 (gate\_slope\_amplifier) |
| Internal / exit nodes in its own subgraph | 3598 (flat\_slope\_or\_gate), 1650 (gate\_slope\_amplifier) |
| Children | female\_exertional\_angina\_gate |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.29 \operatorname{female\_exertional\_angina\_gate}{\left(x_{sex:0},x_{exang:0},x_{restecg:2\_b} \right)} \\
&\quad + 1.31 \operatorname{relu}{\left(1.51 x_{slope:2\_b} + 1.05 \operatorname{female\_exertional\_angina\_gate}{\left(x_{sex:0},x_{exang:0},x_{restecg:2\_b} \right)} - 0.761 \right)} \\
&\quad -0.733 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y=\mathrm{relu}\big(1.29\,G + 1.31\,\mathrm{relu}(1.51\,x_{slope:2} + 1.05\,G - 0.761) - 0.733\big),\ G=\text{female\_exertional\_angina\_gate}(x_{sex:0},x_{exang:0},x_{restecg:2})\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: serial composition H(G(sex:0, exang:0, restecg:2), slope:2). Node 3598 = relu(1.51 slope:2 + 1.05 G - 0.76) and node 1650 = relu(1.29 G + 1.31\*3598 - 0.73). Because G takes only the values 0 or 2.117 and slope:2 is binary, 1650 has exactly four regimes on train: G=0 & slope not flat -\> 0 (n=131, 27% positive); G=0 & flat slope (slope:2) -\> 0.446 (n=103, 63% positive); G on & slope not flat -\> 2.097 (n=1, negative); G on & flat slope -\> 6.092 (n=7, all positive). So without the gate it is a small additional flat-ST-slope bonus (+0.34 logit after the 0.772 output weight, on top of the direct slope:2 weight); with the gate it amplifies the female-exertional-angina signal, and the flat-slope ReLU 3598 makes the two conditions super-additive (6.09 vs 0.45+2.10). Evidence: activation\_profile on node 1650 (train): active 45.9%, mean 0.375, SD 1.018, max 6.092; values confirmed by local forward pass.

**Evidence (2 records, DB):**

- `analytical_formula`: Serial composition of the gate with a flat-slope ReLU.
- `empirical_viz`: Four discrete regimes; gate and flat slope combine super-additively.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// empirical_viz
{
 "n": 242,
 "viz": "activation_profile node 1650",
 "split": "train",
 "regimes": {
  "G0_flat": {
   "n": 103,
   "value": 0.446,
   "pos_rate": 0.63
  },
  "Gon_flat": {
   "n": 7,
   "value": 6.092,
   "pos_rate": 1.0
  },
  "Gon_not_flat": {
   "n": 1,
   "value": 2.097,
   "pos_rate": 0.0
  },
  "G0_slope_not_flat": {
   "n": 131,
   "value": 0,
   "pos_rate": 0.27
  }
 }
}
```

</details>

#### A.1.1 Leaf: `female_exertional_angina_gate`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | -2 (sex:0), -16 (exang:0), -14\_b (restecg:2\_b) |
| Exit nodes | 665 (female\_exang\_noLVH\_gate) |
| Internal / exit nodes in its own subgraph | 2349 (restecg2\_relay\_a), 2771 (restecg2\_relay\_b), 665 (female\_exang\_noLVH\_gate) |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00486 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.71 x_{exang:0} \\
&\quad + 1.41 x_{sex:0} \\
&\quad - 0.717 \operatorname{relu}{\left(0.977 x_{restecg:2\_b} + 2.11 \right)} \\
&\quad - 0.555 \operatorname{relu}{\left(0.883 \operatorname{relu}{\left(0.977 x_{restecg:2\_b} + 2.11 \right)} + 0.975 \right)} \\
&\quad -0.3 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = \mathrm{relu}(1.41\,x_{sex:0} - 1.71\,x_{exang:0} - 0.717\,\mathrm{relu}(0.977 x_{restecg:2}+2.11) - 0.555\,\mathrm{relu}(0.883\,\mathrm{relu}(0.977 x_{restecg:2}+2.11)+0.975) - 0.30) \;=\; \mathrm{relu}(1.41\,x_{sex:0} - 1.71\,x_{exang:0} - 1.18\,x_{restecg:2} - 3.39)\ \text{(inner ReLUs always active on data)}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: node 665 is a three-way AND gate that fires only for women (sex:0 set) with exercise-induced angina (exang:0 unset, i.e. exang=yes) and no left-ventricular hypertrophy on resting ECG (restecg:2 unset). The restecg:2 chain 2349 -\> 2771 never leaves the ReLU linear region on standardized one-hot inputs (2349 in \[1.15,3.11\], 2771 in \[1.99,3.72\]), so the gate collapses exactly to relu(1.41 sex:0 - 1.71 exang:0 - 1.18 restecg:2 - 3.39). Evidence (train, n=242): pre-activation max is 2.117, reached only when all three conditions hold; removing any one condition drives it negative (no female: -0.86; no exang: -2.53; LVH present: -0.24). The gate is active on 8/242 rows (3.3%), always at the single value 2.117, and 7 of those 8 are positive cases. Closed form matches the forward pass (check\_explanation closed-form err 0.0049, numeric err 0).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form; inner restecg:2 ReLUs are always active, so the gate is a single ReLU of a linear form.
- `empirical_viz`: Gate fires on 8/242 train rows (all female, exang=yes, no LVH); 7/8 have disease.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// empirical_viz
{
 "n": 242,
 "viz": "output_distribution+local truth table",
 "split": "train",
 "active_rows": 8,
 "sensitivity": {
  "sex:0": 0.047,
  "exang:0": 0.057,
  "restecg:2": 0.039
 },
 "active_value": 2.117,
 "positives_among_active": 7
}
```

</details>

### A.2 Leaf: `restecg2_relay_2447`

|  |  |
|---|---|
| Annotation id | `ann_29` |
| Entry nodes | -14\_c (restecg:2\_c) |
| Exit nodes | 2447 (restecg2\_relay\_out) |
| Internal / exit nodes in its own subgraph | 2447 (restecg2\_relay\_out) |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00827 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.02 x_{restecg:2\_c} + 1.48 \right)}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y=\mathrm{relu}(1.02\,x_{restecg:2}+1.48)=1.02\,x_{restecg:2}+1.48\ \text{on data}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: node 2447 is an always-active affine relay of restecg:2 (LVH on resting ECG). Evidence (train): pre-activation in \[0.485, 2.516\], so the ReLU never clips; through its 0.026 output weight it adds 0.026\*1.02 = 0.026 per SD of restecg:2 to the logit (logit-term SD 0.026 vs total logit SD 4.3), i.e. it is practically inert and simply nudges the direct restecg:2 weight from 0.224 to 0.250. Closed-form err 0.0083 (3-s.f. rounding).

**Evidence (2 records, DB):**

- `analytical_formula`: Affine relay, never clipped on train.
- `empirical_viz`: Always active; logit contribution SD 0.026.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// empirical_viz
{
 "viz": "activation range",
 "split": "train",
 "pre_max": 2.516,
 "pre_min": 0.485,
 "frac_active": 1.0,
 "logit_term_sd": 0.026
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 4 |
| rename\_node | 32 |
| split\_node | 2 |

2 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -14 into -14\_a, -14\_b, -14\_c. Final copies: -14\_a → heart\_disease\_logit (entry) \| -14\_b → female\_exertional\_angina\_gate (entry); flat\_slope\_amplifier (entry) \| -14\_c → restecg2\_relay\_2447 (entry); heart\_disease\_logit (entry)
- `#27` split -20 into -20\_a, -20\_b. Final copies: -20\_a → heart\_disease\_logit (entry) \| -20\_b → flat\_slope\_amplifier (entry); heart\_disease\_logit (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.78e-15 | 1.78e-15 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.008275 | 0.008275 | 0.008275 | agree |
| Numeric-function max abs error | n/a | 0 | 0 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 2 / 2 | n/a | 2 / 2 | agree |
| Operation counts | 2 splits, 0 identity | {"annotate": 4, "rename\_node": 32, "split\_node": 2} | {"annotate": 4, "rename\_node": 32, "split\_node": 2} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `restecg2_relay_2447` | 0.008275 | 0 |
| `female_exertional_angina_gate` | 0.004859 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Two input splits (restecg:2, slope:2). 3598 and 1650 take input from gate 665 so live only in the flat\_slope\_amplifier composition. Inner restecg:2 ReLUs 2349, 2771 never clip (affine). Relay 2447 near-inert (logit SD 0.026). Gate evidence rests on only 8 active train rows: specificity solid, clinical meaning weakly supported.

**Tool notes (record):** Root validation failed with grandchild entry nodes, so they were left out of the root entry list. get\_formula shows output as ReLU; agent reports train AUC 0.945 = AUC(z) while sigmoid(relu(z)) gives 0.912. compute\_shap \>120s, tiny nonzero SHAP (1e-5) on disconnected inputs (kernel-SHAP noise).

## What to look at

- Largest closed-form error: `restecg2_relay_2447` (0.00827). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `female_exertional_angina_gate`: *“Hypothesis: node 665 is a three-way AND gate that fires only for women (sex:0 set) with exercise-induced angina (exang:0 unset, i.e.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 2 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s0.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s0', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
