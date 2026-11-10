# Verification packet 26 · monk2-s6

Pre-registered verification order: 26 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/4edb86d0-8d75-4aa7-878f-52bd3b790556?experiment=5db6a9ec-f286-4baf-991c-521fd3ef36cf&name=10seed-monk2-s6), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 6 |
| Genome id | `4edb86d0-8d75-4aa7-878f-52bd3b790556` |
| Experiment | 10seed-monk2-s6 (5db6a9ec-f286-4baf-991c-521fd3ef36cf); best at generation 99 of 100, NEAT key 24508 |
| Test AUC (runs csv) | 0.754 \[95% CI 0.634, 0.862\] |
| Train AUC | 0.766 |
| Size | 16 of 17 inputs connected; 6 hidden nodes; 28 enabled connections; depth 4; 3 skip connections + 14 direct input→output edges |
| Annotations | 5 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 6 of 6 |
| Claude's time | 43.2 min (2026-11-03T13:12:36+13:00 → 2026-11-03T13:55:50+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> logit = -0.57 + Linear + 9.54 I(a4!=2) - 6.04 G - 1.62 I(a1=2) + 0.022 I(a2=3); G is a veto gate firing when \>=3 of {a2,a3,a4,a6}=1; dominant a5=1 penalty (\~-17). Approximately 'a5 != 1 AND one or two of {a2,a3,a4,a6} take their first value'.

**Agreement with the MONK-2 rule (record):** Ignores a1=1, treats a5=1 as a veto, counts only a2/a3/a4/a6 and fires on one or two of them: positive for 76% of train rows with n\_first=1 and 74% with n\_first=2.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `a5_veto_count_peak_model`

|  |  |
|---|---|
| Annotation id | `ann_29` |
| Entry nodes | 126, 713, 2408, 951, id\_linear, -1 (a1:1) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | count\_gate\_3of4, a4\_not2\_gate, a1\_2\_linear\_unit, a2\_3\_double\_negation\_chain, direct\_linear\_terms |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{1.24 e^{0.578 a_{1 2 linear unit}{\left(-2_{b} \right)} + 0.37 a_{2 3 double negation chain}{\left(-6_{b} \right)} - 3.95 a_{4 not2 gate}{\left(-10 \right)} + 6.04 \operatorname{count\_gate\_3of4}{\left(-4_{b},-8_{b},-9,-15_{b},-17_{b} \right)} - 1.0 \operatorname{direct\_linear\_terms}{\left(-2_{a},-3,-4_{a},-5,-6_{a},-7,-8_{a},-11,-12,-13,-14,-15_{a},-16,-17_{a} \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Additive composition under the output sigmoid: logit = -0.213 + Linear + 3.945\*A4gate - 6.037\*Count3of4 - 0.578\*A1unit - 0.370\*A2chain, i.e. in indicator terms Linear + 9.54 I(a4!=2) - 6.04 G - 1.62 I(a1=2) + 0.022 I(a2=3) - 0.57, with G the at-least-three-of-{a2=1,a3=1,a4=1,a6=1} gate. Mechanism: (1) Count3of4 is a veto: whenever k4 = #{a2=1,a3=1,a4=1,a6=1} \>= 3 it subtracts at least 41.7 logits and the model predicts class 0 (0/84 configs positive). (2) Otherwise the sign is set by the linear score plus the a4!=2 bonus: a5=1 contributes about -17 relative to the other a5 values and vetoes almost everything (1/87 a5=1 configs with k4\<3 positive); with a5!=1, k4=0 is mostly negative (6/36 positive) and k4 in {1,2} is almost always positive (211/225). So the network approximates 'a5 != 1 AND one or two of {a2,a3,a4,a6} take their first value' (agreement 95.1% over all 432 configurations, 94.8% of train rows). a1:1 (node -1) has no enabled edges and is a zero-effect entry; a1 matters only through the direct a1:2/a1:3 terms and the affine a1:2 unit. The two minor children are linear/near-constant corrections (mean-ablation changes 3.2% and 0.3% of train predictions); the count gate is the dominant nonlinearity (mean-ablation flips 52%). EVIDENCE: composed closed form (sigmoid of the weighted sum of children); exhaustive enumeration over 432 configurations; train performance of the composed model accuracy 0.655, AUC 0.766 (identical to the evolved network). Relative to the MONK-2 target (exactly two of six first values), the network ignores a1=1, treats a5=1 as a veto instead of a counted indicator, and counts only a2,a3,a4,a6, so it fires for one OR two first values among those four.

**Evidence (2 records, DB):**

- `analytical_formula`: Composed closed form: sigmoid of the weighted sum of the five child functions.
- `performance_metrics`: Composed model on train (full sample): acc 0.655, AUC 0.766. Approximating rule agreement 95.1% (432 configs).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// performance_metrics
{
 "rule": "a5!=1 and k4 in {1,2}",
 "split": "train",
 "auc_roc": 0.7656393042416845,
 "accuracy": 0.6550724637681159,
 "n_samples": 345,
 "rule_agreement_432": 0.9514,
 "rule_agreement_train": 0.9478
}
```

</details>

### A.1 Leaf: `count_gate_3of4`

|  |  |
|---|---|
| Annotation id | `ann_24` |
| Entry nodes | -4\_b (a2:1\_b), -8\_b (a3:2\_b), -9 (a4:1), -15\_b (a5:4\_b), -17\_b (a6:2\_b) |
| Exit nodes | 126 |
| Internal / exit nodes in its own subgraph | 126 |
| Dimensionality (in → out, incl. descendants) | 5 → 1 |
| Live closed-form error (3 s.f. display) | 0.0145 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 5.2 x_{a2:1\_b} \\
&\quad - 5.06 x_{a3:2\_b} \\
&\quad + 5.54 x_{a4:1} \\
&\quad + 1.05 x_{a5:4\_b} \\
&\quad - 5.32 x_{a6:2\_b} \\
&\quad -6.19 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> An at-least-three-of-four first-value detector over attributes a2, a3, a4, a6 (a1 and a5 are not consulted, apart from a weak a5=4 term). In indicator terms the pre-activation is -24.87 + 11.00\*I(a2=1) + 10.13\*I(a3=1) + 11.97\*I(a4=1) + 10.64\*I(a6=1) + 2.43\*I(a5=4) (a3 and a6 enter through their complements a3:2, a6:2 with negative weights). Any two indicators sum to at most 22.97 \< 24.87, so with two indicators the unit is off except for a sliver (0.17 or 0.54) when a5=4 is also present; any three indicators sum to at least 31.77, so the unit fires at 6.9-18.9 whenever three or more of {a2=1,a3=1,a4=1,a6=1} hold. Its -6.04 readout then drives the output logit far negative.

**Evidence (2 records, DB):**

- `analytical_formula`: Pre-activation in indicator terms: -24.87 + 11.00 I(a2=1) + 10.13 I(a3=1) + 11.97 I(a4=1) + 10.64 I(a6=1) + 2.43 I(a5=4). Max two-indicator sum (a2+a4) with a5=4 reaches +0.53; without a5=4 every two-indicator pattern is \<= -1.9; min three-indicator sum reaches +6.90. Exhaustive over all 432 configurations: 339 zero, 9 in (0.17, 0.54), 84 in \[6.9, 18.9\].
- `empirical_viz`: Train split (345 rows, full sample, local forward pass on the model state): h126 = 0 for all rows with k4=#{a2=1,a3=1,a4=1,a6=1} in {0,1} (38+115 rows); in \[0, 0.536\] for k4=2 (124 rows); in \[6.91, 11.18\] for k4=3 (58 rows); in \[18.88, 21.30\] for k4=4 (10 rows). Mean-ablating this unit flips 52% of train predictions and drops train accuracy 0.655 -\> 0.597: the most influential hidden unit.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "Pre-activation in indicator terms: -24.87 + 11.00 I(a2=1) + 10.13 I(a3=1) + 11.97 I(a4=1) + 10.64 I(a6=1) + 2.43 I(a5=4). Max two-indicator sum (a2+a4) with a5=4 reaches +0.53; without a5=4 every two-indicator pattern is <= -1.9; min three-indicator sum reaches +6.90. Exhaustive over all 432 configurations: 339 zero, 9 in (0.17, 0.54), 84 in [6.9, 18.9]."
}
```

```json
// empirical_viz
{
 "note": "Train split (345 rows, full sample, local forward pass on the model state): h126 = 0 for all rows with k4=#{a2=1,a3=1,a4=1,a6=1} in {0,1} (38+115 rows); in [0, 0.536] for k4=2 (124 rows); in [6.91, 11.18] for k4=3 (58 rows); in [18.88, 21.30] for k4=4 (10 rows). Mean-ablating this unit flips 52% of train predictions and drops train accuracy 0.655 -> 0.597: the most influential hidden unit.",
 "split": "train",
 "summary": "",
 "viz_type": "activation_profile",
 "n_samples": 345
}
```

</details>

### A.2 Leaf: `a4_not2_gate`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -10 (a4:2) |
| Exit nodes | 713 |
| Internal / exit nodes in its own subgraph | 713 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0021 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.593 - 2.35 x_{a4:2} \right)}\end{gathered}
$$

**Hypothesis (DB):**

> A binary gate on a4 != 2: h713 = 2.418 \* I(a4 != 2) exactly (pre-activation -2.42 when a4=2). With readout +3.945 it adds +9.54 to the logit unless a4=2, i.e. a per-attribute penalty of 9.54 on a4=2. Because a4:2 has no direct edge, this is the only route by which a4=2 is distinguished from a4=1/3 in the linear part. EVIDENCE: closed form, train line plot (two levels, corr -1.00), mean-ablation flips 10% of train predictions.

**Evidence (2 records, DB):**

- `analytical_formula`: ReLU(0.593 - 2.348 z(a4:2)), z(a4:2) in {-0.778, 1.286} -\> {2.418, 0}. Equals 2.418 \* I(a4 != 2) exactly; readout +3.945 gives +9.54 \* I(a4 != 2).
- `empirical_viz`: Train line summary: scatter y in {0, 2.418}, correlation with z(a4:2) = -1.00; h713 = 0 for every a4=2 row and 2.418 for every other row. Mean-ablation flips 10% of train predictions (accuracy 0.655 -\> 0.658).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "ReLU(0.593 - 2.348 z(a4:2)), z(a4:2) in {-0.778, 1.286} -> {2.418, 0}. Equals 2.418 * I(a4 != 2) exactly; readout +3.945 gives +9.54 * I(a4 != 2)."
}
```

```json
// empirical_viz
{
 "note": "Train line summary: scatter y in {0, 2.418}, correlation with z(a4:2) = -1.00; h713 = 0 for every a4=2 row and 2.418 for every other row. Mean-ablation flips 10% of train predictions (accuracy 0.655 -> 0.658).",
 "split": "train",
 "summary": "{\n  \"viz_type\": \"line\",\n  \"dimensionality\": [\n    1,\n    1\n  ],\n  \"entry_names\": [\n    \"-10\"\n  ],\n  \"exit_names\": [\n    \"713\"\n  ],\n  \"x_range\": [\n    -0.8807738202066151,\n    1.3892001340780056\n  ],\n  \"x_points\": 200,\n  \"y_range\": [\n    0.0,\n    2.660473558723311\n  ],\n  \"y_mean\": 0.6674637891639845,\n  \"scatter_points\": 345,\n  \"scatter_x_range\": [\n    -0.7775931859209505,\n    1.286019499792341\n  ],\n  \"scatter_y_range\": [\n    0.0,\n    2.4182479847116736\n  ],\n  \"correlation\": -0.9999999999999994\n}\n",
 "viz_type": "line",
 "n_samples": 345
}
```

</details>

### A.3 Leaf: `a1_2_linear_unit`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -2\_b (a1:2\_b) |
| Exit nodes | 2408 |
| Internal / exit nodes in its own subgraph | 2408 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00407 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.32 x_{a1:2\_b} + 1.47 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> A ReLU unit that is always active on the domain, hence purely affine: h2408 = 0.526 + 2.801 I(a1=2). Its -0.578 readout contributes -0.304 - 1.619 I(a1=2), partly cancelling the direct a1:2 edge (+3.49): net a1=2 effect falls from +3.49 to +1.87 relative to a1 not 2. A redundant linear correction rather than a nonlinearity. EVIDENCE: closed form, train line plot (corr 1.00, always active), mean-ablation flips 3.2% of predictions.

**Evidence (2 records, DB):**

- `analytical_formula`: ReLU(1.323 z(a1:2) + 1.468), z(a1:2) in {-0.712, 1.405} -\> {0.526, 3.327}; never clipped, so affine: 0.526 + 2.801 I(a1=2). Readout -0.578: -0.304 - 1.619 I(a1=2).
- `empirical_viz`: Train line summary: y in \[0.526, 3.327\], correlation 1.00 with input; ReLU never inactive on the train split. Mean-ablation flips 3.2% of train predictions.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "ReLU(1.323 z(a1:2) + 1.468), z(a1:2) in {-0.712, 1.405} -> {0.526, 3.327}; never clipped, so affine: 0.526 + 2.801 I(a1=2). Readout -0.578: -0.304 - 1.619 I(a1=2)."
}
```

```json
// empirical_viz
{
 "note": "Train line summary: y in [0.526, 3.327], correlation 1.00 with input; ReLU never inactive on the train split. Mean-ablation flips 3.2% of train predictions.",
 "split": "train",
 "summary": "{\n  \"viz_type\": \"line\",\n  \"dimensionality\": [\n    1,\n    1\n  ],\n  \"entry_names\": [\n    \"-2_b\"\n  ],\n  \"exit_names\": [\n    \"2408\"\n  ],\n  \"x_range\": [\n    -0.8175615943451056,\n    1.5108783685364524\n  ],\n  \"x_points\": 200,\n  \"y_range\": [\n    0.38642960787393776,\n    3.4670339771975387\n  ],\n  \"y_mean\": 1.9267317925357383,\n  \"scatter_points\": 345,\n  \"scatter_x_range\": [\n    -0.7117234142141257,\n    1.4050401884054724\n  ],\n  \"scatter_y_range\": [\n    0.5264570792068288,\n    3.327006505864648\n  ],\n  \"correlation\": 1.0\n}\n",
 "viz_type": "line",
 "n_samples": 345
}
```

</details>

### A.4 Leaf: `a2_3_double_negation_chain`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | -6\_b (a2:3\_b) |
| Exit nodes | 951 |
| Internal / exit nodes in its own subgraph | 28, 399, 951 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 1.26e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.138 \\
&\quad - 0.0711 \operatorname{relu}{\left(0.838 - 0.667 \operatorname{relu}{\left(0.886 - 0.786 x_{a2:3\_b} \right)} \right)} \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A three-unit serial chain (NOT of NOT of a2=3) that re-encodes I(a2=3) and attenuates it: h951 = 0.138 - 0.0596 I(a2=3), so its logit contribution is -0.051 + 0.022 I(a2=3). Functionally near-inert vestigial structure: the effect is 0.022 logits against a direct a2=3 effect of -3.5. EVIDENCE: closed form, train line plot (two levels, corr -1.00), mean-ablation changes 1/345 train predictions.

**Evidence (2 records, DB):**

- `analytical_formula`: h28 = ReLU(0.886 - 0.786 z(a2:3)) = 1.453 I(a2!=3); h399 = ReLU(0.838 - 0.667 h28) = 0.838 I(a2=3); h951 = ReLU(0.138 - 0.0711 h399) = 0.138 - 0.0596 I(a2=3). Readout -0.370: -0.051 + 0.022 I(a2=3).
- `empirical_viz`: Train line summary: h951 in {0.0785, 0.1381}, correlation -1.00 with z(a2:3). Contribution to logit spans 0.022. Mean-ablation flips 1 of 345 train predictions (0.3%).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "h28 = ReLU(0.886 - 0.786 z(a2:3)) = 1.453 I(a2!=3); h399 = ReLU(0.838 - 0.667 h28) = 0.838 I(a2=3); h951 = ReLU(0.138 - 0.0711 h399) = 0.138 - 0.0596 I(a2=3). Readout -0.370: -0.051 + 0.022 I(a2=3)."
}
```

```json
// empirical_viz
{
 "note": "Train line summary: h951 in {0.0785, 0.1381}, correlation -1.00 with z(a2:3). Contribution to logit spans 0.022. Mean-ablation flips 1 of 345 train predictions (0.3%).",
 "split": "train",
 "summary": "{\n  \"viz_type\": \"line\",\n  \"dimensionality\": [\n    1,\n    1\n  ],\n  \"entry_names\": [\n    \"-6_b\"\n  ],\n  \"exit_names\": [\n    \"951\"\n  ],\n  \"x_range\": [\n    -0.8263867429394942,\n    1.4923841919628202\n  ],\n  \"x_points\": 200,\n  \"y_range\": [\n    0.07854383169417635,\n    0.1380941833816499\n  ],\n  \"y_mean\": 0.10820424439200384,\n  \"scatter_points\": 345,\n  \"scatter_x_range\": [\n    -0.720988064080298,\n    1.3869855131036242\n  ],\n  \"scatter_y_range\": [\n    0.07854383169417635,\n    0.1380941833816499\n  ],\n  \"correlation\": -0.9999999999999977\n}\n",
 "viz_type": "line",
 "n_samples": 345
}
```

</details>

### A.5 Leaf: `direct_linear_terms`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | -2\_a (a1:2\_a), -3 (a1:3), -4\_a (a2:1\_a), -5 (a2:2), -6\_a (a2:3\_a), -7 (a3:1), -8\_a (a3:2\_a), -11 (a4:3), -12 (a5:1), -13 (a5:2), -14 (a5:3), -15\_a (a5:4\_a), -16 (a6:1), -17\_a (a6:2\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 14 → 1 |
| Live closed-form error (3 s.f. display) | 0.0357 |
| Live numeric-function error | 7.11e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.65 x_{a1:2\_a} \\
&\quad + 1.26 x_{a1:3} \\
&\quad + 1.05 x_{a2:1\_a} \\
&\quad - 1.83 x_{a2:2} \\
&\quad - 2.91 x_{a2:3\_a} \\
&\quad + 2.12 x_{a3:1} \\
&\quad - 1.87 x_{a3:2\_a} \\
&\quad - 1.92 x_{a4:3} \\
&\quad - 7.29 x_{a5:1} \\
&\quad + 0.022 x_{a5:2} \\
&\quad + 1.44 x_{a5:3} \\
&\quad + 0.155 x_{a5:4\_a} \\
&\quad + 3.83 x_{a6:1} \\
&\quad - 1.03 x_{a6:2\_a}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> The model's additive per-attribute score. Per-attribute contributions: a1=(1:-2.02, 2:+1.47, 3:+0.71); a2=(1:+4.84, 2:-1.31, 3:-3.52); a3=(1:+3.89, 2:-4.10); a4=(1:+1.30, 2:+1.30, 3:-2.85); a5=(1:-13.49, 2:+3.36, 3:+6.66, 4:+3.67); a6=(1:+4.74, 2:-5.00). It rewards first values of a2, a3, a6 (swings of 8-10 logits each) and weakly a4=1 vs a4=3, but carries a dominant veto on a5=1 (about -17 relative to other a5 values) and a mild penalty on a1=1 (-3). So it is not a symmetric count of first values: it is a count over a2/a3/a6 (a4 completed by the a4!=2 gate) with a strong a5=1 veto. EVIDENCE: closed form, train sensitivity (a5:1 largest at 7.30).

**Evidence (2 records, DB):**

- `analytical_formula`: Sum of 14 direct edges into the identity node. Per attribute value (z-scaled one-hots): a1=(1:-2.02, 2:+1.47, 3:+0.71); a2=(1:+4.84, 2:-1.31, 3:-3.52); a3=(1:+3.89, 2:-4.10); a4=(1:+1.30, 2:+1.30, 3:-2.85); a5=(1:-13.49, 2:+3.36, 3:+6.66, 4:+3.67); a6=(1:+4.74, 2:-5.00). Range on train: -30.98 to +22.91.
- `empirical_viz`: Train sensitivity (\|d id\_linear / d input\|) equals the edge magnitudes; largest is a5:1 (7.30) then a6:1 (3.83), a2:3 (2.91), a3:1 (2.12). Contribution std on train 10.9 logits.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "Sum of 14 direct edges into the identity node. Per attribute value (z-scaled one-hots): a1=(1:-2.02, 2:+1.47, 3:+0.71); a2=(1:+4.84, 2:-1.31, 3:-3.52); a3=(1:+3.89, 2:-4.10); a4=(1:+1.30, 2:+1.30, 3:-2.85); a5=(1:-13.49, 2:+3.36, 3:+6.66, 4:+3.67); a6=(1:+4.74, 2:-5.00). Range on train: -30.98 to +22.91."
}
```

```json
// empirical_viz
{
 "note": "Train sensitivity (|d id_linear / d input|) equals the edge magnitudes; largest is a5:1 (7.30) then a6:1 (3.83), a2:3 (2.91), a3:1 (2.12). Contribution std on train 10.9 logits.",
 "split": "train",
 "summary": "{\n  \"viz_type\": \"sensitivity\",\n  \"dimensionality\": [\n    14,\n    1\n  ],\n  \"entry_names\": [\n    \"-2_a\",\n    \"-3\",\n    \"-4_a\",\n    \"-5\",\n    \"-6_a\",\n    \"-7\",\n    \"-8_a\",\n    \"-11\",\n    \"-12\",\n    \"-13\",\n    \"-14\",\n    \"-15_a\",\n    \"-16\",\n    \"-17_a\"\n  ],\n  \"exit_names\": [\n    \"id_linear\"\n  ],\n  \"sensitivities\": {\n    \"id_linear\": [\n      1.6496077713502337,\n      1.259627656506985,\n      1.0515807777631372,\n      1.8333110758822677,\n      2.9114258256214818,\n      2.123594729617117,\n      1.8735013046153124,\n      1.9247360383437244,\n      7.295065036536487,\n      0.021982076150496476,\n      1.4431855841977193,\n      0.15466161009339469,\n      3.8324900413452747,\n      1.0340508910495427\n    ]\n  },\n  \"input_labels\": [\n    \"-2_a\",\n    \"-3\",\n    \"-4_a\",\n    \"-5\",\n    \"-6_a\",\n    \"-7\",\n    \"-8_a\",\n    \"-11\",\n    \"-12\",\n    \"-13\",\n    \"-14\",\n    \"-15_a\",\n    \"-16\",\n    \"-17_a\"\n  ]\n}\n",
 "viz_type": "sensitivity",
 "n_samples": 345
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 6 |
| rename\_node | 17 |
| split\_node | 6 |

6 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -2 into -2\_a, -2\_b. Final copies: -2\_a → direct\_linear\_terms (entry) \| -2\_b → a1\_2\_linear\_unit (entry)
- `#18` split -4 into -4\_a, -4\_b. Final copies: -4\_a → direct\_linear\_terms (entry) \| -4\_b → count\_gate\_3of4 (entry)
- `#19` split -6 into -6\_a, -6\_b. Final copies: -6\_a → direct\_linear\_terms (entry) \| -6\_b → a2\_3\_double\_negation\_chain (entry)
- `#20` split -8 into -8\_a, -8\_b. Final copies: -8\_a → direct\_linear\_terms (entry) \| -8\_b → count\_gate\_3of4 (entry)
- `#21` split -15 into -15\_a, -15\_b. Final copies: -15\_a → direct\_linear\_terms (entry) \| -15\_b → count\_gate\_3of4 (entry)
- `#22` split -17 into -17\_a, -17\_b. Final copies: -17\_a → direct\_linear\_terms (entry) \| -17\_b → count\_gate\_3of4 (entry)
- `#23` identity node id\_linear intercepts 14 connection(s) into 0 (from -2\_a (a1:2\_a), -3 (a1:3), -4\_a (a2:1\_a), -5 (a2:2), -6\_a (a2:3\_a), -7 (a3:1), -8\_a (a3:2\_a), -11 (a4:3), … (6 more)); sits in direct\_linear\_terms (exit); a5\_veto\_count\_peak\_model (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T12:25:06+13:00 (commit `77513b9`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 5.00e-16 | 5.00e-16 | 5.00e-16 | agree |
| Closed-form max abs error (all leaves) | 0.0357 | 0.03573 | 0.03573 | agree |
| Numeric-function max abs error | n/a | 7.11e-15 | 7.11e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | a5\_veto\_count\_peak\_model | a5\_veto\_count\_peak\_model | agree |
| Leaf / composition counts | 5 / 1 | n/a | 5 / 1 | agree |
| Operation counts | 6 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 6, "rename\_node": 17, "split\_node": 6} | {"add\_identity\_node": 1, "annotate": 6, "rename\_node": 17, "split\_node": 6} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `direct_linear_terms` | 0.03573 | 7.11e-15 |
| `count_gate_3of4` | 0.01453 | 3.55e-15 |
| `a1_2_linear_unit` | 0.004068 | 0 |
| `a4_not2_gate` | 0.002096 | 0 |
| `a2_3_double_negation_chain` | 1.26e-04 | 0 |

## What resisted (from the record)

> Nothing structural. Network is an approximation (train acc 0.655, AUC 0.766); root states an approximating rule agreeing with the network on 95.1% of 432 configurations. a2:3 three-node chain near-inert; a1:2 unit affine.

**Tool notes (record):** activation\_profile returned empty for a 5-input gate (tabulated locally). Parallel add\_evidence/update\_narrative calls on different annotations of the same genome race on the operations JSONB and lose writes (ann\_25 redone serially and verified). Early scratch files briefly written to repo root (moved).

## What to look at

- Largest closed-form error: `direct_linear_terms` (0.0357), then `count_gate_3of4` (0.0145). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `a2_3_double_negation_chain`: *“A three-unit serial chain (NOT of NOT of a2=3) that re-encodes I(a2=3) and attenuates it: h951 = 0.138 - 0.0596 I(a2=3), so its logit contribution is -0.051 + 0.022 I(a2=3).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `a4_not2_gate`: *“A binary gate on a4 != 2: h713 = 2.418 \* I(a4 != 2) exactly (pre-activation -2.42 when a4=2).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 6 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s6.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s6', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
