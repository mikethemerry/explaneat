# Verification packet 06 · monk2-s3

Pre-registered verification order: 6 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/70ab29b1-bcfa-48a3-babf-cfc336efaa1c?experiment=5849dfd3-5d64-4e59-8faa-ce984857f01f&name=10seed-monk2-s3), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 3 |
| Genome id | `70ab29b1-bcfa-48a3-babf-cfc336efaa1c` |
| Experiment | 10seed-monk2-s3 (5849dfd3-5d64-4e59-8faa-ce984857f01f); best at generation 99 of 100, NEAT key 24414 |
| Test AUC (runs csv) | 0.85 \[95% CI 0.746, 0.937\] |
| Train AUC | 0.915 |
| Size | 13 of 17 inputs connected; 3 hidden nodes; 21 enabled connections; depth 3; 2 skip connections + 10 direct input→output edges |
| Annotations | 3 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 3 of 3 |
| Claude's time | 48.5 min (2026-11-03T13:12:35+13:00 → 2026-11-03T14:01:02+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> sigmoid(1.855 + Ramp - 7.41 Gate + 0.618 Chain): Ramp adds 3.6-10.5 logit per first value of a1,a2,a3,a4,a6 and \~-8 for a5=1; Gate fires iff \>=3 of {a1,a2,a3,a4,a6}=1. Predicts positive roughly iff exactly two of those five AND a5!=1. Train acc 0.901, AUC 0.915.

**Agreement with the MONK-2 rule (record):** 'Exactly two of six' with a5 mishandled: a5=1 is a veto rather than an indicator, so all 20 train positives with a5=1 plus one other indicator are missed; approximate rule agrees with the model on 418/432 configurations.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `two_of_five_with_a5_veto`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | 419, id\_linear, 2480, -5 (a2:2), -6 (a2:3), -10 (a4:2), -14 (a5:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | count\_gate\_atleast3\_of5, a2\_a5\_bonus\_chain, linear\_indicator\_ramp |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{0.618 a_{2 a5 bonus chain}{\left(-4_{c},-15 \right)} - 7.41 \operatorname{count\_gate\_atleast3\_of5}{\left(-1,-4_{b},-8,-9_{b},-16_{b},-17_{b} \right)} + 1.0 \operatorname{linear\_indicator\_ramp}{\left(-2,-3,-4_{a},-7,-9_{a},-11,-12,-13,-16_{a},-17_{a} \right)}}}{e^{0.618 a_{2 a5 bonus chain}{\left(-4_{c},-15 \right)} - 7.41 \operatorname{count\_gate\_atleast3\_of5}{\left(-1,-4_{b},-8,-9_{b},-16_{b},-17_{b} \right)} + 1.0 \operatorname{linear\_indicator\_ramp}{\left(-2,-3,-4_{a},-7,-9_{a},-11,-12,-13,-16_{a},-17_{a} \right)}} + 0.156}\end{gathered}
$$

**Hypothesis (DB):**

> The output composes the three children additively under the sigmoid: logit = 1.855 + Ramp - 7.413\*Gate + 0.618\*Chain. Ramp: the first value of a1,a2,a3,a4,a6 raises the sum (by 3.6-10.5 each), so with the bias it crosses zero between one and two active indicators among {a1=1,a2=1,a3=1,a4=1,a6=1} when a5!=1 (lower edge of a response peak); a5=1 instead costs about 8 logit units, pushing count-two cases below zero (an a5=1 veto). Gate: when three or more of the same five indicators are active it fires (pre-activation \>=1.76) and its -7.41 readout overwhelms the Ramp (upper edge). Chain: +0.89 when a2!=1 and a5!=4; flips only 2 train predictions and is near-vestigial. Net: the model predicts positive approximately iff exactly two of {a1=1,a2=1,a3=1,a4=1,a6=1} hold AND a5!=1. Evidence (train, n=345): this rule agrees with the model on 331/345 rows (418/432 of all configurations); the 14 disagreements are near-boundary cases (\|logit\| \< 1.8). Train accuracy 0.901, AUC 0.915; every train positive with a5=1 and exactly one other indicator (20 rows) is missed. Ablation: zeroing the gate drops train accuracy to 0.58; zeroing the chain gives 0.907. Inputs a2:2, a2:3, a4:2, a5:3 have no enabled edges and are zero-effect entries (sensitivity exactly 0).

**Evidence (4 records, DB):**

- `analytical_formula`: Output = sigmoid(1.855 + Ramp - 7.41 Gate + 0.618 Chain); 0.156 = exp(-1.855).
- `empirical_viz`: Output sensitivity on full train split; the four unconnected inputs have exactly zero effect.
- `input_output_sampling`: cnt5 = number of {a1=1,a2=1,a3=1,a4=1,a6=1}. The model predicts positive almost only for cnt5==2 with a5!=1; all 20 train positives with a5=1 plus one other indicator are missed (the a5=1 veto).
- `ablation_study`: The gate is essential (removing it drops train accuracy to 0.58). The chain flips only 2 train predictions and its removal slightly improves train accuracy (0.907): it is a near-vestigial nudge.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "labels": [
  "419",
  "id_linear",
  "2480",
  "-5",
  "-6",
  "-10",
  "-14"
 ],
 "viz_type": "sensitivity",
 "sensitivities": {
  "0": [
   0.25225779973386325,
   0.03420638836909089,
   0.021149611675033004,
   0.0,
   0.0,
   0.0,
   0.0
  ]
 }
}
```

```json
// input_output_sampling
{
 "n": 345,
 "rows": [
  {
   "n": 19,
   "y": 0.0,
   "a5=1": false,
   "cnt5": 0,
   "pred1": 0.0
  },
  {
   "n": 6,
   "y": 0.0,
   "a5=1": true,
   "cnt5": 0,
   "pred1": 0.0
  },
  {
   "n": 64,
   "y": 0.0,
   "a5=1": false,
   "cnt5": 1,
   "pred1": 0.08
  },
  {
   "n": 20,
   "y": 1.0,
   "a5=1": true,
   "cnt5": 1,
   "pred1": 0.0
  },
  {
   "n": 93,
   "y": 1.0,
   "a5=1": false,
   "cnt5": 2,
   "pred1": 0.91
  },
  {
   "n": 31,
   "y": 0.0,
   "a5=1": true,
   "cnt5": 2,
   "pred1": 0.03
  },
  {
   "n": 112,
   "y": 0.0,
   "a5=1": "any",
   "cnt5": ">=3",
   "pred1": 0.0
  }
 ],
 "split": "train",
 "train_acc": 0.901,
 "agreement_with_two_of_five_and_a5_not_1": {
  "train": 0.959,
  "all_432": 0.968
 }
}
```

```json
// ablation_study
{
 "n": 345,
 "split": "train",
 "method": "local numpy forward pass, child readout zeroed",
 "acc_full": 0.901,
 "acc_ramp_only": 0.594,
 "acc_without_gate": 0.583,
 "acc_without_chain": 0.907,
 "chain_flips_predictions": 2
}
```

</details>

### A.1 Leaf: `count_gate_atleast3_of5`

|  |  |
|---|---|
| Annotation id | `ann_22` |
| Entry nodes | -1 (a1:1), -4\_b (a2:1\_b), -8 (a3:2), -9\_b (a4:1\_b), -16\_b (a6:1\_b), -17\_b (a6:2\_b) |
| Exit nodes | 419 |
| Internal / exit nodes in its own subgraph | 419 |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |
| Live closed-form error (3 s.f. display) | 0.00943 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 4.54 x_{a1:1} \\
&\quad + 5.63 x_{a2:1\_b} \\
&\quad - 7.62 x_{a3:2} \\
&\quad + 5.53 x_{a4:1\_b} \\
&\quad + 6.54 x_{a6:1\_b} \\
&\quad - 1.37 x_{a6:2\_b} \\
&\quad -4.58 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> An at-least-three-of-five indicator detector over {a1=1, a2=1, a3=1, a4=1, a6=1} that ignores a5. Node 419 = ReLU(-4.585 + 4.540 z(a1:1) + 5.631 z(a2:1) - 7.622 z(a3:2) + 5.529 z(a4:1) + 6.541 z(a6:1) - 1.371 z(a6:2)). In indicator terms the pre-activation is -31.60 (no indicators active) plus a1 +9.50, a2 +11.92, a3 +15.25 (via the complement a3:2), a4 +11.95, a6 +15.83 (via a6:1 and the complement a6:2). The largest two-indicator sum (a3+a6 = 31.08) leaves it at -0.52 (off); the smallest three-indicator sum (a1+a2+a4 = 33.37) reaches +1.76 (on). So the unit fires iff at least three of these five indicators are active. Evidence (train, n=345, and all 432 configurations): fires iff \>=3 of the five indicators hold, agreement 1.0; max pre-activation with \<=2 active is -0.525. Sensitivity is spread over all six entries, smallest for a6:2 (0.44).

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (inputs are train-standardised one-hot columns).
- `empirical_viz`: Sensitivity of the exit to each entry on the full train split (n=345).
- `input_output_sampling`: On all 345 train rows (and all 432 possible configurations) node 419 is \>0 iff at least three of {a1=1,a2=1,a3=1,a4=1,a6=1} hold; with \<=2 active its pre-activation is at most -0.525. a5 plays no role.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "labels": [
  "-1",
  "-4_b",
  "-8",
  "-9_b",
  "-16_b",
  "-17_b"
 ],
 "viz_type": "sensitivity",
 "sensitivities": {
  "419": [
   1.473911386490396,
   1.828112286804042,
   2.4743751714736084,
   1.7947634311326295,
   2.1234985428848328,
   0.4449373111481926
  ]
 }
}
```

```json
// input_output_sampling
{
 "n": 345,
 "split": "train",
 "also_exhaustive_432": "fires on 31.5% of all 432 configs, exactly those with >=3 of {a1=1,a2=1,a3=1,a4=1,a6=1}",
 "gate_fires_iff_cnt5_ge_3": 1.0,
 "max_preact_when_cnt5_le_2": -0.525
}
```

</details>

### A.2 Leaf: `a2_a5_bonus_chain`

|  |  |
|---|---|
| Annotation id | `ann_23` |
| Entry nodes | -4\_c (a2:1\_c), -15 (a5:4) |
| Exit nodes | 2480 |
| Internal / exit nodes in its own subgraph | 1701, 2480 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 7.59e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.12 x_{a5:4} \\
&\quad - 0.915 \operatorname{relu}{\left(0.732 x_{a2:1\_c} + 0.632 \right)} \\
&\quad + 0.9 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A small serial chain giving a bonus when neither a2=1 nor a5=4. Node 1701 = ReLU(0.632 + 0.732 z(a2:1)) is always positive (0.111 if a2!=1, 1.660 if a2=1), so it is a linear relay of a2:1. Node 2480 = ReLU(0.900 - 1.120 z(a5:4) - 0.915 \* n1701) takes three values: 1.445 when a2!=1 and a5!=4, 0.026 when a2=1 and a5!=4, and 0 when a5=4 (ReLU clips). Through its +0.618 readout it adds about +0.89 to the output logit only when a2!=1 and a5!=4: a weak, AND-of-negations nudge. Evidence (train, n=345): n2480 \> 1 iff a2!=1 and a5!=4, agreement 1.0. Local ablation: zeroing its readout flips only 2 train predictions and slightly raises train accuracy (0.901 -\> 0.907), so it is near-vestigial.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (inputs are train-standardised one-hot columns).
- `empirical_viz`: Sensitivity of the exit to each entry on the full train split (n=345).
- `input_output_sampling`: On all 345 train rows, n2480 \> 1 iff a2!=1 and a5!=4 (agreement 1.0); otherwise it is 0.026 or 0. Readout +0.618, so the effect on the logit is +0.89 or \~0.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "labels": [
  "-4_c",
  "-15"
 ],
 "viz_type": "sensitivity",
 "sensitivities": {
  "2480": [
   0.5030023127262998,
   0.8407578921784863
  ]
 }
}
```

```json
// input_output_sampling
{
 "n": 345,
 "split": "train",
 "values": {
  "a5=4": 0.0,
  "a2=1 & a5!=4": 0.026,
  "a2!=1 & a5!=4": 1.445
 },
 "bonus_iff_not_a2_1_and_not_a5_4": 1.0
}
```

</details>

### A.3 Leaf: `linear_indicator_ramp`

|  |  |
|---|---|
| Annotation id | `ann_24` |
| Entry nodes | -2 (a1:2), -3 (a1:3), -4\_a (a2:1\_a), -7 (a3:1), -9\_a (a4:1\_a), -11 (a4:3), -12 (a5:1), -13 (a5:2), -16\_a (a6:1\_a), -17\_a (a6:2\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 10 → 1 |
| Live closed-form error (3 s.f. display) | 0.0148 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 3.16 x_{a1:2} \\
&\quad - 2.39 x_{a1:3} \\
&\quad + 4.98 x_{a2:1\_a} \\
&\quad + 4.04 x_{a3:1} \\
&\quad + 1.67 x_{a4:1\_a} \\
&\quad - 0.722 x_{a4:3} \\
&\quad - 3.54 x_{a5:1} \\
&\quad + 0.266 x_{a5:2} \\
&\quad + 0.779 x_{a6:1\_a} \\
&\quad - 3.73 x_{a6:2\_a}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A linear indicator ramp with an a5=1 veto. Evaluated per attribute value, the direct connections contribute: a1 = 1/2/3: +3.85/-2.84/-1.31; a2 = 1/other: +7.00/-3.54; a3 = 1/2: +3.93/-4.14; a4 = 1/2/3: +2.98/-0.63/-2.19; a6 = 1/2: +4.40/-4.63. So for a1,a2,a3,a4,a6 the first value raises the sum by 3.6 to 10.5 units, making the sum rise roughly linearly with the number of these indicators. Attribute a5 is inverted: a5 = 1/2/3/4 contributes -6.25/+2.51/+1.90/+1.90, i.e. a5=1 is an 8-unit penalty rather than a positive indicator. Together with the output bias (+1.855) the ramp crosses zero between one and two active indicators among {a1,a2,a3,a4,a6} when a5!=1, and an a5=1 pushes it back below zero. Evidence: the per-value contributions are exact (linear in the one-hot columns); sensitivity on train is largest for a2:1 (4.98), a3:1 (4.04), a6:2 (3.73), a5:1 (3.54), a1:2 (3.16). Note a5=1 acts as a veto, not an indicator.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (inputs are train-standardised one-hot columns).
- `empirical_viz`: Sensitivity of the exit to each entry on the full train split (n=345).
- `sign_analysis`: Per attribute value contribution of the direct edges (a2:2, a2:3, a4:2, a5:3 have no edge, so they enter only via the shift of the other columns). First value is positive for a1,a2,a3,a4,a6 but strongly negative for a5.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=true)"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "labels": [
  "-2",
  "-3",
  "-4_a",
  "-7",
  "-9_a",
  "-11",
  "-12",
  "-13",
  "-16_a",
  "-17_a"
 ],
 "viz_type": "sensitivity",
 "sensitivities": {
  "id_linear": [
   3.161782945475703,
   2.38894136769207,
   4.979944084722777,
   4.035992882860966,
   1.6692492359041105,
   0.7224567476408361,
   3.5371592965753496,
   0.26569424216548887,
   0.7790374255098085,
   3.7318910317611307
  ]
 }
}
```

```json
// sign_analysis
{
 "per_value_contribution": {
  "a1": [
   3.852,
   -2.841,
   -1.313
  ],
  "a2": [
   6.997,
   -3.544,
   -3.544
  ],
  "a3": [
   3.932,
   -4.143
  ],
  "a4": [
   2.977,
   -0.632,
   -2.19
  ],
  "a5": [
   -6.246,
   2.512,
   1.9,
   1.9
  ],
  "a6": [
   4.395,
   -4.63
  ]
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 4 |
| rename\_node | 17 |
| split\_node | 4 |

4 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -4 into -4\_a, -4\_b, -4\_c. Final copies: -4\_a → linear\_indicator\_ramp (entry) \| -4\_b → count\_gate\_atleast3\_of5 (entry) \| -4\_c → a2\_a5\_bonus\_chain (entry)
- `#18` split -9 into -9\_a, -9\_b. Final copies: -9\_a → linear\_indicator\_ramp (entry) \| -9\_b → count\_gate\_atleast3\_of5 (entry)
- `#19` split -16 into -16\_a, -16\_b. Final copies: -16\_a → linear\_indicator\_ramp (entry) \| -16\_b → count\_gate\_atleast3\_of5 (entry)
- `#20` split -17 into -17\_a, -17\_b. Final copies: -17\_a → linear\_indicator\_ramp (entry) \| -17\_b → count\_gate\_atleast3\_of5 (entry)
- `#21` identity node id\_linear intercepts 10 connection(s) into 0 (from -2 (a1:2), -3 (a1:3), -4\_a (a2:1\_a), -7 (a3:1), -9\_a (a4:1\_a), -11 (a4:3), -12 (a5:1), -13 (a5:2), … (2 more)); sits in linear\_indicator\_ramp (exit); two\_of\_five\_with\_a5\_veto (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 6.10e-16 | 6.11e-16 | 6.11e-16 | agree |
| Closed-form max abs error (all leaves) | 0.0148 | 0.01476 | 0.01476 | agree |
| Numeric-function max abs error | n/a | 3.55e-15 | 3.55e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | two\_of\_five\_with\_a5\_veto | two\_of\_five\_with\_a5\_veto | agree |
| Leaf / composition counts | 3 / 1 | n/a | 3 / 1 | agree |
| Operation counts | 4 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 17, "split\_node": 4} | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 17, "split\_node": 4} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `linear_indicator_ramp` | 0.01476 | 3.55e-15 |
| `count_gate_atleast3_of5` | 0.009426 | 3.55e-15 |
| `a2_a5_bonus_chain` | 7.59e-04 | 0 |

## What resisted (from the record)

> Chain 1701-\>2480 (+0.89 logit when a2!=1 and a5!=4) has no clear role: flips 2 train predictions and removing it slightly raises accuracy; explained as a near-vestigial nudge.

**Tool notes (record):** add\_identity\_node docstring params wrong.

## What to look at

- Largest closed-form error: `linear_indicator_ramp` (0.0148). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `a2_a5_bonus_chain`: *“Through its +0.618 readout it adds about +0.89 to the output logit only when a2!=1 and a5!=4: a weak, AND-of-negations nudge.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `linear_indicator_ramp`: *“Attribute a5 is inverted: a5 = 1/2/3/4 contributes -6.25/+2.51/+1.90/+1.90, i.e.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 4 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s3.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s3', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
