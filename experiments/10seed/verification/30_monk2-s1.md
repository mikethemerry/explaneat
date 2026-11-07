# Verification packet 30 · monk2-s1

Pre-registered verification order: 30 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/d32476a4-0b34-4678-ac6f-bf22ac17389d?experiment=3327bdb3-f6da-45f1-8321-54567c04f867&name=10seed-monk2-s1), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 1 |
| Genome id | `d32476a4-0b34-4678-ac6f-bf22ac17389d` |
| Experiment | 10seed-monk2-s1 (3327bdb3-f6da-45f1-8321-54567c04f867); best at generation 87 of 100, NEAT key 21319 |
| Test AUC (runs csv) | 0.998 \[95% CI 0.992, 1\] |
| Train AUC | 1 |
| Size | 12 of 17 inputs connected; 1 hidden nodes; 20 enabled connections; depth 2; 0 skip connections + 10 direct input→output edges |
| Annotations | 2 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 1 of 1 |
| Claude's time | 43.6 min (2026-11-03T13:12:36+13:00 → 2026-11-03T13:56:12+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> One hidden ReLU (an at-most-one-of-six gate, readout -7.61) plus a falling linear ramp (7-12 logit per first value): output = sigmoid(Ramp - 7.61 Gate - 0.93), peaking at count two. Mirrors the paper's MONK-2 construction with reversed ramp direction.

**Agreement with the MONK-2 rule (record):** Logit peaks at exactly two first values; ranks every count-2 row above all others (AUC 1.0) but only 86/113 count-2 rows exceed 0.5: computes the rule's ordering, not its threshold.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `two_indicator_peak`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | 773, id\_linear, -3 (a1:3), -5 (a2:2), -8 (a3:2), -13 (a5:2), -15 (a5:4) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | few\_indicators\_gate, descending\_count\_ramp |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.0 \operatorname{descending\_count\_ramp}{\left(-1_{b},-4_{b},-6,-7_{b},-9_{b},-10_{b},-11,-12_{b},-16_{b},-17 \right)} - 7.61 \operatorname{few\_indicators\_gate}{\left(-1_{a},-2,-4_{a},-7_{a},-9_{a},-10_{a},-12_{a},-14,-16_{a} \right)}}}{e^{1.0 \operatorname{descending\_count\_ramp}{\left(-1_{b},-4_{b},-6,-7_{b},-9_{b},-10_{b},-11,-12_{b},-16_{b},-17 \right)} - 7.61 \operatorname{few\_indicators\_gate}{\left(-1_{a},-2,-4_{a},-7_{a},-9_{a},-10_{a},-12_{a},-14,-16_{a} \right)}} + 2.53}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = \sigma(\mathrm{Ramp} - 7.61\,\mathrm{Gate} - 0.929)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Output = sigmoid(Ramp - 7.61 \* Gate - 0.93): an additive composition in logit space. The descending ramp makes counts \>= 3 negative; the at-most-one gate overrides the ramp's high values at counts 0-1. The logit therefore peaks at exactly two first-value indicators. Inputs a1:3, a2:2, a3:2, a5:2, a5:4 have no enabled connections and are zero-effect entries. Evidence (train, full sample): AUC 1.0 - every count-2 row outranks every other row - but the margin is thin (min positive logit -3.52 vs max negative -3.75), and the threshold is miscalibrated: only 86/113 count-2 rows exceed 0.5 (accuracy 0.922); the misses are pairs involving a3/a6/a5/a1 where the ramp's heavier weights push the logit below zero. No count != 2 row is predicted positive.

**Evidence (2 records, DB):**

- `analytical_formula`: Composition: additive in logit space under the output sigmoid; gate enters with weight -7.61, ramp with weight 1.
- `empirical_viz`: Train split, full sample. Logit peaks at count two; all count-2 rows outrank all other rows (AUC 1.0) but with a thin 0.23-logit margin, and 27/113 count-2 rows (mostly pairs involving a3 or a6, whose ramp weights are largest) fall below 0.5, giving accuracy 0.922.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "latex_tool": "y_0 = 1/(2.53 e^{7.61 x_{773} - x_{id\\_linear}} + 1)",
 "zero_effect_entries": [
  "a1:3",
  "a2:2",
  "a3:2",
  "a5:2",
  "a5:4"
 ]
}
```

```json
// empirical_viz
{
 "n": 345,
 "auc": 1.0,
 "split": "train",
 "max_logit_neg": -3.75,
 "min_logit_pos": -3.52,
 "auc_gate_alone": 0.673,
 "auc_ramp_alone": 0.616,
 "logit_by_count": {
  "0": [
   -99.5,
   -67.7
  ],
  "1": [
   -53.4,
   -12.5
  ],
  "2": [
   -3.52,
   7.72
  ],
  "3": [
   -13.7,
   -3.75
  ],
  "4": [
   -23.7,
   -14.0
  ],
  "5": [
   -31.9,
   -26.7
  ],
  "6": [
   -38.8,
   -38.8
  ]
 },
 "accuracy_at_0.5": 0.922,
 "low_count2_pairs": "(a3,a6) min -3.52, (a1,a6) -1.63, (a5,a6) -1.42, (a3,a5) -0.78, (a1,a3) -0.99",
 "predicted_positive_by_count": {
  "2": "86/113",
  "others": 0
 }
}
```

</details>

### A.1 Leaf: `few_indicators_gate`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -1\_a (a1:1\_a), -2 (a1:2), -4\_a (a2:1\_a), -7\_a (a3:1\_a), -9\_a (a4:1\_a), -10\_a (a4:2\_a), -12\_a (a5:1\_a), -14 (a5:3), -16\_a (a6:1\_a) |
| Exit nodes | 773 |
| Internal / exit nodes in its own subgraph | 773 |
| Dimensionality (in → out, incl. descendants) | 9 → 1 |
| Live closed-form error (3 s.f. display) | 0.0201 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 3.54 x_{a1:1\_a} \\
&\quad - 0.0739 x_{a1:2} \\
&\quad - 4.17 x_{a2:1\_a} \\
&\quad - 4.41 x_{a3:1\_a} \\
&\quad - 2.27 x_{a4:1\_a} \\
&\quad + 1.52 x_{a4:2\_a} \\
&\quad - 3.72 x_{a5:1\_a} \\
&\quad + 0.211 x_{a5:3} \\
&\quad - 4.16 x_{a6:1\_a} \\
&\quad -4.47 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = \operatorname{relu}(-3.54 x_{a1:1} - 0.0739 x_{a1:2} - 4.17 x_{a2:1} - 4.41 x_{a3:1} - 2.27 x_{a4:1} + 1.52 x_{a4:2} - 3.72 x_{a5:1} + 0.211 x_{a5:3} - 4.16 x_{a6:1} - 4.47)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> An at-most-one-of-six detector (lower edge of the peak). ReLU unit 773 = relu(-3.54 a1:1 - 4.17 a2:1 - 4.41 a3:1 - 2.27 a4:1 - 3.72 a5:1 - 4.16 a6:1 + 1.52 a4:2 - 0.07 a1:2 + 0.21 a5:3 - 4.47) on z-scored inputs. In indicator terms its pre-activation is about +12 to +16 when no attribute takes its first value, and each first-value indicator subtracts 4.9-8.8. Evidence (train, n=345, full sample): active on 19/19 count-0 and 70/70 count-1 rows, on only 7/113 count-2 rows (pairs a1&a4 and one a1&a6, activation \<= 0.52) and on 0/143 rows with count \>= 3. Its -7.61 readout drives low-count logits to between -99 and -12. Alone it ranks the classes with AUC 0.67.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form (sympy, 3 s.f.). Each first-value indicator lowers the pre-activation by 4.9-8.8 from a base of \~12-16, so the unit is on for counts 0-1 and off for counts \>= 3.
- `empirical_viz`: Train split, full sample (local forward pass on read-only code + get\_viz\_summary sensitivity). Gate active on every count-0/1 row, 7/113 count-2 rows with small activation (\<=0.52, i.e. \<=4 logit units after the -7.61 readout), never for count \>= 3.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "inputs": "z-scored one-hot",
 "per_indicator_drop": {
  "a1": 7.41,
  "a2": 8.81,
  "a3": 8.83,
  "a4": "4.91 (vs a4=3) / 8.04 (vs a4=2)",
  "a5": 8.56,
  "a6": 8.32
 },
 "preactivation_all_off": "about 12.4 to 16.0"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "count2_active_pairs": "a1&a4 (6 rows, h<=0.52), a1&a6 (1 row, h=0.24)",
 "sensitivity_mean_abs": {
  "a1:1": 0.98,
  "a1:2": 0.02,
  "a2:1": 1.14,
  "a3:1": 1.21,
  "a4:1": 0.63,
  "a4:2": 0.42,
  "a5:1": 1.02,
  "a5:3": 0.06,
  "a6:1": 1.14
 },
 "fraction_active_by_count": {
  "0": "19/19",
  "1": "70/70",
  "2": "7/113",
  "3": "0/92",
  "4": "0/42",
  "5": "0/8",
  "6": "0/1"
 }
}
```

</details>

### A.2 Leaf: `descending_count_ramp`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -1\_b (a1:1\_b), -4\_b (a2:1\_b), -6 (a2:3), -7\_b (a3:1\_b), -9\_b (a4:1\_b), -10\_b (a4:2\_b), -11 (a4:3), -12\_b (a5:1\_b), -16\_b (a6:1\_b), -17 (a6:2) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 10 → 1 |
| Live closed-form error (3 s.f. display) | 0.0259 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 4.88 x_{a1:1\_b} \\
&\quad - 3.85 x_{a2:1\_b} \\
&\quad + 1.77 x_{a2:3} \\
&\quad - 6.05 x_{a3:1\_b} \\
&\quad - 3.49 x_{a4:1\_b} \\
&\quad + 0.00806 x_{a4:2\_b} \\
&\quad - 0.28 x_{a4:3} \\
&\quad - 4.34 x_{a5:1\_b} \\
&\quad - 0.131 x_{a6:1\_b} \\
&\quad + 6.24 x_{a6:2}
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y = -4.88 x_{a1:1} - 3.85 x_{a2:1} + 1.77 x_{a2:3} - 6.05 x_{a3:1} - 3.49 x_{a4:1} + 0.00806 x_{a4:2} - 0.28 x_{a4:3} - 4.34 x_{a5:1} - 0.131 x_{a6:1} + 6.24 x_{a6:2}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> A linear ramp that falls with the number of first-value indicators (upper edge of the peak). The ten direct input-\>output edges give each attribute's first value a contribution 7-12 lower than its other values (a1 -10.2, a2 -8.2 vs a2=2 / -11.9 vs a2=3, a3 -12.1, a4 -7.6/-6.9, a5 -10.0, a6 -12.2). Evidence (train, full sample): ramp + output bias lies in \[21.3, 25.7\] at count 0, \[8.6, 18.2\] at count 1, \[-3.5, 7.9\] at count 2 and below -3.7 for every count \>= 3 row, so it crosses zero between two and three indicators. Weights are not uniform: a3 and a6 (binary attributes) carry the largest drops, so count-2 rows involving a3 or a6 sit closest to zero. Alone it ranks poorly (AUC 0.62) because counts 0-1 score highest.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form. Every attribute's first value contributes 7-12 less than its other values, so the ramp falls linearly with the count of first-value indicators.
- `empirical_viz`: Train split, full sample. Ramp+bias is positive on all count 0-1 rows, mostly positive on count-2 rows and negative on every count \>= 3 row. Alone it ranks poorly (AUC 0.62) because count 0-1 rows score highest.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "per_attribute_value_contribution": {
  "a1": {
   "1": -6.6,
   "2/3": 3.61
  },
  "a2": {
   "1": -6.69,
   "2": 1.46,
   "3": 5.2
  },
  "a3": {
   "1": -5.9,
   "2": 6.21
  },
  "a4": {
   "1": -5.02,
   "2": 2.54,
   "3": 1.92
  },
  "a5": {
   "1": -7.48,
   "other": 2.52
  },
  "a6": {
   "1": -5.95,
   "2": 6.28
  }
 }
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "auc_ramp_alone": 0.616,
 "ramp_plus_bias_range_by_count": {
  "0": [
   21.34,
   25.7
  ],
  "1": [
   8.59,
   18.15
  ],
  "2": [
   -3.52,
   7.93
  ],
  "3": [
   -13.74,
   -3.75
  ],
  "4": [
   -23.74,
   -13.96
  ],
  "5": [
   -31.89,
   -26.72
  ],
  "6": [
   -38.82,
   -38.82
  ]
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 3 |
| rename\_node | 17 |
| split\_node | 7 |

7 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` identity node id\_linear intercepts 10 connection(s) into 0 (from -1, -4, -6 (a2:3), -7, -9, -10, -11 (a4:3), -12, … (2 more)); sits in descending\_count\_ramp (exit); two\_indicator\_peak (entry)
- `#18` split -1 into -1\_a, -1\_b. Final copies: -1\_a → few\_indicators\_gate (entry) \| -1\_b → descending\_count\_ramp (entry)
- `#19` split -4 into -4\_a, -4\_b. Final copies: -4\_a → few\_indicators\_gate (entry) \| -4\_b → descending\_count\_ramp (entry)
- `#20` split -7 into -7\_a, -7\_b. Final copies: -7\_a → few\_indicators\_gate (entry) \| -7\_b → descending\_count\_ramp (entry)
- `#21` split -9 into -9\_a, -9\_b. Final copies: -9\_a → few\_indicators\_gate (entry) \| -9\_b → descending\_count\_ramp (entry)
- `#22` split -10 into -10\_a, -10\_b. Final copies: -10\_a → few\_indicators\_gate (entry) \| -10\_b → descending\_count\_ramp (entry)
- `#23` split -12 into -12\_a, -12\_b. Final copies: -12\_a → few\_indicators\_gate (entry) \| -12\_b → descending\_count\_ramp (entry)
- `#24` split -16 into -16\_a, -16\_b. Final copies: -16\_a → few\_indicators\_gate (entry) \| -16\_b → descending\_count\_ramp (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.44e-15 | 1.44e-15 | 1.44e-15 | agree |
| Closed-form max abs error (all leaves) | 0.02593 | 0.02593 | 0.02593 | agree |
| Numeric-function max abs error | n/a | 3.55e-15 | 3.55e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | two\_indicator\_peak | two\_indicator\_peak | agree |
| Leaf / composition counts | 2 / 1 | n/a | 2 / 1 | agree |
| Operation counts | 7 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 3, "rename\_node": 17, "split\_node": 7} | {"add\_identity\_node": 1, "annotate": 3, "rename\_node": 17, "split\_node": 7} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `descending_count_ramp` | 0.02593 | 3.55e-15 |
| `few_indicators_gate` | 0.02013 | 3.55e-15 |

## What resisted (from the record)

> Nothing. Non-uniform ramp weights (a3, a6 \~12) give perfect ranking (AUC 1.0) with a thin margin and a miscalibrated threshold: 27/113 count-2 rows below 0.5 (train acc 0.922).

**Tool notes (record):** Shell-quoted apply\_operation JSON failed silently. Root annotate fails if subgraph\_nodes include child exits.

## What to look at

- Largest closed-form error: `descending_count_ramp` (0.0259), then `few_indicators_gate` (0.0201). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `few_indicators_gate`: *“Evidence (train, n=345, full sample): active on 19/19 count-0 and 70/70 count-1 rows, on only 7/113 count-2 rows (pairs a1&a4 and one a1&a6, activation \<= 0.52) and on 0/143 rows with count \>= 3.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 7 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s1.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s1', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
