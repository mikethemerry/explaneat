# Verification packet 16 · monk2-s0

Pre-registered verification order: 16 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/e324239d-f591-4c8a-b149-7699ec2d0346?experiment=18010622-b77f-4dc4-8dc3-b08f35d7bb22&name=10seed-monk2-s0), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 0 |
| Genome id | `e324239d-f591-4c8a-b149-7699ec2d0346` |
| Experiment | 10seed-monk2-s0 (18010622-b77f-4dc4-8dc3-b08f35d7bb22); best at generation 99 of 100, NEAT key 23293 |
| Test AUC (runs csv) | 0.639 \[95% CI 0.517, 0.749\] |
| Train AUC | 0.807 |
| Size | 14 of 17 inputs connected; 3 hidden nodes; 23 enabled connections; depth 2; 0 skip connections + 11 direct input→output edges |
| Annotations | 4 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 3 of 3 |
| Claude's time | 42.9 min (2026-11-03T13:12:36+13:00 → 2026-11-03T13:55:29+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Additive logit under a saturating sigmoid, dominated by ReLU node 263 (about 19 with no first values, dropping 6.8-9.3 per first value of a2, a3, a4, a6). Direct linear term penalises a1=1 and a5=1 (about -15 each) and gives a1=3 +24.9. Nodes 1855 and 3471 are minor.

**Agreement with the MONK-2 rule (record):** Learned only the upper edge (\>=3 first values, \>=4 when a1=2, makes output negative); no lower edge, so 0-1 first values get the highest logits; ignores the count entirely when a1=3. Predicts positive on 85.5% of train rows; train accuracy 0.472.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `upper_edge_count_classifier`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | 263, 1855, 3471, id\_linear, -10 (a4:2), -11 (a4:3), -14 (a5:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | few\_first\_values\_ramp, a5\_not4\_unit, a6\_tiny\_unit, direct\_linear |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.53 a_{5 not4 unit}{\left(-7,-15_{b} \right)} + 0.105 a_{6 tiny unit}{\left(-16 \right)} + 1.0 \operatorname{direct\_linear}{\left(-1,-3_{a},-4_{a},-5,-6,-8_{a},-9_{a},-12,-13,-15_{a},-17_{a} \right)} + 5.14 \operatorname{few\_first\_values\_ramp}{\left(-2,-3_{b},-4_{b},-8_{b},-9_{b},-17_{b} \right)}}}{e^{1.53 a_{5 not4 unit}{\left(-7,-15_{b} \right)} + 0.105 a_{6 tiny unit}{\left(-16 \right)} + 1.0 \operatorname{direct\_linear}{\left(-1,-3_{a},-4_{a},-5,-6,-8_{a},-9_{a},-12,-13,-15_{a},-17_{a} \right)} + 5.14 \operatorname{few\_first\_values\_ramp}{\left(-2,-3_{b},-4_{b},-8_{b},-9_{b},-17_{b} \right)}} + 0.0758}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: additive composition under the output sigmoid, logit = 2.58 + Linear + 5.14\*Ramp + 1.53\*A5not4 + 0.105\*A6. a4:2, a4:3, a5:3 have no enabled connections (zero-effect entries). The model is an upper-edge-only count detector: for a1 in {1,2} it is positive while few attributes take their first value and negative once about three of {a2,a3,a4,a6} (a1=1) or four (a1=2) do; a5=1 counts only through Linear's -14.8, which is not enough to flip a1=1 inputs with one other first value. There is no lower edge: inputs with zero or one first value get the largest logits. a1=3 inputs are all positive (107/107 on train) via Linear's +24.9. Evidence (train, n=345): positive rate by (a1, count) - a1=1: count\<=2 100%, 3 74%, \>=4 0%; a1=2: count\<=3 100%, 4 29%, 5 0%; a1=3: 100% at every count. Predicts positive on 85.5% of train rows (accuracy 0.472 at 0.5). Logits reach \|\~130\| so the sigmoid saturates; the sigmoid-output AUC 0.807 is inflated by ties at 1.0, the logit ranking AUC is 0.734. Compared with the MONK-2 target (exactly two first values): the network captures only the upper edge (\>=3-4 first values -\> negative) and none of the lower edge (0-1 -\> negative), and ignores the count entirely when a1=3.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: get\_viz\_summary sensitivity on the full training split (345 rows).
- `input_output_sampling`: Local forward pass (read-only repo weights) on the training split: positive unless many attributes take their first value; a1=3 always positive.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "entry_names": [
  "263",
  "1855",
  "3471",
  "id_linear",
  "-10",
  "-11",
  "-14"
 ],
 "sensitivities": {
  "0": [
   0.053256434764345016,
   0.015911335636611724,
   0.0010907177173617806,
   0.010408227565812675,
   0.0,
   0.0,
   0.0
  ]
 },
 "dataset_split_id": "63257a45-49c9-42a6-8b28-8e346eab2bee"
}
```

```json
// input_output_sampling
{
 "n": 345,
 "note": "sigmoid saturates (|logit| up to ~130) so many outputs tie at 1.0; the sigmoid-output AUC is inflated by ties relative to the logit ranking AUC",
 "split": "train",
 "train_auc_logit": 0.7335,
 "train_auc_sigmoid_output": 0.8065,
 "train_predicted_positive_rate": 0.855,
 "positive_rate_by_a1_and_first_value_count": {
  "a1=1": {
   "1": "9/9 pos (y=0)",
   "2": "38/38 pos (y=1)",
   "3": "74% pos (y=0)",
   "4": "0/26",
   "5": "0/6",
   "6": "0/1"
  },
  "a1=2": {
   "0": "11/11 pos (y=0)",
   "1": "29/29 pos (y=0)",
   "2": "40/40 pos (y=1)",
   "3": "28/28 pos (y=0)",
   "4": "29% pos",
   "5": "0/1"
  },
  "a1=3": {
   "all counts": "108/108 pos"
  }
 }
}
```

</details>

### A.1 Leaf: `few_first_values_ramp`

|  |  |
|---|---|
| Annotation id | `ann_24` |
| Entry nodes | -2 (a1:2), -3\_b (a1:3\_b), -4\_b (a2:1\_b), -8\_b (a3:2\_b), -9\_b (a4:1\_b), -17\_b (a6:2\_b) |
| Exit nodes | 263 |
| Internal / exit nodes in its own subgraph | 263 |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |
| Live closed-form error (3 s.f. display) | 0.0413 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.74 x_{a1:2} \\
&\quad - 11.2 x_{a1:3\_b} \\
&\quad - 4.0 x_{a2:1\_b} \\
&\quad + 4.65 x_{a3:2\_b} \\
&\quad - 3.12 x_{a4:1\_b} \\
&\quad + 4.11 x_{a6:2\_b} \\
&\quad -0.47 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: a falling count ramp over first-value indicators. In indicator terms node 263 = ReLU(18.98 - 8.46\[a2=1\] - 9.29\[a3=1\] - 6.75\[a4=1\] - 8.23\[a6=1\] + 5.81\[a1=2\] - 24.11\[a1=3\]); a3:2 and a6:2 enter as complements of a3=1, a6=1. a1=1: positive iff at most two of {a2,a3,a4,a6}=1 (all pairs \>= 1.23, all triples \<= -4.46). a1=2: +5.8 shift, three indicators leave it at 0.3-1.4 except a2+a3+a6 (-1.19); four switch it off. a1=3: always 0 (max pre -5.13). a5 is not consulted. Evidence (train, n=345): closed form from get\_formula (z-scored inputs; closed-form max err 0.041 from 3-s.f. rounding); sensitivity largest for a1:3 (6.66); values 0..24.79, zero on 40.3% of rows; ablation flips 112/345 decisions and drops logit AUC 0.734 -\> 0.537. Readout 5.14: the dominant decision-bearing term.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: get\_viz\_summary sensitivity on the full training split (345 rows).
- `ablation_study`: Removing the ramp flips 112/345 training decisions and drops logit AUC 0.734 -\> 0.537: it is the main decision-bearing unit.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "entry_names": [
  "-2",
  "-3_b",
  "-4_b",
  "-8_b",
  "-9_b",
  "-17_b"
 ],
 "sensitivities": {
  "263": [
   1.6382505895701316,
   6.659920196138448,
   2.3875515035123724,
   2.773809549701637,
   1.8630206298562064,
   2.4551468952758646
  ]
 },
 "dataset_split_id": "63257a45-49c9-42a6-8b28-8e346eab2bee"
}
```

```json
// ablation_study
{
 "n": 345,
 "split": "train",
 "ablate_to_zero": {
  "auc_pre_logit": [
   0.7335,
   0.5372
  ],
  "decision_flips": 112
 },
 "values_on_train": "0 to 24.79; zero on 40.3% of rows"
}
```

</details>

### A.2 Leaf: `a5_not4_unit`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -7 (a3:1), -15\_b (a5:4\_b) |
| Exit nodes | 1855 |
| Internal / exit nodes in its own subgraph | 1855 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00262 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.341 x_{a3:1} \\
&\quad - 1.42 x_{a5:4\_b} \\
&\quad -0.171 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: an a5-not-4 indicator weighted by a3. Node 1855 = ReLU(0.295 + 0.682\[a3=1\] - 3.271\[a5=4\]) takes values 0 (a5=4), 0.295 (a3=2) or 0.977 (a3=1). Evidence (train, n=345): closed form (err 0.0026 from rounding); sensitivity a5:4 1.06 vs a3:1 0.26; readout 1.53 adds at most 1.49 to the logit; ablation flips 3/345 decisions, logit AUC 0.7335 -\> 0.7329. A minor correction term.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: get\_viz\_summary sensitivity on the full training split (345 rows).
- `ablation_study`: Ablating flips 3/345 decisions; AUC essentially unchanged.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "entry_names": [
  "-7",
  "-15_b"
 ],
 "sensitivities": {
  "1855": [
   0.25600889782758346,
   1.0624454188912973
  ]
 },
 "dataset_split_id": "63257a45-49c9-42a6-8b28-8e346eab2bee"
}
```

```json
// ablation_study
{
 "n": 345,
 "split": "train",
 "unique_values": [
  0,
  0.295,
  0.977
 ],
 "ablate_to_zero": {
  "auc_pre_logit": [
   0.7335,
   0.7329
  ],
  "decision_flips": 3
 }
}
```

</details>

### A.3 Leaf: `a6_tiny_unit`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -16 (a6:1) |
| Exit nodes | 3471 |
| Internal / exit nodes in its own subgraph | 3471 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 3.05e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.0186 - 0.706 x_{a6:1} \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: a near-inert a6 indicator. Node 3471 = ReLU(0.743 - 1.412\[a6=1\]) = 0.743\[a6=2\], linear in binary a6. Evidence (train, n=345): closed form (err 0.0003); readout 0.105 adds at most 0.078 to the logit; ablation flips no decisions. Functionally negligible, unearned structure.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: get\_viz\_summary sensitivity on the full training split (345 rows).
- `ablation_study`: Ablating flips no decisions; contribution at most 0.078 to the logit.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "entry_names": [
  "-16"
 ],
 "sensitivities": {
  "3471": [
   0.34365997729713504
  ]
 },
 "dataset_split_id": "63257a45-49c9-42a6-8b28-8e346eab2bee"
}
```

```json
// ablation_study
{
 "n": 345,
 "split": "train",
 "unique_values": [
  0,
  0.743
 ],
 "ablate_to_zero": {
  "auc_pre_logit": [
   0.7335,
   0.7336
  ],
  "decision_flips": 0
 }
}
```

</details>

### A.4 Leaf: `direct_linear`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | -1 (a1:1), -3\_a (a1:3\_a), -4\_a (a2:1\_a), -5 (a2:2), -6 (a2:3), -8\_a (a3:2\_a), -9\_a (a4:1\_a), -12 (a5:1), -13 (a5:2), -15\_a (a5:4\_a), -17\_a (a6:2\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 11 → 1 |
| Live closed-form error (3 s.f. display) | 0.065 |
| Live numeric-function error | 7.11e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 7.5 x_{a1:1} \\
&\quad + 11.5 x_{a1:3\_a} \\
&\quad + 0.566 x_{a2:1\_a} \\
&\quad + 0.342 x_{a2:2} \\
&\quad + 0.291 x_{a2:3} \\
&\quad - 1.09 x_{a3:2\_a} \\
&\quad - 2.27 x_{a4:1\_a} \\
&\quad - 6.43 x_{a5:1} \\
&\quad + 1.03 x_{a5:2} \\
&\quad - 1.55 x_{a5:4\_a} \\
&\quad - 0.12 x_{a6:2\_a}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis: the direct linear part. Per attribute in indicator terms (relative to the reference value): a1=1 -15.69, a1=3 +24.94 (a1=2: 0); a5=1 -14.80, a5=2 +2.38, a5=4 -3.58 (a5=3: 0); a4=1 -4.90; a3=2 -2.19; a2=1 +1.20, a2=2 +0.73, a2=3 +0.61; a6=2 -0.24. It penalises the two first values the ramp does not see (a1=1, a5=1) by \~15 each, and gives a1=3 a +24.9 bonus that keeps every a1=3 input positive although the ramp is off there. Evidence (train, n=345): closed form (linear; err 0.065 from 3-s.f. rounding on values -33.4..+32.2); sensitivities equal the z-scale weights (a1:3 11.5, a1:1 7.5, a5:1 6.4).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: get\_viz\_summary sensitivity on the full training split (345 rows).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "entry_names": [
  "-1",
  "-3_a",
  "-4_a",
  "-5",
  "-6",
  "-8_a",
  "-9_a",
  "-12",
  "-13",
  "-15_a",
  "-17_a"
 ],
 "sensitivities": {
  "id_linear": [
   7.5027915901589495,
   11.533657645065572,
   0.5658672960665069,
   0.34242398489174514,
   0.29073330237895684,
   1.0921364759583199,
   2.265616515262535,
   6.427502303670707,
   1.03210734238609,
   1.5496885550259567,
   0.12033521313459289
  ]
 },
 "dataset_split_id": "63257a45-49c9-42a6-8b28-8e346eab2bee"
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 5 |
| rename\_node | 17 |
| split\_node | 6 |

6 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -3 into -3\_a, -3\_b. Final copies: -3\_a → direct\_linear (entry) \| -3\_b → few\_first\_values\_ramp (entry)
- `#18` split -4 into -4\_a, -4\_b. Final copies: -4\_a → direct\_linear (entry) \| -4\_b → few\_first\_values\_ramp (entry)
- `#19` split -8 into -8\_a, -8\_b. Final copies: -8\_a → direct\_linear (entry) \| -8\_b → few\_first\_values\_ramp (entry)
- `#20` split -9 into -9\_a, -9\_b. Final copies: -9\_a → direct\_linear (entry) \| -9\_b → few\_first\_values\_ramp (entry)
- `#21` split -17 into -17\_a, -17\_b. Final copies: -17\_a → direct\_linear (entry) \| -17\_b → few\_first\_values\_ramp (entry)
- `#22` split -15 into -15\_a, -15\_b. Final copies: -15\_a → direct\_linear (entry) \| -15\_b → a5\_not4\_unit (entry)
- `#23` identity node id\_linear intercepts 11 connection(s) into 0 (from -1 (a1:1), -3\_a (a1:3\_a), -4\_a (a2:1\_a), -5 (a2:2), -6 (a2:3), -8\_a (a3:2\_a), -9\_a (a4:1\_a), -12 (a5:1), … (3 more)); sits in direct\_linear (exit); upper\_edge\_count\_classifier (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 8.33e-16 | 8.33e-16 | 8.33e-16 | agree |
| Closed-form max abs error (all leaves) | 0.065 | 0.06501 | 0.06501 | agree |
| Numeric-function max abs error | n/a | 7.11e-15 | 7.11e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | upper\_edge\_count\_classifier | upper\_edge\_count\_classifier | agree |
| Leaf / composition counts | 4 / 1 | n/a | 4 / 1 | agree |
| Operation counts | 6 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 5, "rename\_node": 17, "split\_node": 6} | {"add\_identity\_node": 1, "annotate": 5, "rename\_node": 17, "split\_node": 6} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `direct_linear` | 0.06501 | 7.11e-15 |
| `few_first_values_ramp` | 0.04129 | 8.88e-16 |
| `a5_not4_unit` | 0.002618 | 0 |
| `a6_tiny_unit` | 3.05e-04 | 0 |

## What resisted (from the record)

> Nothing structural. Closed-form error 0.065 is 3-s.f. rounding on a linear leaf with range -33..+32. The a1=1/count-3 boundary is pattern-specific (74% positive), described via per-indicator weights rather than a clean rule. One root evidence payload says a1=3 '108/108 positive' (correct: 107/107; narrative corrected, payload not).

**Tool notes (record):** Root annotate failed when subgraph\_nodes included child exits ('already covered'); worked with output + unconnected inputs. Sigmoid saturation (\|logit\| up to \~130) makes many outputs tie at exactly 1.0, so the sigmoid AUC (0.807, = fitness) is higher than the logit ranking AUC (0.734).

## What to look at

- Largest closed-form error: `direct_linear` (0.065), then `few_first_values_ramp` (0.0413). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- 6 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s0.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s0', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
