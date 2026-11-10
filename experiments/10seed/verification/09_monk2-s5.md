# Verification packet 09 · monk2-s5

Pre-registered verification order: 9 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/5715927f-d138-489c-884a-392704480cfe?experiment=a85000c9-0b94-47f1-9c73-eb95934c66db&name=10seed-monk2-s5), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 5 |
| Genome id | `5715927f-d138-489c-884a-392704480cfe` |
| Experiment | 10seed-monk2-s5 (a85000c9-0b94-47f1-9c73-eb95934c66db); best at generation 55 of 100, NEAT key 13307 |
| Test AUC (runs csv) | 1 \[95% CI 1, 1\] |
| Train AUC | 1 |
| Size | 11 of 17 inputs connected; 1 hidden nodes; 19 enabled connections; depth 2; 0 skip connections + 10 direct input→output edges |
| Annotations | 2 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 1 of 1 |
| Claude's time | 43.6 min (2026-11-03T13:12:35+13:00 → 2026-11-03T13:56:11+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Descending linear ramp switches the output off at \>=3 first values; a single ReLU gate (node 217) fires at 0-1 and suppresses via -5.64: sigmoid(Ramp - 1.20 - 5.64 Gate), peaking at exactly two. Mirrors the paper's construction with reversed ramp.

**Agreement with the MONK-2 rule (record):** Ranking matches 'exactly two of six' on all 432 configurations (AUC 1.0); at 0.5 it computes 'exactly two except most pairs with a6=1 and some a2&a4 / a4&a5', 89% of 432 correct.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `two_indicator_band`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | 217, id\_linear, -5 (a2:2), -10 (a4:2), -11 (a4:3), -13 (a5:2), -14 (a5:3), -15 (a5:4) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | few\_indicators\_gate, descending\_count\_ramp |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.0 \operatorname{descending\_count\_ramp}{\left(-1_{a},-2,-4_{a},-6_{a},-7,-8_{a},-9_{a},-12_{a},-16,-17_{a} \right)} - 5.64 \operatorname{few\_indicators\_gate}{\left(-1_{b},-3,-4_{b},-6_{b},-8_{b},-9_{b},-12_{b},-17_{b} \right)}}}{e^{1.0 \operatorname{descending\_count\_ramp}{\left(-1_{a},-2,-4_{a},-6_{a},-7,-8_{a},-9_{a},-12_{a},-16,-17_{a} \right)} - 5.64 \operatorname{few\_indicators\_gate}{\left(-1_{b},-3,-4_{b},-6_{b},-8_{b},-9_{b},-12_{b},-17_{b} \right)}} + 3.33}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{1}{3.33 e^{5.64 x_{217} - 1.0 x_{id\_linear}} + 1}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Output = sigmoid(Ramp - 1.20 - 5.64\*Gate): an additive composition under the output sigmoid. Unconnected inputs a2:2, a4:2, a4:3, a5:2, a5:3 and a5:4 are zero-effect entries; the model never consults them. When no indicator is active the gate is about 13, and its -5.64 readout overwhelms the ramp (logit about -54). With one active indicator the gate still wins (logit \<= -8.96). With two, the gate is (almost) off and the logit equals the ramp, -3.34 to +5.63. With three or more, the ramp itself is \<= -3.41. So the network produces a peak at count two of the six rule indicators. The ramp makes the upper edge and the gate makes the lower edge: the mirror image of a single gate plus ramp construction. Because the ramp weights are uneven, the peak is not level. Pairs involving a6=1 (with a1, a2, a4 or a5) get logits -3.34 to -1.07 and fall below p=0.5, and so do some a2&a4 and a4&a5 configurations; a3&a6 stays positive because I3 is light. Train evidence (n=345): AUC 1.0, since every count-2 logit (\>= -3.337) exceeds every other logit (\<= -3.408), a margin of only 0.07. Accuracy at 0.5 is 0.890: 38 false negatives, all count-2 rows, and no false positives. Local ablation with the gate set to 0 drops train accuracy to 0.632 and turns every count\<=1 row positive.

**Evidence (2 records, DB):**

- `analytical_formula`: Output composes ramp and gate additively under the sigmoid.
- `empirical_viz`: Logit by count and per indicator pair (all 432 configs), plus local gate ablation on train.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "form": "y = sigmoid(Ramp - 1.20 - 5.64 * Gate)"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "train_auc": 1.0,
 "max_logit_y0": -3.408,
 "min_logit_y1": -3.337,
 "false_negatives": 38,
 "false_positives": 0,
 "per_count_logit": "count 0: about -54; count 1: <= -8.96; count 2: -3.34..+5.63; count>=3: <= -3.41",
 "gate_ablation_h0": "train acc 0.632; every count-0 and count-1 row becomes positive",
 "train_acc_at_0.5": 0.89,
 "count2_pairs_all_432": {
  "a1,a2": 1.43,
  "a1,a3": "5.18..5.63",
  "a1,a4": "0.43..0.87",
  "a1,a5": "1.40..1.84",
  "a1,a6": "-1.52..-1.07",
  "a2,a3": "4.36..4.89",
  "a2,a4": "-0.39..0.14",
  "a2,a5": "0.58..1.11",
  "a2,a6": "-2.34..-1.81",
  "a3,a4": "1.04..4.34",
  "a3,a5": "4.33..5.31",
  "a3,a6": "1.42..2.40",
  "a4,a5": "-0.42..0.55",
  "a4,a6": "-3.34..-2.36",
  "a5,a6": "-2.37..-1.39"
 },
 "heatmap_corr_217_idlinear": 0.73
}
```

</details>

### A.1 Leaf: `few_indicators_gate`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -1\_b (a1:1\_b), -3 (a1:3), -4\_b (a2:1\_b), -6\_b (a2:3\_b), -8\_b (a3:2\_b), -9\_b (a4:1\_b), -12\_b (a5:1\_b), -17\_b (a6:2\_b) |
| Exit nodes | 217 |
| Internal / exit nodes in its own subgraph | 217 |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |
| Live closed-form error (3 s.f. display) | 0.0136 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 4.19 x_{a1:1\_b} \\
&\quad - 0.0912 x_{a1:3} \\
&\quad - 3.46 x_{a2:1\_b} \\
&\quad + 0.184 x_{a2:3\_b} \\
&\quad + 3.28 x_{a3:2\_b} \\
&\quad - 2.97 x_{a4:1\_b} \\
&\quad - 4.06 x_{a5:1\_b} \\
&\quad + 3.57 x_{a6:2\_b} \\
&\quad -3.86 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A fewer-than-two-indicators gate. Write I\_k = 1 when attribute k takes its first value (a3:2 = 1-I3 and a6:2 = 1-I6, since a3 and a6 are binary). In these terms node 217 computes ReLU(13.02 - 8.77 I1 - 7.32 I2 - 6.57 I3 - 6.43 I4 - 9.36 I5 - 7.15 I6 - 0.20\[a1=3\] + 0.39\[a2=3\]), a near-uniform weighted count of active indicators (weights 6.4 to 9.4) subtracted from 13. It outputs about 13 when no indicator is active, 3.5 to 7.0 when exactly one is, and zero once two or more are. The one leak: the lightest pair (I3+I4) leaves at most +0.41, so the gate fires weakly on 6% of count-2 configurations. Evidence (train, n=345): node 217 ranges 12.82-13.40 at count 0, 3.46-6.98 at count 1, 0-0.41 at count 2, and exactly 0 at count\>=3; sensitivities are about 0.8-1.15 for the six indicator columns and about 0.03-0.05 for a1:3 and a2:3, which are negligible.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of node 217, rewritten in rule-indicator terms.
- `empirical_viz`: Train activations of node 217 grouped by number of active rule indicators: on for count\<=1, off for count\>=2.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are z-scored one-hot columns; converted to indicator form using the two train values of each column",
 "indicator_form": "ReLU(13.02 - 8.77 I1 - 7.32 I2 - 6.57 I3 - 6.43 I4 - 9.36 I5 - 7.15 I6 - 0.20[a1=3] + 0.39[a2=3])"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "all_432_configs": "h>0 for 100% of count-0 and count-1 configs, 6% of count-2 (max 0.41), 0% of count>=3",
 "sensitivity_train": {
  "a1:1": 1.15,
  "a1:3": 0.025,
  "a2:1": 0.95,
  "a2:3": 0.05,
  "a3:2": 0.9,
  "a4:1": 0.82,
  "a5:1": 1.11,
  "a6:2": 0.98
 },
 "per_count_217_range": {
  "0": [
   12.82,
   13.4
  ],
  "1": [
   3.46,
   6.98
  ],
  "2": [
   0.0,
   0.41
  ],
  "3": [
   0,
   0
  ],
  "4": [
   0,
   0
  ],
  "5": [
   0,
   0
  ],
  "6": [
   0,
   0
  ]
 }
}
```

</details>

### A.2 Leaf: `descending_count_ramp`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -1\_a (a1:1\_a), -2 (a1:2), -4\_a (a2:1\_a), -6\_a (a2:3\_a), -7 (a3:1), -8\_a (a3:2\_a), -9\_a (a4:1\_a), -12\_a (a5:1\_a), -16 (a6:1), -17\_a (a6:2\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 10 → 1 |
| Live closed-form error (3 s.f. display) | 0.0197 |
| Live numeric-function error | 3.55e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 4.17 x_{a1:1\_a} \\
&\quad - 0.25 x_{a1:2} \\
&\quad - 4.47 x_{a2:1\_a} \\
&\quad - 0.212 x_{a2:3\_a} \\
&\quad - 3.32 x_{a3:1} \\
&\quad - 0.692 x_{a3:2\_a} \\
&\quad - 4.63 x_{a4:1\_a} \\
&\quad - 3.92 x_{a5:1\_a} \\
&\quad - 4.77 x_{a6:1} \\
&\quad + 1.2 x_{a6:2\_a}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A descending linear count ramp. The direct input-to-output connections sum, in rule-indicator terms, to 20.80 - 8.72 I1 - 9.45 I2 - 5.25 I3 - 10.01 I4 - 9.04 I5 - 11.95 I6 - 0.53\[a1=2\] - 0.45\[a2=3\]. a3 and a6 enter through both of their one-hot columns, which combine into single indicator weights. Each active indicator subtracts 5 to 12, and the weights are not uniform: I6 is heaviest (11.95) and I3 lightest (5.25). With the output bias (-1.20) the ramp is +18.6 to +19.6 at count 0, +6.7 to +14.4 at count 1, -3.3 to +5.6 at count 2, and at most -3.41 at every count\>=3 (train). It forms the upper edge of the response: three or more active indicators always switch the output off. Evidence: closed form, plus per-count ranges on the train split (n=345).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form of the direct input-to-output sum, rewritten in rule-indicator terms.
- `empirical_viz`: Ramp plus output bias falls monotonically with the number of active indicators; it is below -3.4 for every count\>=3 row.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "indicator_form": "20.80 - 8.72 I1 - 9.45 I2 - 5.25 I3 - 10.01 I4 - 9.04 I5 - 11.95 I6 - 0.53[a1=2] - 0.45[a2=3]"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "sensitivity_train": "|w| per column: a6:1 4.77, a4:1 4.63, a2:1 4.47, a1:1 4.17, a5:1 3.92, a3:1 3.32, a6:2 1.20, a3:2 0.69, a1:2 0.25, a2:3 0.21",
 "per_count_ramp_plus_output_bias_range": {
  "0": [
   18.62,
   19.6
  ],
  "1": [
   6.67,
   14.35
  ],
  "2": [
   -3.34,
   5.63
  ],
  "3": [
   -12.37,
   -3.41
  ],
  "4": [
   -21.38,
   -12.86
  ],
  "5": [
   -29.57,
   -24.81
  ],
  "6": [
   -34.82,
   -34.82
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

- `#17` split -1 into -1\_a, -1\_b. Final copies: -1\_a → descending\_count\_ramp (entry) \| -1\_b → few\_indicators\_gate (entry)
- `#18` split -4 into -4\_a, -4\_b. Final copies: -4\_a → descending\_count\_ramp (entry) \| -4\_b → few\_indicators\_gate (entry)
- `#19` split -6 into -6\_a, -6\_b. Final copies: -6\_a → descending\_count\_ramp (entry) \| -6\_b → few\_indicators\_gate (entry)
- `#20` split -8 into -8\_a, -8\_b. Final copies: -8\_a → descending\_count\_ramp (entry) \| -8\_b → few\_indicators\_gate (entry)
- `#21` split -9 into -9\_a, -9\_b. Final copies: -9\_a → descending\_count\_ramp (entry) \| -9\_b → few\_indicators\_gate (entry)
- `#22` split -12 into -12\_a, -12\_b. Final copies: -12\_a → descending\_count\_ramp (entry) \| -12\_b → few\_indicators\_gate (entry)
- `#23` split -17 into -17\_a, -17\_b. Final copies: -17\_a → descending\_count\_ramp (entry) \| -17\_b → few\_indicators\_gate (entry)
- `#24` identity node id\_linear intercepts 10 connection(s) into 0 (from -1\_a (a1:1\_a), -2 (a1:2), -4\_a (a2:1\_a), -6\_a (a2:3\_a), -7 (a3:1), -8\_a (a3:2\_a), -9\_a (a4:1\_a), -12\_a (a5:1\_a), … (2 more)); sits in descending\_count\_ramp (exit); two\_indicator\_band (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T12:25:06+13:00 (commit `77513b9`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 1.11e-15 | 1.11e-15 | 1.11e-15 | agree |
| Closed-form max abs error (all leaves) | 0.01966 | 0.01966 | 0.01966 | agree |
| Numeric-function max abs error | n/a | 3.55e-15 | 3.55e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | two\_indicator\_band | two\_indicator\_band | agree |
| Leaf / composition counts | 2 / 1 | n/a | 2 / 1 | agree |
| Operation counts | 7 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 3, "rename\_node": 17, "split\_node": 7} | {"add\_identity\_node": 1, "annotate": 3, "rename\_node": 17, "split\_node": 7} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `descending_count_ramp` | 0.01966 | 3.55e-15 |
| `few_indicators_gate` | 0.0136 | 3.55e-15 |

## What resisted (from the record)

> Nothing structural. Uneven count-2 peak (I6 weight 11.95 vs I3 5.25): 38 train false negatives at 0.5 (acc 0.890) though train AUC 1.0, logit ranking margin 0.07.

**Tool notes (record):** apply\_operation docstring gives add\_identity\_node params {target\_node\_id, connection\_sources}; engine requires {target\_node, connections:\[\[src,tgt\]\], new\_node\_id}.

## What to look at

- Largest closed-form error: `descending_count_ramp` (0.0197), then `few_indicators_gate` (0.0136). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `few_indicators_gate`: *“It outputs about 13 when no indicator is active, 3.5 to 7.0 when exactly one is, and zero once two or more are.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `descending_count_ramp`: *“It forms the upper edge of the response: three or more active indicators always switch the output off.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 7 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s5.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s5', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
