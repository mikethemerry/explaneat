# Verification packet 15 · monk2-s8

Pre-registered verification order: 15 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/11c1e069-6fd8-4be6-a846-0d5f559fd560?experiment=8b76a90c-dc7e-4722-8d7a-4fe7bdf0a522&name=10seed-monk2-s8), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 8 |
| Genome id | `11c1e069-6fd8-4be6-a846-0d5f559fd560` |
| Experiment | 10seed-monk2-s8 (8b76a90c-dc7e-4722-8d7a-4fe7bdf0a522); best at generation 99 of 100, NEAT key 23784 |
| Test AUC (runs csv) | 0.87 \[95% CI 0.79, 0.937\] |
| Train AUC | 0.946 |
| Size | 13 of 17 inputs connected; 2 hidden nodes; 23 enabled connections; depth 3; 7 skip connections + 10 direct input→output edges |
| Annotations | 2 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 1 of 2 |
| Claude's time | 45.0 min (2026-11-03T13:12:35+13:00 → 2026-11-03T13:57:38+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> sigmoid(2.236 + Ramp - 0.918 P - 5.804 L): Ramp a falling linear count (upper edge); P a graded at-least-two-of-{a2,a3,a6} counter; L a low-count gate firing at 0-1 first values (lower edge). Train acc 0.855, AUC 0.946.

**Agreement with the MONK-2 rule (record):** Weighted 'about two first values' peak: a4=1 and a5=1 count almost double, a1=1 light; only 87/142 exactly-two inputs predicted positive; agrees with the rule on 84.3% of 432 inputs.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `weighted_count_peak`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | id\_linear, 516, 584, -10 (a4:2), -11 (a4:3), -13 (a5:2), -14 (a5:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | count\_stage, linear\_count\_ramp |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.107 e^{0.918 \operatorname{count\_stage\_0}{\left(-4,-6_{b},-7_{a},-17_{b},-2,-3_{b},-7_{b},-9_{b},-12_{b},-16_{b} \right)} + 5.8 \operatorname{count\_stage\_1}{\left(-4,-6_{b},-7_{a},-17_{b},-2,-3_{b},-7_{b},-9_{b},-12_{b},-16_{b} \right)} - 1.0 \operatorname{linear\_count\_ramp}{\left(-1,-3_{a},-5,-6_{a},-8,-9_{a},-12_{a},-15,-16_{a},-17_{a} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{1}{0.107 e^{0.918 x_{516} + 5.8 x_{584} - 1.0 x_{id\_linear}} + 1}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: score = sigmoid(2.236 + Ramp - 0.918 P - 5.804 L). Ramp falls with weighted indicator count (upper edge); L subtracts up to 41.6 when there are 0-1 indicators (lower edge); P adds an extra penalty (0.918 x 0.95..13.4) when \>=2 of a2/a3/a6 are first values and, through L, closes the lower edge. Unconnected inputs a4:2, a4:3, a5:2, a5:3 (no enabled edges) have zero effect; a4 and a5 are read only through a4:1, a5:1, a5:4. Evidence: mean output pre-activation by number of first values (432 inputs): -26.9, -15.4, +0.55, -5.4, -13.2, -21.3, -29.7 for 0..6, a peak at two. Positive predictions: 0/24, 9/92 (8 of them a1=1 alone), 87/142, 4/113, 0/61. The peak is imperfect because the indicators are weighted unequally: a4=1 and a5=1 each count almost double, so pairs containing a5 and {a2,a4},{a4,a5} mostly score negative, while a1=1 alone often scores positive. Train (n=345): accuracy 0.855, AUC 0.946; agreement with the MONK-2 rule on all 432 inputs 0.843. Train sensitivity of output: L 0.31, Ramp 0.053, P 0.049.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Output pre-activation mean by count 0..6: -26.9, -15.4, +0.55, -5.4, -13.2, -21.3, -29.7. Positive predictions per count: 0/24, 9/92, 87/142, 4/113, 0/61. compute\_performance(train, n=345): accuracy 0.855, AUC 0.946. Agreement with MONK-2 rule on all 432 inputs: 0.843.

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
 "viz_type": "sensitivity",
 "enumeration": "Output pre-activation mean by count 0..6: -26.9, -15.4, +0.55, -5.4, -13.2, -21.3, -29.7. Positive predictions per count: 0/24, 9/92, 87/142, 4/113, 0/61. compute_performance(train, n=345): accuracy 0.855, AUC 0.946. Agreement with MONK-2 rule on all 432 inputs: 0.843.",
 "sensitivities": {
  "0": [
   0.05343931082176785,
   0.049043889121769865,
   0.31033898543167376,
   0.0,
   0.0,
   0.0,
   0.0
  ]
 }
}
```

</details>

### A.1 Composition: `count_stage`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -4 (a2:1), -6\_b (a2:3\_b), -7\_a (a3:1\_a), -17\_b (a6:2\_b), -2 (a1:2), -3\_b (a1:3\_b), -7\_b (a3:1\_b), -9\_b (a4:1\_b), -12\_b (a5:1\_b), -16\_b (a6:1\_b) |
| Exit nodes | 516, 584 |
| Internal / exit nodes in its own subgraph | 584 |
| Children | pair\_gate\_a2a3a6 |
| Dimensionality (in → out, incl. descendants) | 10 → 2 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \operatorname{pair\_gate\_a2a3a6}{\left(x_{a2:1},x_{a2:3\_b},x_{a3:1\_a},x_{a6:2\_b} \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 2.38 x_{a1:2} \\
&\quad + 2.71 x_{a1:3\_b} \\
&\quad - 0.792 x_{a3:1\_b} \\
&\quad - 2.56 x_{a4:1\_b} \\
&\quad - 2.4 x_{a5:1\_b} \\
&\quad - 0.302 x_{a6:1\_b} \\
&\quad - 1.88 \operatorname{pair\_gate\_a2a3a6}{\left(x_{a2:1},x_{a2:3\_b},x_{a3:1\_a},x_{a6:2\_b} \right)} \\
&\quad + 0.587 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.96 x_{a2:1} \\
&\quad - 2.11 x_{a2:3\_b} \\
&\quad + 3.11 x_{a3:1\_a} \\
&\quad - 4.21 x_{a6:2\_b} \\
&\quad + 0.628 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 2.38 x_{a1:2} \\
&\quad + 2.71 x_{a1:3\_b} \\
&\quad - 0.792 x_{a3:1\_b} \\
&\quad - 2.56 x_{a4:1\_b} \\
&\quad - 2.4 x_{a5:1\_b} \\
&\quad - 0.302 x_{a6:1\_b} \\
&\quad - 1.88 \operatorname{relu}{\left(2.96 x_{a2:1} - 2.11 x_{a2:3\_b} + 3.11 x_{a3:1\_a} - 4.21 x_{a6:2\_b} + 0.628 \right)} \\
&\quad + 0.587 \bigr)
\end{aligned}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Serial composition: pair gate P feeds the low-count gate L (node 584); outputs (P, L). L = ReLU(1.308 + 5.043\[a1=2\] + 5.853\[a1=3\] - 1.583\[a3=1\] - 5.530\[a4=1\] - 5.532\[a5=1\] - 0.603\[a6=1\] - 1.885 P) \~= ReLU(6.7 - 5.4\[a1=1\] - 5.53\[a4=1\] - 5.53\[a5=1\] - 1.58\[a3=1\] - 0.60\[a6=1\] - 1.885 P). L is a 'too few indicators' detector: large (up to 7.16) when none of a1=1, a4=1, a5=1 hold and P is quiet; closed by a first value of a1/a4/a5 plus any further indicator, or by P (the P -\> L inhibition routes a2/a3/a6 count evidence into the lower edge). Evidence: over all 432 inputs, L\>0 for 100% of inputs with 0 or 1 first values (means 6.76, 4.04), 30% with exactly 2 (mean 0.22; L stays \~1.2-1.3 for pairs {a1,a2}, {a2,a4}, {a2,a5}), 0% with \>=3. Train sensitivity of L: a1:3 0.96, a4:1 0.90, a5:1 0.85, a1:2 0.84, a3:1 0.28, others \<=0.23. Local train ablation: P=0 everywhere -\> acc 0.672 / AUC 0.614; L removed -\> 0.667 / 0.632 (full 0.855 / 0.946).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Exhaustive enumeration over 432 inputs: L\>0 for 24/24 inputs with 0 first values, 92/92 with 1, 30% of 142 with 2, 0 with \>=3. Local ablation on train: setting P to 0 everywhere (output and L) drops accuracy 0.855-\>0.672, AUC 0.946-\>0.614; removing L drops to 0.667 / 0.632.

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
 "viz_type": "sensitivity",
 "enumeration": "Exhaustive enumeration over 432 inputs: L>0 for 24/24 inputs with 0 first values, 92/92 with 1, 30% of 142 with 2, 0 with >=3. Local ablation on train: setting P to 0 everywhere (output and L) drops accuracy 0.855->0.672, AUC 0.946->0.614; removing L drops to 0.667 / 0.632.",
 "sensitivities": {
  "516": [
   1.5261895806174166,
   1.087159948077413,
   1.6035963861034324,
   2.172185391062739,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "584": [
   0.16158869523788455,
   0.11510546249021622,
   0.16978431186368095,
   0.22998492953583655,
   0.8424817591076174,
   0.9574045239988936,
   0.279884705569403,
   0.9045495840576765,
   0.8495191378812803,
   0.10665619531001626
  ]
 }
}
```

</details>

#### A.1.1 Leaf: `pair_gate_a2a3a6`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -4 (a2:1), -6\_b (a2:3\_b), -7\_a (a3:1\_a), -17\_b (a6:2\_b) |
| Exit nodes | 516 |
| Internal / exit nodes in its own subgraph | 516 |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.00635 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.96 x_{a2:1} \\
&\quad - 2.11 x_{a2:3\_b} \\
&\quad + 3.11 x_{a3:1\_a} \\
&\quad - 4.21 x_{a6:2\_b} \\
&\quad + 0.628 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Pair gate over a2=1, a3=1, a6=1. P = ReLU(0.628 + 2.96 z(a2:1) - 2.11 z(a2:3) + 3.11 z(a3:1) - 4.21 z(a6:2)) = ReLU(-7.469 + 6.26\[a2=1\] + 6.22\[a3=1\] + 8.42\[a6=1\] - 4.44\[a2=3\]). Evidence (closed form; exhaustive enumeration of the 12 (a2,a3,a6) patterns): all three indicators -\> 13.43; two -\> 7.22 (a2,a6), 7.17 (a3,a6), 5.01 (a2,a3), 2.73 (a3,a6 with a2=3); one -\> 0 except a6=1 alone with a2=2 -\> 0.95; none -\> 0. So P is a graded at-least-two-of-three counter over the indicators of a2, a3, a6. Train sensitivity is largest for a6:2 (2.17), then a3:1 (1.60), a2:1 (1.53), a2:3 (1.09). It supplies count information for a2 and a6, which the direct linear path barely weighs.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Exhaustive enumeration of (a2,a3,a6) \[12 patterns\]: P = 13.43 (1,1,1); 5.01 (1,1,2); 7.22 (1,2,1); 7.17 (2,1,1); 2.73 (3,1,1); 0.95 (2,2,1); 0 otherwise.

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
 "viz_type": "sensitivity",
 "enumeration": "Exhaustive enumeration of (a2,a3,a6) [12 patterns]: P = 13.43 (1,1,1); 5.01 (1,1,2); 7.22 (1,2,1); 7.17 (2,1,1); 2.73 (3,1,1); 0.95 (2,2,1); 0 otherwise.",
 "sensitivities": {
  "516": [
   1.5261895806174166,
   1.087159948077413,
   1.6035963861034324,
   2.172185391062739
  ]
 }
}
```

</details>

### A.2 Leaf: `linear_count_ramp`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | -1 (a1:1), -3\_a (a1:3\_a), -5 (a2:2), -6\_a (a2:3\_a), -8 (a3:2), -9\_a (a4:1\_a), -12\_a (a5:1\_a), -15 (a5:4), -16\_a (a6:1\_a), -17\_a (a6:2\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 10 → 1 |
| Live closed-form error (3 s.f. display) | 0.0136 |
| Live numeric-function error | 2.66e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 2.03 x_{a1:1} \\
&\quad + 0.344 x_{a1:3\_a} \\
&\quad + 0.893 x_{a2:2} \\
&\quad + 0.169 x_{a2:3\_a} \\
&\quad + 2.73 x_{a3:2} \\
&\quad - 3.7 x_{a4:1\_a} \\
&\quad - 3.98 x_{a5:1\_a} \\
&\quad + 0.989 x_{a5:4} \\
&\quad - 0.756 x_{a6:1\_a} \\
&\quad - 0.478 x_{a6:2\_a}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Decreasing weighted linear count ramp (direct input-\>output connections). Per attribute value: a1: =1 -2.97, =2 +1.27, =3 +2.01; a2: =1 -0.74, =2 +1.17, =3 -0.38; a3: =1 -2.66, =2 +2.80; a4: =1 -5.52, =2/3 +2.48; a5: =1 -7.42, =2/3 +1.74, =4 +4.03; a6: =1 -0.27, =2 +0.29. Each first-value indicator lowers the sum by an unequal amount: a5 (9.2-11.4) and a4 (8.0) count heavily, a3 (5.5) and a1 (4.2-5.0) moderately, a2 (0.4-1.9) and a6 (0.56) barely. Evidence (closed form; enumeration of 432 inputs): ramp + output bias averages 12.3, 8.1, 3.6, -1.1, -6.2, -11.6, -17.3 for 0..6 first values (range -17.3 to 15.0). It is the upper edge of the response peak. Alone (with bias) it gives train accuracy 0.565, AUC 0.602.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true).
- `empirical_viz`: Ramp+bias by number of first values (432 inputs): mean 12.3, 8.1, 3.6, -1.1, -6.2, -11.6, -17.3 for 0..6. Ramp alone (with bias) gives train accuracy 0.565, AUC 0.602.

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
 "viz_type": "sensitivity",
 "enumeration": "Ramp+bias by number of first values (432 inputs): mean 12.3, 8.1, 3.6, -1.1, -6.2, -11.6, -17.3 for 0..6. Ramp alone (with bias) gives train accuracy 0.565, AUC 0.602.",
 "sensitivities": {
  "id_linear": [
   2.026832803846015,
   0.3438842876965726,
   0.8927766218744783,
   0.16945802128366444,
   2.7278979946904665,
   3.7010509887576744,
   3.976914203444886,
   0.9891032994344033,
   0.7564349899263962,
   0.478014811298786
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
| split\_node | 7 |

7 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -3 into -3\_a, -3\_b. Final copies: -3\_a → linear\_count\_ramp (entry) \| -3\_b → count\_stage (entry)
- `#18` split -6 into -6\_a, -6\_b. Final copies: -6\_a → linear\_count\_ramp (entry) \| -6\_b → pair\_gate\_a2a3a6 (entry); count\_stage (entry)
- `#19` split -7 into -7\_a, -7\_b. Final copies: -7\_a → pair\_gate\_a2a3a6 (entry); count\_stage (entry) \| -7\_b → count\_stage (entry)
- `#20` split -9 into -9\_a, -9\_b. Final copies: -9\_a → linear\_count\_ramp (entry) \| -9\_b → count\_stage (entry)
- `#21` split -12 into -12\_a, -12\_b. Final copies: -12\_a → linear\_count\_ramp (entry) \| -12\_b → count\_stage (entry)
- `#22` split -16 into -16\_a, -16\_b. Final copies: -16\_a → linear\_count\_ramp (entry) \| -16\_b → count\_stage (entry)
- `#23` split -17 into -17\_a, -17\_b. Final copies: -17\_a → linear\_count\_ramp (entry) \| -17\_b → pair\_gate\_a2a3a6 (entry); count\_stage (entry)
- `#24` identity node id\_linear intercepts 10 connection(s) into 0 (from -1 (a1:1), -3\_a (a1:3\_a), -5 (a2:2), -6\_a (a2:3\_a), -8 (a3:2), -9\_a (a4:1\_a), -12\_a (a5:1\_a), -15 (a5:4), … (2 more)); sits in linear\_count\_ramp (exit); weighted\_count\_peak (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 6.10e-16 | 6.11e-16 | 6.11e-16 | agree |
| Closed-form max abs error (all leaves) | 0.0136 | 0.01359 | 0.01359 | agree |
| Numeric-function max abs error | n/a | 2.66e-15 | 2.66e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | weighted\_count\_peak | weighted\_count\_peak | agree |
| Leaf / composition counts | 2 / 2 | n/a | 2 / 2 | agree |
| Operation counts | 7 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 17, "split\_node": 7} | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 17, "split\_node": 7} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `linear_count_ramp` | 0.01359 | 2.66e-15 |
| `pair_gate_a2a3a6` | 0.006347 | 8.88e-16 |

## What resisted (from the record)

> Low-count gate 584 takes the pair-gate leaf's exit (which also feeds the output) so it sits as the own node of composition count\_stage. Network only approximates the rule (unequal indicator weights; \~16% of 432 inputs disagree).

**Tool notes (record):** add\_identity\_node docstring params wrong (engine needs target\_node, new\_node\_id, connections). annotate fails if an entry node is not also in subgraph\_nodes.

## What to look at

- Largest closed-form error: `linear_count_ramp` (0.0136). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- 7 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s8.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s8', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
