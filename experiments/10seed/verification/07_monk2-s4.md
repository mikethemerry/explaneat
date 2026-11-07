# Verification packet 07 · monk2-s4

Pre-registered verification order: 7 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/60ee493c-bce5-4a2b-b7af-d06843835ae6?experiment=f5ce4c65-e0d7-4a1e-bd29-97f82bac9d3b&name=10seed-monk2-s4), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 4 |
| Genome id | `60ee493c-bce5-4a2b-b7af-d06843835ae6` |
| Experiment | 10seed-monk2-s4 (f5ce4c65-e0d7-4a1e-bd29-97f82bac9d3b); best at generation 99 of 100, NEAT key 20353 |
| Test AUC (runs csv) | 0.861 \[95% CI 0.771, 0.937\] |
| Train AUC | 0.925 |
| Size | 14 of 17 inputs connected; 7 hidden nodes; 35 enabled connections; depth 3; 7 skip connections + 12 direct input→output edges |
| Annotations | 5 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 7 of 7 |
| Claude's time | 52.8 min (2026-11-03T13:12:35+13:00 → 2026-11-03T14:05:23+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> sigmoid(-2.36 + Lin + Veto + 2.55 Gate - 0.60 A5): Lin +4..+12 per first value of a1,a2,a3,a4,a6; Veto a ReLU count unit, 0 for \<=2 and down to -70 for \>=3; Gate +9..+11 when a5!=1 (bonus with a3=1). Predicts 'exactly two of {a1,a2,a3,a4,a6}, preferably a5!=1'. Train acc 0.887, AUC 0.925.

**Agreement with the MONK-2 rule (record):** Counts five attributes (not a5); a5=1 acts as a handicap. 113/114 two-of-six correct when a5!=1, misses all 28 with a5=1 as one of the two, accepts 23/38 three-of-six with a5=1: 88.0% agreement over 432.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `two_of_five_peak`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | id\_linear, id\_suppress, 1223, 1751, 927, -10 (a4:2), -13 (a5:2), -17 (a6:2) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | linear\_first\_value\_score, atleast3\_of\_five\_veto, a5\_a3\_bonus\_gate, a5\_in\_2\_4\_penalty, inert\_a32\_unit |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{10.6 e^{- 2.55 a_{5 a3 bonus gate}{\left(-3_{b},-12_{b},-12_{c},-16_{c},-7_{c},-11_{b} \right)} + 0.599 a_{5 in 2 4 penalty}{\left(-12_{d},-14 \right)} - 1.0 \operatorname{atleast\_3\_of\_five\_veto}{\left(-1_{b},-4_{b},-7_{b},-9_{b},-16_{b} \right)} - 0.735 \operatorname{inert\_a32\_unit}{\left(-8 \right)} - 1.0 \operatorname{linear\_first\_value\_score}{\left(-1_{a},-2,-3_{a},-4_{a},-5,-6,-7_{a},-9_{a},-11_{a},-12_{a},-15,-16_{a} \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Two-of-five peak with an a5 handicap. The output is sigmoid(s) with s = -2.363 + Lin + Veto + 2.547\*Gate - 0.599\*A5 + 0.735\*Inert, where Inert is identically 0. Lin (linear\_first\_value\_score) rises by +4..+12 for each first value among a1, a2, a3, a4, a6 and is the rising edge: with at most one such first value s \< 0 on all 432 configurations (max -0.42). Veto (atleast3\_of\_five\_veto) is exactly 0 for at most two first values and \<= -70 for three or more, so it is the falling edge: s \< 0 for every configuration with three or more. Gate (a5\_a3\_bonus\_gate) adds +8.5..+11 when a5 != 1 (more when a3 = 1) and nothing when a5 = 1 and a3 = 2; A5 is a -0.72 penalty for a5 in {2,4}. Net: with exactly two first values among {a1,a2,a3,a4,a6} the model predicts positive in 113 of 114 configurations when a5 != 1 but only 23 of 38 (61%) when a5 = 1 (the a5\_a3 gate is weak or off there; pairs (2,3) and (3,4) always fail). So the network computes approximately 'exactly two of the five attributes a1,a2,a3,a4,a6 take their first value, and preferably a5 != 1': it agrees with \[count5 == 2\] on 96.3% of all 432 configurations and 95.9% of training rows, and reaches 88.7% train accuracy (AUC 0.925). Inputs a4:2, a5:2, a6:2 have no enabled connections and are zero-effect entries. Evidence: on train, ablating the veto drops accuracy 0.887 -\> 0.562 and ablating the gate 0.887 -\> 0.690; the score sign is negative for every configuration with \<=1 or \>=3 such first values. Comparison with the target rule (exactly two of all six attributes take their first value): the network has replaced a5=1 as a rule indicator with a5 != 1 as a bonus, so it is correct on two-of-six cases where a5 != 1 (113/114) but misses all 28 two-of-six cases in which a5 = 1 is one of the two, and wrongly accepts 23 of 38 three-of-six cases that include a5 = 1 (agreement with the rule 88.0% over all 432 configurations).

**Evidence (2 records, DB):**

- `analytical_formula`: Additive composition of five children under the output sigmoid.
- `empirical_viz`: Score is negative for \<=1 and \>=3 first-values among a1,a2,a3,a4,a6 on every configuration; positive for exactly two except mostly when a5=1.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "score": "s = -2.363 + Lin + Veto + 2.547*Gate - 0.599*A5 + 0.735*Inert"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "sensitivities": {
  "0": [
   0.02417381948425206,
   0.024173819484251992,
   0.06159981191916847,
   0.014472892772605878,
   0.017760471108297978,
   0.0,
   0.0,
   0.0
  ]
 },
 "exhaustive_432": {
  "count5<=1_positive": 0,
  "count5>=3_positive": 0,
  "agree_with_count5_eq_2": 0.963,
  "agree_with_target_rule": 0.88,
  "count5==2_a5!=1_positive": "113/114",
  "count5==2_a5==1_positive": "23/38"
 },
 "performance_train": {
  "auc": 0.925,
  "accuracy": 0.887
 },
 "ablations_train_acc": {
  "none": 0.887,
  "gate_removed": 0.69,
  "veto_removed": 0.562,
  "a5_24_removed": 0.89
 },
 "train_agree_with_count5_eq_2": 0.959
}
```

</details>

### A.1 Leaf: `linear_first_value_score`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | -1\_a (a1:1\_a), -2 (a1:2), -3\_a (a1:3\_a), -4\_a (a2:1\_a), -5 (a2:2), -6 (a2:3), -7\_a (a3:1\_a), -9\_a (a4:1\_a), -11\_a (a4:3\_a), -12\_a (a5:1\_a), -15 (a5:4), -16\_a (a6:1\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | 1948, id\_linear |
| Dimensionality (in → out, incl. descendants) | 12 → 1 |
| Live closed-form error (3 s.f. display) | 0.0242 |
| Live numeric-function error | 7.11e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 4.89 x_{a1:1\_a} \\
&\quad - 1.34 x_{a1:2} \\
&\quad - 0.789 x_{a1:3\_a} \\
&\quad + 3.13 x_{a2:1\_a} \\
&\quad - 0.892 x_{a2:2} \\
&\quad - 1.46 x_{a2:3} \\
&\quad + 2.06 x_{a3:1\_a} \\
&\quad + 2.88 x_{a4:1\_a} \\
&\quad - 2.13 x_{a4:3\_a} \\
&\quad + 1.14 x_{a5:1\_a} \\
&\quad - 0.112 x_{a5:4} \\
&\quad + 5.97 x_{a6:1\_a} \\
&\quad + 0.899 \operatorname{relu}{\left(0.827 - 0.0898 x_{a1:2} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Additive per-attribute score (the linear backbone). Measured per attribute value relative to the all-low baseline, the direct edges give large positive credit when an attribute takes its first value: a6=1 +11.94, a1=1 +10.22, a2=1 +6.63, a4=1 +6.22, a3=1 +4.12, but only +2.63 for a5=1; non-first values add 0 or a penalty (a4=3 -4.60, a2=3 -3.07, a1=2 -3.00, a1=3 -1.71, a2=2 -1.91, a5=4 -0.26). Node 1948 = ReLU(0.827 - 0.090 z(a1:2)) never clips (it takes only 0.70 or 0.89), so it is an affine relay of a1:2 folded into the a1=2 penalty. The score rises roughly linearly with the number of first-values among a1, a2, a3, a4, a6 and is the rising edge of the model's response peak. Evidence: closed form is linear (sensitivities on train equal the direct weights exactly; 1948 takes only 0.70/0.89 on train, never clipping).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form is linear in the 12 entries plus one never-clipping relu relay of a1:2.
- `empirical_viz`: Sensitivities equal the direct weights (constant gradient = linear). First values of a1,a2,a3,a4,a6 earn +4.1..+11.9; a5=1 only +2.6.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "Affine on domain: 1948 relu never clips (values 0.70/0.89 on train)."
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "sensitivities": {
  "a1:1": 4.89,
  "a1:2": 1.42,
  "a1:3": 0.79,
  "a2:1": 3.13,
  "a2:2": 0.89,
  "a2:3": 1.46,
  "a3:1": 2.06,
  "a4:1": 2.88,
  "a4:3": 2.13,
  "a5:1": 1.14,
  "a5:4": 0.11,
  "a6:1": 5.97
 },
 "per_value_effect_vs_all_low": {
  "a1": [
   10.22,
   -3.0,
   -1.71
  ],
  "a2": [
   6.63,
   -1.91,
   -3.07
  ],
  "a3": [
   4.12,
   0
  ],
  "a4": [
   6.22,
   0,
   -4.6
  ],
  "a5": [
   2.63,
   0,
   0,
   -0.26
  ],
  "a6": [
   11.94,
   0
  ]
 }
}
```

</details>

### A.2 Leaf: `atleast3_of_five_veto`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | -1\_b (a1:1\_b), -4\_b (a2:1\_b), -7\_b (a3:1\_b), -9\_b (a4:1\_b), -16\_b (a6:1\_b) |
| Exit nodes | id\_suppress |
| Internal / exit nodes in its own subgraph | 274, 2786, id\_suppress |
| Dimensionality (in → out, incl. descendants) | 5 → 1 |
| Live closed-form error (3 s.f. display) | 0.17 |
| Live numeric-function error | 2.84e-14 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 6.56 \operatorname{relu}{\left(0.605 \operatorname{relu}{\left(3.46 x_{a1:1\_b} + 4.56 x_{a2:1\_b} + 4.67 x_{a3:1\_b} + 3.98 x_{a4:1\_b} + 4.48 x_{a6:1\_b} - 1.34 \right)} - 0.609 \right)} \\
&\quad - 7.14 \operatorname{relu}{\left(3.46 x_{a1:1\_b} + 4.56 x_{a2:1\_b} + 4.67 x_{a3:1\_b} + 3.98 x_{a4:1\_b} + 4.48 x_{a6:1\_b} - 1.34 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> At-least-three-of-five veto. Node 274 = ReLU(-1.34 + 3.46 z(a1:1) + 4.56 z(a2:1) + 4.67 z(a3:1) + 3.98 z(a4:1) + 4.48 z(a6:1)) counts first-values of a1, a2, a3, a4 and a6 (a5 is NOT included). Per-indicator increments are 7.2 to 9.7 on a base of -19.2: with two or fewer indicators the pre-activation is at most -0.2 (exactly 0 output), with three it is at least +5.6. Node 2786 = ReLU(0.605\*274 - 0.609) never clips while 274 is on, so the exit id\_suppress = -7.14\*274 - 6.56\*2786 = -11.11\*274 + 4.00 whenever 274 fires, i.e. a penalty of -70 or more (-269 with all five), which forces the output to 0. Exactly zero otherwise. Forms the falling edge of the response peak at three-of-five. Evidence: on all 345 training rows 274 \> 0 exactly when three or more of a1,a2,a3,a4,a6 take their first value (112 rows); removing the veto drops train accuracy from 0.887 to 0.562.

**Evidence (2 records, DB):**

- `empirical_viz`: Count threshold at three-of-five confirmed on every training row; the veto is essential.
- `analytical_formula`: Serial relu chain 274 -\> 2786; 2786 never clips while 274\>0.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "train_check": "274>0 iff count of first-values among {a1,a2,a3,a4,a6} >=3: true on all 345 train rows (112 rows fire); min firing pre-activation 5.6; max non-firing -0.2 (exhaustive over 432 configs)",
 "sensitivities": [
  12.48,
  16.46,
  16.83,
  14.34,
  16.15
 ],
 "ablation_train": "removing the veto drops train accuracy 0.887 -> 0.562"
}
```

```json
// analytical_formula
{
 "simplified": "id_suppress = 0 if 274=0 else -11.109*274 + 3.995"
}
```

</details>

### A.3 Leaf: `a5_a3_bonus_gate`

|  |  |
|---|---|
| Annotation id | `ann_29` |
| Entry nodes | -3\_b (a1:3\_b), -12\_b (a5:1\_b), -12\_c (a5:1\_c), -16\_c (a6:1\_c), -7\_c (a3:1\_c), -11\_b (a4:3\_b) |
| Exit nodes | 1223 |
| Internal / exit nodes in its own subgraph | 1717, 1223 |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |
| Live closed-form error (3 s.f. display) | 0.00673 |
| Live numeric-function error | 2.22e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.41 x_{a3:1\_c} \\
&\quad + 0.298 x_{a4:3\_b} \\
&\quad - 1.86 x_{a5:1\_b} \\
&\quad - 0.691 \operatorname{relu}{\left(- 0.0341 x_{a1:3\_b} - 0.0515 x_{a5:1\_c} - 0.559 x_{a6:1\_c} - 0.525 \right)} \\
&\quad + 2.47 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A clipped bonus that rewards a5 != 1 and a3 = 1. Node 1223 = ReLU(2.47 + 1.41 z(a3:1) + 0.30 z(a4:3) - 1.86 z(a5:1) - 0.69\*1717). Readout to the output (x2.547): a5 != 1 gives +12.0..13.7 when a3=1 and +4.7..6.5 when a3=2; a5 = 1 gives +1.1..2.75 when a3=1 and exactly 0 when a3=2 (clipped). So it is mostly an additive a5 != 1 bonus of \~+8.5 to +11 plus an a3=1 bonus, with one clip interaction (a5=1 and a3=2). Sub-unit 1717 = ReLU(-0.52 - 0.03 z(a1:3) - 0.05 z(a5:1) - 0.56 z(a6:1)) is near-inert: it is nonzero only when a6 != 1 and never exceeds 0.10, moving 1223 by at most 0.07. Net effect: a5 = 1 is treated as evidence AGAINST the positive class, cancelling the small direct +2.63 credit for a5=1. Evidence: on all 345 training rows 1223 = 0 exactly when a5 = 1 and a3 = 2 (40 rows); train sensitivities are dominated by a5:1 (1.65) and a3:1 (1.25), with the 1717 path \< 0.15; removing the gate drops train accuracy from 0.887 to 0.690.

**Evidence (2 records, DB):**

- `empirical_viz`: a5:1 and a3:1 dominate; 1717 path effect \<=0.07.
- `analytical_formula`: Single clipped relu over a3:1, a4:3, a5:1 with near-inert 1717 sub-unit.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "viz_type": "sensitivity",
 "train_check": "1223==0 iff a5=1 and a3=2 on all 345 rows (40 rows clipped)",
 "sensitivities": {
  "a1:3": 0.009,
  "a3:1": 1.25,
  "a4:3": 0.26,
  "a6:1": 0.14,
  "a5:1(b)": 1.65,
  "a5:1(c)": 0.013
 },
 "ablation_train": "removing it drops train accuracy 0.887 -> 0.690",
 "output_contribution_table_x2.547": {
  "a5=1,a3=1": "1.1..2.75",
  "a5=1,a3=2": "0",
  "a5!=1,a3=1": "12.0..13.7",
  "a5!=1,a3=2": "4.7..6.5"
 }
}
```

```json
// analytical_formula
{}
```

</details>

### A.4 Leaf: `a5_in_2_4_penalty`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -12\_d (a5:1\_d), -14 (a5:3) |
| Exit nodes | 1751 |
| Internal / exit nodes in its own subgraph | 1751 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00313 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.32 x_{a5:1\_d} \\
&\quad - 1.26 x_{a5:3} \\
&\quad -0.286 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Exact indicator of a5 in {2,4}. Node 1751 = ReLU(-0.29 - 1.32 z(a5:1) - 1.26 z(a5:3)) is 1.198 when a5 is 2 or 4 and exactly 0 when a5 is 1 or 3. With the -0.599 readout it is a constant -0.72 penalty for a5 in {2,4}: a small per-attribute lookup on a5, not an interaction. Evidence: on train the exit takes only 0 and 1.198, and is positive exactly when a5 in {2,4}; removing it changes train accuracy 0.887 -\> 0.890 (negligible).

**Evidence (2 records, DB):**

- `analytical_formula`: Two-input relu over a5 one-hot columns.
- `empirical_viz`: Exact a5 in {2,4} indicator; negligible for accuracy.

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
 "viz_type": "line",
 "train_check": "1751>0 iff a5 in {2,4} on all 345 rows",
 "ablation_train": "removing it changes train accuracy 0.887 -> 0.890",
 "train_exit_values": [
  0.0,
  1.198
 ]
}
```

</details>

### A.5 Leaf: `inert_a32_unit`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | -8 (a3:2) |
| Exit nodes | 927 |
| Internal / exit nodes in its own subgraph | 927 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(- 1.93 x_{a3:2} - 2.43 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Inert unit. Node 927 = ReLU(-2.43 - 1.93 z(a3:2)); z(a3:2) takes only -0.974 or +1.026, so the pre-activation is at most -0.55 and the node outputs exactly 0 on every input in the domain. Its +0.73 readout contributes nothing. Evidence: the exit is 0 on all 345 training rows (line plot y range \[0,0\]).

**Evidence (2 records, DB):**

- `analytical_formula`: Pre-activation negative for both values of a3:2.
- `empirical_viz`: Exit is identically 0 on all 345 training rows.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "max_preactivation": -0.55
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "y_range": [
  0,
  0
 ],
 "viz_type": "line"
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 2 |
| annotate | 6 |
| rename\_node | 17 |
| split\_node | 8 |

8 splits, 2 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -1 into -1\_a, -1\_b. Final copies: -1\_a → linear\_first\_value\_score (entry) \| -1\_b → atleast3\_of\_five\_veto (entry)
- `#18` split -3 into -3\_a, -3\_b. Final copies: -3\_a → linear\_first\_value\_score (entry) \| -3\_b → a5\_a3\_bonus\_gate (entry)
- `#19` split -4 into -4\_a, -4\_b. Final copies: -4\_a → linear\_first\_value\_score (entry) \| -4\_b → atleast3\_of\_five\_veto (entry)
- `#20` split -7 into -7\_a, -7\_b, -7\_c. Final copies: -7\_a → linear\_first\_value\_score (entry) \| -7\_b → atleast3\_of\_five\_veto (entry) \| -7\_c → a5\_a3\_bonus\_gate (entry)
- `#21` split -9 into -9\_a, -9\_b. Final copies: -9\_a → linear\_first\_value\_score (entry) \| -9\_b → atleast3\_of\_five\_veto (entry)
- `#22` split -11 into -11\_a, -11\_b. Final copies: -11\_a → linear\_first\_value\_score (entry) \| -11\_b → a5\_a3\_bonus\_gate (entry)
- `#23` split -12 into -12\_a, -12\_b, -12\_c, -12\_d. Final copies: -12\_a → linear\_first\_value\_score (entry) \| -12\_b → a5\_a3\_bonus\_gate (entry) \| -12\_c → a5\_a3\_bonus\_gate (entry) \| -12\_d → a5\_in\_2\_4\_penalty (entry)
- `#24` split -16 into -16\_a, -16\_b, -16\_c. Final copies: -16\_a → linear\_first\_value\_score (entry) \| -16\_b → atleast3\_of\_five\_veto (entry) \| -16\_c → a5\_a3\_bonus\_gate (entry)
- `#25` identity node id\_linear intercepts 13 connection(s) into 0 (from -1\_a (a1:1\_a), -2 (a1:2), -3\_a (a1:3\_a), -4\_a (a2:1\_a), -5 (a2:2), -6 (a2:3), -7\_a (a3:1\_a), -9\_a (a4:1\_a), … (5 more)); sits in linear\_first\_value\_score (exit); two\_of\_five\_peak (entry)
- `#26` identity node id\_suppress intercepts 2 connection(s) into 0 (from 274, 2786); sits in atleast3\_of\_five\_veto (exit); two\_of\_five\_peak (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 4.44e-16 | 4.44e-16 | 4.44e-16 | agree |
| Closed-form max abs error (all leaves) | 0.1699 | 0.1699 | 0.1699 | agree |
| Numeric-function max abs error | n/a | 2.84e-14 | 2.84e-14 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | two\_of\_five\_peak | two\_of\_five\_peak | agree |
| Leaf / composition counts | 5 / 1 | n/a | 5 / 1 | agree |
| Operation counts | 8 splits, 2 identity | {"add\_identity\_node": 2, "annotate": 6, "rename\_node": 17, "split\_node": 8} | {"add\_identity\_node": 2, "annotate": 6, "rename\_node": 17, "split\_node": 8} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `atleast3_of_five_veto` | 0.1699 | 2.84e-14 |
| `linear_first_value_score` | 0.02419 | 7.11e-15 |
| `a5_a3_bonus_gate` | 0.006729 | 2.22e-16 |
| `a5_in_2_4_penalty` | 0.003135 | 0 |
| `inert_a32_unit` | 0 | 0 |

## What resisted (from the record)

> a5\_a3\_bonus\_gate is not a clean logical unit (additive a5!=1 / a3=1 bonus with one clip interaction, plus near-inert sub-unit 1717). Nodes 1948, 2786 affine on the domain; node 927 inert. Closed-form error 0.17 is 3-s.f. rounding at magnitudes up to 269 (numeric 2.8e-14).

**Tool notes (record):** add\_identity\_node docstring params wrong. Parallel add\_evidence on the same genome races and lost one record (re-added serially).

## What to look at

- Largest closed-form error: `atleast3_of_five_veto` (0.17), then `linear_first_value_score` (0.0242). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. It is large enough that a reader using the displayed coefficients would get visibly different exit values: check which coefficient drives it.
- Strong claim in `atleast3_of_five_veto`: *“Evidence: on all 345 training rows 274 \> 0 exactly when three or more of a1,a2,a3,a4,a6 take their first value (112 rows);”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `a5_a3_bonus_gate`: *“Evidence: on all 345 training rows 1223 = 0 exactly when a5 = 1 and a3 = 2 (40 rows);”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 8 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s4.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s4', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
