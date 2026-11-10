# Verification packet 14 · monk2-s7

Pre-registered verification order: 14 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/4cacc8f5-e09f-44eb-83e3-afd74922e268?experiment=a72d23bf-64bc-4beb-8495-051d5c6794d6&name=10seed-monk2-s7), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 7 |
| Genome id | `4cacc8f5-e09f-44eb-83e3-afd74922e268` |
| Experiment | 10seed-monk2-s7 (a72d23bf-64bc-4beb-8495-051d5c6794d6); best at generation 78 of 100, NEAT key 18758 |
| Test AUC (runs csv) | 1 \[95% CI 1, 1\] |
| Train AUC | 1 |
| Size | 10 of 17 inputs connected; 3 hidden nodes; 19 enabled connections; depth 4; 9 skip connections + 6 direct input→output edges |
| Annotations | 2 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 2 of 3 |
| Claude's time | 41.0 min (2026-11-03T13:12:36+13:00 → 2026-11-03T13:53:38+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Output = sigmoid(1.383 + Ramp - 4.009 Veto - 0.888 Pool). Ramp is a descending linear count of first-value indicators (positive only when \<=2 active); Pool is a rectified count of I4 and I5; Veto fires when \<=1 indicator is active and overrides the ramp. Net: a window on exactly two active indicators. Train accuracy 0.991, AUC 1.0.

**Agreement with the MONK-2 rule (record):** 428/432 configurations agree with the MONK-2 rule; all 4 disagreements are a2=1 & a5=1 veto-leak cases. Implements the window in reverse to the paper's model: descending ramp sets the upper edge, low-count veto sets the lower edge.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `exactly_two_window`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | 389, 288, id\_ramp, -2 (a1:2), -3 (a1:3), -5 (a2:2), -8 (a3:2), -11 (a4:3), -13 (a5:2), -14 (a5:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | low\_count\_veto\_gate, linear\_count\_ramp |
| Dimensionality (in → out, incl. descendants) | 10 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.251 e^{0.888 x_{389} - 1.0 \operatorname{linear\_count\_ramp}{\left(-1_{a},-4_{a},-7_{a},-9_{a},-12_{a},-17 \right)} + 4.01 \operatorname{low\_count\_veto\_gate}{\left(-1_{b},-4_{b},-6,-7_{b},-16,x_{389} \right)}} + 1}\end{gathered}
$$

**Hypothesis (DB):**

> Exactly-two window = descending ramp minus low-count veto. Output = sigmoid(1.383 + Ramp - 4.009 Veto - 0.888 Pool). The linear part (bias + ramp + pool readout) by number of active first-value indicators: 0 -\> 18.1, 1 -\> 9.2 to 13.8, 2 -\> 0.49 to 5.67, 3 -\> -8.0 to -2.75, 4 or more -\> below -11.5; it is positive iff at most two indicators are active (upper edge of the window). The veto is at least 5.16 whenever at most one indicator is active, so -4.009 Veto \<= -20.7 overrides the ramp's maximum of 18.1 (lower edge). The output is therefore above 0.5 exactly when two indicators are active, with one exception: for count two made of a2=1 and a5=1, the veto leaks 0.62, and 0.62 x 4.009 = 2.49 exceeds the ramp margin 1.56 (pre-activation -0.93), so the 4 such configurations (a1 in {2,3}, a2=1, a3=2, a4 in {2,3}, a5=1, a6=2) are scored negative. The inputs a1:2, a1:3, a2:2, a3:2, a4:3, a5:2, a5:3 have no enabled connections and zero effect; they are listed here as zero-effect entry nodes. EVIDENCE: Train split (n=345, full sample): accuracy 0.991, AUC 1.000. Linear part (bias + ramp - 0.888 pool) per count: 0 -\> 18.11; 1 -\> 9.22..13.75; 2 -\> 0.49..5.67; 3 -\> -7.98..-2.75; 4 -\> -16.39..-11.48; 5 -\> -24.48..-20.37; 6 -\> -29.50. Output \> 0.5 for 0% of count 0/1/3/4/5/6 rows and 97.3% of count 2 rows; the 3 misses are the a2=1,a5=1 veto leak (output 0.283). Sensitivity of the output: veto 0.213, ramp 0.053, pool 0.047, all seven unconnected inputs 0.0. Over all 432 configurations the network agrees with 'exactly two indicators' on 428 (the 4 a2=1,a5=1 count-2 configurations are the only disagreements).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: Train split (n=345, full sample): accuracy 0.991, AUC 1.000. Linear part (bias + ramp - 0.888 pool) per count: 0 -\> 18.11; 1 -\> 9.22..13.75; 2 -\> 0.49..5.67; 3 -\> -7.98..-2.75; 4 -\> -16.39..-11.48; 5 -\> -24.48..-20.37; 6 -\> -29.50. Output \> 0.5 for 0% of count 0/1/3/4/5/6 rows and 97.3% of count 2 rows; the 3 misses are the a2=1,a5=1 veto leak (output 0.283). Sensitivity of the output: veto 0.213, ramp 0.053, pool 0.047, all seven unconnected inputs 0.0. Over all 432 configurations the network agrees with 'exactly two indicators' on 428 (the 4 a2=1,a5=1 count-2 configurations are the only disagreements).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are z-scored one-hot columns; indicator-form coefficients are given in the hypothesis"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "entry_names": [
  "389",
  "288",
  "id_ramp",
  "-2",
  "-3",
  "-5",
  "-8",
  "-11",
  "-13",
  "-14"
 ],
 "max_samples": 100000,
 "sensitivity": {
  "0": [
   0.04693582536041474,
   0.21290971495841535,
   0.05266675259524126,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ]
 },
 "sample_fraction": 1.0,
 "count_table_train": {
  "0": {
   "n": 19,
   "y_mean": 0.0,
   "gate_on": 1.0,
   "lin_max": 18.11375839362858,
   "lin_min": 18.11375839362858,
   "out_pos": 0.0,
   "gate_max": 17.86576556441778,
   "gate_min": 17.125078279911335
  },
  "1": {
   "n": 70,
   "y_mean": 0.0,
   "gate_on": 1.0,
   "lin_max": 13.748552015836477,
   "lin_min": 9.220662459336808,
   "out_pos": 0.0,
   "gate_max": 9.314838076657248,
   "gate_min": 5.162989739748502
  },
  "2": {
   "n": 113,
   "y_mean": 1.0,
   "gate_on": 0.02654867256637168,
   "lin_max": 5.665290966186106,
   "lin_min": 0.491075385159792,
   "out_pos": 0.9734513274336283,
   "gate_max": 0.6205508402550902,
   "gate_min": 0.0
  },
  "3": {
   "n": 92,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -2.7457729699001243,
   "lin_min": -7.982546074335429,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  },
  "4": {
   "n": 42,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -11.475360044077142,
   "lin_min": -16.393610010421657,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  },
  "5": {
   "n": 8,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -20.368455978368907,
   "lin_min": -24.47687106007203,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  },
  "6": {
   "n": 1,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -29.502954419630402,
   "lin_min": -29.502954419630402,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  }
 }
}
```

</details>

### A.1 Composition: `low_count_veto_gate`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -1\_b (a1:1\_b), -4\_b (a2:1\_b), -6 (a2:3), -7\_b (a3:1\_b), -16 (a6:1), 389 |
| Exit nodes | 288 |
| Internal / exit nodes in its own subgraph | 288 |
| Children | a4\_a5\_first\_value\_pool |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 5.12 x_{a1:1\_b} \\
&\quad - 3.69 x_{a2:1\_b} \\
&\quad + 0.351 x_{a2:3} \\
&\quad - 4.92 x_{a3:1\_b} \\
&\quad - 5.91 x_{a6:1} \\
&\quad - 3.38 a_{4 a5 first value pool}{\left(-10,-9_{b},-12_{b},-15 \right)} \\
&\quad -0.156 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Low-count veto gate, a serial composition Veto(a1, a2, a3, a6, Pool(a4, a5)). Node 288 = ReLU(17.125 - 10.71 I1 - 7.81 I2 + 0.74 \[a2=3\] - 9.85 I3 - 11.82 I6 - 3.378 Pool). Through the pool, I4 alone subtracts 10.3 (12.0 when a5=4), I5 alone 8.7, both 21.5. Every active first-value indicator thus removes 7.8 to 11.8 from a bias of 17.1: the gate is on (5.16 to 17.87) for every configuration with zero or one active indicators and off for two or more, with one exception: I2 + I5 (a2=1 and a5=1) leaves 17.125 - 7.81 - 8.69 = 0.62 \> 0, a small leak. So 288 detects 'at most one of the six rule indicators is active'. EVIDENCE: Train split (n=345, full sample), by number of active first-value indicators: count 0 (19 rows) gate 17.13..17.87, on 100%; count 1 (70) gate 5.16..9.32, on 100%; count 2 (113) on 2.7% (3 rows, all a2=1 and a5=1, value 0.62); count \>=3 (143) off 100%. Sensitivities: a6:1 1.58, a1:1 1.37, a3:1 1.31, a2:1 0.98, pool 0.90, a2:3 0.09. Over all 432 configurations the same pattern holds (on for count \<=1; on at count 2 only for the 4 a2=1,a5=1 configurations).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: Train split (n=345, full sample), by number of active first-value indicators: count 0 (19 rows) gate 17.13..17.87, on 100%; count 1 (70) gate 5.16..9.32, on 100%; count 2 (113) on 2.7% (3 rows, all a2=1 and a5=1, value 0.62); count \>=3 (143) off 100%. Sensitivities: a6:1 1.58, a1:1 1.37, a3:1 1.31, a2:1 0.98, pool 0.90, a2:3 0.09. Over all 432 configurations the same pattern holds (on for count \<=1; on at count 2 only for the 4 a2=1,a5=1 configurations).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are z-scored one-hot columns; indicator-form coefficients are given in the hypothesis"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "entry_names": [
  "-1_b",
  "-4_b",
  "-6",
  "-7_b",
  "-16",
  "389"
 ],
 "max_samples": 100000,
 "sensitivity": {
  "288": [
   1.3654626763032651,
   0.9839222094948598,
   0.0936997556988346,
   1.3130741117165152,
   1.5760542481988298,
   0.9009280876530745
  ]
 },
 "sample_fraction": 1.0,
 "count_table_train": {
  "0": {
   "n": 19,
   "y_mean": 0.0,
   "gate_on": 1.0,
   "lin_max": 18.11375839362858,
   "lin_min": 18.11375839362858,
   "out_pos": 0.0,
   "gate_max": 17.86576556441778,
   "gate_min": 17.125078279911335
  },
  "1": {
   "n": 70,
   "y_mean": 0.0,
   "gate_on": 1.0,
   "lin_max": 13.748552015836477,
   "lin_min": 9.220662459336808,
   "out_pos": 0.0,
   "gate_max": 9.314838076657248,
   "gate_min": 5.162989739748502
  },
  "2": {
   "n": 113,
   "y_mean": 1.0,
   "gate_on": 0.02654867256637168,
   "lin_max": 5.665290966186106,
   "lin_min": 0.491075385159792,
   "out_pos": 0.9734513274336283,
   "gate_max": 0.6205508402550902,
   "gate_min": 0.0
  },
  "3": {
   "n": 92,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -2.7457729699001243,
   "lin_min": -7.982546074335429,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  },
  "4": {
   "n": 42,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -11.475360044077142,
   "lin_min": -16.393610010421657,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  },
  "5": {
   "n": 8,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -20.368455978368907,
   "lin_min": -24.47687106007203,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  },
  "6": {
   "n": 1,
   "y_mean": 0.0,
   "gate_on": 0.0,
   "lin_max": -29.502954419630402,
   "lin_min": -29.502954419630402,
   "out_pos": 0.0,
   "gate_max": 0.0,
   "gate_min": 0.0
  }
 }
}
```

</details>

#### A.1.1 Leaf: `a4_a5_first_value_pool`

|  |  |
|---|---|
| Annotation id | `ann_23` |
| Entry nodes | -10 (a4:2), -9\_b (a4:1\_b), -12\_b (a5:1\_b), -15 (a5:4) |
| Exit nodes | 389 |
| Internal / exit nodes in its own subgraph | 2443, 389 |
| Dimensionality (in → out, incl. descendants) | 4 → 1 |
| Live closed-form error (3 s.f. display) | 0.00908 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.75 x_{a4:1\_b} \\
&\quad + 1.44 x_{a5:1\_b} \\
&\quad + 0.213 x_{a5:4} \\
&\quad - 0.000589 \operatorname{relu}{\left(0.483 x_{a4:2} - 0.906 \right)} \\
&\quad + 1.39 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A4/a5 first-value pool (with an inert a4:2 sub-unit). Write I\_k for 'attribute k takes its first value'. In indicator form node 389 = ReLU(-0.744 + 3.793 I4 + 3.317 I5 + 0.492 \[a5=4\] - 0.0006 u), where u = node 2443 = ReLU(0.997 \[a4=2\] - 1.281) is identically zero on the whole domain (maximum pre-activation -0.284), so the a4:2 branch is inert (and its readout weight is -0.0006 anyway). With neither a4 nor a5 at its first value the pool outputs 0 (pre-activation -0.744 or -0.252); a4=1 alone gives 3.05 (3.54 when a5=4), a5=1 alone gives 2.57, both give 6.37. The pool is therefore a rectified, near-additive count of the two rule indicators I4 and I5, which the downstream gate does not receive directly. EVIDENCE: Train split (n=345, full sample): node 2443 is 0 on every row; sensitivity of the pool to a4:2 is 0.0, to a4:1 0.86, a5:1 0.71, a5:4 0.10. The pool takes exactly five values on train: 0 (176 rows), 2.573 (a5=1, 62), 3.049 (a4=1, 57), 3.541 (a4=1, a5=4, 25), 6.366 (a4=1 and a5=1, 25).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: Train split (n=345, full sample): node 2443 is 0 on every row; sensitivity of the pool to a4:2 is 0.0, to a4:1 0.86, a5:1 0.71, a5:4 0.10. The pool takes exactly five values on train: 0 (176 rows), 2.573 (a5=1, 62), 3.049 (a4=1, 57), 3.541 (a4=1, a5=4, 25), 6.366 (a4=1 and a5=1, 25).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are z-scored one-hot columns; indicator-form coefficients are given in the hypothesis"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "entry_names": [
  "-10",
  "-9_b",
  "-12_b",
  "-15"
 ],
 "max_samples": 100000,
 "sensitivity": {
  "389": [
   0.0,
   0.8593540276364865,
   0.705676683942183,
   0.1042395205202392
  ]
 },
 "sample_fraction": 1.0
}
```

</details>

### A.2 Leaf: `linear_count_ramp`

|  |  |
|---|---|
| Annotation id | `ann_24` |
| Entry nodes | -1\_a (a1:1\_a), -4\_a (a2:1\_a), -7\_a (a3:1\_a), -9\_a (a4:1\_a), -12\_a (a5:1\_a), -17 (a6:2) |
| Exit nodes | id\_ramp |
| Internal / exit nodes in its own subgraph | id\_ramp |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |
| Live closed-form error (3 s.f. display) | 0.0162 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 4.25 x_{a1:1\_a} \\
&\quad - 3.82 x_{a2:1\_a} \\
&\quad - 4.2 x_{a3:1\_a} \\
&\quad - 0.766 x_{a4:1\_a} \\
&\quad - 2.69 x_{a5:1\_a} \\
&\quad + 4.36 x_{a6:2}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear descending count ramp. In indicator form the direct input-to-output connections sum to 16.73 - 8.89 I1 - 8.08 I2 - 8.41 I3 - 1.66 I4 - 6.19 I5 - 8.73 I6 (exact; a6 enters through its complement a6:2 with +8.73, the same signal inverted). Every first-value indicator lowers the score by 6 to 9, except I4 (-1.66), whose shortfall is made up at the root by the pool readout (-0.888 x Pool, i.e. about -2.7 for I4 and -2.3 for I5). The ramp falls approximately linearly with the number of active rule indicators. EVIDENCE: Train split (n=345, full sample): sensitivities equal the absolute weights (a1:1 4.25, a2:1 3.82, a3:1 4.20, a4:1 0.77, a5:1 2.69, a6:2 4.36), as expected for a linear map. Indicator-form least squares fit over all 432 configurations is exact (residual 5.5e-14): 16.73 - 8.89 I1 - 8.08 I2 - 8.41 I3 - 1.66 I4 - 6.19 I5 - 8.73 I6.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: Train split (n=345, full sample): sensitivities equal the absolute weights (a1:1 4.25, a2:1 3.82, a3:1 4.20, a4:1 0.77, a5:1 2.69, a6:2 4.36), as expected for a linear map. Indicator-form least squares fit over all 432 configurations is exact (residual 5.5e-14): 16.73 - 8.89 I1 - 8.08 I2 - 8.41 I3 - 1.66 I4 - 6.19 I5 - 8.73 I6.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "inputs are z-scored one-hot columns; indicator-form coefficients are given in the hypothesis"
}
```

```json
// empirical_viz
{
 "n": 345,
 "split": "train",
 "entry_names": [
  "-1_a",
  "-4_a",
  "-7_a",
  "-9_a",
  "-12_a",
  "-17"
 ],
 "max_samples": 100000,
 "sensitivity": {
  "id_ramp": [
   4.25173540016523,
   3.8186886054007307,
   4.204100730637564,
   0.7662979004370214,
   2.6869319317554674,
   4.363308099378071
  ]
 },
 "sample_fraction": 1.0
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 4 |
| rename\_node | 17 |
| split\_node | 5 |

5 splits, 1 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -1 into -1\_a, -1\_b. Final copies: -1\_a → linear\_count\_ramp (entry) \| -1\_b → low\_count\_veto\_gate (entry)
- `#18` split -4 into -4\_a, -4\_b. Final copies: -4\_a → linear\_count\_ramp (entry) \| -4\_b → low\_count\_veto\_gate (entry)
- `#19` split -7 into -7\_a, -7\_b. Final copies: -7\_a → linear\_count\_ramp (entry) \| -7\_b → low\_count\_veto\_gate (entry)
- `#20` split -9 into -9\_a, -9\_b. Final copies: -9\_a → linear\_count\_ramp (entry) \| -9\_b → a4\_a5\_first\_value\_pool (entry)
- `#21` split -12 into -12\_a, -12\_b. Final copies: -12\_a → linear\_count\_ramp (entry) \| -12\_b → a4\_a5\_first\_value\_pool (entry)
- `#22` identity node id\_ramp intercepts 6 connection(s) into 0 (from -1\_a (a1:1\_a), -4\_a (a2:1\_a), -7\_a (a3:1\_a), -9\_a (a4:1\_a), -12\_a (a5:1\_a), -17 (a6:2)); sits in linear\_count\_ramp (exit); exactly\_two\_window (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 3.33e-16 | 3.33e-16 | 3.33e-16 | agree |
| Closed-form max abs error (all leaves) | 0.0162 | 0.01621 | 0.01621 | agree |
| Numeric-function max abs error | n/a | 1.78e-15 | 1.78e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | exactly\_two\_window | exactly\_two\_window | agree |
| Leaf / composition counts | 2 / 2 | n/a | 2 / 2 | agree |
| Operation counts | 5 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 17, "split\_node": 5} | {"add\_identity\_node": 1, "annotate": 4, "rename\_node": 17, "split\_node": 5} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `linear_count_ramp` | 0.01621 | 1.78e-15 |
| `a4_a5_first_value_pool` | 0.009081 | 8.88e-16 |

## What resisted (from the record)

> Nothing unexplained. Veto gate 288 takes the a4/a5 pool node as input so sits in a serial composition rather than a leaf. Small veto leak: when the two active indicators are a2=1 and a5=1 the veto outputs 0.62 (x4.009 = 2.49 \> ramp margin 1.56), giving 4/432 configurations (3 train rows) as false negatives.

**Tool notes (record):** None. Limitation: a leaf cannot take another leaf's exit as an entry, so serial chains are compositions.

## What to look at

- Largest closed-form error: `linear_count_ramp` (0.0162). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `a4_a5_first_value_pool`: *“The pool takes exactly five values on train: 0 (176 rows), 2.573 (a5=1, 62), 3.049 (a4=1, 57), 3.541 (a4=1, a5=4, 25), 6.366 (a4=1 and a5=1, 25).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `linear_count_ramp`: *“…s sum to 16.73 - 8.89 I1 - 8.08 I2 - 8.41 I3 - 1.66 I4 - 6.19 I5 - 8.73 I6 (exact;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 5 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s7.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s7', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
