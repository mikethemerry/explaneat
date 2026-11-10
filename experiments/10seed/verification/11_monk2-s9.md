# Verification packet 11 · monk2-s9

Pre-registered verification order: 11 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/7992e0c3-de64-412f-a501-71e0a89b057f?experiment=a821f1ba-b90c-4df8-a8ea-2605510531a6&name=10seed-monk2-s9), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 9 |
| Genome id | `7992e0c3-de64-412f-a501-71e0a89b057f` |
| Experiment | 10seed-monk2-s9 (a821f1ba-b90c-4df8-a8ea-2605510531a6); best at generation 99 of 100, NEAT key 23182 |
| Test AUC (runs csv) | 0.682 \[95% CI 0.557, 0.797\] |
| Train AUC | 0.794 |
| Size | 15 of 17 inputs connected; 5 hidden nodes; 24 enabled connections; depth 3; 4 skip connections + 8 direct input→output edges |
| Annotations | 5 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 5 of 5 |
| Claude's time | 49.5 min (2026-11-03T13:12:36+13:00 → 2026-11-03T14:02:07+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Linear ramp over first values of a2,a3,a4,a6 (a1 weak) plus a count gate (1158, -6.92) that fires at three of {a2,a3,a4,a6} (or a3 plus one other with a1!=3), a direct a5=1 veto (-15.7), a small a5=3 penalty, an always-zero chain and an always-active affine node. Train acc 0.786, AUC 0.794.

**Agreement with the MONK-2 rule (record):** Rough 'about two first values' peak counting a2,a3,a4,a6 (a1 weakly) with a5=1 as a veto: 65 false negatives (all exactly-two: 28 a5 veto, 16 gate, 21 weak ramp) and 34 false positives on 432 inputs.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `peak_with_a5_veto`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | id\_linear, id\_a5veto, 1158, 708, 1057, -2 (a1:2), -11 (a4:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | inert\_chain\_a3\_a5, count\_gate\_a2\_a3\_a4\_a6, a5\_3\_minor\_penalty, linear\_indicator\_ramp, a5\_1\_veto |
| Dimensionality (in → out, incl. descendants) | 7 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.69 e^{- 1.0 a_{5 1 veto}{\left(-12_{a} \right)} + 0.207 a_{5 3 minor penalty}{\left(-10,-12_{b},-14 \right)} + 6.92 \operatorname{count\_gate\_a2\_a3\_a4\_a6}{\left(-3,-4,-8_{b},-9_{b},-16 \right)} + 0.125 \operatorname{inert\_chain\_a3\_a5}{\left(-7_{b},-15 \right)} - 1.0 \operatorname{linear\_indicator\_ramp}{\left(-1,-5,-6,-7_{a},-8_{a},-9_{a},-13,-17 \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{1}{0.69 e^{0.125 x_{1057} + 6.92 x_{1158} + 0.207 x_{708} - 1.0 x_{id\_a5veto} - 1.0 x_{id\_linear}} + 1}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> The output composes the five children additively under the final sigmoid: logit = 0.371 + Ramp + Veto - 6.921\*Gate - 0.207\*A5\_3 - 0.125\*Inert (Inert identically zero). Inputs a1:2 and a4:3 have no enabled connections and are zero-effect entries. With a5 in {3,4} the baseline logit is about -4.99; the ramp adds a1=1 +1.25, a2=1 +5.24, a3=1 +3.70, a4=1 +2.28, a6=1 +4.05, so it turns positive once about two strong indicators (a2,a3,a4,a6) are present (lower edge of a response peak). The count gate pulls the logit back down when three of {a2,a3,a4,a6} are first-valued (upper edge), weakly also for a3 plus one other. The a5=1 veto makes the output negative whenever a5=1; a5=3 subtracts a further 0.46-0.60. Net: an approximate 'about two of {a2,a3,a4,a6} (a1 weak), and a5 must not be 1' peak detector. Evidence: closed form composition; train sensitivity of the output (gate 0.53, ramp and veto 0.076 each, 708 0.016, 1057 0.010, unconnected 0); enumeration: 111/432 positive; by total indicator count k: k=0 0%, k=1 4%, k=2 54%, k=3 27%, k\>=4 0%. Against the MONK-2 target: 65 false negatives all at k=2 (28 with a5=1 vetoed; 16 a3+one-other pairs suppressed by the gate; 21 where the ramp is too weak: 20 pairs a1=1 with a3, a4 or a6, plus one a3+a4 pair with a1=3), 34 false positives (30 at k=3, mostly a1=1 plus two others where the gate is weak or silent; 4 at k=1 with a2=1 alone). Train accuracy 0.786, AUC 0.794; function preservation error 3.3e-16.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (z-scored inputs).
- `empirical_viz`: Train-split sensitivity (full sample, 345 rows).
- `exhaustive_enumeration`: Exhaustive enumeration: 111/432 inputs predicted positive. By total rule-indicator count k: k=0 0%, k=1 4.3%, k=2 54.2%, k=3 26.5%, k\>=4 0%. Versus the MONK-2 target: 65 false negatives (all at k=2; 28 of them have a5=1, killed by the veto, the rest killed by the gate firing on a3+one other or by insufficient ramp, e.g. pairs involving a1=1 or a5=1), 34 false positives (4 at k=1, 30 at k=3 where the gate does not fire, e.g. a1=1 plus two of a2/a4/a6). Train: accuracy 0.786, AUC 0.794. Function preservation max abs err 3.3e-16.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "0"
  ],
  "entry_names": [
   "id_linear",
   "id_a5veto",
   "1158",
   "708",
   "1057",
   "-2",
   "-11"
  ],
  "input_labels": [
   "id_linear",
   "id_a5veto",
   "1158",
   "708",
   "1057",
   "-2",
   "-11"
  ],
  "sensitivities": {
   "0": [
    0.07615793056980859,
    0.07615793056980841,
    0.5262369138149191,
    0.015796834665940788,
    0.00952119907397345,
    0.0,
    0.0
   ]
  },
  "dimensionality": [
   7,
   1
  ]
 },
 "viz_type": "sensitivity",
 "max_samples": 100000
}
```

```json
// exhaustive_enumeration
{
 "method": "local read-only forward pass over all 432 one-hot configurations, z-scored with the train scaler",
 "result": "Exhaustive enumeration: 111/432 inputs predicted positive. By total rule-indicator count k: k=0 0%, k=1 4.3%, k=2 54.2%, k=3 26.5%, k>=4 0%. Versus the MONK-2 target: 65 false negatives (all at k=2; 28 of them have a5=1, killed by the veto, the rest killed by the gate firing on a3+one other or by insufficient ramp, e.g. pairs involving a1=1 or a5=1), 34 false positives (4 at k=1, 30 at k=3 where the gate does not fire, e.g. a1=1 plus two of a2/a4/a6). Train: accuracy 0.786, AUC 0.794. Function preservation max abs err 3.3e-16."
}
```

</details>

### A.1 Leaf: `inert_chain_a3_a5`

|  |  |
|---|---|
| Annotation id | `ann_23` |
| Entry nodes | -7\_b (a3:1\_b), -15 (a5:4) |
| Exit nodes | 1057 |
| Internal / exit nodes in its own subgraph | 611, 1057 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.143 x_{a5:4} \\
&\quad + 0.572 \operatorname{relu}{\left(0.212 x_{a3:1\_b} - 1.46 \right)} \\
&\quad -2.65 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> An inert serial chain. Node 611 = ReLU(0.212\*z(a3:1) - 1.458); z(a3:1) takes only two values (-1.03, +0.97), so its pre-activation is at most -1.25 and 611 is identically zero. Node 1057 = ReLU(0.143\*z(a5:4) + 0.572\*h611 - 2.649); with h611 = 0 its pre-activation is at most -2.40, so 1057 is identically zero and its -0.125 readout contributes nothing. Evidence: closed form; train sensitivity 0 for both entries; exhaustive enumeration of all 432 inputs gives h611 = h1057 = 0 everywhere. Structurally present, functionally absent.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (z-scored inputs).
- `empirical_viz`: Train-split sensitivity (full sample, 345 rows).
- `exhaustive_enumeration`: Exhaustive enumeration over all 432 MONK-2 configurations (local forward pass, read-only): h611 = 0 and h1057 = 0 on every input; on the 345 train rows both are constant 0.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "1057"
  ],
  "entry_names": [
   "-7_b",
   "-15"
  ],
  "input_labels": [
   "-7_b",
   "-15"
  ],
  "sensitivities": {
   "1057": [
    0.0,
    0.0
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "viz_type": "sensitivity",
 "max_samples": 100000
}
```

```json
// exhaustive_enumeration
{
 "method": "local read-only forward pass over all 432 one-hot configurations, z-scored with the train scaler",
 "result": "Exhaustive enumeration over all 432 MONK-2 configurations (local forward pass, read-only): h611 = 0 and h1057 = 0 on every input; on the 345 train rows both are constant 0."
}
```

</details>

### A.2 Leaf: `count_gate_a2_a3_a4_a6`

|  |  |
|---|---|
| Annotation id | `ann_24` |
| Entry nodes | -3 (a1:3), -4 (a2:1), -8\_b (a3:2\_b), -9\_b (a4:1\_b), -16 (a6:1) |
| Exit nodes | 1158 |
| Internal / exit nodes in its own subgraph | 1158 |
| Dimensionality (in → out, incl. descendants) | 5 → 1 |
| Live closed-form error (3 s.f. display) | 0.00757 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.775 x_{a1:3} \\
&\quad + 3.11 x_{a2:1} \\
&\quad - 4.14 x_{a3:2\_b} \\
&\quad + 2.58 x_{a4:1\_b} \\
&\quad + 2.81 x_{a6:1} \\
&\quad -3.0 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A graded count gate over four of the six rule indicators. In indicator terms the pre-activation of 1158 is -13.56 + 8.28\[a3=1\] + 6.59\[a2=1\] + 5.58\[a4=1\] + 5.62\[a6=1\] - 1.68\[a1=3\] (a3 enters through its complement a3:2, weight -4.14). It fires iff at least three of {a2=1,a3=1,a4=1,a6=1} hold (activation \>= 2.56) or a3=1 plus exactly one other with a1!=3 (activation only 0.30..1.31). a1=1 and a5=1 are not consulted. Readout -6.92, so the penalty to the output logit is graded: about -2.1 to -9.1 for the a3+one pairs, -17.7 to -86.6 for three or more. Evidence: closed form; train sensitivities (a3:2 1.44, a2:1 1.08, a6:1 0.98, a4:1 0.90, a1:3 0.27); exhaustive enumeration confirms the firing rule on 432/432 inputs. Role: the upper edge of the response peak (suppresses inputs with too many indicators), but the weak pair-level activation is not always enough to cancel the ramp (e.g. a1=1,a3=1,a6=1 stays positive).

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (z-scored inputs).
- `empirical_viz`: Train-split sensitivity (full sample, 345 rows).
- `exhaustive_enumeration`: Exhaustive enumeration over all 432 configurations: the rule 'fires iff count{a2=1,a3=1,a4=1,a6=1}\>=3 or (a3=1 and count=2 and a1!=3)' matches h1158\>0 exactly (432/432). Active range 0.30..12.51. Fraction firing by total rule-indicator count k (all 6 attributes): k\<=1: 0, k=2: 0.17, k=3: 0.60, k=4: 0.90, k\>=5: 1.0. Train: 34.8% of rows active.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "1158"
  ],
  "entry_names": [
   "-3",
   "-4",
   "-8_b",
   "-9_b",
   "-16"
  ],
  "input_labels": [
   "-3",
   "-4",
   "-8_b",
   "-9_b",
   "-16"
  ],
  "sensitivities": {
   "1158": [
    0.2695270794837353,
    1.0831537425330229,
    1.4402151485787815,
    0.8971258106749774,
    0.9767636406810282
   ]
  },
  "dimensionality": [
   5,
   1
  ]
 },
 "viz_type": "sensitivity",
 "max_samples": 100000
}
```

```json
// exhaustive_enumeration
{
 "method": "local read-only forward pass over all 432 one-hot configurations, z-scored with the train scaler",
 "result": "Exhaustive enumeration over all 432 configurations: the rule 'fires iff count{a2=1,a3=1,a4=1,a6=1}>=3 or (a3=1 and count=2 and a1!=3)' matches h1158>0 exactly (432/432). Active range 0.30..12.51. Fraction firing by total rule-indicator count k (all 6 attributes): k<=1: 0, k=2: 0.17, k=3: 0.60, k=4: 0.90, k>=5: 1.0. Train: 34.8% of rows active."
}
```

</details>

### A.3 Leaf: `a5_3_minor_penalty`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -10 (a4:2), -12\_b (a5:1\_b), -14 (a5:3) |
| Exit nodes | 708 |
| Internal / exit nodes in its own subgraph | 708 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00137 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.336 x_{a4:2} \\
&\quad - 0.311 x_{a5:1\_b} \\
&\quad + 2.41 x_{a5:3} \\
&\quad -1.91 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> A small a5=3 detector. In indicator terms 708 = ReLU(-3.37 + 5.59\[a5=3\] + 0.69\[a4=2\] - 0.72\[a5=1\]): 2.22 when a5=3 (2.92 if also a4=2), zero otherwise. Readout -0.207 lowers the logit by 0.46/0.60 when a5=3. Evidence: closed form; train sensitivity dominated by a5:3 (0.59 vs 0.08 for a4:2 and a5:1); enumeration shows three values {0, 2.22, 2.92}; removing it flips the decision on only 9/432 inputs. A minor nuisance term.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (z-scored inputs).
- `empirical_viz`: Train-split sensitivity (full sample, 345 rows).
- `exhaustive_enumeration`: Exhaustive enumeration: h708 takes values {0, 2.22, 2.92}; nonzero iff a5=3 (2.92 when also a4=2). Train: 24.6% of rows active. Removing its readout flips the sign of the output logit on 9/432 inputs.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "708"
  ],
  "entry_names": [
   "-10",
   "-12_b",
   "-14"
  ],
  "input_labels": [
   "-10",
   "-12_b",
   "-14"
  ],
  "sensitivities": {
   "708": [
    0.08290119715720888,
    0.07668492334519335,
    0.593689429213078
   ]
  },
  "dimensionality": [
   3,
   1
  ]
 },
 "viz_type": "sensitivity",
 "max_samples": 100000
}
```

```json
// exhaustive_enumeration
{
 "method": "local read-only forward pass over all 432 one-hot configurations, z-scored with the train scaler",
 "result": "Exhaustive enumeration: h708 takes values {0, 2.22, 2.92}; nonzero iff a5=3 (2.92 when also a4=2). Train: 24.6% of rows active. Removing its readout flips the sign of the output logit on 9/432 inputs."
}
```

</details>

### A.4 Leaf: `linear_indicator_ramp`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -1 (a1:1), -5 (a2:2), -6 (a2:3), -7\_a (a3:1\_a), -8\_a (a3:2\_a), -9\_a (a4:1\_a), -13 (a5:2), -17 (a6:2) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | 3057, id\_linear |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |
| Live closed-form error (3 s.f. display) | 0.0294 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.6 x_{a1:1} \\
&\quad - 2.45 x_{a2:2} \\
&\quad + 2.75 x_{a3:1\_a} \\
&\quad + 0.901 x_{a3:2\_a} \\
&\quad + 1.05 x_{a4:1\_a} \\
&\quad - 0.454 x_{a5:2} \\
&\quad - 2.02 x_{a6:2} \\
&\quad - 1.41 \operatorname{relu}{\left(1.51 x_{a2:3} + 1.73 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> An additive ramp over the first-value indicators of a1,a2,a3,a4,a6. Node 3057 = ReLU(1.51\*z(a2:3)+1.73) is always active (0.641 or 3.826), so it is a linear term in a2:3. Per attribute, relative to all attributes at their second value (ramp = -10.37): a1=1 +1.25; a2=1 +5.24 (a2=3 +0.73); a3=1 +3.70; a4=1 +2.28; a6=1 +4.05; a5=2 -1.05 relative to a5=1/3/4. The ramp therefore rises roughly linearly with the number of strong indicators (a2,a3,a4,a6), with a1=1 weighted only weakly; with the output bias, the a5 veto offset and no gate it crosses zero once about two strong indicators are present (or one strong indicator plus a1=1 in some combinations): the lower edge of the response peak. Evidence: closed form; train sensitivities match the edge weights; exhaustive enumeration confirms exact additivity.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (z-scored inputs).
- `empirical_viz`: Train-split sensitivity (full sample, 345 rows).
- `exhaustive_enumeration`: Exhaustive enumeration: the ramp is exactly additive per attribute (node 3057 always active: 0.641 or 3.826). Per-level effects relative to all-second-level (-10.37): a1=(+1.25,0,0), a2=(+5.24,0,+0.73), a3=(+3.70,0), a4=(+2.28,0,0), a5=(+1.05,0,+1.05,+1.05), a6=(+4.05,0). Range -10.37..7.20.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "id_linear"
  ],
  "entry_names": [
   "-1",
   "-5",
   "-6",
   "-7_a",
   "-8_a",
   "-9_a",
   "-13",
   "-17"
  ],
  "input_labels": [
   "-1",
   "-5",
   "-6",
   "-7_a",
   "-8_a",
   "-9_a",
   "-13",
   "-17"
  ],
  "sensitivities": {
   "id_linear": [
    0.5997419541714447,
    2.445529947592462,
    2.1368647786922574,
    2.751169458236915,
    0.9006527312785745,
    1.0524542760652225,
    0.4540665770708612,
    2.0231408112511677
   ]
  },
  "dimensionality": [
   8,
   1
  ]
 },
 "viz_type": "sensitivity",
 "max_samples": 100000
}
```

```json
// exhaustive_enumeration
{
 "method": "local read-only forward pass over all 432 one-hot configurations, z-scored with the train scaler",
 "result": "Exhaustive enumeration: the ramp is exactly additive per attribute (node 3057 always active: 0.641 or 3.826). Per-level effects relative to all-second-level (-10.37): a1=(+1.25,0,0), a2=(+5.24,0,+0.73), a3=(+3.70,0), a4=(+2.28,0,0), a5=(+1.05,0,+1.05,+1.05), a6=(+4.05,0). Range -10.37..7.20."
}
```

</details>

### A.5 Leaf: `a5_1_veto`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | -12\_a (a5:1\_a) |
| Exit nodes | id\_a5veto |
| Internal / exit nodes in its own subgraph | id\_a5veto |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00508 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = - 6.82 x_{a5:1\_a}\end{gathered}
$$

**Hypothesis (DB):**

> A veto on a5=1. The direct edge a5:1 -\> output has weight -6.82 on the z-score: +3.96 when a5!=1, -11.74 when a5=1, a jump of -15.70. The remaining terms cannot overcome it (max ramp gain from indicators is +16.5 from a baseline of -20.7 with a5=1, and the gate subtracts once several indicators are present), so the network never predicts class 1 when a5=1. Evidence: closed form; train sensitivity 6.82; enumeration: 0 of 108 a5=1 inputs positive.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form of the subgraph (z-scored inputs).
- `empirical_viz`: Train-split sensitivity (full sample, 345 rows).
- `exhaustive_enumeration`: Exhaustive enumeration: veto takes values +3.96 (a5!=1) and -11.74 (a5=1). Output logit is negative on all 108 inputs with a5=1 (0 positive predictions).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula force=true"
}
```

```json
// empirical_viz
{
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "id_a5veto"
  ],
  "entry_names": [
   "-12_a"
  ],
  "input_labels": [
   "-12_a"
  ],
  "sensitivities": {
   "id_a5veto": [
    6.81704937045634
   ]
  },
  "dimensionality": [
   1,
   1
  ]
 },
 "viz_type": "sensitivity",
 "max_samples": 100000
}
```

```json
// exhaustive_enumeration
{
 "method": "local read-only forward pass over all 432 one-hot configurations, z-scored with the train scaler",
 "result": "Exhaustive enumeration: veto takes values +3.96 (a5!=1) and -11.74 (a5=1). Output logit is negative on all 108 inputs with a5=1 (0 positive predictions)."
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 2 |
| annotate | 6 |
| rename\_node | 17 |
| split\_node | 4 |

4 splits, 2 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -7 into -7\_a, -7\_b. Final copies: -7\_a → linear\_indicator\_ramp (entry) \| -7\_b → inert\_chain\_a3\_a5 (entry)
- `#18` split -8 into -8\_a, -8\_b. Final copies: -8\_a → linear\_indicator\_ramp (entry) \| -8\_b → count\_gate\_a2\_a3\_a4\_a6 (entry)
- `#19` split -9 into -9\_a, -9\_b. Final copies: -9\_a → linear\_indicator\_ramp (entry) \| -9\_b → count\_gate\_a2\_a3\_a4\_a6 (entry)
- `#20` split -12 into -12\_a, -12\_b. Final copies: -12\_a → a5\_1\_veto (entry) \| -12\_b → a5\_3\_minor\_penalty (entry)
- `#21` identity node id\_linear intercepts 8 connection(s) into 0 (from -1 (a1:1), -5 (a2:2), -7\_a (a3:1\_a), -8\_a (a3:2\_a), -9\_a (a4:1\_a), -13 (a5:2), -17 (a6:2), 3057); sits in linear\_indicator\_ramp (exit); peak\_with\_a5\_veto (entry)
- `#22` identity node id\_a5veto intercepts 1 connection(s) into 0 (from -12\_a (a5:1\_a)); sits in a5\_1\_veto (exit); peak\_with\_a5\_veto (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T12:25:06+13:00 (commit `77513b9`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 3.33e-16 | 3.33e-16 | 3.33e-16 | agree |
| Closed-form max abs error (all leaves) | 0.02936 | 0.02936 | 0.02936 | agree |
| Numeric-function max abs error | n/a | 1.78e-15 | 1.78e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | peak\_with\_a5\_veto | peak\_with\_a5\_veto | agree |
| Leaf / composition counts | 5 / 1 | n/a | 5 / 1 | agree |
| Operation counts | 4 splits, 2 identity | {"add\_identity\_node": 2, "annotate": 6, "rename\_node": 17, "split\_node": 4} | {"add\_identity\_node": 2, "annotate": 6, "rename\_node": 17, "split\_node": 4} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `linear_indicator_ramp` | 0.02936 | 1.78e-15 |
| `count_gate_a2_a3_a4_a6` | 0.007567 | 1.78e-15 |
| `a5_1_veto` | 0.005081 | 0 |
| `a5_3_minor_penalty` | 0.00137 | 0 |
| `inert_chain_a3_a5` | 0 | 0 |

## What resisted (from the record)

> (empty)

**Tool notes (record):** add\_identity\_node docstring params wrong.

## What to look at

- Largest closed-form error: `linear_indicator_ramp` (0.0294). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `count_gate_a2_a3_a4_a6`: *“It fires iff at least three of {a2=1,a3=1,a4=1,a6=1} hold (activation \>= 2.56) or a3=1 plus exactly one other with a1!=3 (activation only 0.30..1.31).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `a5_3_minor_penalty`: *“removing it flips the decision on only 9/432 inputs.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 4 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s9.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s9', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
