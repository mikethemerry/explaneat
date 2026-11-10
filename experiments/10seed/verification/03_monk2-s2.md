# Verification packet 03 · monk2-s2

Pre-registered verification order: 3 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/651cf7a8-b9e9-458b-b212-cdd3ce7c2a78?experiment=07b4e641-7d22-4eae-8755-0f740c81ba8f&name=10seed-monk2-s2), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | monk2 / 2 |
| Genome id | `651cf7a8-b9e9-458b-b212-cdd3ce7c2a78` |
| Experiment | 10seed-monk2-s2 (07b4e641-7d22-4eae-8755-0f740c81ba8f); best at generation 70 of 100, NEAT key 16945 |
| Test AUC (runs csv) | 0.767 \[95% CI 0.703, 0.831\] |
| Train AUC | 0.808 |
| Size | 11 of 17 inputs connected; 6 hidden nodes; 26 enabled connections; depth 5; 9 skip connections + 9 direct input→output edges |
| Annotations | 6 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 6 of 6 |
| Claude's time | 54.2 min (2026-11-03T13:12:36+13:00 → 2026-11-03T14:06:51+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Saturated logistic count: logit \~ 80 - 20k (k = number of first values); per-indicator drops 17.5-25 via 9 direct edges; hidden pathway is a lookup on (a3, a6, a1=3) supplying the a3 indicator. Output \~1 iff k\<=3.

**Agreement with the MONK-2 rule (record):** Only an upper edge one step too high (k\>=4/5 negative) and no lower edge: agrees with the rule on 41.4% of 432; output \~1 on 91.5% of train rows (acc 0.41, AUC 0.81).

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 345 training rows.

### A Composition: `count_at_most_three`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | id\_linear, id\_hidden, -2 (a1:2), -10 (a4:2), -11 (a4:3), -14 (a5:3), -15 (a5:4), -16 (a6:1) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | linear\_count\_down, a3\_pathway |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.0 a_{3 pathway}{\left(-3,-7_{a},-7_{b},-8_{b},-8_{c},-8_{d},-17_{b},-17_{c} \right)} + 1.0 \operatorname{linear\_count\_down}{\left(-1,-4,-5,-6,-8_{a},-9,-12,-13,-17_{a} \right)}}}{e^{1.0 a_{3 pathway}{\left(-3,-7_{a},-7_{b},-8_{b},-8_{c},-8_{d},-17_{b},-17_{c} \right)} + 1.0 \operatorname{linear\_count\_down}{\left(-1,-4,-5,-6,-8_{a},-9,-12,-13,-17_{a} \right)}} + 0.299}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{e^{1.0 x_{id\_hidden} + 1.0 x_{id\_linear}}}{e^{1.0 x_{id\_hidden} + 1.0 x_{id\_linear}} + 0.299}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: output = sigmoid(1.208 + Linear + A3Pathway), an additive composition of linear\_count\_down and a3\_pathway under the output sigmoid. Unconnected inputs a1:2, a4:2, a4:3, a5:3, a5:4, a6:1 are zero-effect entries (their one-hot siblings carry the same information). Net: logit \~ 80 - 20\*k, where k = number of attributes taking their first value; per-indicator drops a1 19.2, a2 22.5/25.4, a3 22.2 (29.5 if a1=3), a4 17.5, a5 24.3/20.1, a6 22.4. Exhaustive check over all 432 configurations: logit 82..96 (k=0), 60..79 (k=1), 37..56 (k=2), 15..32 (k=3), -5.2..7.7 (k=4, 49% positive, depending on which attributes are first-valued: the a1/a4 indicators drop the least), \<-16 (k\>=5). The decision equals 'k \<= 3' on 94.4% of the 432 configurations (all disagreements at k=4). The model is a saturated monotone 'at most three (sometimes four) first values' detector. Train (n=345): output median 1.0, mean 0.915; accuracy 0.41, AUC 0.81. Compared with the target rule (exactly two first values) it agrees on 41.4% of the 432 configurations: it learned only the upper edge of the k=2 peak (k\>=4/5 -\> negative), placed one step too high, and has no lower edge (k=0,1 -\> positive).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 1.0,
   "min": 6.033392768421105e-19,
   "std": 0.2552785256808342,
   "mean": 0.9154925335825149,
   "count": 345,
   "median": 1.0
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "0"
  ],
  "entry_names": [
   "id_linear",
   "id_hidden",
   "-2",
   "-10",
   "-11",
   "-14",
   "-15",
   "-16"
  ],
  "dimensionality": [
   8,
   1
  ]
 }
}
```

</details>

### A.1 Leaf: `linear_count_down`

|  |  |
|---|---|
| Annotation id | `ann_29` |
| Entry nodes | -1 (a1:1), -4 (a2:1), -5 (a2:2), -6 (a2:3), -8\_a (a3:2\_a), -9 (a4:1), -12 (a5:1), -13 (a5:2), -17\_a (a6:2\_a) |
| Exit nodes | id\_linear |
| Internal / exit nodes in its own subgraph | id\_linear |
| Dimensionality (in → out, incl. descendants) | 9 → 1 |
| Live closed-form error (3 s.f. display) | 0.0271 |
| Live numeric-function error | 7.11e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 9.2 x_{a1:1} \\
&\quad - 8.55 x_{a2:1} \\
&\quad + 2.03 x_{a2:2} \\
&\quad + 3.45 x_{a2:3} \\
&\quad - 2.63 x_{a3:2\_a} \\
&\quad - 8.11 x_{a4:1} \\
&\quad - 8.75 x_{a5:1} \\
&\quad + 1.8 x_{a5:2} \\
&\quad + 9.43 x_{a6:2\_a}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear count of first-value indicators, read through nine direct edges (exact linear, no nonlinearity). Per-attribute logit contributions (first value vs others): a1 -12.43 vs +6.80; a2 -15.91 vs +6.55 (a2=2) / +9.48 (a2=3); a4 -12.09 vs +5.44; a5 -16.11 vs +8.18 (a5=2) / +4.04 (a5=3,4); a6 -9.18 vs +9.68. Each of these indicators lowers the logit by 17.5..24.3. a3 has the opposite (small) sign directly: a3=1 +2.56 vs a3=2 -2.70; the a3\_pathway reverses this. Train (n=345): id\_linear ranges -68.4..42.1 (mean 0, sd 21.5), and per indicator count k its range is k=0: 29.8..36.9, k=2: -12.8..24.6, k=4: -50.9..-13.5, k=6: -63.2.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 42.12393766091575,
   "min": -68.42229332394037,
   "std": 21.548464533631627,
   "mean": -1.4828717963689047e-15,
   "count": 345,
   "median": -1.3302613568385613
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "id_linear"
  ],
  "entry_names": [
   "-1",
   "-4",
   "-5",
   "-6",
   "-8_a",
   "-9",
   "-12",
   "-13",
   "-17_a"
  ],
  "dimensionality": [
   9,
   1
  ]
 }
}
```

</details>

### A.2 Composition: `a3_pathway`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -3 (a1:3), -7\_a (a3:1\_a), -7\_b (a3:1\_b), -8\_b (a3:2\_b), -8\_c (a3:2\_c), -8\_d (a3:2\_d), -17\_b (a6:2\_b), -17\_c (a6:2\_c) |
| Exit nodes | id\_hidden |
| Internal / exit nodes in its own subgraph | id\_hidden |
| Children | dead\_a3a6\_unit\_1083, dead\_a3\_unit\_2445, a3\_offset\_639, a3\_gate\_435, linear\_relay\_821\_1957 |
| Dimensionality (in → out, incl. descendants) | 8 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 3.49 a_{3 gate 435}{\left(x_{a1:3},x_{a3:2\_b},0.284 \operatorname{dead\_a3a6\_unit\_1083}{\left(x_{a3:1\_a},x_{a6:2\_c} \right)} \right)} \\
&\quad - 0.721 a_{3 offset 639}{\left(x_{a3:2\_c} \right)} \\
&\quad - 0.109 \operatorname{dead\_a3\_unit\_2445}{\left(x_{a3:1\_b} \right)} \\
&\quad - 0.673 \operatorname{linear\_relay\_821\_1957\_0}{\left(x_{a3:2\_d},x_{a6:2\_b},2.54 a_{3 gate 435}{\left(x_{a1:3},x_{a3:2\_b},0.284 \operatorname{dead\_a3a6\_unit\_1083}{\left(x_{a3:1\_a},x_{a6:2\_c} \right)} \right)} - 1.32 \operatorname{dead\_a3a6\_unit\_1083}{\left(x_{a3:1\_a},x_{a6:2\_c} \right)} \right)} \\
&\quad + 2.15 \operatorname{linear\_relay\_821\_1957\_1}{\left(x_{a3:2\_d},x_{a6:2\_b},2.54 a_{3 gate 435}{\left(x_{a1:3},x_{a3:2\_b},0.284 \operatorname{dead\_a3a6\_unit\_1083}{\left(x_{a3:1\_a},x_{a6:2\_c} \right)} \right)} - 1.32 \operatorname{dead\_a3a6\_unit\_1083}{\left(x_{a3:1\_a},x_{a6:2\_c} \right)} \right)}
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.109 \operatorname{relu}{\left(0.706 x_{a3:1\_b} - 1.49 \right)} \\
&\quad - 0.721 \operatorname{relu}{\left(2.16 x_{a3:2\_c} + 3.14 \right)} \\
&\quad + 2.15 \operatorname{relu}{\left(1.63 \operatorname{relu}{\left(- 0.893 x_{a3:2\_d} + 0.627 x_{a6:2\_b} - 1.32 \operatorname{relu}{\left(0.298 x_{a3:1\_a} + 1.13 x_{a6:2\_c} - 1.95 \right)} + 2.54 \operatorname{relu}{\left(0.317 x_{a1:3} + 6.72 x_{a3:2\_b} + 0.284 \operatorname{relu}{\left(0.298 x_{a3:1\_a} + 1.13 x_{a6:2\_c} - 1.95 \right)} - 3.36 \right)} + 5.77 \right)} + 1.71 \right)} \\
&\quad + 3.49 \operatorname{relu}{\left(0.317 x_{a1:3} + 6.72 x_{a3:2\_b} + 0.284 \operatorname{relu}{\left(0.298 x_{a3:1\_a} + 1.13 x_{a6:2\_c} - 1.95 \right)} - 3.36 \right)} \\
&\quad - 0.673 \operatorname{relu}{\left(- 0.893 x_{a3:2\_d} + 0.627 x_{a6:2\_b} - 1.32 \operatorname{relu}{\left(0.298 x_{a3:1\_a} + 1.13 x_{a6:2\_c} - 1.95 \right)} + 2.54 \operatorname{relu}{\left(0.317 x_{a1:3} + 6.72 x_{a3:2\_b} + 0.284 \operatorname{relu}{\left(0.298 x_{a3:1\_a} + 1.13 x_{a6:2\_c} - 1.95 \right)} - 3.36 \right)} + 5.77 \right)}
\end{aligned}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> The hidden pathway (children: dead\_a3a6\_unit\_1083, dead\_a3\_unit\_2445, a3\_offset\_639, a3\_gate\_435, linear\_relay\_821\_1957). id\_hidden = 3.489\*h435 - 0.721\*h639 - 0.673\*h821 + 2.148\*h1957 - 0.109\*h2445; the relay is fed by 2.542\*h435 - 1.324\*h1083 and the gate by 0.284\*h1083; both dead units are 0, and the relay and offset are linear on the domain, so the only nonlinearity is the a3 gate, which reaches the output with total gain 3.489 + 2.542\*2.837 = 10.70. Result: a lookup on (a3, a6, a1=3): +20.01 (a3=1,a6=1), +23.57 (a3=1,a6=2), +47.44 (a3=2,a6=1), +51.00 (a3=2,a6=2), plus +7.33 when a3=2 and a1=3. Train (n=345): id\_hidden takes exactly these six values (min 20.01, max 58.33). Function: it supplies the a3 'first value' indicator (a3=1 lowers the logit by 27.4, or 34.8 if a1=3; net -22.2 / -29.5 after the linear part's +5.25) and adds a small a6 term (a6=1 lowers by a further 3.56, total a6 drop 22.4).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 58.33295876704883,
   "min": 20.009216921052175,
   "std": 15.029376625781838,
   "mean": 36.09922134410331,
   "count": 345,
   "median": 23.56815555620442
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "id_hidden"
  ],
  "entry_names": [
   "-3",
   "-7_a",
   "-7_b",
   "-8_b",
   "-8_c",
   "-8_d",
   "-17_b",
   "-17_c"
  ],
  "dimensionality": [
   8,
   1
  ]
 }
}
```

</details>

#### A.2.1 Leaf: `dead_a3a6_unit_1083`

|  |  |
|---|---|
| Annotation id | `ann_24` |
| Entry nodes | -7\_a (a3:1\_a), -17\_c (a6:2\_c) |
| Exit nodes | 1083 |
| Internal / exit nodes in its own subgraph | 1083 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.298 x_{a3:1\_a} \\
&\quad + 1.13 x_{a6:2\_c} \\
&\quad -1.95 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Dead unit 1083 = ReLU(0.298 z(a3:1) + 1.13 z(a6:2) - 1.95). Analytically the maximum pre-activation over the whole input domain is -0.502 (a3=1, a6=2), so the unit is identically zero on all 432 configurations. Empirically (train, n=345, full sample) its output distribution is min=max=0. It reads a3 and a6, but contributes nothing to the gate (0.284) or the relay (-1.324).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 0.0,
   "min": 0.0,
   "std": 0.0,
   "mean": 0.0,
   "count": 345,
   "median": 0.0
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "1083"
  ],
  "entry_names": [
   "-7_a",
   "-17_c"
  ],
  "dimensionality": [
   2,
   1
  ]
 }
}
```

</details>

#### A.2.2 Leaf: `dead_a3_unit_2445`

|  |  |
|---|---|
| Annotation id | `ann_25` |
| Entry nodes | -7\_b (a3:1\_b) |
| Exit nodes | 2445 |
| Internal / exit nodes in its own subgraph | 2445 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.706 x_{a3:1\_b} - 1.49 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Dead unit 2445 = ReLU(0.706 z(a3:1) - 1.49). Maximum pre-activation -0.802 \< 0 over the input domain, so it is identically zero; train output distribution (n=345) min=max=0. Its -0.109 readout into the output contributes nothing.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 0.0,
   "min": 0.0,
   "std": 0.0,
   "mean": 0.0,
   "count": 345,
   "median": 0.0
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "2445"
  ],
  "entry_names": [
   "-7_b"
  ],
  "dimensionality": [
   1,
   1
  ]
 }
}
```

</details>

#### A.2.3 Leaf: `a3_offset_639`

|  |  |
|---|---|
| Annotation id | `ann_26` |
| Entry nodes | -8\_c (a3:2\_c) |
| Exit nodes | 639 |
| Internal / exit nodes in its own subgraph | 639 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00375 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(2.16 x_{a3:2\_c} + 3.14 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Always-on a3 offset: 639 = ReLU(2.16 z(a3:2) + 3.14) takes exactly two values, 1.04 (a3=1) and 5.36 (a3=2), never clipped (train n=345: min 1.04, max 5.36). Through its -0.721 readout it adds -0.75 (a3=1) or -3.86 (a3=2) to the logit: a linear a3 term of -3.11 for a3=2.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 5.356975056866642,
   "min": 1.0393767214987983,
   "std": 2.158064479624637,
   "mean": 3.1418593891561826,
   "count": 345,
   "median": 1.0393767214987983
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "639"
  ],
  "entry_names": [
   "-8_c"
  ],
  "dimensionality": [
   1,
   1
  ]
 }
}
```

</details>

#### A.2.4 Leaf: `a3_gate_435`

|  |  |
|---|---|
| Annotation id | `ann_27` |
| Entry nodes | -3 (a1:3), -8\_b (a3:2\_b), id\_1083\_435 |
| Exit nodes | 435 |
| Internal / exit nodes in its own subgraph | 435 |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.00367 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.317 x_{a1:3} \\
&\quad + 6.72 x_{a3:2\_b} \\
&\quad + 1.0 x_{id\_1083\_435} \\
&\quad -3.36 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> a3=2 gate with a1=3 modulation: 435 = ReLU(0.317 z(a1:3) + 6.72 z(a3:2) + 0.284\*h1083 - 3.36), h1083 = 0. Analytically: a3=1 gives pre-activation -10.12 / -9.44 (off); a3=2 gives 3.33 (a1 in {1,2}) or 4.01 (a1=3). Train (n=345): output in {0, 3.33, 4.01}, mean 1.71, median 0, two regimes (regime\_map), on for 48.7% of rows (= the a3=2 rows). This is the network's only active nonlinearity; it acts as an a3 indicator (a3=2 -\> on) plus the network's only interaction (a1=3 & a3=2 adds 0.685 to the gate, i.e. +7.33 to the logit after the gate's total gain of 10.70).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "regime_map": {
  "viz_type": "regime_map",
  "exit_names": [
   "435"
  ],
  "entry_names": [
   "-3",
   "-8_b",
   "id_1083_435"
  ],
  "num_regimes": 2,
  "total_samples": 345,
  "dimensionality": [
   3,
   1
  ]
 },
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 4.0140991484178405,
   "min": 0.0,
   "std": 1.772657905676689,
   "mean": 1.714287216132105,
   "count": 345,
   "median": 0.0
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "435"
  ],
  "entry_names": [
   "-3",
   "-8_b",
   "id_1083_435"
  ],
  "dimensionality": [
   3,
   1
  ]
 }
}
```

</details>

#### A.2.5 Leaf: `linear_relay_821_1957`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | -8\_d (a3:2\_d), -17\_b (a6:2\_b), id\_into\_821 |
| Exit nodes | 821, 1957 |
| Internal / exit nodes in its own subgraph | 821, 1957 |
| Dimensionality (in → out, incl. descendants) | 3 → 2 |
| Live closed-form error (3 s.f. display) | 0.0493 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.893 x_{a3:2\_d} \\
&\quad + 0.627 x_{a6:2\_b} \\
&\quad + 1.0 x_{id\_into\_821} \\
&\quad + 5.77 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 1.63 \operatorname{relu}{\left(- 0.893 x_{a3:2\_d} + 0.627 x_{a6:2\_b} + 1.0 x_{id\_into\_821} + 5.77 \right)} \\
&\quad + 1.71 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear relay: 821 = ReLU(5.77 - 0.893 z(a3:2) + 0.627 z(a6:2) + id), id = 2.542\*h435 - 1.324\*h1083 \>= 0; 1957 = ReLU(1.63\*h821 + 1.71). Over all 432 inputs 821 pre-activation ranges 6.03..15.70 and 1957 11.55..27.35: neither ReLU ever clips, so the pair is exactly linear on the domain. Train (n=345) output distribution: 821 in \[6.03, 15.70\]. Combined with the readouts (-0.673 on 821, 2.148 on 1957) the relay multiplies its input by 2.837 on the logit: gate x 7.21, a6=2 -\> +3.56, a3=2 -\> -5.07 relative to a3=1.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula.
- `empirical_viz`: Train-split output distribution of the annotation exit(s).

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
 "n": 345,
 "split": "train",
 "regime_map": {
  "viz_type": "regime_map",
  "exit_names": [
   "821",
   "1957"
  ],
  "entry_names": [
   "-8_d",
   "-17_b",
   "id_into_821"
  ],
  "num_regimes": 2,
  "total_samples": 345,
  "dimensionality": [
   3,
   2
  ]
 },
 "max_samples": 100000,
 "sample_fraction": 1.0,
 "output_distribution": {
  "stats": {
   "max": 15.698948465946266,
   "min": 6.026640371633421,
   "std": 3.6785072747940593,
   "mean": 10.125528368555008,
   "count": 345,
   "median": 7.281463663775586
  },
  "viz_type": "output_distribution",
  "exit_names": [
   "821",
   "1957"
  ],
  "entry_names": [
   "-8_d",
   "-17_b",
   "id_into_821"
  ],
  "dimensionality": [
   3,
   2
  ]
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 4 |
| annotate | 8 |
| rename\_node | 17 |
| split\_node | 3 |

3 splits, 4 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#17` split -8 into -8\_a, -8\_b, -8\_c, -8\_d. Final copies: -8\_a → linear\_count\_down (entry) \| -8\_b → a3\_gate\_435 (entry); a3\_pathway (entry) \| -8\_c → a3\_offset\_639 (entry); a3\_pathway (entry) \| -8\_d → linear\_relay\_821\_1957 (entry); a3\_pathway (entry)
- `#18` split -17 into -17\_a, -17\_b, -17\_c. Final copies: -17\_a → linear\_count\_down (entry) \| -17\_b → linear\_relay\_821\_1957 (entry); a3\_pathway (entry) \| -17\_c → dead\_a3a6\_unit\_1083 (entry); a3\_pathway (entry)
- `#19` split -7 into -7\_a, -7\_b. Final copies: -7\_a → dead\_a3a6\_unit\_1083 (entry); a3\_pathway (entry) \| -7\_b → dead\_a3\_unit\_2445 (entry); a3\_pathway (entry)
- `#20` identity node id\_1083\_435 intercepts 1 connection(s) into 435 (from 1083); sits in a3\_gate\_435 (entry)
- `#21` identity node id\_into\_821 intercepts 2 connection(s) into 821 (from 435, 1083); sits in linear\_relay\_821\_1957 (entry)
- `#22` identity node id\_linear intercepts 9 connection(s) into 0 (from -1 (a1:1), -4 (a2:1), -5 (a2:2), -6 (a2:3), -8\_a (a3:2\_a), -9 (a4:1), -12 (a5:1), -13 (a5:2), … (1 more)); sits in linear\_count\_down (exit); count\_at\_most\_three (entry)
- `#23` identity node id\_hidden intercepts 5 connection(s) into 0 (from 435, 639, 821, 1957, 2445); sits in a3\_pathway (exit, composition); count\_at\_most\_three (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 2.44e-15 | 2.44e-15 | 2.44e-15 | agree |
| Closed-form max abs error (all leaves) | 0.0493 | 0.04926 | 0.04926 | agree |
| Numeric-function max abs error | n/a | 7.11e-15 | 7.11e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | count\_at\_most\_three | count\_at\_most\_three | agree |
| Leaf / composition counts | 6 / 2 | n/a | 6 / 2 | agree |
| Operation counts | 3 splits, 4 identity | {"add\_identity\_node": 4, "annotate": 8, "rename\_node": 17, "split\_node": 3} | {"add\_identity\_node": 4, "annotate": 8, "rename\_node": 17, "split\_node": 3} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `linear_relay_821_1957` | 0.04926 | 0 |
| `linear_count_down` | 0.02714 | 7.11e-15 |
| `a3_offset_639` | 0.003745 | 0 |
| `a3_gate_435` | 0.003671 | 0 |
| `dead_a3a6_unit_1083` | 0 | 0 |
| `dead_a3_unit_2445` | 0 | 0 |

## What resisted (from the record)

> Two hidden units (1083, 2445) dead on the whole domain, annotated as inert. Closed-form error 0.049 is 3-s.f. rounding amplified through the 821-\>1957 relay.

**Tool notes (record):** add\_identity\_node docstring params wrong. activation\_profile needs node\_id. Composition first built without cross-leaf edges and redone (agent error). Calls \~40-50 s under CPU load.

## What to look at

- Largest closed-form error: `linear_relay_821_1957` (0.0493), then `linear_count_down` (0.0271). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `a3_offset_639`: *“Always-on a3 offset: 639 = ReLU(2.16 z(a3:2) + 3.14) takes exactly two values, 1.04 (a3=1) and 5.36 (a3=2), never clipped (train n=345: min 1.04, max 5.36).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `a3_gate_435`: *“it acts as an a3 indicator (a3=2 -\> on) plus the network's only interaction (a1=3 & a3=2 adds 0.685 to the gate, i.e.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 3 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/monk2-s2.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('monk2-s2', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
