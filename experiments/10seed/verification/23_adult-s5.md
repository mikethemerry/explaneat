# Verification packet 23 · adult-s5

Pre-registered verification order: 23 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/7a8237f3-74cb-40bd-8268-b605f64dac2b?experiment=3aca0bd1-85cc-4a28-9ee7-951912e48476&name=10seed-adult-s5), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 5 |
| Genome id | `7a8237f3-74cb-40bd-8268-b605f64dac2b` |
| Experiment | 10seed-adult-s5 (3aca0bd1-85cc-4a28-9ee7-951912e48476); best at generation 99 of 100, NEAT key 23140 |
| Test AUC (runs csv) | 0.907 \[95% CI 0.901, 0.914\] |
| Train AUC | 0.912 |
| Size | 66 of 107 inputs connected; 15 hidden nodes; 95 enabled connections; depth 4; 17 skip connections + 55 direct input→output edges |
| Annotations | 9 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 15 of 15 |
| Claude's time | 16.9 min (2026-11-05T12:22:16+13:00 → 2026-11-05T12:39:12+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> GAM: P(income\<=50K) = sigmoid(3.385 + additive terms): sparse linear backbone (ablation AUC 0.912 -\> 0.790); concave capital-gain penalty (steepening \~\$4.8k and \$5.2k, with country/education switches; ablation AUC 0.894); a V-shaped age term (vertex at 46); a short-hours hinge; small category offsets. Train acc 0.858, AUC 0.912.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit`

|  |  |
|---|---|
| Annotation id | `ann_131` |
| Entry nodes | 2451, 5559, identity\_age, identity\_cg, identity\_core, identity\_country, identity\_lin\_gates, identity\_rare, -106 (native-country:8), -103 (native-country:5), -101 (native-country:40), -96 (native-country:36), -93 (native-country:33), -87 (native-country:28), -85 (native-country:26), -84 (native-country:25), -83 (native-country:24), -81 (native-country:22), -77 (native-country:19), -75 (native-country:17), -74 (native-country:16), -71 (native-country:13), -70 (native-country:12), -68 (native-country:10), -66 (native-country:0), -61 (race:4), -60 (race:3), -59 (race:2), -52 (relationship:1), -51 (relationship:0), -50 (occupation:9), -44 (occupation:3), -43 (occupation:2), -42 (occupation:14), -37 (occupation:1), -34 (marital-status:5), -33 (marital-status:4), -32 (marital-status:3), -27 (education:9), -26 (education:8), … (9 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | age\_profile, capital\_gain\_effect, self\_emp\_not\_inc\_offset, short\_hours\_hinge, always\_on\_linear\_gates, rare\_category\_indicators, country\_linear\_offsets, core\_linear\_backbone |
| Dimensionality (in → out, incl. descendants) | 49 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.0339 e^{- 1.0 \operatorname{age\_profile}{\left(-1,-80,-97 \right)} - 1.0 \operatorname{always\_on\_linear\_gates}{\left(-53,-19_{a} \right)} - 1.0 \operatorname{capital\_gain\_effect}{\left(identity_{cg main},262 \right)} - 1.0 \operatorname{core\_linear\_backbone}{\left(-10,-11,-12,-13,-18,-19_{b},-23,-24_{b},-25,-28,-29,-3,-30,-31,-35,-36,-38,-39_{b},-40,-41,-45,-46,-47,-48,-49,-5,-54,-55,-56,-57,-58,-62,-64,-7,-9_{b} \right)} - 1.0 \operatorname{country\_linear\_offsets}{\left(-100,-104,-107,-67,-69,-72,-76,-78,-79,-88,-89,-91,-92,-94_{b},-95,-99 \right)} - 1.0 \operatorname{rare\_category\_indicators}{\left(-9_{a},-102,-39_{a},-98,-17 \right)} - 0.433 \operatorname{self\_emp\_not\_inc\_offset}{\left(-8 \right)} - 0.814 \operatorname{short\_hours\_hinge}{\left(-65 \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{e^{0.814 x_{2451} + 0.433 x_{5559} + 1.0 x_{identity\_age} + 1.0 x_{identity\_cg} + 1.0 x_{identity\_core} + 1.0 x_{identity\_country} + 1.0 x_{identity\_lin\_gates} + 1.0 x_{identity\_rare}}}{e^{0.814 x_{2451} + 0.433 x_{5559} + 1.0 x_{identity\_age} + 1.0 x_{identity\_cg} + 1.0 x_{identity\_core} + 1.0 x_{identity\_country} + 1.0 x_{identity\_lin\_gates} + 1.0 x_{identity\_rare}} + 0.0339}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root. P(income\<=50K) = sigmoid(3.385 + core\_linear\_backbone + country\_linear\_offsets + capital\_gain\_effect + age\_profile + 0.814\*short\_hours\_hinge + 0.433\*self\_emp\_not\_inc\_offset + always\_on\_linear\_gates + rare\_category\_indicators). The logit is purely additive (GAM-like): no child interacts with another; all nonlinearity is univariate relu hinges in age, capital gain and hours, plus category indicators. 41 one-hot inputs have no edges and zero effect (they are listed as entries, sensitivity 0). EVIDENCE: Function preservation 5.6e-16; local reconstruction 3.3853 + sum of child contributions matches the pre-sigmoid logit to 3e-14 on all 39,073 train rows. Sensitivity of 41 unconnected inputs = 0. Whole-model KernelSHAP on 200 train rows (base 0.753): top features marital-status:2 Married-civ-spouse 0.139, education-num 0.075, age 0.047, capital-gain 0.040, hours 0.030, occupation:4 Exec-managerial 0.018, sex 0.016, capital-loss 0.014. Train acc 0.858, AUC 0.912.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "sensitivity",
 "note": "Function preservation 5.6e-16; local reconstruction 3.3853 + sum of child contributions matches the pre-sigmoid logit to 3e-14 on all 39,073 train rows. Sensitivity of 41 unconnected inputs = 0. Whole-model KernelSHAP on 200 train rows (base 0.753): top features marital-status:2 Married-civ-spouse 0.139, education-num 0.075, age 0.047, capital-gain 0.040, hours 0.030, occupation:4 Exec-managerial 0.018, sex 0.016, capital-loss 0.014. Train acc 0.858, AUC 0.912.",
 "shap": {
  "top": [
   [
    "marital-status:2",
    0.1387
   ],
   [
    "education-num",
    0.0754
   ],
   [
    "age",
    0.0467
   ],
   [
    "capital-gain",
    0.04
   ],
   [
    "hours-per-week",
    0.0302
   ],
   [
    "occupation:4",
    0.0177
   ],
   [
    "sex",
    0.0161
   ],
   [
    "capital-loss",
    0.0141
   ],
   [
    "occupation:10",
    0.0085
   ],
   [
    "relationship:5",
    0.0083
   ],
   [
    "occupation:8",
    0.0081
   ],
   [
    "workclass:6",
    0.0078
   ],
   [
    "relationship:3",
    0.0073
   ],
   [
    "marital-status:0",
    0.0066
   ],
   [
    "fnlwgt",
    0.0063
   ]
  ],
  "split": "train",
  "n_samples": 200
 },
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "0"
  ],
  "entry_names": [
   "2451",
   "5559",
   "identity_age",
   "identity_cg",
   "identity_core",
   "identity_country",
   "identity_lin_gates",
   "identity_rare",
   "-106",
   "-103",
   "-101",
   "-96",
   "-93",
   "-87",
   "-85",
   "-84",
   "-83",
   "-81",
   "-77",
   "-75",
   "-74",
   "-71",
   "-70",
   "-68",
   "-66",
   "-61",
   "-60",
   "-59",
   "-52",
   "-51",
   "-50",
   "-44",
   "-43",
   "-42",
   "-37",
   "-34",
   "-33",
   "-32",
   "-27",
   "-26",
   "-22",
   "-21",
   "-20",
   "-16",
   "-15",
   "-14",
   "-6",
   "-4",
   "-2"
  ],
  "input_labels": [
   "2451",
   "5559",
   "identity_age",
   "identity_cg",
   "identity_core",
   "identity_country",
   "identity_lin_gates",
   "identity_rare",
   "-106",
   "-103",
   "-101",
   "-96",
   "-93",
   "-87",
   "-85",
   "-84",
   "-83",
   "-81",
   "-77",
   "-75",
   "-74",
   "-71",
   "-70",
   "-68",
   "-66",
   "-61",
   "-60",
   "-59",
   "-52",
   "-51",
   "-50",
   "-44",
   "-43",
   "-42",
   "-37",
   "-34",
   "-33",
   "-32",
   "-27",
   "-26",
   "-22",
   "-21",
   "-20",
   "-16",
   "-15",
   "-14",
   "-6",
   "-4",
   "-2"
  ],
  "sensitivities": {
   "0": [
    0.071545
… (truncated; full payload in the DB evidence record)
```

</details>

### A.1 Leaf: `age_profile`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -1 (age), -80 (native-country:21), -97 (native-country:37) |
| Exit nodes | identity\_age |
| Internal / exit nodes in its own subgraph | 2134, 2990, 5070, identity\_age |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.0173 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 1.03 x_{age} \\
&\quad - 0.0464 x_{native-country:21} \\
&\quad + 0.0192 x_{native-country:37} \\
&\quad + 0.975 \operatorname{relu}{\left(1.03 \operatorname{relu}{\left(1.45 x_{age} + 0.137 \operatorname{relu}{\left(- 0.673 x_{native-country:21} + 0.854 x_{native-country:37} - 0.705 \right)} - 0.481 \right)} - 0.317 \right)} \\
&\quad - 0.124 \operatorname{relu}{\left(- 0.673 x_{native-country:21} + 0.854 x_{native-country:37} - 0.705 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Piecewise-linear age term in the logit of P(income\<=50K), with a Thailand override. Direct age edge gives -1.03 per SD (-0.075/yr). 2990=relu(1.446 age-0.48) and 5070=relu(1.031\*2990-0.32) form a stacked hinge that switches on at age \~46 (z=0.545) and adds +1.45/SD, so the net slope above 46 is +0.42/SD (+0.031/yr): the probability of earning \>50K rises with age to a peak at \~46 and then declines. 2134 is a Thailand indicator (native-country:37); for Thai respondents (27 rows) it adds -4.0 directly and pushes 2990 permanently on (removing the hinge). Ireland (native-country:21) only adds a direct -1.57 and keeps 2134 off. EVIDENCE: Train evidence (39,073 rows): mean contribution by age (non-Thai) +1.63 at 17, +0.65 at 30, -0.10 at 40, minimum -0.55 at exactly 46, -0.29 at 55, +0.17 at 70, +0.79 at 90, which confirms a V shape with its vertex at 46. Thailand (26 rows): the -4.0 direct term and the released hinge almost cancel, so being Thai changes the logit by only +0.10 on average. Mean-ablation of the leaf: acc 0.8578-\>0.8534, AUC 0.9118-\>0.9051.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "line",
 "note": "Train evidence (39,073 rows): mean contribution by age (non-Thai) +1.63 at 17, +0.65 at 30, -0.10 at 40, minimum -0.55 at exactly 46, -0.29 at 55, +0.17 at 70, +0.79 at 90, which confirms a V shape with its vertex at 46. Thailand (26 rows): the -4.0 direct term and the released hinge almost cancel, so being Thai changes the logit by only +0.10 on average. Mean-ablation of the leaf: acc 0.8578->0.8534, AUC 0.9118->0.9051.",
 "split": "train",
 "summary": {
  "y_mean": 0.37761045519715,
  "x_range": [
   -1.8403933369383092,
   4.00486082593868
  ],
  "y_range": [
   -0.5553889089247198,
   1.9008095931953832
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_age"
  ],
  "correlation": -0.7808748273651204,
  "entry_names": [
   "-1",
   "-80",
   "-97"
  ],
  "dimensionality": [
   3,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -1.574699965898446,
   3.739167454898817
  ],
  "scatter_y_range": [
   -2.1256953608601568,
   1.6265203104109272
  ]
 },
 "mean_ablation": {
  "acc": 0.8534,
  "auc": 0.9051,
  "contrib_sd": 0.64,
  "contrib_mean": 0.303
 }
}
```

</details>

### A.2 Composition: `capital_gain_effect`

|  |  |
|---|---|
| Annotation id | `ann_130` |
| Entry nodes | identity\_cg\_main, 262 |
| Exit nodes | identity\_cg |
| Internal / exit nodes in its own subgraph | identity\_cg |
| Children | capital\_gain\_hinge, capital\_gain\_country\_gate |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 1.84 \operatorname{capital\_gain\_country\_gate}{\left(-63_{a},-24_{a},-82,-86,-90,-94_{a} \right)} \\
&\quad + 1.0 \operatorname{capital\_gain\_hinge}{\left(-63_{bc},-105,-73 \right)}
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = - 1.84 x_{262} + 1.0 x_{identity\_cg\_main}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Additive composition: identity\_cg = capital\_gain\_hinge - 1.842\*capital\_gain\_country\_gate. Together a concave piecewise-linear penalty on the logit of P(\<=50K): -0.337 per \$1000 everywhere, steepening to -0.485 per \$1000 above \~\$4.8k and -0.644 per \$1000 above \~\$5.2k, plus a constant -2.44. Ecuador removes the \$5.2k kink; 9th grade, Jamaica, Nicaragua, Philippines and Scotland remove the \$4.8k kink. EVIDENCE: Train evidence: mean contribution by gain band: \$1-3k -2.74, \$3-4.5k -3.29, \$4.5-5k -3.66, \$5-5.5k -3.85, \$5.5-8k -5.28, \$8-15k -7.76, \$15-30k -11.4, \>\$30k -63 (saturated). Observed P(\<=50K) falls from 0.71 to 0.10 across \$4.5k-8k, where the kinks sit. Ablating the whole composition: AUC 0.912-\>0.894, acc -\>0.837.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "sensitivity",
 "note": "Train evidence: mean contribution by gain band: $1-3k -2.74, $3-4.5k -3.29, $4.5-5k -3.66, $5-5.5k -3.85, $5.5-8k -5.28, $8-15k -7.76, $15-30k -11.4, >$30k -63 (saturated). Observed P(<=50K) falls from 0.71 to 0.10 across $4.5k-8k, where the kinks sit. Ablating the whole composition: AUC 0.912->0.894, acc ->0.837.",
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_cg"
  ],
  "entry_names": [
   "identity_cg_main",
   "262"
  ],
  "input_labels": [
   "identity_cg_main",
   "262"
  ],
  "sensitivities": {
   "identity_cg": [
    0.9999999999999811,
    1.8424460716995172
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "mean_ablation": {
  "acc": 0.8372,
  "auc": 0.8935
 }
}
```

</details>

#### A.2.1 Leaf: `capital_gain_hinge`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -63\_bc (capital-gain\_bc), -105 (native-country:7), -73 (native-country:15) |
| Exit nodes | identity\_cg\_main |
| Internal / exit nodes in its own subgraph | 1485, 4262, identity\_cg\_main |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |
| Live closed-form error (3 s.f. display) | 0.0408 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 2.52 x_{capital-gain\_bc} \\
&\quad - 2.73 \operatorname{relu}{\left(0.245 x_{capital-gain\_bc} - 2.17 x_{native-country:7} - 0.2 \right)} \\
&\quad - 2.79 \operatorname{relu}{\left(- 1.38 x_{native-country:15} + 0.77 \operatorname{relu}{\left(0.245 x_{capital-gain\_bc} - 2.17 x_{native-country:7} - 0.2 \right)} + 0.875 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Main capital-gain penalty (more gain -\> less likely \<=50K). Linear -2.524/SD (-0.337 per \$1000, SD\~\$7.5k) everywhere, plus hinge 1485=relu(0.245 cg-0.2-2.168 Ecuador) that switches on above \~\$5.2k and adds -1.19/SD (-0.159 per \$1000) via its direct edge (-2.728) and via 4262 (0.77\*-2.786). 4262=relu(0.875+0.77\*1485-1.379 Holand-Netherlands) is always on (that country never occurs in train), so it is linear and contributes a constant -2.44. Ecuador (native-country:7) disables the hinge. EVIDENCE: Train evidence: the line plot is almost linear (corr -0.999) from -2.07 (no gain) to -50.8 (max gain). Kink at \$5.16k computed analytically. Mean-ablation: AUC 0.9118-\>0.9028, acc -\>0.8463 (the largest single leaf after the backbone).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "line",
 "note": "Train evidence: the line plot is almost linear (corr -0.999) from -2.07 (no gain) to -50.8 (max gain). Kink at $5.16k computed analytically. Mean-ablation: AUC 0.9118->0.9028, acc ->0.8463 (the largest single leaf after the backbone).",
 "split": "train",
 "summary": {
  "y_mean": -26.115879694909662,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   -53.30633061332649,
   -0.38714923423708236
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "identity_cg_main"
  ],
  "correlation": -0.9989218853531618,
  "entry_names": [
   "-63_bc",
   "-105",
   "-73"
  ],
  "dimensionality": [
   3,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.14552350385541227,
   13.187777170087646
  ],
  "scatter_y_range": [
   -50.82733617020367,
   -2.0696337551565733
  ]
 },
 "mean_ablation": {
  "acc": 0.8463,
  "auc": 0.9028,
  "contrib_sd": 3.615,
  "contrib_mean": -2.553
 }
}
```

</details>

#### A.2.2 Leaf: `capital_gain_country_gate`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -63\_a (capital-gain\_a), -24\_a (education:6\_a), -82 (native-country:23), -86 (native-country:27), -90 (native-country:30), -94\_a (native-country:34\_a) |
| Exit nodes | 262 |
| Internal / exit nodes in its own subgraph | 2391, 262 |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |
| Live closed-form error (3 s.f. display) | 8.45e-04 |
| Live numeric-function error | 2.22e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.603 x_{capital-gain\_a} \\
&\quad - 0.984 x_{education:6\_a} \\
&\quad - 1.27 x_{native-country:23} \\
&\quad - 0.254 x_{native-country:30} \\
&\quad - 1.28 \operatorname{relu}{\left(1.01 x_{native-country:23} + 0.347 x_{native-country:27} + 0.964 x_{native-country:30} + 1.02 x_{native-country:34\_a} - 0.913 \right)} \\
&\quad -0.5 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Second capital-gain hinge 262=relu(0.603 cg-0.5-0.984 9th-grade-1.274 Jamaica-0.254 Philippines-1.285\*2391), weight -1.842 into the output: above \~\$4.8k gain it adds a further -1.11/SD (-0.148 per \$1000). It is switched off for education=9th, and for Jamaica, Nicaragua, Philippines and Scotland via 2391 (a relu OR-detector of those four countries, native-country 23/27/30/34). So those groups get a shallower capital-gain slope. EVIDENCE: Train evidence: 262 is active for 5.1% of rows (gains above \~\$4.8k), exit range 0..7.65 (contribution down to -14.1). Mean-ablation: AUC 0.9118-\>0.9114 (small, because it is collinear with the main capital-gain terms).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "line",
 "note": "Train evidence: 262 is active for 5.1% of rows (gains above ~$4.8k), exit range 0..7.65 (contribution down to -14.1). Mean-ablation: AUC 0.9118->0.9114 (small, because it is collinear with the main capital-gain terms).",
 "split": "train",
 "summary": {
  "y_mean": 3.67129904302894,
  "x_range": [
   -0.8121885375525651,
   13.8544422037848
  ],
  "y_range": [
   0.0,
   8.056544979358186
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "262"
  ],
  "correlation": 0.9798699588386481,
  "entry_names": [
   "-63_a",
   "-24_a",
   "-82",
   "-86",
   "-90",
   "-94_a"
  ],
  "dimensionality": [
   6,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.14552350385541227,
   13.187777170087646
  ],
  "scatter_y_range": [
   0.0,
   7.654531953296284
  ]
 },
 "mean_ablation": {
  "acc": 0.8572,
  "auc": 0.9114,
  "contrib_sd": 1.018,
  "contrib_mean": -0.109
 }
}
```

</details>

### A.3 Leaf: `self_emp_not_inc_offset`

|  |  |
|---|---|
| Annotation id | `ann_124` |
| Entry nodes | -8 (workclass:6) |
| Exit nodes | 5559 |
| Internal / exit nodes in its own subgraph | 8292, 5559 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00119 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.94 \operatorname{relu}{\left(0.827 x_{workclass:6} - 1.03 \right)} \\
&\quad -0.222 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Serial relu chain 8292-\>5559 that is an indicator of workclass=Self-emp-not-inc (workclass:6): 8292 = 1.822 when self-employed-not-inc, else 0; 5559=relu(0.94\*8292-0.22). Adds +0.646 to the logit of P(\<=50K) for that group (7.8% of rows), 0 otherwise. EVIDENCE: Train evidence: exit is 1.49 for all 7.8% Self-emp-not-inc rows and 0 otherwise (line corr 1.0); contribution +0.646. Mean-ablation: acc -\>0.8557, AUC -\>0.9111.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "line",
 "note": "Train evidence: exit is 1.49 for all 7.8% Self-emp-not-inc rows and 0 otherwise (line corr 1.0); contribution +0.646. Mean-ablation: acc ->0.8557, AUC ->0.9111.",
 "split": "train",
 "summary": {
  "y_mean": 0.42065166324575537,
  "x_range": [
   -0.47698180846301114,
   3.6333349183086665
  ],
  "y_range": [
   0.0,
   1.6354238785290374
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "5559"
  ],
  "correlation": 1.0,
  "entry_names": [
   "-8"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -0.29014922997338943,
   3.4465023398190446
  ],
  "scatter_y_range": [
   0.0,
   1.4902279087332224
  ]
 },
 "mean_ablation": {
  "acc": 0.8557,
  "auc": 0.9111,
  "contrib_sd": 0.173,
  "contrib_mean": 0.05
 }
}
```

</details>

### A.4 Leaf: `short_hours_hinge`

|  |  |
|---|---|
| Annotation id | `ann_125` |
| Entry nodes | -65 (hours-per-week) |
| Exit nodes | 2451 |
| Internal / exit nodes in its own subgraph | 2451 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 4.84e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.598 - 0.777 x_{hours-per-week} \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hours-per-week hinge 2451=relu(0.598-0.777 hours\_z), weight +0.814: below \~50 h/week each fewer hour adds +0.051 to the logit of P(\<=50K) (part-time workers are more likely \<=50K); flat (0) above \~50 h/week. Hours has no other path. EVIDENCE: Train evidence: mean contribution by hours: 10h +2.04, 20h +1.53, 30h +1.02, 40h +0.51, 45h +0.25, \>=50h 0. Active for 80% of rows. Mean-ablation: acc -\>0.8556, AUC -\>0.9076.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "line",
 "note": "Train evidence: mean contribution by hours: 10h +2.04, 20h +1.53, 30h +1.02, 40h +0.51, 45h +0.25, >=50h 0. Active for 80% of rows. Mean-ablation: acc ->0.8556, AUC ->0.9076.",
 "split": "train",
 "summary": {
  "y_mean": 0.8469234581858079,
  "x_range": [
   -3.5745149488954633,
   5.123585616693855
  ],
  "y_range": [
   0.0,
   3.374903156230918
  ],
  "viz_type": "line",
  "x_points": 200,
  "exit_names": [
   "2451"
  ],
  "correlation": -0.9143586316932566,
  "entry_names": [
   "-65"
  ],
  "dimensionality": [
   1,
   1
  ],
  "scatter_points": 39073,
  "scatter_x_range": [
   -3.179146741368676,
   4.728217409167068
  ],
  "scatter_y_range": [
   0.0,
   3.0677128975238466
  ]
 },
 "mean_ablation": {
  "acc": 0.8556,
  "auc": 0.9076,
  "contrib_sd": 0.494,
  "contrib_mean": 0.56
 }
}
```

</details>

### A.5 Leaf: `always_on_linear_gates`

|  |  |
|---|---|
| Annotation id | `ann_126` |
| Entry nodes | -53 (relationship:2), -19\_a (education:15\_a) |
| Exit nodes | identity\_lin\_gates |
| Internal / exit nodes in its own subgraph | 3843, 7420, identity\_lin\_gates |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00211 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.037 \operatorname{relu}{\left(1.44 - 0.188 x_{education:15\_a} \right)} \\
&\quad + 0.306 \operatorname{relu}{\left(0.533 x_{relationship:2} + 1.74 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Two relu gates that never switch off on train, so they are exactly linear: 3843=1.738+0.533 Other-relative (relationship:2) -\> +0.95 logit for Other-relative; 7420=1.439-0.188 Some-college (education:15) -\> +0.017 for Some-college. Constant part about +0.48. EVIDENCE: Train evidence: both relus are active for 100% of rows (min pre-activations 1.64 and 1.09), so the leaf is exactly linear; sensitivity 0.163 (Other-relative) vs 0.007 (Some-college). Mean-ablation: no measurable change (AUC 0.9117).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "sensitivity",
 "note": "Train evidence: both relus are active for 100% of rows (min pre-activations 1.64 and 1.09), so the leaf is exactly linear; sensitivity 0.163 (Other-relative) vs 0.007 (Some-college). Mean-ablation: no measurable change (AUC 0.9117).",
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_lin_gates"
  ],
  "entry_names": [
   "-53",
   "-19_a"
  ],
  "input_labels": [
   "-53",
   "-19_a"
  ],
  "sensitivities": {
   "identity_lin_gates": [
    0.16346714619238678,
    0.006974106554382004
   ]
  },
  "dimensionality": [
   2,
   1
  ]
 },
 "mean_ablation": {
  "acc": 0.8576,
  "auc": 0.9117,
  "contrib_sd": 0.164,
  "contrib_mean": 0.479
 }
}
```

</details>

### A.6 Leaf: `rare_category_indicators`

|  |  |
|---|---|
| Annotation id | `ann_127` |
| Entry nodes | -9\_a (workclass:7\_a), -102 (native-country:41), -39\_a (occupation:11\_a), -98 (native-country:38), -17 (education:13) |
| Exit nodes | identity\_rare |
| Internal / exit nodes in its own subgraph | 1624, 3389, 16072, identity\_rare |
| Dimensionality (in → out, incl. descendants) | 5 → 1 |
| Live closed-form error (3 s.f. display) | 0.00207 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.0604 \operatorname{relu}{\left(0.999 x_{education:13} - 0.908 \right)} \\
&\quad + 0.0467 \operatorname{relu}{\left(0.998 x_{native-country:38} - 0.0079 x_{occupation:11\_a} - 2.13 \right)} \\
&\quad - 0.0353 \operatorname{relu}{\left(0.812 x_{native-country:41} + 0.0583 x_{workclass:7\_a} - 0.127 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Three relu indicators of rare categories (zero for everyone else): 1624 = Yugoslavia (native-country:41) -\> -1.22 (State-gov adds only -0.005); 3389 = Trinadad&Tobago (native-country:38) -\> +1.87 (Protective-serv only modulates within it); 16072 = Preschool education (education:13) -\> +1.43. Affects \~0.3% of rows. EVIDENCE: Train evidence: 1624 is \>0.2 only for Yugoslavia (State-gov gives 0.155, contribution -0.005); 3389 is active for 0.06% of rows (Trinadad&Tobago); 16072 for 0.16% (Preschool). Mean-ablation: no measurable change in acc/AUC (rare groups).

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "sensitivity",
 "note": "Train evidence: 1624 is >0.2 only for Yugoslavia (State-gov gives 0.155, contribution -0.005); 3389 is active for 0.06% of rows (Trinadad&Tobago); 16072 for 0.16% (Preschool). Mean-ablation: no measurable change in acc/AUC (rare groups).",
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_rare"
  ],
  "entry_names": [
   "-9_a",
   "-102",
   "-39_a",
   "-98",
   "-17"
  ],
  "input_labels": [
   "-9_a",
   "-102",
   "-39_a",
   "-98",
   "-17"
  ],
  "sensitivities": {
   "identity_rare": [
    8.517121315593588e-05,
    0.0011855671768540848,
    2.0784826818485815e-07,
    2.6268469500143678e-05,
    9.894392256285392e-05
   ]
  },
  "dimensionality": [
   5,
   1
  ]
 },
 "mean_ablation": {
  "acc": 0.8577,
  "auc": 0.9117,
  "contrib_sd": 0.078,
  "contrib_mean": 0.003
 }
}
```

</details>

### A.7 Leaf: `country_linear_offsets`

|  |  |
|---|---|
| Annotation id | `ann_128` |
| Entry nodes | -100 (native-country:4), -104 (native-country:6), -107 (native-country:9), -67 (native-country:1), -69 (native-country:11), -72 (native-country:14), -76 (native-country:18), -78 (native-country:2), -79 (native-country:20), -88 (native-country:29), -89 (native-country:3), -91 (native-country:31), -92 (native-country:32), -94\_b (native-country:34\_b), -95 (native-country:35), -99 (native-country:39) |
| Exit nodes | identity\_country |
| Internal / exit nodes in its own subgraph | identity\_country |
| Dimensionality (in → out, incl. descendants) | 16 → 1 |
| Live closed-form error (3 s.f. display) | 0.00983 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.0192 x_{native-country:11} \\
&\quad + 0.00525 x_{native-country:14} \\
&\quad - 0.00301 x_{native-country:18} \\
&\quad - 0.0146 x_{native-country:1} \\
&\quad - 0.00105 x_{native-country:20} \\
&\quad + 0.0301 x_{native-country:29} \\
&\quad - 0.0366 x_{native-country:2} \\
&\quad - 0.00654 x_{native-country:31} \\
&\quad - 0.0494 x_{native-country:32} \\
&\quad + 0.0359 x_{native-country:34\_b} \\
&\quad + 0.0561 x_{native-country:35} \\
&\quad - 0.108 x_{native-country:39} \\
&\quad + 0.0381 x_{native-country:3} \\
&\quad + 0.234 x_{native-country:4} \\
&\quad + 0.0259 x_{native-country:6} \\
&\quad - 0.0402 x_{native-country:9}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Linear per-country offsets from the direct native-country edges (standardised one-hot, so each acts as a switch). Largest: Columbia +5.8 (rare, 64 rows), Scotland +1.77, Portugal -1.40, South +1.16, England -0.83, United-States -0.36. Population SD of the term 0.29 logit. EVIDENCE: Train evidence: per-switch effects w\*(hi-lo): Columbia +5.80, Scotland +1.77, Portugal -1.40, South +1.16, Peru +0.97, England -0.83, China +0.77, US -0.36. Contribution SD 0.29. Mean-ablation: AUC -\>0.9111.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "sensitivity",
 "note": "Train evidence: per-switch effects w*(hi-lo): Columbia +5.80, Scotland +1.77, Portugal -1.40, South +1.16, Peru +0.97, England -0.83, China +0.77, US -0.36. Contribution SD 0.29. Mean-ablation: AUC ->0.9111.",
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_country"
  ],
  "entry_names": [
   "-100",
   "-104",
   "-107",
   "-67",
   "-69",
   "-72",
   "-76",
   "-78",
   "-79",
   "-88",
   "-89",
   "-91",
   "-92",
   "-94_b",
   "-95",
   "-99"
  ],
  "input_labels": [
   "-100",
   "-104",
   "-107",
   "-67",
   "-69",
   "-72",
   "-76",
   "-78",
   "-79",
   "-88",
   "-89",
   "-91",
   "-92",
   "-94_b",
   "-95",
   "-99"
  ],
  "sensitivities": {
   "identity_country": [
    0.23440515901726822,
    0.025868214908049934,
    0.04021473668807674,
    0.014560268759111192,
    0.019222793527159707,
    0.005247384940729923,
    0.0030129586647128885,
    0.03657511885750696,
    0.0010465199933609411,
    0.03012976558100132,
    0.03806367738947334,
    0.006540547117095558,
    0.04942813863590975,
    0.03587529731188617,
    0.05608577818962823,
    0.10794078006427332
   ]
  },
  "dimensionality": [
   16,
   1
  ]
 },
 "mean_ablation": {
  "acc": 0.8565,
  "auc": 0.9111,
  "contrib_sd": 0.293,
  "contrib_mean": 0.0
 }
}
```

</details>

### A.8 Leaf: `core_linear_backbone`

|  |  |
|---|---|
| Annotation id | `ann_129` |
| Entry nodes | -10 (workclass:8), -11 (fnlwgt), -12 (education:0), -13 (education:1), -18 (education:14), -19\_b (education:15\_b), -23 (education:5), -24\_b (education:6\_b), -25 (education:7), -28 (education-num), -29 (marital-status:0), -3 (workclass:1), -30 (marital-status:1), -31 (marital-status:2), -35 (marital-status:6), -36 (occupation:0), -38 (occupation:10), -39\_b (occupation:11\_b), -40 (occupation:12), -41 (occupation:13), -45 (occupation:4), -46 (occupation:5), -47 (occupation:6), -48 (occupation:7), -49 (occupation:8), -5 (workclass:3), -54 (relationship:3), -55 (relationship:4), -56 (relationship:5), -57 (race:0), -58 (race:1), -62 (sex), -64 (capital-loss), -7 (workclass:5), -9\_b (workclass:7\_b) |
| Exit nodes | identity\_core |
| Internal / exit nodes in its own subgraph | identity\_core |
| Dimensionality (in → out, incl. descendants) | 35 → 1 |
| Live closed-form error (3 s.f. display) | 0.017 |
| Live numeric-function error | 1.42e-14 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.282 x_{capital-loss} \\
&\quad - 0.986 x_{education-num} \\
&\quad - 0.0948 x_{education:0} \\
&\quad - 0.00674 x_{education:14} \\
&\quad - 0.0578 x_{education:15\_b} \\
&\quad - 0.0318 x_{education:1} \\
&\quad - 0.155 x_{education:5} \\
&\quad - 0.0957 x_{education:6\_b} \\
&\quad + 0.0714 x_{education:7} \\
&\quad - 0.102 x_{fnlwgt} \\
&\quad - 0.142 x_{marital-status:0} \\
&\quad - 1.62 x_{marital-status:1} \\
&\quad - 1.43 x_{marital-status:2} \\
&\quad - 0.18 x_{marital-status:6} \\
&\quad + 0.0864 x_{occupation:0} \\
&\quad - 0.122 x_{occupation:10} \\
&\quad - 0.0791 x_{occupation:11\_b} \\
&\quad - 0.0643 x_{occupation:12} \\
&\quad - 0.0895 x_{occupation:13} \\
&\quad - 0.219 x_{occupation:4} \\
&\quad + 0.164 x_{occupation:5} \\
&\quad + 0.136 x_{occupation:6} \\
&\quad + 0.0913 x_{occupation:7} \\
&\quad + 0.255 x_{occupation:8} \\
&\quad + 0.0318 x_{race:0} \\
&\quad - 0.0612 x_{race:1} \\
&\quad + 0.186 x_{relationship:3} \\
&\quad - 0.0149 x_{relationship:4} \\
&\quad - 0.195 x_{relationship:5} \\
&\quad - 0.237 x_{sex} \\
&\quad - 0.0758 x_{workclass:1} \\
&\quad + 0.45 x_{workclass:3} \\
&\quad - 0.033 x_{workclass:5} \\
&\quad + 0.0735 x_{workclass:7\_b} \\
&\quad + 0.0106 x_{workclass:8}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Sparse logistic-regression backbone on the remaining direct edges (logit of P(\<=50K)). Main effects: education-num -0.98/SD (-0.38 per education year), Married-civ-spouse -2.87, male -0.50, capital-loss -0.28/SD, Exec-managerial -0.67, Other-service +0.85, Wife -0.91, Own-child +0.51, Widowed -1.05, fnlwgt -0.10/SD. Two rare one-hots act as extreme switches: Married-AF-spouse -59 (29 rows) and Never-worked +30 (9 rows). EVIDENCE: Train evidence: ablating the backbone to its mean drops acc 0.858-\>0.801 and AUC 0.912-\>0.790. Whole-model SHAP (200 train rows) ranks Married-civ-spouse (0.139), education-num (0.075), age, capital-gain, hours, Exec-managerial, sex, capital-loss highest, so the backbone holds the dominant features. Backbone+country alone (others at their mean): acc 0.833, AUC 0.878.

**Evidence (2 records, DB):**

- `analytical_formula`: (no narrative)
- `empirical_viz`: (no narrative)

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "source": "get_formula(force=True)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "sensitivity",
 "note": "Train evidence: ablating the backbone to its mean drops acc 0.858->0.801 and AUC 0.912->0.790. Whole-model SHAP (200 train rows) ranks Married-civ-spouse (0.139), education-num (0.075), age, capital-gain, hours, Exec-managerial, sex, capital-loss highest, so the backbone holds the dominant features. Backbone+country alone (others at their mean): acc 0.833, AUC 0.878.",
 "split": "train",
 "summary": {
  "viz_type": "sensitivity",
  "exit_names": [
   "identity_core"
  ],
  "entry_names": [
   "-10",
   "-11",
   "-12",
   "-13",
   "-18",
   "-19_b",
   "-23",
   "-24_b",
   "-25",
   "-28",
   "-29",
   "-3",
   "-30",
   "-31",
   "-35",
   "-36",
   "-38",
   "-39_b",
   "-40",
   "-41",
   "-45",
   "-46",
   "-47",
   "-48",
   "-49",
   "-5",
   "-54",
   "-55",
   "-56",
   "-57",
   "-58",
   "-62",
   "-64",
   "-7",
   "-9_b"
  ],
  "input_labels": [
   "-10",
   "-11",
   "-12",
   "-13",
   "-18",
   "-19_b",
   "-23",
   "-24_b",
   "-25",
   "-28",
   "-29",
   "-3",
   "-30",
   "-31",
   "-35",
   "-36",
   "-38",
   "-39_b",
   "-40",
   "-41",
   "-45",
   "-46",
   "-47",
   "-48",
   "-49",
   "-5",
   "-54",
   "-55",
   "-56",
   "-57",
   "-58",
   "-62",
   "-64",
   "-7",
   "-9_b"
  ],
  "sensitivities": {
   "identity_core": [
    0.010600963144025586,
    0.102155125874071,
    0.09481906608972439,
    0.03180410798216136,
    0.006744352512041396,
    0.057806268590086675,
    0.15491064527761167,
    0.095653924478032,
    0.07138285451399608,
    0.9862991092310913,
    0.14192187214252527,
    0.07580221478545712,
    1.6196157197053476,
    1.4286679498819632,
    0.17992927938514788,
    0.08636752131374645,
    0.1216836624891699,
    0.07911744248493426,
    0.06429848370774652,
    0.0894936622706504,
    0.2191811013011077,
    0.16372202887134418,
    0.13625770589313896,
    0.09126267808985403,
    0.2547203194008078,
    0.45025137061731385,
    0.1863167488172346,
    0.014894720024670146,
    0.1954196931438438,
    0.031769524784070656,
    0.06119550639285271,
    0.23675265109874924,
    0.28235299343647147,
    0.032995834826702305,
    0.07355267905184192
   ]
  },
  "dimensionality": [
   35,
   1
  ]
 },
 "mean_ablation": {
  "acc": 0.8007,
  "auc": 0.7898,
  "contrib_sd": 2.735,
  "contrib_mean": 0.0
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 7 |
| annotate | 11 |
| consolidate\_node | 1 |
| rename\_node | 107 |
| split\_node | 6 |

6 splits, 7 identity nodes, 1 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node identity\_age intercepts 5 connection(s) into 0 (from -1 (age), -80 (native-country:21), -97 (native-country:37), 2134, 5070); sits in age\_profile (exit); income\_logit (entry)
- `#108` identity node identity\_cg\_main intercepts 3 connection(s) into 0 (from -63, 1485, 4262); sits in capital\_gain\_hinge (exit); capital\_gain\_effect (entry)
- `#109` identity node identity\_cg intercepts 2 connection(s) into 0 (from identity\_cg\_main, 262); sits in capital\_gain\_effect (exit, composition); income\_logit (entry)
- `#110` identity node identity\_lin\_gates intercepts 2 connection(s) into 0 (from 3843, 7420); sits in always\_on\_linear\_gates (exit); income\_logit (entry)
- `#111` identity node identity\_rare intercepts 3 connection(s) into 0 (from 1624, 3389, 16072); sits in rare\_category\_indicators (exit); income\_logit (entry)
- `#112` identity node identity\_country intercepts 16 connection(s) into 0 (from -67 (native-country:1), -69 (native-country:11), -72 (native-country:14), -76 (native-country:18), -78 (native-country:2), -79 (native-country:20), -88 (native-country:29), -89 (native-country:3), … (8 more)); sits in country\_linear\_offsets (exit); income\_logit (entry)
- `#113` identity node identity\_core intercepts 35 connection(s) into 0 (from -3 (workclass:1), -5 (workclass:3), -7 (workclass:5), -9, -10 (workclass:8), -11 (fnlwgt), -12 (education:0), -13 (education:1), … (27 more)); sits in core\_linear\_backbone (exit); income\_logit (entry)
- `#114` split -63 into -63\_a, -63\_b, -63\_c. Final copies: -63\_a → capital\_gain\_country\_gate (entry) \| -63\_bc → capital\_gain\_hinge (entry) \| -63\_bc → capital\_gain\_hinge (entry)
- `#115` split -9 into -9\_a, -9\_b. Final copies: -9\_a → rare\_category\_indicators (entry) \| -9\_b → core\_linear\_backbone (entry)
- `#116` split -19 into -19\_a, -19\_b. Final copies: -19\_a → always\_on\_linear\_gates (entry) \| -19\_b → core\_linear\_backbone (entry)
- `#117` split -24 into -24\_a, -24\_b. Final copies: -24\_a → capital\_gain\_country\_gate (entry) \| -24\_b → core\_linear\_backbone (entry)
- `#118` split -39 into -39\_a, -39\_b. Final copies: -39\_a → rare\_category\_indicators (entry) \| -39\_b → core\_linear\_backbone (entry)
- `#119` split -94 into -94\_a, -94\_b. Final copies: -94\_a → capital\_gain\_country\_gate (entry) \| -94\_b → country\_linear\_offsets (entry)
- `#120` consolidate -63\_b, -63\_c into -63\_bc

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 5.55e-16 | 5.55e-16 | 5.55e-16 | agree |
| Closed-form max abs error (all leaves) | 0.04082 | 0.04082 | 0.04082 | agree |
| Numeric-function max abs error | n/a | 1.42e-14 | 1.42e-14 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit | income\_logit | agree |
| Leaf / composition counts | 9 / 2 | n/a | 9 / 2 | agree |
| Operation counts | 6 splits, 7 identity | {"add\_identity\_node": 7, "annotate": 11, "consolidate\_node": 1, "rename\_node": 107, "split\_node": 6} | {"add\_identity\_node": 7, "annotate": 11, "consolidate\_node": 1, "rename\_node": 107, "split\_node": 6} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `capital_gain_hinge` | 0.04082 | 1.78e-15 |
| `age_profile` | 0.01733 | 8.88e-16 |
| `core_linear_backbone` | 0.01698 | 1.42e-14 |
| `country_linear_offsets` | 0.00983 | 4.44e-16 |
| `always_on_linear_gates` | 0.00211 | 0 |
| `rare_category_indicators` | 0.002069 | 0 |
| `self_emp_not_inc_offset` | 0.001186 | 0 |
| `capital_gain_country_gate` | 8.45e-04 | 2.22e-16 |
| `short_hours_hinge` | 4.84e-04 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Degenerate gates: 4262 always on (Holand-Netherlands never occurs in train), 3843 and 7420 always on (linear). Thailand gate's -4.0 direct effect and age-hinge release nearly cancel (net +0.10 over 26 rows). Rare standardised one-hots act as extreme switches (Married-AF-spouse -59, Never-worked +30 logit). Category names and kinks inferred.

**Tool notes (record):** None. KernelSHAP (200) \~3 min.

## What to look at

- Largest closed-form error: `capital_gain_hinge` (0.0408), then `age_profile` (0.0173). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `capital_gain_country_gate`: *“It is switched off for education=9th, and for Jamaica, Nicaragua, Philippines and Scotland via 2391 (a relu OR-detector of those four countries, native-country 23/27/30/34).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `core_linear_backbone`: *“Two rare one-hots act as extreme switches: Married-AF-spouse -59 (29 rows) and Never-worked +30 (9 rows).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 6 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- 1 consolidation(s): copies merged back after a split. Check that the merged copies carry the same role.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s5.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s5', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
