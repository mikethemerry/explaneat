# Verification packet 08 · adult-s1

Pre-registered verification order: 8 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/f2932f2e-0de5-42e9-b03a-4ee6395549b3?experiment=d69d564c-1585-4b42-a951-2bc9ed1717ca&name=10seed-adult-s1), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | adult / 1 |
| Genome id | `f2932f2e-0de5-42e9-b03a-4ee6395549b3` |
| Experiment | 10seed-adult-s1 (d69d564c-1585-4b42-a951-2bc9ed1717ca); best at generation 99 of 100, NEAT key 24519 |
| Test AUC (runs csv) | 0.904 \[95% CI 0.898, 0.911\] |
| Train AUC | 0.907 |
| Size | 50 of 107 inputs connected; 16 hidden nodes; 80 enabled connections; depth 7; 14 skip connections + 46 direct input→output edges |
| Annotations | 6 leaves, 2 compositions |
| Hidden nodes in exactly one leaf, unsplit | 14 of 16 |
| Claude's time | 44.9 min (2026-11-05T11:08:23+13:00 → 2026-11-05T11:53:18+13:00); within the 3 h box: yes |
| Protocol | v3 |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> P(income\<=50K) = sigmoid(z): essentially logistic regression over 44 direct inputs (married-civ-spouse -3.4 logit, education-num -1.15/SD, male -1.2, wife -1.5) plus one real nonlinearity, a concave piecewise-linear capital-gain penalty (kinks \~3.8k and 4.8k USD); removing it: train acc 0.852 -\> 0.824, AUC 0.907 -\> 0.888. Other hidden nodes are one-hot indicators; rare categories with standardized values 20-66 act as saturating switches.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 39,073 training rows.

### A Composition: `income_logit`

|  |  |
|---|---|
| Annotation id | `ann_123` |
| Entry nodes | -107 (native-country:9), -106 (native-country:8), -105 (native-country:7), -103 (native-country:5), -101 (native-country:40), -100 (native-country:4), -99 (native-country:39), -98 (native-country:38), -96 (native-country:36), -95 (native-country:35), -94 (native-country:34), -93 (native-country:33), -92 (native-country:32), -91 (native-country:31), -90 (native-country:30), -89 (native-country:3), -88 (native-country:29), -87 (native-country:28), -86 (native-country:27), -85 (native-country:26), -84 (native-country:25), -83 (native-country:24), -82 (native-country:23), -80 (native-country:21), -78 (native-country:2), -77 (native-country:19), -76 (native-country:18), -75 (native-country:17), -74 (native-country:16), -73 (native-country:15), -72 (native-country:14), -71 (native-country:13), -70 (native-country:12), -69 (native-country:11), -68 (native-country:10), -67 (native-country:1), -66 (native-country:0), -65 (hours-per-week), -64 (capital-loss), -62 (sex), … (61 more) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | capital\_gain\_term, native\_country37\_gate, relationship0\_indicator, occupation4\_indicator, occupation2\_gate, native\_country22\_switch |
| Dimensionality (in → out, incl. descendants) | 115 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.191 e^{0.441 x_{age} + 0.284 x_{capital-loss} + 1.15 x_{education-num} - 0.113 x_{education:12} - 1.01 x_{education:13} + 0.0555 x_{education:15} + 0.138 x_{education:3} + 0.201 x_{education:4} - 0.0322 x_{education:8} + 0.0982 x_{fnlwgt} + 0.414 x_{hours-per-week} + 1.68 x_{marital-status:2} - 0.14 x_{marital-status:4} - 0.00997 x_{native-country:0} - 1.64 x_{native-country:13} + 1.3 x_{native-country:15} + 0.0259 x_{native-country:22\_a} - 0.0138 x_{native-country:24} - 0.0338 x_{native-country:25} - 0.0948 x_{native-country:26} - 0.0598 x_{native-country:35} - 0.0035 x_{native-country:36} + 0.0288 x_{native-country:41\_a} - 0.236 x_{native-country:4} - 1.94 x_{native-country:6\_a} + 0.109 x_{occupation:10} + 0.0596 x_{occupation:11} + 0.0601 x_{occupation:12} + 0.0136 x_{occupation:1} + 0.00964 x_{occupation:2\_a} + 0.0433 x_{occupation:3\_a} - 0.18 x_{occupation:5} - 0.121 x_{occupation:6} - 0.332 x_{occupation:8} - 0.0672 x_{race:2} + 0.447 x_{relationship:1} + 0.254 x_{relationship:4} + 0.325 x_{relationship:5} + 0.556 x_{sex} - 0.227 x_{workclass:0} + 0.0952 x_{workclass:1\_a} - 0.925 x_{workclass:3} - 0.193 x_{workclass:6} - 0.0898 x_{workclass:7} - 1.0 \operatorname{capital\_gain\_term}{\left(x_{capital-gain},x_{native-country:20} \right)} + 0.0237 \operatorname{native\_country22\_switch}{\left(x_{native-country:22\_b},\operatorname{relu}{\left(1.24 x_{native-country:6\_b} + 0.942 x_{workclass:1\_bc} + 0.912 \operatorname{relu}{\left(0.96 \operatorname{relu}{\left(0.638 x_{workclass:5} + 0.27 \right)} - 0.0379 \right)} + 0.943 \operatorname{relu}{\left(- 0.869 x_{occupation:2\_c} + 0.998 x_{workclass:1\_bc} + 0.463 \right)} + 0.916 \operatorname{relu}{\left(0.663 x_{native-country:41\_b} + 1.06 x_{occupation:3\_bc} + 1.08 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(0.998 x_{occupation:3\_bc} + 0.37 \right)} - 0.846 \right)} - 0.00707 \right)} + 0.3 \right)} \right)} - 0.0302 \operatorname{native\_country37\_gate}{\left(x_{native-country:37} \right)} - 0.00861 \operatorname{occupation\_2\_gate}{\left(x_{occupation:2\_b},x_{occupation:3\_d} \right)} + 0.459 \operatorname{occupation\_4\_indicator}{\left(x_{occupation:4} \right)} - 0.205 \operatorname{relationship\_0\_indicator}{\left(x_{relationship:0} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \frac{1}{0.191 e^{0.441 x_{age} + 3.11 x_{capital-gain} + 0.284 x_{capital-loss} + 1.15 x_{education-num} - 0.113 x_{education:12} - 1.01 x_{education:13} + 0.0555 x_{education:15} + 0.138 x_{education:3} + 0.201 x_{education:4} - 0.0322 x_{education:8} + 0.0982 x_{fnlwgt} + 0.414 x_{hours-per-week} + 1.68 x_{marital-status:2} - 0.14 x_{marital-status:4} - 0.00997 x_{native-country:0} - 1.64 x_{native-country:13} + 1.3 x_{native-country:15} - 0.0389 x_{native-country:20} + 0.0259 x_{native-country:22\_a} - 0.0138 x_{native-country:24} - 0.0338 x_{native-country:25} - 0.0948 x_{native-country:26} - 0.0598 x_{native-country:35} - 0.0035 x_{native-country:36} + 0.0288 x_{native-country:41\_a} - 0.236 x_{native-country:4} - 1.94 x_{native-country:6\_a} + 0.109 x_{occupation:10} + 0.0596 x_{occupation:11} + 0.0601 x_{occupation:12} + 0.0136 x_{occupation:1} + 0.00964 x_{occupation:2\_a} + 0.0433 x_{occupation:3\_a} - 0.18 x_{occupation:5} - 0.121 x_{occupation:6} - 0.332 x_{occupation:8} - 0.0672 x_{race:2} + 0.447 x_{relationship:1} + 0.254 x_{relationship:4} + 0.325 x_{relationship:5} + 0.556 x_{sex} - 0.227 x_{workclass:0} + 0.0952 x_{workclass:1\_a} - 0.925 x_{workclass:3} - 0.193 x_{workclass:6} - 0.0898 x_{workclass:7} + 0.0237 \operatorname{relu}{\left(2.05 - 1.24 \operatorname{relu}{\left(0.953 x_{native-country:22\_b} - 0.175 \operatorname{relu}{\left(1.24 x_{native-country:6\_b} + 0.942 x_{workclass:1\_bc} + 0.912 \operatorname{relu}{\left(0.96 \operatorname{relu}{\left(0.638 x_{workclass:5} + 0.27 \right)} - 0.0379 \right)} + 0.943 \operatorname{relu}{\left(- 0.869 x_{occupation:2\_c} + 0.998 x_{workclass:1\_bc} + 0.463 \right)} + 0.916 \operatorname{relu}{\left(0.663 x_{native-country:41\_b} + 1.06 x_{occupation:3\_bc} + 1.08 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(0.998 x_{occupation:3\_bc} + 0.37 \right)} - 0.846 \right)} - 0.00707 \right)} + 0.3 \right)} - 1.07 \right)} \right)} + 2.19 \operatorname{relu}{\left(0.976 x_{capital-gain} - 0.35 \right)} - 0.0302 \operatorname{relu}{\left(1.0 x_{native-country:37} - 2.08 \right)} + 0.459 \operatorname{relu}{\left(0.717 x_{occupation:4} - 0.214 \right)} - 0.205 \operatorname{relu}{\left(1.03 x_{relationship:0} - 0.827 \right)} + 2.38 \operatorname{relu}{\left(0.893 \operatorname{relu}{\left(0.976 x_{capital-gain} - 0.35 \right)} - 0.12 \right)} - 1.2 \operatorname{relu}{\left(1.24 x_{capital-gain} - 0.485 x_{native-country:20} + 0.867 \right)} - 0.00861 \operatorname{relu}{\left(1.0 x_{occupation:2\_b} - 1.76 x_{occupation:3\_d} - 1.32 \right)}} + 1}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis (supported): additive composition. Output = P(income\<=50K) = sigmoid(z) (class 1 is '\<=50k'). z = 1.653 + a sparse linear score over 44 directly connected standardized inputs + capital\_gain\_term + five one-hot indicator terms (native\_country37\_gate x0.030, relationship0\_indicator x0.205, occupation4\_indicator x-0.459, occupation2\_gate x0.0086, native\_country22\_switch x-0.024). 57 of the 107 inputs (e.g. most native-country and several education/occupation one-hots) have no edges and are listed here as zero-effect entries; native-country:15 has a direct edge but is constant on train, so it is also inert.
> Mechanism: the network is essentially logistic regression plus one nonlinear feature, a concave piecewise-linear capital-gain penalty. The direct linear part lowers P(\<=50K) for married-civ-spouse (-3.38 logit when set), wives (-1.51), males (-1.18), not-in-family (-1.02), each SD of education-num (-1.15), age (-0.44), hours-per-week (-0.41) and capital-loss (-0.28); it raises it for other-service (+1.10) and farming-fishing (+1.05) occupations. Counter-intuitively, education:3/4 (1st-4th, 5th-6th grade; n=189, 406) lower P(\<=50K) by \~2.0 logit when set, which partly cancels the education-num slope for these rows; I cannot give a substantive reason for this. Very rare one-hots (workclass Never-worked n=9, two native countries n=72-82, Preschool n=64) have standardized values of 20-66, so their small weights become +25 to +61 logit 'saturators' that force a \<=50K prediction. Category names are inferred from PMLB's alphabetical coding.
> Evidence (train, n=39073): acc 0.852, AUC 0.907, logit SD 7.25. Term SDs: capital\_gain\_term 5.54, direct linear 4.16, occupation4 0.26, others \<0.05. Mean-ablation: capital\_gain\_term acc 0.852-\>0.824, AUC 0.907-\>0.888 (3571 flips); occupation4 777 flips, AUC -\>0.905; relationship0 181 flips with no accuracy loss; the three rare-country/occupation gates flip \<=6 predictions; the 7-node workclass\_occupation\_aggregator inside native\_country22\_switch is masked (output change exactly 0). SHAP (KernelExplainer, 500 train rows, probability scale) ranks marital-status:2 (0.173), education-num (0.092), sex and capital-gain (0.042 each), age (0.033), relationship:1 (0.030), hours-per-week (0.026) and occupation:4 (0.025) highest; capital-gain's mean \|SHAP\| is modest because 91% of rows have zero gain, while its ablation effect is the largest.

**Evidence (3 records, DB):**

- `analytical_formula`: Additive composition at the output; 57 unconnected inputs are zero-effect entries.
- `empirical_viz`: Full train split. Effects are logit changes of P(\<=50K) when a one-hot is set (w\*(hi-lo)) or per SD for numeric inputs.
- `shap`: Top features by mean \|SHAP\| on 500 train rows.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "structure": "P(income<=50K) = sigmoid(z); z = 1.653 + sum_i w_i x_i (44 direct standardized inputs) + capital_gain_term + 0.0302*native_country37_gate + 0.205*relationship0_indicator - 0.459*occupation4_indicator + 0.0086*occupation2_gate - 0.0237*native_country22_switch",
 "unconnected_inputs_zero_effect": [
  "-107",
  "-106",
  "-105",
  "-103",
  "-101",
  "-99",
  "-98",
  "-94",
  "-93",
  "-92",
  "-91",
  "-90",
  "-89",
  "-88",
  "-87",
  "-86",
  "-82",
  "-80",
  "-78",
  "-77",
  "-76",
  "-75",
  "-74",
  "-72",
  "-70",
  "-69",
  "-68",
  "-67",
  "-61",
  "-60",
  "-58",
  "-57",
  "-54",
  "-53",
  "-50",
  "-48",
  "-42",
  "-41",
  "-36",
  "-35",
  "-34",
  "-32",
  "-30",
  "-29",
  "-27",
  "-25",
  "-24",
  "-23",
  "-20",
  "-18",
  "-15",
  "-14",
  "-13",
  "-12",
  "-10",
  "-6",
  "-4"
 ]
}
```

```json
// empirical_viz
{
 "n": 39073,
 "auc": 0.9069,
 "tool": "compute_performance + local ablation",
 "split": "train",
 "per_sd": {
  "age": -0.44,
  "fnlwgt": -0.1,
  "capital-loss": -0.28,
  "education-num": -1.15,
  "hours-per-week": -0.41
 },
 "term_sd": {
  "direct_linear": 4.163,
  "occupation2_gate": 0.009,
  "capital_gain_term": 5.536,
  "native_country37_gate": 0.029,
  "occupation4_indicator": 0.256,
  "native_country22_switch": 0.002,
  "relationship0_indicator": 0.042
 },
 "accuracy": 0.8517,
 "logit_sd": 7.25,
 "mean_ablation": {
  "direct_linear": {
   "acc": 0.8012,
   "auc": 0.5041
  },
  "occupation2_gate": {
   "flips": 2
  },
  "capital_gain_term": {
   "acc": 0.8244,
   "auc": 0.8875,
   "flips": 3571
  },
  "native_country37_gate": {
   "flips": 6
  },
  "occupation4_indicator": {
   "acc": 0.8503,
   "auc": 0.9046,
   "flips": 777
  },
  "native_country22_switch": {
   "flips": 4
  },
  "relationship0_indicator": {
   "acc": 0.8522,
   "auc": 0.9068,
   "flips": 181
  }
 },
 "category_names_note": "decoded from PMLB alphabetical label coding (inferred; consistent with prevalences)",
 "rare_category_saturators": {
  "native-country:4 (n=64)": 5.8,
  "native-country:6 (n=82)": 42.3,
  "native-country:13 (n=72)": 38.2,
  "education:13 (Preschool, n=64)": 25.0,
  "workclass:3 (Never-worked, n=9)": 60.9
 },
 "largest_direct_effects_when_set_logit": {
  "sex (male)": -1.18,
  "workclass:0 (?)": 0.98,
  "education:3 (1st-4th)": -1.99,
  "education:4 (5th-6th)": -1.98,
  "relationship:5 (Wife)": -1.51,
  "occupation:8 (Other-service)": 1.1,
  "occupation:5 (Farming-fishing)": 1.05,
  "relationship:1 (Not-in-family)": -1.02,
  "marital-status:2 (Married-civ-spouse)": -3.38
 }
}
```

```json
// shap
{
 "note": "500 train rows explained against a 500-row background, per task instruction for Adult",
 "tool": "compute_shap (KernelExplainer, whole model, probability scale)",
 "split": "train",
 "base_value": 0.7176,
 "max_samples": 500,
 "top_mean_abs_shap": {
  "age": 0.0331,
  "sex": 0.0415,
  "capital-gain": 0.0415,
  "capital-loss": 0.0185,
  "occupation:4": 0.0245,
  "occupation:8": 0.0123,
  "education-num": 0.0921,
  "hours-per-week": 0.0262,
  "relationship:1": 0.0302,
  "relationship:4": 0.0107,
  "relationship:5": 0.0165,
  "marital-status:2": 0.1727
 }
}
```

</details>

### A.1 Leaf: `capital_gain_term`

|  |  |
|---|---|
| Annotation id | `ann_116` |
| Entry nodes | -63 (capital-gain), -79 (native-country:20) |
| Exit nodes | cg\_term |
| Internal / exit nodes in its own subgraph | 194, 1266, 8342, cg\_term |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.07 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 3.11 x_{capital-gain} \\
&\quad + 0.0389 x_{native-country:20} \\
&\quad - 2.19 \operatorname{relu}{\left(0.976 x_{capital-gain} - 0.35 \right)} \\
&\quad - 2.38 \operatorname{relu}{\left(0.893 \operatorname{relu}{\left(0.976 x_{capital-gain} - 0.35 \right)} - 0.12 \right)} \\
&\quad + 1.2 \operatorname{relu}{\left(1.24 x_{capital-gain} - 0.485 x_{native-country:20} + 0.867 \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis (supported): the full capital-gain contribution to the logit of P(income\<=50K) (identity node cg\_term collects the direct edge, relay 194 and hinges 1266/8342). It is a concave, monotonically decreasing piecewise-linear function of standardized capital-gain: slope -1.62 up to x=0.359 (\~3.8k USD), -3.76 up to x=0.496 (\~4.8k USD), then -5.83, because ReLU 1266 = relu(0.976x-0.35) and 8342 = relu(0.893\*1266-0.12) switch on in sequence and both carry negative output weights. Relay 194 = relu(1.24x - 0.485 nc20 + 0.867) is always active except for native-country:20 (Iran, 50 train rows), where it is zeroed, giving those rows a steeper linear slope (-3.11) and +0.25 logit at zero gain. Evidence (train, n=39073): term = +1.30 at zero gain (91% of rows), -0.19 at x=0.52, -7.2 at x=1.73, -74 at the 99999 cap; logit-term SD 5.54 (largest single term). Mean-ablating it drops train accuracy 0.852-\>0.824 and AUC 0.907-\>0.888 (3571 predictions flip). Interpretation: any sizeable capital gain is strong, increasingly steep evidence of income \>50K. Closed-form (3 s.f.) max err 0.070.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form; concave piecewise-linear in capital-gain with two kinks.
- `empirical_viz`: Full train split; ablation replaces the term by its train mean.
- `empirical_viz`: Tool line plot over all 39073 train rows: the term falls monotonically from +1.54 to -74.0 across the capital-gain range (r=-0.996).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "piecewise": "For native-country:20 unset, 194 never clips on train (kink at x_cg=-0.698 < min -0.146), so y = 1.04 - 1.617 x_cg for x_cg<0.359; slope -3.758 for 0.359<=x_cg<0.496; slope -5.830 for x_cg>=0.496 (standardized units; raw kinks ~3.8k and ~4.8k USD). native-country:20 (Iran) kills 194: y = -3.108 x_cg + 1.087."
}
```

```json
// empirical_viz
{
 "n": 39073,
 "viz": "local forward pass, term vs capital-gain",
 "split": "train",
 "nc20_rows": 50,
 "term_at_cg0": 1.295,
 "P_le50k_cg_0": 0.795,
 "rows_cg_gt_0": 3176,
 "logit_term_sd": 5.536,
 "mean_ablation": {
  "acc": [
   0.8517,
   0.8244
  ],
  "auc": [
   0.9069,
   0.8875
  ],
  "changed_preds": 3571
 },
 "P_le50k_cg_gt_0": 0.373,
 "term_at_cg_1.73": -7.24,
 "term_at_cg_0.523": -0.191,
 "term_at_cg_13.19(99999)": -74.0
}
```

```json
// empirical_viz
{
 "n": 39073,
 "tool": "get_viz_summary line_plot",
 "split": "train",
 "correlation": -0.9956754279861324,
 "sample_fraction": 1.0,
 "scatter_x_range": [
  -0.14552350385541227,
  13.187777170087646
 ],
 "scatter_y_range": [
  -74.00866093859727,
  1.5377766860511963
 ]
}
```

</details>

### A.2 Leaf: `native_country37_gate`

|  |  |
|---|---|
| Annotation id | `ann_117` |
| Entry nodes | -97 (native-country:37) |
| Exit nodes | 14807 |
| Internal / exit nodes in its own subgraph | 14807 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00178 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.0 x_{native-country:37} - 2.08 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis (supported): node 14807 is an exact indicator of native-country:37 (Thailand under PMLB's alphabetical coding, inferred). Because standardized one-hot columns take the value 38.75 when set (prevalence 0.07%), relu(1.0 x - 2.08) is 0 for all unset rows and 36.67 for the 26 set rows; through output weight 0.030 it adds +1.11 to the logit of P(\<=50K). Evidence (train): exact indicator on 39073 rows; 21/26 set rows are \<=50K (0.81 vs base 0.76). Effect on the dataset is negligible (logit-term SD 0.029; mean-ablation flips 6 predictions). Rare-category bump learned from 26 examples.

**Evidence (2 records, DB):**

- `analytical_formula`: Single ReLU; off for every unset row.
- `empirical_viz`: Truth table on train.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "exact_on_data": "y = 36.67*[native-country:37]"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "rows_set": 26,
 "P_le50k_set": 0.808,
 "P_le50k_unset": 0.761,
 "value_when_set": 36.671,
 "exact_indicator": true,
 "logit_contribution_when_set": 1.107,
 "mean_ablation_changed_preds": 6
}
```

</details>

### A.3 Leaf: `relationship0_indicator`

|  |  |
|---|---|
| Annotation id | `ann_118` |
| Entry nodes | -51 (relationship:0) |
| Exit nodes | 17026 |
| Internal / exit nodes in its own subgraph | 17026 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00364 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.03 x_{relationship:0} - 0.827 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis (supported, but weak effect): node 17026 is an exact indicator of relationship:0 (Husband, inferred from PMLB alphabetical coding; 40% of train rows): 0.423 when set, 0 otherwise. Its output weight 0.205 adds only +0.087 to the logit of P(\<=50K) for husbands, a small correction that runs against the univariate pattern (husbands are 55% \<=50K vs 90% for others); the husband/married effect is carried mainly by the direct marital-status:2 weight (-1.68) and the absence of relationship:0 from the direct edges. Evidence (train): logit-term SD 0.042; mean-ablation flips 181 predictions and changes accuracy 0.8517-\>0.8522, AUC 0.9069-\>0.9068, i.e. practically inert.

**Evidence (2 records, DB):**

- `analytical_formula`: Single ReLU; affine indicator.
- `empirical_viz`: Truth table and ablation on train.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "exact_on_data": "y = 0.423*[relationship:0]"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "rows_set": 15748,
 "P_le50k_set": 0.55,
 "P_le50k_unset": 0.904,
 "logit_term_sd": 0.042,
 "mean_ablation": {
  "acc": [
   0.8517,
   0.8522
  ],
  "auc": [
   0.9069,
   0.9068
  ],
  "changed_preds": 181
 },
 "value_when_set": 0.423,
 "exact_indicator": true,
 "logit_contribution_when_set": 0.087
}
```

</details>

### A.4 Leaf: `occupation4_indicator`

|  |  |
|---|---|
| Annotation id | `ann_119` |
| Entry nodes | -45 (occupation:4) |
| Exit nodes | 4578 |
| Internal / exit nodes in its own subgraph | 4578 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 9.07e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.717 x_{occupation:4} - 0.214 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis (supported): node 4578 is an exact indicator of occupation:4 (Exec-managerial, inferred from PMLB alphabetical coding; 12.4% of train rows): relu(0.717x-0.214) = 1.689 when set, 0 otherwise. With output weight -0.459 it lowers the logit of P(\<=50K) by 0.78 for managers, consistent with the data (52% \<=50K vs 80% otherwise). There is no direct occupation:4 edge, so this node is the network's whole managerial effect. Evidence (train): logit-term SD 0.256; mean-ablation flips 777 predictions, accuracy 0.8517-\>0.8503, AUC 0.9069-\>0.9046.

**Evidence (3 records, DB):**

- `analytical_formula`: Single ReLU; affine indicator.
- `empirical_viz`: Truth table and ablation on train.
- `empirical_viz`: Tool line plot: exit takes only 0 and 1.689, perfectly correlated with occupation:4.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "exact_on_data": "y = 1.689*[occupation:4]"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "rows_set": 4853,
 "P_le50k_set": 0.522,
 "P_le50k_unset": 0.795,
 "logit_term_sd": 0.256,
 "mean_ablation": {
  "acc": [
   0.8517,
   0.8503
  ],
  "auc": [
   0.9069,
   0.9046
  ],
  "changed_preds": 777
 },
 "value_when_set": 1.689,
 "exact_indicator": true,
 "logit_contribution_when_set": -0.776
}
```

```json
// empirical_viz
{
 "n": 39073,
 "tool": "get_viz_summary line_plot",
 "split": "train",
 "correlation": 1.0,
 "sample_fraction": 1.0,
 "scatter_y_range": [
  0.0,
  1.689
 ]
}
```

</details>

### A.5 Leaf: `occupation2_gate`

|  |  |
|---|---|
| Annotation id | `ann_120` |
| Entry nodes | -43\_b (occupation:2\_b), -44\_d (occupation:3\_d) |
| Exit nodes | 4030 |
| Internal / exit nodes in its own subgraph | 4030 |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.104 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 x_{occupation:2\_b} \\
&\quad - 1.76 x_{occupation:3\_d} \\
&\quad -1.32 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis (supported): node 4030 = relu(1.0 occ2 - 1.76 occ3 - 1.32) is an exact indicator of occupation:2 (Armed-Forces, inferred; 12 train rows). The standardized one-hot takes 57.05 when set, so the gate outputs 56.5 there and 0 everywhere else; the occupation:3 (Craft-repair) input is irrelevant because the categories are mutually exclusive and occupation:3 can only push the gate further off. Output weight 0.0086 adds +0.49 to the logit of P(\<=50K) for Armed-Forces rows. Evidence (train): exact indicator; 9/12 set rows \<=50K; logit-term SD 0.009, mean-ablation flips 2 predictions. Negligible rare-category bump. Closed-form err 0.10 comes from 3 s.f. rounding times the 57.05 input value.

**Evidence (2 records, DB):**

- `analytical_formula`: Single ReLU over two one-hots.
- `empirical_viz`: Truth table on train.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "exact_on_data": "y = 56.50*[occupation:2] (occupation:3 is mutually exclusive, so its term only deepens the off state)"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "rows_set": 12,
 "P_le50k_set": 0.75,
 "logit_term_sd": 0.009,
 "value_when_set": 56.498,
 "exact_indicator": true,
 "logit_contribution_when_set": 0.486,
 "mean_ablation_changed_preds": 2
}
```

</details>

### A.6 Composition: `native_country22_switch`

|  |  |
|---|---|
| Annotation id | `ann_122` |
| Entry nodes | -81\_b (native-country:22\_b), 3996 |
| Exit nodes | 1083 |
| Internal / exit nodes in its own subgraph | 594, 1083 |
| Children | workclass\_occupation\_aggregator |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.05 \\
&\quad - 1.24 \operatorname{relu}{\left(0.953 x_{native-country:22\_b} - 0.175 \operatorname{workclass\_occupation\_aggregator}{\left(-104_{b},-3_{bc},-43_{c},-102_{b},-44_{bc},-7 \right)} - 1.07 \right)} \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.05 \\
&\quad - 1.24 \operatorname{relu}{\left(- 0.175 x_{3996} + 0.953 x_{native-country:22\_b} - 1.07 \right)} \bigr)
\end{aligned}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis (supported): serial composition H(nc22, A) with A = workclass\_occupation\_aggregator. 594 = relu(0.953 nc22 - 0.175 A - 1.07) and 1083 = relu(2.05 - 1.24\*594). The standardized native-country:22 one-hot (Italy, inferred) is 21.05 when set, so 594 is between 17.0 and 18.9 for the 88 set rows and 0 otherwise; 1083 is therefore an exact NOT-indicator: 2.053 for everyone except native-country:22 rows, where it is 0. With output weight -0.024 this is a constant -0.049 logit offset that is lifted for Italy-born rows (added to the direct native-country:22 weight -0.026\*21.05 = -0.55). A enters only inside the clipped region and has no effect. Evidence (train, n=39073): 1083==0 iff native-country:22 holds on every row; replacing 1083 by its constant changes P by at most 0.012 and flips 2 predictions. A negligible country bump built from a needlessly deep chain.

**Evidence (2 records, DB):**

- `analytical_formula`: Serial composition; inner aggregator is masked.
- `empirical_viz`: Truth table and ablation on full train.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "exact_on_data": "y = 2.053*(1 - [native-country:22])"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "exact": "1083==0 iff native-country:22 on all train rows",
 "split": "train",
 "rows_nc22": 88,
 "P_le50k_nc22": 0.682,
 "node_1083_values": [
  0,
  2.053
 ],
 "logit_contribution": "-0.0487 when unset, 0 when set (net +0.049 for nc22 rows)",
 "ablation_1083_const": {
  "max_abs_dP": 0.0122,
  "changed_preds": 2
 },
 "node_594_values_when_set": [
  16.96,
  18.095,
  18.39,
  18.92
 ]
}
```

</details>

#### A.6.1 Leaf: `workclass_occupation_aggregator`

|  |  |
|---|---|
| Annotation id | `ann_121` |
| Entry nodes | -104\_b (native-country:6\_b), -3\_bc (workclass:1\_bc), -43\_c (occupation:2\_c), -102\_b (native-country:41\_b), -44\_bc (occupation:3\_bc), -7 (workclass:5) |
| Exit nodes | 3996 |
| Internal / exit nodes in its own subgraph | 15610, 3479, 16792, 2322, 2447, 9651, 3996 |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |
| Live closed-form error (3 s.f. display) | 0.0189 |
| Live numeric-function error | 7.11e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.24 x_{native-country:6\_b} \\
&\quad + 0.942 x_{workclass:1\_bc} \\
&\quad + 0.912 \operatorname{relu}{\left(0.96 \operatorname{relu}{\left(0.638 x_{workclass:5} + 0.27 \right)} - 0.0379 \right)} \\
&\quad + 0.943 \operatorname{relu}{\left(- 0.869 x_{occupation:2\_c} + 0.998 x_{workclass:1\_bc} + 0.463 \right)} \\
&\quad + 0.916 \operatorname{relu}{\left(0.663 x_{native-country:41\_b} + 1.06 x_{occupation:3\_bc} + 1.08 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(0.998 x_{occupation:3\_bc} + 0.37 \right)} - 0.846 \right)} - 0.00707 \right)} \\
&\quad + 0.3 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hypothesis (supported, functionally inert): node 3996 adds up affine relays of six one-hot inputs: 3996 = relu(1.24 nc6 + 0.942 wc1 + 0.943\*15610 + 0.916\*2322 + 0.912\*9651 + 0.30), where 9651/2447 are always-active affine relays of workclass:5 (Self-emp-inc, inferred), 3479-\>16792 are exact indicators of occupation:3 (Craft-repair, inferred) that feed 2322 together with occupation:3 and native-country:41, and 15610 is a workclass:1 relay that switches off only for occupation:2 rows. On train it takes 13 distinct values in \[0.46, 32.3\] and is always positive. Its only consumer is node 594 in native\_country22\_switch, which outputs \>=13 whenever native-country:22 is set and 0 otherwise regardless of 3996 (3996 would need to exceed \~99). So the whole 7-node subtree is masked: setting 3996 to 0 or to its train maximum changes the network output by exactly 0 on all 39073 train rows. It is evolutionary residue with no effect on the predictions; I found no meaningful named function for it beyond 'workclass/occupation/country one-hot score'.

**Evidence (3 records, DB):**

- `analytical_formula`: Closed form; tree of affine relays and one-hot indicators.
- `empirical_viz`: Local ablation on full train: replacing 3996 by a constant leaves the network output exactly unchanged.
- `empirical_viz`: Tool line plot: exit always positive, range \[0.46, 32.3\].

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "on_data": "2447,9651 always active (affine relays of workclass:5); 3479,16792 = exact occupation:3 indicators (3.03, 2.18); 15610 = relu(0.998 wc1 - 0.869 occ2 + 0.463) is 0 only for occupation:2 rows; 2322 active only for occupation:3 (12.4%). 3996 = sum, range [0.46, 32.3], 13 distinct values."
}
```

```json
// empirical_viz
{
 "n": 39073,
 "split": "train",
 "ablation": {
  "3996_set_to_0_max_abs_dP": 0.0,
  "3996_set_to_train_max_max_abs_dP": 0.0
 },
 "distinct_values": 13,
 "node_3996_range": [
  0.464,
  32.27
 ],
 "unmask_threshold": "3996 would need to exceed ~99 for the native-country:22 switch to pass it; train max 32.3"
}
```

```json
// empirical_viz
{
 "n": 39073,
 "tool": "get_viz_summary line_plot",
 "split": "train",
 "sample_fraction": 1.0,
 "scatter_y_range": [
  0.464,
  32.27
 ],
 "correlation_with_first_entry": 0.435
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 1 |
| annotate | 8 |
| consolidate\_node | 2 |
| rename\_node | 107 |
| split\_node | 6 |

6 splits, 1 identity nodes, 2 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#107` identity node cg\_term intercepts 5 connection(s) into 0 (from -63 (capital-gain), 194, 1266, 8342, -79 (native-country:20)); sits in capital\_gain\_term (exit)
- `#108` split -3 into -3\_a, -3\_b, -3\_c. Final copies: -3\_a → income\_logit (entry) \| -3\_bc → workclass\_occupation\_aggregator (entry) \| -3\_bc → workclass\_occupation\_aggregator (entry)
- `#109` split -43 into -43\_a, -43\_b, -43\_c. Final copies: -43\_a → income\_logit (entry) \| -43\_b → occupation2\_gate (entry) \| -43\_c → workclass\_occupation\_aggregator (entry)
- `#110` split -44 into -44\_a, -44\_b, -44\_c, -44\_d. Final copies: -44\_a → income\_logit (entry) \| -44\_bc → workclass\_occupation\_aggregator (entry) \| -44\_bc → workclass\_occupation\_aggregator (entry) \| -44\_d → occupation2\_gate (entry)
- `#111` split -102 into -102\_a, -102\_b. Final copies: -102\_a → income\_logit (entry) \| -102\_b → workclass\_occupation\_aggregator (entry)
- `#112` split -104 into -104\_a, -104\_b. Final copies: -104\_a → income\_logit (entry) \| -104\_b → workclass\_occupation\_aggregator (entry)
- `#113` split -81 into -81\_a, -81\_b. Final copies: -81\_a → income\_logit (entry) \| -81\_b → native\_country22\_switch (entry)
- `#114` consolidate -3\_b, -3\_c into -3\_bc
- `#115` consolidate -44\_b, -44\_c into -44\_bc

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T07:45:47+13:00 (commit `dfc261d`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 4.44e-16 | 4.44e-16 | 4.44e-16 | agree |
| Closed-form max abs error (all leaves) | 0.1039 | 0.1039 | 0.1039 | agree |
| Numeric-function max abs error | n/a | 7.11e-15 | 7.11e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | income\_logit | income\_logit | agree |
| Leaf / composition counts | 6 / 2 | n/a | 6 / 2 | agree |
| Operation counts | 6 splits, 1 identity | {"add\_identity\_node": 1, "annotate": 8, "consolidate\_node": 2, "rename\_node": 107, "split\_node": 6} | {"add\_identity\_node": 1, "annotate": 8, "consolidate\_node": 2, "rename\_node": 107, "split\_node": 6} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `occupation2_gate` | 0.1039 | 0 |
| `capital_gain_term` | 0.07001 | 1.78e-15 |
| `workclass_occupation_aggregator` | 0.01886 | 7.11e-15 |
| `relationship0_indicator` | 0.003643 | 0 |
| `native_country37_gate` | 0.001781 | 0 |
| `occupation4_indicator` | 9.07e-04 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. A 7-node workclass/occupation subtree (3996) is fully masked by a native-country:22 switch (exactly zero effect on all train rows), described as inert residue. relationship0 indicator practically inert. Unexplained: education:3/4 lower P(\<=50K) by \~2 logits. Category names inferred from PMLB alphabetical coding.

**Tool notes (record):** Compositions accept only their own nodes plus child exits as entries. get\_formula on root \~10 min; compute\_shap (500 vs 500 background) \~36 min under contention. One transient empty get\_model\_state.

## What to look at

- Largest closed-form error: `occupation2_gate` (0.104), then `capital_gain_term` (0.07). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. It is large enough that a reader using the displayed coefficients would get visibly different exit values: check which coefficient drives it.
- Strong claim in `workclass_occupation_aggregator`: *“Hypothesis (supported, functionally inert): node 3996 adds up affine relays of six one-hot inputs: 3996 = relu(1.24 nc6 + 0.942 wc1 + 0.943\*15610 + 0.916\*2322 + 0.912\*9651 + 0.30), where 9651/2447 are always-active affine relays o…”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `native_country37_gate`: *“Evidence (train): exact indicator on 39073 rows;”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 6 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- 2 consolidation(s): copies merged back after a split. Check that the merged copies carry the same role.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/adult-s1.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s1', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
