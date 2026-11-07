# Verification packet 27 · heart-s3

Pre-registered verification order: 27 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/c426b094-0843-44e9-ab9a-7d9553d52601?experiment=deb9a13d-9f27-442a-b153-fd7ac103a621&name=10seed-heart-s3), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 3 |
| Genome id | `c426b094-0843-44e9-ab9a-7d9553d52601` |
| Experiment | 10seed-heart-s3 (deb9a13d-9f27-442a-b153-fd7ac103a621); best at generation 39 of 40, NEAT key 9854 |
| Test AUC (runs csv) | 0.874 \[95% CI 0.776, 0.952\] |
| Train AUC | 0.941 |
| Size | 22 of 26 inputs connected; 5 hidden nodes; 30 enabled connections; depth 2; 0 skip connections + 18 direct input→output edges |
| Annotations | 6 leaves, 5 compositions |
| Hidden nodes in exactly one leaf, unsplit | 5 of 5 |
| Claude's time | 15.4 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:39:20+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Generalised additive logistic model: logit = -0.054 + linear backbone (11 features, logit SD 3.49) + four per-feature-group shape functions: inverted-U age effect peaking near 59; saturating resting-BP effect discounted by upsloping ST; per-category chest-pain lookup; additive thallium terms. Train AUC 0.9407; hidden-unit nonlinearity adds only \~0.002 AUC.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_44` |
| Entry nodes | -1\_a (age\_a), -8\_a (trestbps\_a), -4 (cp:1), -6\_a (cp:3\_a), -7 (cp:4), -23 (thal:3), -25 (thal:6), -3 (sex:1), -9 (chol), -10 (fbs:0), -12 (restecg:0), -14 (restecg:2), -15 (thalach), -16 (exang:0), -17 (exang:1), -18 (oldpeak), -20 (slope:2), -22 (ca), -2 (sex:0), -11 (fbs:1), -13 (restecg:1), -26 (thal:7) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | linear\_core, age\_effect, bp\_slope\_effect, chest\_pain\_effect, thal\_effect |
| Dimensionality (in → out, incl. descendants) | 29 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{e^{1.0 \operatorname{age\_effect}{\left(x_{age\_a},x_{age\_b} \right)} + 1.0 \operatorname{bp\_slope\_effect}{\left(x_{trestbps\_a},x_{trestbps\_b},x_{slope:1} \right)} + 1.0 \operatorname{chest\_pain\_effect}{\left(x_{cp:1},x_{cp:3\_a},x_{cp:4},x_{cp:2},x_{cp:3\_b},x_{slope:3} \right)} + 1.0 \operatorname{linear\_core}{\left(x_{sex:1},x_{chol},x_{fbs:0},x_{restecg:0},x_{restecg:2},x_{thalach},x_{exang:0},x_{exang:1},x_{oldpeak},x_{slope:2},x_{ca} \right)} + 1.0 \operatorname{thal\_effect}{\left(x_{thal:3},x_{thal:6},x_{thal:4} \right)}}}{e^{1.0 \operatorname{age\_effect}{\left(x_{age\_a},x_{age\_b} \right)} + 1.0 \operatorname{bp\_slope\_effect}{\left(x_{trestbps\_a},x_{trestbps\_b},x_{slope:1} \right)} + 1.0 \operatorname{chest\_pain\_effect}{\left(x_{cp:1},x_{cp:3\_a},x_{cp:4},x_{cp:2},x_{cp:3\_b},x_{slope:3} \right)} + 1.0 \operatorname{linear\_core}{\left(x_{sex:1},x_{chol},x_{fbs:0},x_{restecg:0},x_{restecg:2},x_{thalach},x_{exang:0},x_{exang:1},x_{oldpeak},x_{slope:2},x_{ca} \right)} + 1.0 \operatorname{thal\_effect}{\left(x_{thal:3},x_{thal:6},x_{thal:4} \right)}} + 1.06}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(-0.0542 \right)}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: z = -0.054 + linear\_core + age\_effect + bp\_slope\_effect + chest\_pain\_effect + thal\_effect, and P(disease) = sigmoid(z). get\_formula shows the output as ReLU, but the evaluator applies a sigmoid to the pre-activation for binary classification. The composition is purely additive, so the model is a generalised additive logistic model: one linear backbone over 11 features plus four per-feature-group shape functions (age inverted-U, saturating BP with an upsloping-ST discount, a per-category chest-pain lookup, and an additive thallium term). Unused inputs: sex:0, fbs:1, restecg:1, thal:7 (mostly redundant one-hot complements; thal:7 acts as the reference via thal:3/thal:6). Only two pieces are materially nonlinear: the age hinge (1499) and the BP/slope:1 hinge (3392). The cp:2 and thal:4 gates are exactly affine on the binary data, and the cp:3 x slope:3 gate is near-inert. Replacing all five hidden gates with affine fits moves the logit by SD 0.16 and train AUC 0.9407 -\> 0.9384, with identical accuracy (0.876). EVIDENCE (train split, n=242, inputs z-scored): {"train\_auc": 0.9407, "train\_acc": 0.876, "logit\_sd\_by\_component": {"linear\_core": 3.489, "chest\_pain\_effect": 1.212, "thal\_effect": 0.682, "bp\_slope\_effect": 0.388, "age\_effect": 0.125}, "auc\_without\_component": {"linear\_core": 0.81, "chest\_pain\_effect": 0.9195, "thal\_effect": 0.9329, "bp\_slope\_effect": 0.9374, "age\_effect": 0.9378}, "all\_gates\_linearised\_auc": 0.9384, "shap\_mean\_abs\_prob\_scale": {"ca": 0.157, "slope:2": 0.12, "sex:1": 0.114, "cp:4": 0.094, "thal:3": 0.041, "chol": 0.038, "oldpeak": 0.032, "exang:0": 0.02, "trestbps": 0.017, "cp:2": 0.014}, "operations": {"splits": 3, "identity\_nodes": 5, "function\_changing": 0}, "function\_preservation\_max\_abs\_err": 3.55e-15, "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "Root: z = -0.054 + linear_core + age_effect + bp_slope_effect + chest_pain_effect + thal_effect, and P(disease) = sigmoid(z)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "tool_note": "get_formula latex/latex_collapsed for the root shows only relu(-0.0542): it drops the composition children; root formula written by hand from the identity-node edges (all weight 1.0)",
 "train_acc": 0.876,
 "train_auc": 0.9407,
 "operations": {
  "splits": 3,
  "identity_nodes": 5,
  "function_changing": 0
 },
 "tool_sensitivity": {
  "0": [
   0.14829229993421209,
   0.051707825413128676,
   0.01397286558934751,
   0.09595214956008971,
   0.11259811507221028,
   0.0016369767720067967,
   0.198823008688918,
   0.05998938248229428,
   0.07192637857867173,
   0.08581384269458249,
   0.511154022851544,
   0.0069411497252660475,
   0.7974518887667589,
   0.18717234509703998,
   0.09189639147494634,
   0.22635184473374403,
   0.5693587948635936,
   0.15401327413213228,
   0.008346076893533352,
   0.06389834279940114,
   0.017991476491660573,
   0.3898678057844419,
   0.23428140800028233,
   0.0883183673468854,
   0.2663994510733892
  ]
 },
 "auc_without_component": {
  "age_effect": 0.9378,
  "linear_core": 0.81,
  "thal_effect": 0.9329,
  "bp_slope_effect": 0.9374,
  "chest_pain_effect": 0.9195
 },
 "logit_sd_by_component": {
  "age_effect": 0.125,
  "linear_core": 3.489,
  "thal_effect": 0.682,
  "bp_slope_effect": 0.388,
  "chest_pain_effect": 1.212
 },
 "tool_activation_profile": {
  "max": 0.9999964640474389,
  "min": 0.48644224170190903,
  "std": 0.1924409149195549,
  "mean": 0.7695324313933757,
  "count": 242,
  "median": 0.7773220208171578,
  "zero_fraction": 0.0,
  "activation_rate": 1.0
 },
 "all_gates_linearised_auc": 0.9384,
 "shap_mean_abs_prob_scale": {
  "ca": 0.157,
  "chol": 0.038,
  "cp:2": 0.014,
  "cp:4": 0.094,
  "sex:1": 0.114,
  "thal:3": 0.041,
  "exang:0": 0.02,
  "oldpeak": 0.032,
  "slope:2": 0.12,
  "trestbps": 0.017
 },
 "function_preservation_max_abs_err": 3.55e-15
}
```

</details>

### A.1 Leaf: `linear_core`

|  |  |
|---|---|
| Annotation id | `ann_39` |
| Entry nodes | -3 (sex:1), -9 (chol), -10 (fbs:0), -12 (restecg:0), -14 (restecg:2), -15 (thalach), -16 (exang:0), -17 (exang:1), -18 (oldpeak), -20 (slope:2), -22 (ca) |
| Exit nodes | identity\_linear (linear\_core\_sum) |
| Internal / exit nodes in its own subgraph | identity\_linear (linear\_core\_sum) |
| Dimensionality (in → out, incl. descendants) | 11 → 1 |
| Live closed-form error (3 s.f. display) | 0.0152 |
| Live numeric-function error | 1.78e-15 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= 1.98 x_{ca} \\
&\quad + 0.665 x_{chol} \\
&\quad - 0.281 x_{exang:0} \\
&\quad - 0.00408 x_{exang:1} \\
&\quad + 0.37 x_{fbs:0} \\
&\quad + 0.496 x_{oldpeak} \\
&\quad - 0.129 x_{restecg:0} \\
&\quad - 0.0349 x_{restecg:2} \\
&\quad + 1.41 x_{sex:1} \\
&\quad + 1.27 x_{slope:2} \\
&\quad - 0.239 x_{thalach}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> identity\_linear = 1.98 ca + 1.41 sex:1 + 1.27 slope:2 + 0.665 chol + 0.496 oldpeak + 0.370 fbs:0 - 0.281 exang:0 - 0.239 thalach - 0.129 restecg:0 - 0.035 restecg:2 - 0.004 exang:1 (z-scored inputs). This is a purely linear logit term, the logistic-regression backbone. The directions mostly agree with cardiology: more fluoroscopy-coloured vessels (ca), male sex, flat ST slope (slope:2), cholesterol and ST depression (oldpeak) raise risk; high maximum heart rate and no exercise angina lower it. One exception: fbs:0 (normal fasting sugar) has a small positive weight. It is the largest component of the logit. EVIDENCE (train split, n=242, inputs z-scored): {"logit\_contribution\_sd": 3.489, "range": \[-6.566, 10.409\], "auc\_alone": 0.902, "auc\_full\_without\_it": 0.81, "note": "get\_viz\_summary activation\_profile on identity\_linear reports min 0/zero\_fraction 0.545 because the tool overrides hidden activations to ReLU; the true identity value is unclipped (ActivationExtractor; range -6.57..10.41)", "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "identity_linear = 1.98 ca + 1.41 sex:1 + 1.27 slope:2 + 0.665 chol + 0.496 oldpeak + 0.370 fbs:0 - 0.281 exang:0 - 0.239 thalach - 0.129 restecg:0 - 0.035 restecg:2 - 0.004 exang:1 (z-scored inputs)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "note": "get_viz_summary activation_profile on identity_linear reports min 0/zero_fraction 0.545 because the tool overrides hidden activations to ReLU; the true identity value is unclipped (ActivationExtractor; range -6.57..10.41)",
 "range": [
  -6.566,
  10.409
 ],
 "split": "train",
 "auc_alone": 0.902,
 "tool_sensitivity": {
  "identity_linear": [
   0.3699663565368968,
   0.12900302835027977,
   0.03486013889301143,
   0.23938577519114862,
   0.2809148850255062,
   0.004084003905416736,
   0.4960326608527706,
   1.269608840352321,
   1.9765892847551938,
   1.4133389916473746,
   0.6646254346367058
  ]
 },
 "auc_full_without_it": 0.81,
 "logit_contribution_sd": 3.489,
 "tool_activation_profile": {
  "max": 10.408822583227678,
  "min": 0.0,
  "std": 2.2796819406868556,
  "mean": 1.413342317510626,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.5454545454545454,
  "activation_rate": 0.45454545454545453
 }
}
```

</details>

### A.2 Composition: `age_effect`

|  |  |
|---|---|
| Annotation id | `ann_40` |
| Entry nodes | -1\_a (age\_a), -1\_b (age\_b) |
| Exit nodes | identity\_age (age\_shape) |
| Internal / exit nodes in its own subgraph | identity\_age (age\_shape) |
| Children | age\_gate |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = 0.179 x_{age\_a} - 0.423 \operatorname{age\_gate}{\left(x_{age\_b} \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Age shape function: identity\_age = 0.179 age\_z - 0.423 age\_gate(age\_z) = 0.179 age\_z - 0.423 relu(1.045 age\_z - 0.478). This is an inverted-U in the logit: it rises with age to a peak near 59 years (+0.05), then falls (-0.09 at 65, -0.24 at 70, -0.44 at 77; -0.39 at 35). So the model treats middle-to-late-50s as the riskiest age and gives the oldest patients a lower risk than the direct slope alone would. This is an evolved non-monotone age effect, a small one. EVIDENCE (train split, n=242, inputs z-scored): {"logit\_contribution\_range": \[-0.509, 0.081\], "logit\_sd": 0.125, "auc\_without": 0.9378, "auc\_full": 0.9407, "curve\_logit\_by\_age": {"35": -0.391, "45": -0.194, "55": 0.004, "60": 0.052, "65": -0.092, "70": -0.237, "77": -0.439}, "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "Age shape function: identity_age = 0.179 age_z - 0.423 age_gate(age_z) = 0.179 age_z - 0.423 relu(1.045 age_z - 0.478)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "auc_full": 0.9407,
 "logit_sd": 0.125,
 "auc_without": 0.9378,
 "tool_sensitivity": {
  "identity_age": [
   0.17944519191792166,
   0.16249855318760986
  ]
 },
 "curve_logit_by_age": {
  "35": -0.391,
  "45": -0.194,
  "55": 0.004,
  "60": 0.052,
  "65": -0.092,
  "70": -0.237,
  "77": -0.439
 },
 "tool_activation_profile": {
  "max": 0.0809832935320632,
  "min": 0.0,
  "std": 0.02449334143114878,
  "mean": 0.013230276475847907,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.7148760330578512,
  "activation_rate": 0.28512396694214875
 },
 "logit_contribution_range": [
  -0.509,
  0.081
 ]
}
```

</details>

#### A.2.1 Leaf: `age_gate`

|  |  |
|---|---|
| Annotation id | `ann_34` |
| Entry nodes | -1\_b (age\_b) |
| Exit nodes | 1499 (older\_age\_hinge) |
| Internal / exit nodes in its own subgraph | 1499 (older\_age\_hinge) |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.0117 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.05 x_{age\_b} - 0.478 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> 1499 = relu(1.045 age\_z - 0.478): an older-age hinge. It is zero below age\_z=0.457 (about 59 years: train mean 54.8, SD 9.09) and rises linearly above it. It is read out to the logit with weight -0.423 (through identity\_age), so for patients over \~59 it cancels and then reverses the direct +0.179 age slope (net -0.263 per SD). EVIDENCE (train split, n=242, inputs z-scored): {"activation\_rate": 0.368, "zero\_fraction": 0.632, "mean": 0.216, "sd": 0.399, "max": 2.074, "threshold\_years": 58.96, "mean\_abs\_grad\_age": 0.384, "linearise\_ablation": "replacing 1499 with its best affine fit in age drops train AUC 0.9407 -\> 0.9387 (affine R2 only 0.55, so it is a genuine hinge)", "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "1499 = relu(1.045 age_z - 0.478): an older-age hinge"
}
```

```json
// empirical_viz
{
 "n": 242,
 "sd": 0.399,
 "max": 2.074,
 "mean": 0.216,
 "split": "train",
 "zero_fraction": 0.632,
 "activation_rate": 0.368,
 "threshold_years": 58.96,
 "tool_sensitivity": {
  "1499": [
   0.3843823749885533
  ]
 },
 "mean_abs_grad_age": 0.384,
 "linearise_ablation": "replacing 1499 with its best affine fit in age drops train AUC 0.9407 -> 0.9387 (affine R2 only 0.55, so it is a genuine hinge)",
 "tool_activation_profile": {
  "max": 2.0736120629689028,
  "min": 0.0,
  "std": 0.3987288380134462,
  "mean": 0.2162937198252049,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.6322314049586777,
  "activation_rate": 0.3677685950413223
 }
}
```

</details>

### A.3 Composition: `bp_slope_effect`

|  |  |
|---|---|
| Annotation id | `ann_41` |
| Entry nodes | -8\_a (trestbps\_a), -8\_b (trestbps\_b), -19 (slope:1) |
| Exit nodes | identity\_bp (bp\_shape) |
| Internal / exit nodes in its own subgraph | identity\_bp (bp\_shape) |
| Children | bp\_slope1\_gate |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= 0.584 x_{trestbps\_a} \\
&\quad - 0.294 \operatorname{bp\_slope1\_gate}{\left(x_{trestbps\_b},x_{slope:1} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Resting-BP x upsloping-ST effect: identity\_bp = 0.584 trestbps\_z - 0.293 relu(0.587 + 1.332 trestbps\_z + 0.907 slope:1\_z). For BP below the hinge it is linear (+0.584 per SD). Above the hinge the BP slope drops to +0.194 per SD (a concave, saturating BP effect), and upsloping ST (slope:1) lowers the logit by up to \~0.53 by opening the gate earlier. Logit by BP (no upslope / upslope): 120 mmHg -0.40/-0.58; 140 mmHg +0.16/-0.37; 160 mmHg +0.38/-0.16; 180 mmHg +0.59/+0.06. slope:1 reaches the output only through this gate. EVIDENCE (train split, n=242, inputs z-scored): {"logit\_contribution\_range": \[-1.232, 0.803\], "logit\_sd": 0.388, "auc\_without": 0.9374, "hinge\_threshold\_mmHg": {"no\_upslope": 136.1, "upslope": 111.3}, "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "Resting-BP x upsloping-ST effect: identity_bp = 0.584 trestbps_z - 0.293 relu(0.587 + 1.332 trestbps_z + 0.907 slope:1_z)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "logit_sd": 0.388,
 "auc_without": 0.9374,
 "tool_sensitivity": {
  "identity_bp": [
   0.17058534262204197,
   0.5844958838769891,
   0.25093664514706493
  ]
 },
 "hinge_threshold_mmHg": {
  "upslope": 111.3,
  "no_upslope": 136.1
 },
 "tool_activation_profile": {
  "max": 0.8025275166377198,
  "min": 0.0,
  "std": 0.13717385246142938,
  "mean": 0.06714857643970926,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.7396694214876033,
  "activation_rate": 0.2603305785123967
 },
 "logit_contribution_range": [
  -1.232,
  0.803
 ]
}
```

</details>

#### A.3.1 Leaf: `bp_slope1_gate`

|  |  |
|---|---|
| Annotation id | `ann_35` |
| Entry nodes | -8\_b (trestbps\_b), -19 (slope:1) |
| Exit nodes | 3392 (bp\_upslope\_hinge) |
| Internal / exit nodes in its own subgraph | 3392 (bp\_upslope\_hinge) |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00658 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.907 x_{slope:1} \\
&\quad + 1.33 x_{trestbps\_b} \\
&\quad + 0.587 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> 3392 = relu(0.587 + 1.332 trestbps\_z + 0.907 slope:1\_z): a resting-BP hinge whose threshold depends on upsloping ST (slope:1). Without upsloping ST it switches on above trestbps of about 136 mmHg; with upsloping ST, above about 111 mmHg (train mean 132.3, SD 18.2). Read out with weight -0.293, it damps the direct BP slope at high BP and gives upsloping ST a negative (protective) effect whenever the gate is open. EVIDENCE (train split, n=242, inputs z-scored): {"activation\_rate": 0.636, "zero\_fraction": 0.364, "mean": 0.946, "sd": 1.126, "max": 5.917, "active\_frac\_no\_upslope": 0.425, "active\_frac\_upslope": 0.87, "mean\_abs\_grad": {"trestbps": 0.855, "slope:1": 0.581}, "linearise\_ablation": "affine fit R2 0.866; linearising drops train AUC 0.9407 -\> 0.9400", "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "3392 = relu(0.587 + 1.332 trestbps_z + 0.907 slope:1_z): a resting-BP hinge whose threshold depends on upsloping ST (slope:1)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "sd": 1.126,
 "max": 5.917,
 "mean": 0.946,
 "split": "train",
 "mean_abs_grad": {
  "slope:1": 0.581,
  "trestbps": 0.855
 },
 "zero_fraction": 0.364,
 "activation_rate": 0.636,
 "tool_sensitivity": {
  "3392": [
   0.5812155602924035,
   0.854987190371897
  ]
 },
 "linearise_ablation": "affine fit R2 0.866; linearising drops train AUC 0.9407 -> 0.9400",
 "active_frac_upslope": 0.87,
 "active_frac_no_upslope": 0.425,
 "tool_activation_profile": {
  "max": 5.916512856690517,
  "min": 0.0,
  "std": 1.126141867641565,
  "mean": 0.9459498747480478,
  "count": 242,
  "median": 0.6395131754027521,
  "zero_fraction": 0.36363636363636365,
  "activation_rate": 0.6363636363636364
 }
}
```

</details>

### A.4 Composition: `chest_pain_effect`

|  |  |
|---|---|
| Annotation id | `ann_42` |
| Entry nodes | -4 (cp:1), -6\_a (cp:3\_a), -7 (cp:4), -5 (cp:2), -6\_b (cp:3\_b), -21 (slope:3) |
| Exit nodes | identity\_cp (chest\_pain\_shape) |
| Internal / exit nodes in its own subgraph | identity\_cp (chest\_pain\_shape) |
| Children | cp2\_indicator, cp3\_slope3\_gate |
| Dimensionality (in → out, incl. descendants) | 6 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.384 x_{cp:1} \\
&\quad - 0.159 x_{cp:3\_a} \\
&\quad + 0.97 x_{cp:4} \\
&\quad + 0.482 \operatorname{cp\_2\_indicator}{\left(x_{cp:2} \right)} \\
&\quad - 0.0721 \operatorname{cp\_3\_slope3\_gate}{\left(x_{cp:3\_b},x_{slope:3} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Chest-pain-type effect: identity\_cp = -0.384 cp:1 - 0.159 cp:3 + 0.970 cp:4 + 0.482 cp2\_indicator(cp:2) - 0.072 cp3\_slope3\_gate(cp:3, slope:3). Since cp is one-hot, this reduces to one logit value per category (train): typical angina cp:1 -2.01 (n=22), non-anginal cp:3 -1.13 (n=70; -1.07 if downsloping ST), atypical angina cp:2 +0.32 (n=42), asymptomatic cp:4 +1.27 (n=108). Asymptomatic chest pain is the high-risk category, as in the Cleveland data. The two hidden gates are, on this domain, an exact cp:2 indicator and a negligible cp:3 x slope:3 interaction (at most 0.07 logit). EVIDENCE (train split, n=242, inputs z-scored): {"logit\_contribution\_range": \[-2.016, 1.304\], "logit\_sd": 1.212, "auc\_without": 0.9195, "per\_category\_logit": {"cp:1": -2.012, "cp:2": 0.323, "cp:3": -1.133, "cp:4": 1.274}, "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "Chest-pain-type effect: identity_cp = -0.384 cp:1 - 0.159 cp:3 + 0.970 cp:4 + 0.482 cp2_indicator(cp:2) - 0.072 cp3_slope3_gate(cp:3, slope:3)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "logit_sd": 1.212,
 "auc_without": 0.9195,
 "tool_sensitivity": {
  "identity_cp": [
   0.017738493742344364,
   0.3842393024739681,
   0.08763380738210019,
   0.1594164840974623,
   0.04597821770090593,
   0.9701061410641643
  ]
 },
 "per_category_logit": {
  "cp:1": -2.012,
  "cp:2": 0.323,
  "cp:3": -1.133,
  "cp:4": 1.274
 },
 "tool_activation_profile": {
  "max": 1.3037935417145032,
  "min": 0.0,
  "std": 0.5936204631506435,
  "mean": 0.6245910597607929,
  "count": 242,
  "median": 0.3212345705699946,
  "zero_fraction": 0.38016528925619836,
  "activation_rate": 0.6198347107438017
 },
 "logit_contribution_range": [
  -2.016,
  1.304
 ]
}
```

</details>

#### A.4.1 Leaf: `cp2_indicator`

|  |  |
|---|---|
| Annotation id | `ann_36` |
| Entry nodes | -5 (cp:2) |
| Exit nodes | 3683 (atypical\_angina\_ind) |
| Internal / exit nodes in its own subgraph | 3683 (atypical\_angina\_ind) |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00638 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(1.05 x_{cp:2} - 0.21 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> 3683 = relu(1.047 cp:2\_z - 0.210). cp:2 (atypical angina) is binary (z = -0.458 or 2.182), so the gate is exactly 0 when cp:2=0 and 2.075 when cp:2=1: an exact indicator, i.e. a direct cp:2 term in disguise. Read out with weight +0.482, so it adds +1.00 to the logit for atypical angina. It is the only route by which cp:2 reaches the output. EVIDENCE (train split, n=242, inputs z-scored): {"activation\_rate": 0.174, "values": {"cp:2=0": 0.0, "cp:2=1": 2.075}, "affine\_fit\_R2": 1.0, "note": "on the binary data domain the ReLU is exactly affine; linearising changes nothing (AUC 0.9407)", "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "3683 = relu(1.047 cp:2_z - 0.210)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "note": "on the binary data domain the ReLU is exactly affine; linearising changes nothing (AUC 0.9407)",
 "split": "train",
 "values": {
  "cp:2=0": 0.0,
  "cp:2=1": 2.075
 },
 "affine_fit_R2": 1.0,
 "activation_rate": 0.174,
 "tool_sensitivity": {
  "3683": [
   0.18169146754501703
  ]
 },
 "tool_activation_profile": {
  "max": 2.074904935934359,
  "min": 0.0,
  "std": 0.7858189197319585,
  "mean": 0.3601074682200127,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.8264462809917356,
  "activation_rate": 0.17355371900826447
 }
}
```

</details>

#### A.4.2 Leaf: `cp3_slope3_gate`

|  |  |
|---|---|
| Annotation id | `ann_37` |
| Entry nodes | -6\_b (cp:3\_b), -21 (slope:3) |
| Exit nodes | 245 (cp3\_slope3\_lookup) |
| Internal / exit nodes in its own subgraph | 245 (cp3\_slope3\_lookup) |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 8.21e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.671 x_{cp:3\_b} \\
&\quad - 0.259 x_{slope:3} \\
&\quad + 0.798 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> 245 = relu(0.798 + 0.671 cp:3\_z - 0.259 slope:3\_z): a 4-cell lookup over non-anginal pain (cp:3) and downsloping ST (slope:3). Cell values: (cp3=0,s3=0) 0.441; (0,1) 0 (clipped); (1,0) 1.921; (1,1) 0.908. Read out with weight -0.072, so it shifts the logit by at most 0.14. It is practically inert: removing it changes train AUC 0.9407 -\> 0.9404. Its only non-additive part is the clip in the (cp3=0, slope3=1) cell. EVIDENCE (train split, n=242, inputs z-scored): {"activation\_rate": 0.95, "zero\_fraction": 0.05, "mean": 0.827, "sd": 0.674, "logit\_contribution\_sd": 0.049, "affine\_fit\_R2": 0.99, "ablation": "removal AUC 0.9404; linearisation AUC 0.9406", "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "245 = relu(0.798 + 0.671 cp:3_z - 0.259 slope:3_z): a 4-cell lookup over non-anginal pain (cp:3) and downsloping ST (slope:3)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "sd": 0.674,
 "mean": 0.827,
 "split": "train",
 "ablation": "removal AUC 0.9404; linearisation AUC 0.9406",
 "affine_fit_R2": 0.99,
 "zero_fraction": 0.05,
 "activation_rate": 0.95,
 "tool_sensitivity": {
  "245": [
   0.2460521536959174,
   0.637767763866505
  ]
 },
 "logit_contribution_sd": 0.049,
 "tool_activation_profile": {
  "max": 1.921392911631808,
  "min": 0.0,
  "std": 0.6738238250175955,
  "mean": 0.8266962470654297,
  "count": 242,
  "median": 0.4414252821336262,
  "zero_fraction": 0.049586776859504134,
  "activation_rate": 0.9504132231404959
 }
}
```

</details>

### A.5 Composition: `thal_effect`

|  |  |
|---|---|
| Annotation id | `ann_43` |
| Entry nodes | -23 (thal:3), -25 (thal:6), -24 (thal:4) |
| Exit nodes | identity\_thal (thal\_shape) |
| Internal / exit nodes in its own subgraph | identity\_thal (thal\_shape) |
| Children | thal4\_passthrough |
| Dimensionality (in → out, incl. descendants) | 3 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}\begin{aligned}
y_0 &= - 0.467 x_{thal:3} \\
&\quad - 0.565 x_{thal:6} \\
&\quad + 0.23 \operatorname{thal\_4\_passthrough}{\left(x_{thal:4} \right)}
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Thallium-scan effect: identity\_thal = -0.467 thal:3 - 0.565 thal:6 + 0.230 thal4\_passthrough(thal:4). It is additive one-hot, with thal:4 entering affinely through an always-on ReLU. Per category (train): thal:3 normal -0.18 (n=135), thal:6 fixed defect -1.58 (n=15), thal:7 reversible defect +0.76 (n=90), thal:4 +3.30 (n=2, near-empty category). Reversible defect raises risk relative to normal, as expected. The strongly negative fixed-defect value is counterintuitive and rests on 15 patients. EVIDENCE (train split, n=242, inputs z-scored): {"logit\_contribution\_range": \[-1.579, 3.295\], "logit\_sd": 0.682, "auc\_without": 0.9329, "per\_category\_logit": {"thal:3": -0.178, "thal:4": 3.295, "thal:6": -1.579, "thal:7": 0.763}, "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "Thallium-scan effect: identity_thal = -0.467 thal:3 - 0.565 thal:6 + 0.230 thal4_passthrough(thal:4)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "logit_sd": 0.682,
 "auc_without": 0.9329,
 "tool_sensitivity": {
  "identity_thal": [
   0.4669660568400357,
   0.22926728594780607,
   0.5647128497481091
  ]
 },
 "per_category_logit": {
  "thal:3": -0.178,
  "thal:4": 3.295,
  "thal:6": -1.579,
  "thal:7": 0.763
 },
 "tool_activation_profile": {
  "max": 3.295168878131084,
  "min": 0.0,
  "std": 0.45764889157901245,
  "mean": 0.3108973481157296,
  "count": 242,
  "median": 0.0,
  "zero_fraction": 0.6198347107438017,
  "activation_rate": 0.38016528925619836
 },
 "logit_contribution_range": [
  -1.579,
  3.295
 ]
}
```

</details>

#### A.5.1 Leaf: `thal4_passthrough`

|  |  |
|---|---|
| Annotation id | `ann_38` |
| Entry nodes | -24 (thal:4) |
| Exit nodes | 3731 (thal4\_relay) |
| Internal / exit nodes in its own subgraph | 3731 (thal4\_relay) |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00101 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.997 x_{thal:4} + 0.496 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> 3731 = relu(0.496 + 0.997 thal:4\_z). It is always active (min pre-activation 0.405 on train), so the ReLU never clips: on the data it is the affine pass-through 0.496 + 0.997 thal:4\_z. Read out with weight +0.230, so it is a direct thal:4 term (+2.52 logit for the 2 train patients with thal:4=1). thal:4 is a near-empty category (n=2/242), so this unit is effectively irrelevant (removal AUC unchanged, 0.9407). EVIDENCE (train split, n=242, inputs z-scored): {"activation\_rate": 1.0, "min": 0.405, "max": 11.417, "n\_thal4": 2, "affine\_fit\_R2": 1.0, "split": "train", "n": 242}

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula (force=true; latex\_collapsed for compositions).
- `empirical_viz`: Train-split activation profile, sensitivity, and local read-only ablations (train split, n=242, inputs z-scored).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "3731 = relu(0.496 + 0.997 thal:4_z)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "max": 11.417,
 "min": 0.405,
 "split": "train",
 "n_thal4": 2,
 "affine_fit_R2": 1.0,
 "activation_rate": 1.0,
 "tool_sensitivity": {
  "3731": [
   0.9969380324000681
  ]
 },
 "tool_activation_profile": {
  "max": 11.416575792542414,
  "min": 0.404659242138995,
  "std": 0.9969380324000706,
  "mean": 0.4956668169357174,
  "count": 242,
  "median": 0.404659242138995,
  "zero_fraction": 0.0,
  "activation_rate": 1.0
 }
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| add\_identity\_node | 5 |
| annotate | 11 |
| rename\_node | 36 |
| split\_node | 3 |

3 splits, 5 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -1 into -1\_a, -1\_b. Final copies: -1\_a → age\_effect (entry); heart\_disease\_logit (entry) \| -1\_b → age\_gate (entry); age\_effect (entry)
- `#27` split -6 into -6\_a, -6\_b. Final copies: -6\_a → chest\_pain\_effect (entry); heart\_disease\_logit (entry) \| -6\_b → cp3\_slope3\_gate (entry); chest\_pain\_effect (entry)
- `#28` split -8 into -8\_a, -8\_b. Final copies: -8\_a → bp\_slope\_effect (entry); heart\_disease\_logit (entry) \| -8\_b → bp\_slope1\_gate (entry); bp\_slope\_effect (entry)
- `#29` identity node identity\_age intercepts 2 connection(s) into 0 (from -1\_a (age\_a), 1499 (older\_age\_hinge)); sits in age\_effect (exit, composition)
- `#30` identity node identity\_bp intercepts 2 connection(s) into 0 (from -8\_a (trestbps\_a), 3392 (bp\_upslope\_hinge)); sits in bp\_slope\_effect (exit, composition)
- `#31` identity node identity\_cp intercepts 5 connection(s) into 0 (from -4 (cp:1), -6\_a (cp:3\_a), -7 (cp:4), 3683 (atypical\_angina\_ind), 245 (cp3\_slope3\_lookup)); sits in chest\_pain\_effect (exit, composition)
- `#32` identity node identity\_thal intercepts 3 connection(s) into 0 (from -23 (thal:3), -25 (thal:6), 3731 (thal4\_relay)); sits in thal\_effect (exit, composition)
- `#33` identity node identity\_linear intercepts 11 connection(s) into 0 (from -3 (sex:1), -9 (chol), -10 (fbs:0), -12 (restecg:0), -14 (restecg:2), -15 (thalach), -16 (exang:0), -17 (exang:1), … (3 more)); sits in linear\_core (exit)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 3.55e-15 | 3.55e-15 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.01519 | 0.01519 | 0.01519 | agree |
| Numeric-function max abs error | n/a | 1.78e-15 | 1.78e-15 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 6 / 5 | n/a | 6 / 5 | agree |
| Operation counts | 3 splits, 5 identity | {"add\_identity\_node": 5, "annotate": 11, "rename\_node": 36, "split\_node": 3} | {"add\_identity\_node": 5, "annotate": 11, "rename\_node": 36, "split\_node": 3} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `linear_core` | 0.01519 | 1.78e-15 |
| `age_gate` | 0.0117 | 0 |
| `bp_slope1_gate` | 0.006579 | 0 |
| `cp2_indicator` | 0.006383 | 0 |
| `thal4_passthrough` | 0.001012 | 0 |
| `cp3_slope3_gate` | 8.21e-04 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. cp2 (3683) and thal4 (3731) exactly affine; cp3xslope3 (245) near-inert (\<=0.14 logit, AUC change 0.0003). Only the age hinge (1499) and BP/slope:1 hinge (3392) are material nonlinearity; replacing all gates with affine fits: train AUC 0.9407 -\> 0.9384. Counterintuitive: fixed-defect thal:6 (-1.58 logit, n=15) and fbs:0 (+0.37).

**Tool notes (record):** get\_viz\_summary activation\_profile on identity nodes overrides them with relu (clips); computed locally. get\_formula for the root drops composition children in both latex and latex\_collapsed (root formula written by hand). Annotate validator rejects grandchild entry nodes. Output shown/extracted as relu(z); compute\_performance applies sigmoid.

## What to look at

- Largest closed-form error: `linear_core` (0.0152), then `age_gate` (0.0117). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `cp2_indicator`: *“cp:2 (atypical angina) is binary (z = -0.458 or 2.182), so the gate is exactly 0 when cp:2=0 and 2.075 when cp:2=1: an exact indicator, i.e.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `thal4_passthrough`: *“It is always active (min pre-activation 0.405 on train), so the ReLU never clips: on the data it is the affine pass-through 0.496 + 0.997 thal:4\_z.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 3 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s3.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s3', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
