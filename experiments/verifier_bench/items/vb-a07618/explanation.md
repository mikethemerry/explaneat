# Explanation vb-a07618

- Genome: `b19b9536-00dd-4e95-9f5a-def6dd8d6889`
- Dataset: Heart Disease (binary, prepared) (training split 165cebbd-6609-46ca-867e-c09e17b2d6ec, n = 242; test n = 61)
- Outcome: heart disease (more than 50% narrowing of a major coronary artery on angiography)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 20%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 1.7763568394002505e-15
- closed_form_max_abs_err: 0.012934542112016167
- n_leaf_annotations: 5
- n_composition_annotations: 1
- n_splits: 4
- n_identity_nodes: 0

**Summary.** Essentially logistic regression on 15 inputs (logit SD 4.96); ca effective weight 2.60 via direct edge plus affine 148. Four small additive one-sided corrections (combined logit SD 1.41): oldpeak hinge above +0.39 SD; protective no-exercise-angina gate (stronger for women); -3.19 correction for fixed thal defect with normal fbs (n=10); near-inert cp:2 offset. Readouts fixed at means: train AUC 0.9437 -> 0.9295.

**What resisted.** Nothing resisted coverage. Single hidden layer: 5 hidden nodes each read out directly, flat root. 148 never clips (affine); 3229 is a constant -0.044 offset for cp:2. 2034 (fixed thal defect AND normal fbs) acts on 10 training patients, looks like an overfitted correction (flips 4 FP to negatives). 6 disconnected inputs (restecg:0/1/2, exang:1, slope:3, thal:3) added to root as zero-effect entries.

## Plain-language summary (draft)

For a typical patient (a 56-year-old man with no chest pain symptoms and median test results) the model predicts a risk of heart disease of about 20%; it adds up most of its inputs independently, and five small parts add to that sum: one switches on for the 70 of 242 training patients with marked ST depression on exercise, one acts on the 160 without chest pain on exercise, and three more are explained below. [C31, C34, C39]

**Inputs that add up.** These add up independently, and these lower the risk: through this sum alone, compared with no chest pain symptoms, typical angina moves a typical patient's predicted risk from about 20% to below 1%, non-anginal chest pain to about 1% and atypical angina to about 6%; being a woman lowers it to about 1% and fasting blood sugar above 120 mg/dL to about 3%, and a higher maximum heart rate lowers it steadily, from about 32% at 118 to about 14% at 175 beats per minute; chest pain type, sex and fasting blood sugar also act through the parts below. [C10, C15, C22, C14, C19, C18] A flat rather than upsloping ST segment raises it to about 90%, and higher cholesterol (from about 8% at 193 mg/dL to about 53% at 308 mg/dL), higher resting blood pressure (from about 11% at 110 mm Hg to about 36% at 155 mm Hg) and older age (from about 15% at 42 to about 23% at 66 years) raise it steadily. [C04, C05, C07, C23] Through this sum alone, each extra vessel coloured by fluoroscopy adds about the same, from about 20% at 0 to about 96% at 2 vessels, flattening out near the top because the predicted risk cannot go much higher, and a reversible defect on the thallium scan, compared with any other thallium result, raises it to about 39%; both also act through the parts below. [C02, C12] A downsloping ST segment raises it to about 40% (the model has little data on this, n = 17); the model ignores the resting ECG. [C08, C30]

**ST depression hinge.** This part reads only ST depression on exercise and acts on 70 of 242 training patients, 50 of whom had heart disease. [C34] Through this part, more ST depression adds nothing until about 1.5 mm, then raises the predicted risk steadily: for a typical patient from about 20% at 0 mm to about 45% at 2.8 mm and about 61% at 3.5 mm. [C32, C33] Across the whole model ST depression has the same effect, from about 20% at 0 mm to about 45% at 2.8 mm. [C06]

**No chest pain on exercise, by sex.** This part reads sex and chest pain on exercise and acts on the 160 of 242 training patients without chest pain on exercise, 47 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 1% both without it and with it. [C39] Through it, being a woman lowers a typical patient's predicted risk from about 20% to about 7%, and chest pain brought on by exercise raises it to about 27%. [C36, C37] Across the whole model, being a woman lowers the risk more when exercise does not bring on chest pain: from about 20% to below 1%, against about 27% to about 1% when it does. [C38]

**Fixed thallium defect with normal blood sugar.** This part reads fasting blood sugar and the thallium scan and acts on the 15 of 242 training patients with a fixed defect on the thallium scan, 9 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is above 99% without it and about 90% with it (the model has little data on this, n = 15). [C44] Through this part, a fixed defect on the thallium scan lowers a typical patient's predicted risk from about 20% to about 1%, but only if fasting blood sugar is 120 mg/dL or less; with fasting blood sugar above 120 mg/dL it stays at about 3% (the model has little data on this, n = 15). [C41, C43] For the typical patient it acts on, fasting blood sugar above 120 mg/dL raises the risk through it from about 90% to above 99%. [C42]

**Vessel count boost.** This part adds more for each vessel coloured by fluoroscopy: through it, a typical patient's predicted risk rises steadily from about 20% at 0 to about 45% at 2 vessels, each extra vessel adding about the same. [C47] Across the whole model, the vessels raise it steadily, each extra vessel adding about the same, from about 20% at 0 to about 99% at 2 vessels, flattening out near the top because the predicted risk cannot go much higher. [C01] Through this part, a thallium result coded 4 also raises the risk, to about 93% (the model has little data on this, n = 2). [C46]

**Atypical angina offset.** This part reads only whether the patient has atypical angina and hardly changes the predictions: through it, atypical angina moves a typical patient's predicted risk from about 20% to about 19%. [C51, C49]

## Parts, from the root down

### The whole model (`ann_35`, `heart_disease_logit`)

- Kind: composition; parent: none; children: vessel_count_linear_boost, sex_exang_protective_gate, fbs_normal_fixed_defect_gate, oldpeak_hinge, cp2_constant_offset
- Reads: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate reached during the exercise test, exercise-induced angina, ST depression during exercise (compared with rest), slope of the ST segment at peak exercise, number of major vessels coloured by fluoroscopy, thallium stress test result
- Entry nodes: -1, -2_a, -3_a, -4, -5_a, -6, -7, -8, -9, -11, -15, -19, -20, -22_a, -26, -12, -13, -14, -17, -21, -23
- Exit nodes: 0
- Subgraph nodes: -1, -2_a, -3_a, -4, -5_a, -6, -7, -8, -9, -11, -15, -19, -20, -22_a, -26, -12, -13, -14, -17, -21, -23, 0

**Hypothesis (plain).** The model adds up its inputs into one score: the vessels coloured (each extra vessel adding about the same), the ST segment, chest pain type and sex each move the predicted risk on their own, five small parts add to that sum, and the resting ECG is ignored. [C31, C01, C04, C11, C09, C30]

**Evidence (plain).** It consists of the inputs that feed the score directly and five parts, the ST depression hinge, the part for patients without chest pain on exercise, the fixed thallium defect part, the vessel count boost and the atypical angina offset (each explained below); the score is their sum, checked on all 242 training patients. [C31] ST depression on exercise and chest pain on exercise do not feed the score directly, so each acts only through its part. [C31, C32, C37] When one input at a time is changed across the whole model, each extra vessel takes a typical patient from about 20% at 0 to about 99% at 2 vessels, against about 96% through the sum alone, and being a woman lowers the risk to below 1%, against about 1% through the sum alone. [C01, C02, C09, C14] Across the whole model a fixed defect on the thallium scan lowers it to about 1% and a thallium result coded 4 raises it to about 93% (the model has little data on this, n = 15 and 2). [C17, C03] The model ignores the resting ECG. [C30]

**Annotation hypothesis (technical).** Root: logit z = -0.954 + LINEAR CORE + 1.893*vessel_count_linear_boost - 0.452*sex_exang_protective_gate - 0.942*fbs_normal_fixed_defect_gate + 1.108*oldpeak_hinge - 0.154*cp2_constant_offset; prediction = sigmoid(z) (get_formula shows the output node as ReLU, but the evaluator overrides it with sigmoid for binary classification; compute_performance on train, AUC 0.9437 and acc 0.888, matches sigmoid(z)). LINEAR CORE (standardised inputs): +2.065 ca, +1.665 sex:1, -0.070 sex:0, +1.298 slope:2, -0.499 slope:1, +1.075 chol, -0.972 cp:3, -0.916 cp:1, +0.618 cp:4, -0.057 cp:2, -0.667 fbs:1, +0.616 trestbps, +0.461 thal:7, -0.419 thalach, +0.205 age. Inputs restecg:0/1/2, exang:1, slope:3 and thal:3 are disconnected in the phenotype and enter the root with zero effect. CHILDREN (all additive, read out directly by the output; no serial composition): vessel_count_linear_boost is affine (never clips), raising the effective ca weight to 2.60; sex_exang_protective_gate lowers the logit by 0.43 (men) or 1.67 (women) only when there is no exercise angina; oldpeak_hinge adds 1.11 per SD of ST depression above +0.39 SD (the only oldpeak pathway); fbs_normal_fixed_defect_gate subtracts 3.19 for the 10 training patients with a fixed thallium defect and normal fasting sugar (likely an overfitted correction); cp2_constant_offset is a near-inert -0.044 offset for cp:2. EVIDENCE (train, n=242): the linear core dominates (logit sd 4.96 vs 1.41 for all readouts); with every readout fixed at its mean, train AUC is 0.9295 and accuracy 0.860, vs 0.9437 and 0.888 for the full model. SHAP (probability scale) ranks ca 0.158, sex:1 0.104, slope:2 0.092, cp:3 0.053, chol 0.044, cp:4 0.041. So the network is a logistic regression (more vessels, male sex, flat ST slope, high cholesterol, asymptomatic chest pain mean more risk) with two clinically plausible one-sided terms (marked ST depression adds risk; no exertional angina lowers it, more for women) and one small-sample correction.

**Evidence records.**

- *analytical_formula*: Root closed form: affine core in 15 inputs plus five child readouts; sigmoid at output. (note: get_formula(force=true) latex_collapsed. Output shown as ReLU, but the evaluator overrides the output activation with sigmoid for binary classification, so prediction = sigmoid(z). The stored ReLU would clip 59.5% of train logits (z<0) and lower AUC to 0.893; compute_performance (sigmoid) gives AUC 0.9437, matching sigmoid(z).)
- *shap_importance*: Train-split SHAP (probability scale): ca 0.158, sex:1 0.104, slope:2 0.092, cp:3 0.053, chol 0.044, cp:4 0.041 dominate; restecg, exang:1, slope:3, thal:3 are exactly 0 (disconnected). (n: 242; space: probability; split: train; base_value: 0.43139138099781016)
- *ablation_study*: Linear core carries most of the logit variance (sd 4.96 vs 1.41 for all readouts). Readouts add +0.014 train AUC and +2.9 points accuracy; the largest single effects are the fixed-defect gate and the oldpeak hinge. (split: train; method: local logit decomposition z = linear core + sum(readout weight * node activation), verified equal to the network pre-activation to 1.8e-15; each readout replaced by its train mean, no refit)
- *performance_metrics*: Train performance of the explained network (sigmoid output). (n: 242; mse: 0.0898; split: train; source: compute_performance(split=train, max_samples=100000); auc_roc: 0.9437; accuracy: 0.8884)


### ST depression hinge (`ann_33`, `oldpeak_hinge`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: ST depression during exercise (compared with rest)
- Entry nodes: -18
- Exit nodes: 2985
- Subgraph nodes: -18, 2985

**Hypothesis (plain).** This part increases the predicted risk with ST depression on exercise: nothing until about 1.5 mm, then steadily. [C32, C33]

**Evidence (plain).** It is active for 70 of 242 training patients, 50 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is above 99% both without it and with it. [C34] Through this part, more ST depression adds nothing until about 1.5 mm, then raises the risk steadily, for a typical patient from about 20% at 0 mm to about 45% at 2.8 mm and about 61% at 3.5 mm. [C32, C33] Its formula reproduces the part exactly on all 242 training patients. [C35]

**Annotation hypothesis (technical).** Node 2985 = ReLU(1.002*oldpeak - 0.386), read out with +1.108. A hinge on exercise-induced ST depression: zero for patients with oldpeak below +0.385 SD (71% of train), then +1.11 logit per SD above it (max +4.41). Disease rate 0.71 where active vs 0.33 where inactive. It is the network's only oldpeak pathway, so the network treats small ST depression as uninformative and only marked depression as risk. Mean-ablation drops train AUC 0.9437 -> 0.9400 and accuracy 0.888 -> 0.876.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(1.0 x_{oldpeak} - 0.386 \right)}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (latex: y_0 = \operatorname{relu}\left(1.0 x_{oldpeak} - 0.386\right); plain: 2985 = relu(1.002 oldpeak - 0.386); readout 2985 -> 0 weight +1.108)
- *empirical_viz*: Hinge: zero for 71% of train patients (oldpeak below +0.385 SD), linear above with slope 1.11 logit per SD of ST depression; max contribution +4.41. (n: 242; split: train; disease_rate: active 0.71 (n=70) vs inactive 0.33 (n=172); threshold_sd: 0.385)


### No chest pain on exercise, by sex (`ann_31`, `sex_exang_protective_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: sex, exercise-induced angina
- Entry nodes: -2_b, -3_b, -16
- Exit nodes: 791
- Subgraph nodes: -2_b, -3_b, -16, 791

**Hypothesis (plain).** This part reduces the predicted risk for patients without chest pain on exercise, and lowers it most for women without it. [C39, C38]

**Evidence (plain).** It is active for the 160 of 242 training patients without chest pain on exercise, 47 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 1% both without it and with it. [C39] Through it, being a woman lowers a typical patient's predicted risk from about 20% to about 7%, and chest pain brought on by exercise raises it to about 27%. [C36, C37] Across the whole model, being a woman lowers the risk from about 20% to below 1% when exercise does not bring on chest pain, against about 27% to about 1% when it does. [C38] Its formula reproduces the part exactly on all 242 training patients. [C40]

**Annotation hypothesis (technical).** Node 791 = ReLU(0.325 + 1.084*sex:0 - 0.218*sex:1 + 2.200*exang:0), read out with -0.452. Since sex and exang are one-hot binaries it takes only three values on train: 0 for all patients with exercise-induced angina (n=82, disease rate 0.73), 0.959 for men without it (n=96) and 3.702 for women without it (n=64). It is therefore a protective interaction: absence of exertional angina lowers the logit by 0.43 for men but by 1.67 for women, and has no effect if angina is present. This is the network's only exang pathway (no direct exang edge). Readout sd 0.66 logits; mean-ablation drops train AUC 0.9437 -> 0.9418, accuracy unchanged.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.2 x_{exang:0} \\
&\quad + 1.08 x_{sex:0_b} \\
&\quad - 0.218 x_{sex:1_b} \\
&\quad + 0.325 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (latex: y_0 = \operatorname{relu}\bigl(2.2 x_{exang:0} + 1.08 x_{sex:0} - 0.218 x_{sex:1} + 0.325\bigr); plain: 791 = relu(2.200 exang:0 + 1.084 sex:0 - 0.218 sex:1 + 0.325); readout 791 -> 0 weight -0.452)
- *empirical_viz*: Three-valued gate on train: exactly 0 for every patient with exercise-induced angina; 0.959 for men and 3.702 for women without it. Logit contribution 0, -0.43 or -1.67. (n: 242; split: train; disease_rate: active 0.29 (n=160) vs inactive 0.73 (n=82))


### Fixed thallium defect with normal blood sugar (`ann_32`, `fbs_normal_fixed_defect_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: fasting blood sugar, thallium stress test result
- Entry nodes: -10, -25
- Exit nodes: 2034
- Subgraph nodes: -10, -25, 2034

**Hypothesis (plain).** This part reduces the predicted risk for patients with a fixed defect on the thallium scan, but only if their fasting blood sugar is 120 mg/dL or less (the model has little data on this, n = 15). [C41, C43]

**Evidence (plain).** It acts on the 15 of 242 training patients with a fixed defect on the thallium scan, 9 of whom had heart disease; for the typical patient it acts on, the predicted risk is above 99% without it and about 90% with it (the model has little data on this, n = 15). [C44] Through it, a fixed defect on the thallium scan lowers a typical patient's predicted risk from about 20% to about 1% if fasting blood sugar is 120 mg/dL or less, and leaves it at about 3% if fasting blood sugar is above 120 mg/dL (the model has little data on this, n = 15). [C41, C43] For the typical patient it acts on, fasting blood sugar above 120 mg/dL raises the risk through it from about 90% to above 99%. [C42] On all 242 training patients, its formula gives exactly what the part computes. [C45]

**Annotation hypothesis (technical).** Node 2034 = ReLU(-1.027 + 1.156*fbs:0 + 1.016*thal:6), read out with -0.942. An AND gate: it is 3.386 only for patients with a fixed thallium defect (thal:6) AND normal fasting blood sugar (n=10 on train), 0.018 for fixed-defect patients with high sugar (n=5), and 0 for everyone else (93.8%). It subtracts 3.19 logits for that small group. Evidence suggests it is a training-set correction rather than a clinical rule: among the 10, the linear core over-predicted 4 of 5 negatives as positive; the gate moves all 5 negatives below 0 while the 5 positives stay above 0. This is the only thal:6 pathway and the only fbs:0 pathway. Mean-ablation drops train AUC 0.9437 -> 0.9388 and accuracy 0.888 -> 0.872 (largest single-gate effect), but it rests on 10 patients, so treat it as likely overfitted.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(1.16 x_{fbs:0} + 1.02 x_{thal:6} - 1.03 \right)}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (latex: y_0 = \operatorname{relu}\left(1.16 x_{fbs:0} + 1.02 x_{thal:6} - 1.03\right); plain: 2034 = relu(1.156 fbs:0 + 1.016 thal:6 - 1.027); readout 2034 -> 0 weight -0.942)
- *empirical_viz*: AND gate: 3.386 only for fixed-defect patients with normal fasting sugar (n=10), ~0 otherwise. Subtracts 3.19 logits for those 10, which flips 4 false positives to correct negatives while all 5 positives stay positive. (n: 242; split: train; fixed_defect_normal_fbs_labels: y=[1,0,0,1,0,0,0,1,1,1]; logit without gate [5.82,1.61,1.57,7.81,-4.64,0.15,0.60,4.99,8.95,11.92]; with gate [2.63,-1.58,-1.62,4.62,-7.82,-3.04,-2.59,1.81,5.76,8.73])


### Vessel count boost (`ann_30`, `vessel_count_linear_boost`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: number of major vessels coloured by fluoroscopy, thallium stress test result
- Entry nodes: -22_b, -24
- Exit nodes: 148
- Subgraph nodes: -22_b, -24, 148

**Hypothesis (plain).** This part increases the predicted risk steadily with each vessel coloured by fluoroscopy, each extra vessel adding about the same. [C47]

**Evidence (plain).** Through this part, a typical patient's predicted risk rises steadily from about 20% at 0 to about 45% at 2 vessels. [C47] Through this part, a thallium result coded 4 also raises the risk, to about 93% (the model has little data on this, n = 2). [C46] On all 242 training patients, its formula gives exactly what the part computes. [C48]

**Annotation hypothesis (technical).** Node 148 = ReLU(0.552 + 0.282*ca + 0.190*thal:4), read out with +1.893. Not a gate: it is active for 100% of training patients (min 0.330), so it is an affine extra channel. Its effect is to add +0.534 logit per SD of ca (number of fluoroscopy-coloured major vessels) on top of the direct ca edge (+2.065), giving ca an effective weight of 2.60, the largest in the network, plus +0.36 per SD of the rare thal:4 code (n=2 on train). Readout contribution sd 0.62 logits; replacing it by its mean drops train AUC 0.9437 -> 0.9408.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.282 x_{ca_b} \\
&\quad + 0.19 x_{thal:4} \\
&\quad + 0.552 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (latex: y_0 = \operatorname{relu}\bigl(0.282 x_{ca} + 0.19 x_{thal:4} + 0.552\bigr); plain: 148 = relu(0.282 ca + 0.190 thal:4 + 0.552); readout 148 -> 0 weight +1.893)
- *empirical_viz*: Never clips on train (activation rate 100%, min 0.330), so the ReLU is inert and 148 is exactly affine in ca and thal:4. Logit contribution +0.534 per SD of ca, on top of the direct ca edge (+2.065), i.e. effective ca weight 2.598. (n: 242; split: train; by_group: ca=-0.728 (n=139): 0.330; ca=0.394 (n=53): 0.646; ca=1.515 (n=33): 0.963; ca=2.636 (n=12): 1.280; thal:4 present (n=2): 2.428)


### Atypical angina offset (`ann_34`, `cp2_constant_offset`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: chest pain type
- Entry nodes: -5_b
- Exit nodes: 3229
- Subgraph nodes: -5_b, 3229

**Hypothesis (plain).** This part reads only whether the patient has atypical angina and hardly changes the predictions. [C51]

**Evidence (plain).** Through this part, atypical angina moves a typical patient's predicted risk from about 20% to about 19%, and switching it off hardly changes the predictions. [C49, C51] On all 242 training patients, its formula gives exactly what the part computes. [C52]

**Annotation hypothesis (technical).** Node 3229 = ReLU(0.947*cp:2 - 1.779), read out with -0.154. Because cp:2 is a standardised one-hot, the node is exactly an indicator of atypical angina (cp:2, n=42) with value 0.288, and 0 otherwise. Net effect: -0.044 logit for cp:2 patients, which simply adds to the direct cp:2 edge (-0.057 per SD). Practically inert (mean-ablation changes train AUC 0.9437 -> 0.9434).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.947 x_{cp:2_b} - 1.78 \right)}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (latex: y_0 = \operatorname{relu}\left(0.947 x_{cp:2} - 1.78\right); plain: 3229 = relu(0.947 cp:2 - 1.779); readout 3229 -> 0 weight -0.154)
- *empirical_viz*: Binary indicator of cp:2 with value 0.288; logit contribution -0.044 for cp:2 patients. Mean-ablation changes train AUC by 0.0003. (n: 242; split: train)

## Worth checking (draft)

- Worth checking: going from 0 to 2 vessels coloured moves a typical patient's predicted risk from about 20% to about 99%. [C01]
- Worth checking: patients with typical angina get the lowest predicted risk of all chest pain types, below patients with no chest pain symptoms. [C11, C16, C21]
- Worth checking: the model treats fasting blood sugar above 120 mg/dL as lowering the risk of heart disease (for a typical patient from about 20% to about 3%). [C20]
- Worth checking: the model treats a fixed defect on the thallium scan as lowering the risk compared with a normal scan (for a typical patient from about 20% to about 1%), from only 15 training patients. [C17]
- Worth checking: the model gives a higher predicted risk to the 2 training patients whose thallium result is coded 4, a code the dataset documentation does not define. [C03]
