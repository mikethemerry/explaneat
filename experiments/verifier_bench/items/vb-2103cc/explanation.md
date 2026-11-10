# Explanation vb-2103cc

- Genome: `0057df7a-62a2-43c3-b107-9afe3735a7c3`
- Dataset: Heart Disease (binary, prepared) (training split 165cebbd-6609-46ca-867e-c09e17b2d6ec, n = 242; test n = 61)
- Outcome: heart disease (more than 50% narrowing of a major coronary artery on angiography)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 28%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 1.7763568394002505e-15
- closed_form_max_abs_err: 0.011048694034678341
- n_leaf_annotations: 3
- n_composition_annotations: 1
- n_splits: 5
- n_identity_nodes: 0

**Summary.** Mostly logistic regression: 16 direct edges (logit SD 3.28, train AUC 0.930 alone) plus three additive nonlinear corrections (logit SD 0.91) to 0.943: a protective heart-rate-reserve gate ReLU(-1.52 - 1.49 age + 1.02 thalach - 1.54 thal:7) read out -0.58 (largest effect); a flat-ST-slope gate scaled by oldpeak and down by chol; a convex two-kink oldpeak dose-response. Age and oldpeak reach the output only via these gates.

**What resisted.** Nothing resisted coverage. Clinically counter-directional signs reported, not explained: cp:2 +0.42, fbs:1 -0.31 in the linear core, negative chol inside the flat-slope gate. Relays 3195 (thal:7) and 3464 (chol) are affine on the data (100%/93% active).

## Plain-language summary (draft)

For a typical patient (a 56-year-old man with no chest pain symptoms and median test results) the model predicts a risk of heart disease of about 28%; it adds up most of its inputs independently, and three parts add to that sum: one acts on 80 of 242 training patients through age and maximum heart rate, one on 121 through ST depression on exercise, and one on 117 through the ST segment, ST depression and cholesterol. [C29, C38, C42, C52]

**Inputs that add up.** These add up independently, and these lower the risk: compared with no chest pain symptoms, typical angina takes a typical patient's predicted risk from about 28% to about 3%, non-anginal chest pain to about 5% and atypical angina to about 14%; being a woman lowers it to about 3% and fasting blood sugar above 120 mg/dL to about 14%; and through this sum alone a higher maximum heart rate lowers it steadily, from about 32% at 118 to about 27% at 175 beats per minute (maximum heart rate also acts through the age and heart rate gate, below). [C06, C09, C17, C07, C18, C23] Each extra vessel coloured by fluoroscopy adds about the same, from about 28% at 0 to about 94% at 2 vessels, flattening out near the top because the predicted risk cannot go much higher; higher resting blood pressure raises it steadily, from about 19% at 110 mm Hg to about 43% at 155 mm Hg, and chest pain brought on by exercise raises it to about 42%. [C01, C08, C19] Through this sum alone, a flat rather than upsloping ST segment raises it to about 73%, higher cholesterol raises it steadily from about 18% at 193 mg/dL to about 49% at 308 mg/dL, and a reversible defect on the thallium scan raises it to about 42%; all three also act through the parts below. [C03, C04, C20] Some rarer results also count (the model has little data on this, n = 2 to 17): through this sum alone a downsloping ST segment raises the risk to about 51% and a fixed defect on the thallium scan lowers it to about 11%, and an ST-T wave abnormality on the resting ECG raises it to about 46%. [C10, C14, C13]

**Age and heart rate gate.** This part reads age, maximum heart rate and the thallium scan, and acts on 80 of 242 training patients, 9 of whom had heart disease; for the typical patient it acts on, the predicted risk is about 4% without it and about 1% with it. [C38] For patients without a reversible defect on the thallium scan, it switches on when age in years minus about 0.28 × the maximum heart rate in beats per minute is below about 10.9 (for maximum heart rates from about 71 to 202 beats per minute); 80 of 242 training patients are on that side, and 11% of them had heart disease, against 60% of the others. [C54] Through this part, older age raises the predicted risk steadily up to about 56 years (between 45 and 61 years for different patients), then no further for most patients, but for patients with a reversible defect on the thallium scan it rises in a jump: for a typical patient from about 12% at 42 to about 28% at 66 years. [C30, C37] Looking at the whole model, a higher maximum heart rate lowers the risk in an uneven way, more without a reversible defect on the thallium scan (from about 32% at 118 to about 20% at 175 beats per minute, against about 46% to about 40% with one). [C35]

**ST depression hinges.** This part reads only ST depression on exercise and acts on 121 of 242 training patients, 77 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 89% without it and about 91% with it. [C42] Through this part, more ST depression raises the predicted risk slowly at first, then faster, adding risk only once it rises above about 0.8 mm: for a typical patient from about 28% at 0 mm to about 47% at 2.8 mm. [C40, C41]

**Flat ST segment with cholesterol.** This part reads the slope of the ST segment, ST depression on exercise and cholesterol, and acts on 117 of 242 training patients, 78 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 87% without it and about 90% with it. [C52] For patients whose ST segment is not flat, it switches on when ST depression on exercise is above about 2.23 mm (for cholesterol from about 126 to 186 mg/dL), or when ST depression minus about 0.012 × the cholesterol in mg/dL is above about 0.03 (for cholesterol from about 186 to 417 mg/dL); 117 of 242 training patients are on, and 67% of them had heart disease, against 23% of the others. [C55] Looking at the whole model, more ST depression raises the risk slowly at first, then faster, and even more so with a flat ST segment: from about 75% to about 91% then, against about 28% to about 47% otherwise. [C48] Higher cholesterol raises it steadily across the whole model, from about 67% to about 88% with a flat ST segment and from about 18% to about 49% otherwise, while through this part it lowers the risk in an uneven way, for the typical patient the part acts on from about 90% to about 89%. [C47, C46]

## Parts, from the root down

### The whole model (`ann_34`, `heart_disease_logit`)

- Kind: composition; parent: none; children: age_hr_reserve_gate, st_slope_chol_gate, oldpeak_hinges
- Reads: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate reached during the exercise test, exercise-induced angina, ST depression during exercise (compared with rest), slope of the ST segment at peak exercise, number of major vessels coloured by fluoroscopy, thallium stress test result
- Entry nodes: -2, -3, -4, -5, -7, -8, -9_a, -11, -13, -15_a, -17, -19, -20_a, -22, -25, -26_a, -6, -10, -12, -14, -16, -21, -23, -24, -1, -15_b, -26_b, -26_c, -9_b, -18_a, -20_b, -18_b
- Exit nodes: 0
- Subgraph nodes: -2, -3, -4, -5, -7, -8, -9_a, -11, -13, -15_a, -17, -19, -20_a, -22, -25, -26_a, -6, -10, -12, -14, -16, -21, -23, -24, 0

**Hypothesis (plain).** The model adds up its inputs into one score: the vessels coloured (each extra vessel adding about the same), the ST segment, chest pain type and sex each move the predicted risk on their own, and three parts add to that sum: one for age and maximum heart rate, one for ST depression and one for the ST segment with cholesterol. [C29, C01, C02, C06, C07]

**Evidence (plain).** It consists of the inputs that feed the score directly and three parts, the age and heart rate gate, the ST depression hinges and the flat ST segment with cholesterol part (each explained below); the score is their sum, checked on all 242 training patients. [C29] Age and ST depression on exercise do not feed the score directly, so they act only through these parts. [C29, C30, C40] When one input at a time is changed across the whole model, a flat rather than upsloping ST segment moves a typical patient's predicted risk from about 28% to about 77%, against about 73% through the sum alone. [C02, C03] Looking at the whole model, a higher maximum heart rate lowers it in an uneven way, from about 32% at 118 to about 20% at 175 beats per minute, against about 27% through the sum alone, where it falls steadily. [C22, C23] Looking at the whole model, a fixed defect on the thallium scan lowers it to about 11% (the model has little data on this, n = 15), and a reversible defect raises it to about 42%. [C15, C21]

**Annotation hypothesis (technical).** Root: logit z = -0.210 + LINEAR CORE - 0.582*age_hr_reserve_gate + 0.225*st_slope_chol_gate + 0.377*oldpeak_hinges[1946] + 0.411*oldpeak_hinges[3284]; prediction = sigmoid(z) (the genome output node is shown as ReLU, but for binary classification the evaluator applies a sigmoid, so z is a logistic score). LINEAR CORE (16 direct input->output edges, z-scored inputs): ca +1.68, cp:4 +0.99, sex:0 -0.74, chol +0.61, slope:2 +0.49, slope:1 -0.47, trestbps +0.47, sex:1 +0.45, cp:2 +0.42, fbs:1 -0.31, thal:7 +0.29, exang:1 +0.28, thal:6 -0.28, cp:1 -0.18, thalach -0.10, restecg:1 +0.07. The signs mostly match cardiology: more fluoroscopy-coloured vessels (ca), asymptomatic chest pain (cp:4), male sex, higher cholesterol and resting BP, flat ST slope and exertional angina raise risk, and upsloping ST (slope:1) lowers it. Exceptions are cp:2 +0.42 and fbs:1 -0.31, which run against the usual clinical direction and are left unexplained. Eight inputs are unused (cp:3, fbs:0, restecg:0, restecg:2, exang:0, slope:3, thal:3, thal:4). Age and oldpeak have no direct edge: they act only through the gates. COMPOSITION: purely additive in the logit. The three children add three nonlinear corrections to the linear core: (1) age_hr_reserve_gate, a protective ReLU that is on when max heart rate is high for the patient's age and there is no reversible thallium defect; (2) st_slope_chol_gate, a flat-ST-slope detector scaled up by oldpeak and down by cholesterol; (3) oldpeak_hinges, a convex two-kink dose-response in ST depression. The children share no hidden nodes. They are coupled only through shared input features (thal:7, thalach, chol and slope:2 also have direct edges, and oldpeak feeds both (2) and (3)). EVIDENCE (train, n=242): full model AUC 0.943, acc 0.884 (sigmoid(z) reproduces the evaluator exactly). Linear core alone gives AUC 0.930, so the nonlinear children add +0.013. Logit SD is 3.28 for the linear core vs 0.91 for all nonlinear readouts. Removing one readout at a time: 1718 -0.0057, 1476 -0.0017, 1946 -0.0007, 3284 -0.0004 AUC; all oldpeak-driven terms together -0.0064. Top SHAP features: ca, cp:4, sex:0, slope:2, slope:1 (all with direct edges). Interpretation: close to logistic regression, plus a protective heart-rate-reserve-for-age gate (blocked by a reversible defect) and two oldpeak/flat-slope ischaemia corrections.

**Evidence records.**

- *analytical_formula*: Root closed form: additive logit of linear core plus three child readouts. (note: from get_formula force=true latex_collapsed; the tool renders the output as relu(z), but for binary classification the evaluator applies sigmoid: p = sigmoid(z). Confirmed locally: sigmoid(z) reproduces compute_performance train AUC 0.9434 / acc 0.884 exactly.)
- *performance_metrics*: Full coverage with 5 input splits; the explained network computes the same function. (n: 242; auc: 0.9434; split: train; accuracy: 0.884; function_preservation_max_abs_err: 1.78e-15)
- *ablation_study*: Linear core carries most of the signal; the nonlinear children add +0.013 train AUC, mostly via the heart-rate-reserve gate. (split: train; method: local logit decomposition z = linear core + sum(readout weight * node activation), readouts removed without re-fitting; corr_nonlinear_vs_linear: 0.54)
- *shap_importance*: Top drivers are direct-edge features; oldpeak (gate-only) ranks 10th and age (gate-only) 0.013. (n: 242; scale: probability; split: train; base_value: 0.448)


### Age and heart rate gate (`ann_31`, `age_hr_reserve_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: age, maximum heart rate reached during the exercise test, thallium stress test result
- Entry nodes: -1, -15_b, -26_b, -26_c
- Exit nodes: 1718
- Subgraph nodes: -1, -15_b, -26_b, -26_c, 3195, 1718

**Hypothesis (plain).** This part lowers the predicted risk for patients whose age is low relative to their maximum heart rate; for the typical patient it acts on, a reversible defect on the thallium scan raises the risk through it from about 1% to about 4%. [C54, C32]

**Evidence (plain).** For patients without a reversible defect on the thallium scan, it switches on when age in years minus about 0.28 × the maximum heart rate in beats per minute is below about 10.9 (for maximum heart rates from about 71 to 202 beats per minute); 80 of 242 training patients are on that side, and 11% of them had heart disease, against 60% of the others. [C54] Through it, older age raises the predicted risk steadily up to about 56 years (between 45 and 61 years for different patients), then no further for most patients, but for patients with a reversible defect on the thallium scan it rises in a jump: for a typical patient from about 12% at 42 to about 28% at 66 years. [C30, C37] Looking at the whole model, older age raises the risk, then no further, only if the thallium scan shows no reversible defect: from about 12% to about 28%, while with one it stays at about 42%. [C33] A higher maximum heart rate lowers the risk through it in an uneven way, from about 28% at 118 to about 22% at 175 beats per minute, and across the whole model it lowers it more without a reversible defect (from about 32% to about 20%, against about 46% to about 40%). [C31, C35] On all 242 training patients, its formula gives exactly what the part computes. [C39]

**Annotation hypothesis (technical).** Heart-rate-reserve gate 1718 = ReLU(-0.812 - 1.490*age + 1.024*thalach - 0.762*thal:7 - 0.720*R3195), with relay R3195 = ReLU(0.981 + 1.073*thal:7). Hypothesis: 3195 is an always-on affine relay of thal:7 (thal:7 is one-hot, z-scored to {-0.769, 1.300}, so 3195 in {0.155, 2.376} > 0), which makes the gate's net thal:7 coefficient -1.535 and bias -1.518. The gate fires for patients whose achieved max heart rate is high for their age (thalach up, age down) AND who do NOT have a reversible thallium defect; its readout into the output is -0.582, so it is a protective (risk-lowering) term. EVIDENCE (train, n=242): 3195 is active for 100% of patients, so the gate reduces exactly to 1718 = ReLU(-1.518 - 1.490*age + 1.024*thalach - 1.535*thal:7). Active 33% overall: 51% of patients without a reversible defect vs 3% with one. Disease rate 11% when active vs 60% when inactive. Mean |sensitivity|: age 0.49 > thalach 0.34 > thal:7 (two copies) 0.25 each. Removing its readout drops train AUC 0.943 -> 0.938, the largest single nonlinear effect (readout logit SD 0.64). Supported.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.49 x_{age} \\
&\quad - 0.762 x_{thal:7_b} \\
&\quad + 1.02 x_{thalach_b} \\
&\quad - 0.72 \operatorname{relu}{\left(1.07 x_{thal:7_c} + 0.981 \right)} \\
&\quad -0.812 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form; relay 3195 is affine on the data so the gate is a single ReLU of age, thalach and thal:7. (latex: y_0 = \operatorname{relu}(-1.49 x_{age} - 0.762 x_{thal:7_b} + 1.02 x_{thalach_b} - 0.72\,\operatorname{relu}(1.07 x_{thal:7_c} + 0.981) - 0.812); readout: logit += -0.582*1718; simplified: 3195 always active on data (thal:7 in {-0.769,1.300} => 3195 in {0.155,2.376}); exact affine reduction: 1718 = relu(-1.518 - 1.490 age + 1.024 thalach - 1.535 thal:7). Pre-activation bias is -0.337 without reversible defect and -3.514 with it.)
- *empirical_viz*: Gate is a protective 'good heart-rate reserve for age, no reversible defect' detector. (n: 242; viz: sensitivity (max_samples 100000); split: train; outcome: disease rate 11.3% when gate active vs 60.5% when inactive; ablation: removing the 1718 readout lowers train AUC 0.9434 -> 0.9378 (largest single nonlinear term); readout logit SD 0.64; activity: 1718 active 33.1% overall; 50.7% of the 152 patients without reversible defect (thal:7=0) vs 3.3% of the 90 with it)


### ST depression hinges (`ann_33`, `oldpeak_hinges`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: ST depression during exercise (compared with rest)
- Entry nodes: -18_b
- Exit nodes: 1946, 3284
- Subgraph nodes: -18_b, 1946, 3284

**Hypothesis (plain).** This part increases the predicted risk with ST depression on exercise, slowly at first, then faster, adding risk only once it rises above about 0.8 mm. [C40, C41]

**Evidence (plain).** It acts on 121 of 242 training patients, 77 of whom had heart disease; for the typical patient it acts on, the predicted risk is about 89% without it and about 91% with it. [C42] Through it, more ST depression raises the risk slowly at first, then faster, adding risk only once it rises above about 0.8 mm: for a typical patient from about 28% at 0 mm to about 47% at 2.8 mm. [C40, C41] Its formula reproduces the part exactly on all 242 training patients. [C43]

**Annotation hypothesis (technical).** Oldpeak hinge pair: 1946 = ReLU(0.167 + 0.767*oldpeak) and 3284 = ReLU(-0.582 + 1.003*1946). Hypothesis: a convex piecewise-linear function of ST depression alone, with kinks at z-oldpeak = -0.218 (1946 opens) and 0.539 (3284 opens). Both exits read out positively (0.377 and 0.411), so beyond the second kink the logit slope in oldpeak roughly doubles: mild ST depression adds a little risk, marked ST depression adds much more. EVIDENCE (train, n=242): 1946 active 50% (disease rate 64%), 3284 active 24% (disease rate 79%). Contribution to the logit 0.377*1946 + 0.411*3284 has slope 0 below z-oldpeak -0.218, 0.289 between the kinks and 0.605 above 0.539. Mean |sensitivity| to oldpeak: 0.38 for 1946, 0.18 for 3284. Removing either readout changes train AUC by <= 0.001. Supported: a small convex ST-depression dose-response.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.767 x_{oldpeak_b} + 0.167 \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 1.0 \operatorname{relu}{\left(0.767 x_{oldpeak_b} + 0.167 \right)} \\
&\quad -0.582 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form of the hinge pair; a convex piecewise-linear function of oldpeak. (latex: y_0 = \operatorname{relu}(0.767 x_{oldpeak_b} + 0.167), \quad y_1 = \operatorname{relu}(1.0\,\operatorname{relu}(0.767 x_{oldpeak_b} + 0.167) - 0.582); readout: logit += 0.377*1946 + 0.411*3284: slope 0 below -0.218, 0.289 between kinks, 0.605 above 0.539 (per z-unit oldpeak); simplified: kinks at z-oldpeak = -0.218 (1946) and 0.539 (3284); 3284 = relu(0.769 oldpeak - 0.415))
- *empirical_viz*: Hinges confirmed: oldpeak raises risk only above about the median, and more steeply at high values. (n: 242; viz: sensitivity (max_samples 100000); split: train; ablation: removing 1946 readout: AUC 0.9434 -> 0.9427; removing 3284 readout: -> 0.9430; activity: 1946 active 50% (121 patients, disease rate 63.6%); 3284 active 24% (58 patients, disease rate 79.3%))


### Flat ST segment with cholesterol (`ann_32`, `st_slope_chol_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: serum cholesterol, ST depression during exercise (compared with rest), slope of the ST segment at peak exercise
- Entry nodes: -9_b, -18_a, -20_b
- Exit nodes: 1476
- Subgraph nodes: -9_b, -18_a, -20_b, 3464, 1476

**Hypothesis (plain).** This part raises the predicted risk mainly for patients with a flat ST segment, and makes ST depression on exercise count more for them, rising slowly at first, then faster. [C44, C48]

**Evidence (plain).** It acts on 117 of 242 training patients, 78 of whom had heart disease; for the typical patient it acts on, the predicted risk is about 87% without it and about 90% with it. [C52] For patients whose ST segment is not flat, it switches on when ST depression on exercise is above about 2.23 mm (for cholesterol from about 126 to 186 mg/dL), or when ST depression minus about 0.012 × the cholesterol in mg/dL is above about 0.03 (for cholesterol from about 186 to 417 mg/dL); 117 of 242 training patients are on, and 67% of them had heart disease, against 23% of the others. [C55] Through it, a flat ST segment raises a typical patient's predicted risk from about 28% to about 33%; for the typical patient it acts on, more ST depression raises it in an uneven way from about 88% to about 91%, and higher cholesterol lowers it in an uneven way from about 90% to about 89%. [C44, C45, C46] Looking at the whole model, more ST depression raises the risk slowly at first, then faster, from about 75% to about 91% with a flat ST segment, against about 28% to about 47% otherwise. [C48] Its formula reproduces the part exactly on all 242 training patients. [C53]

**Annotation hypothesis (technical).** Flat-ST-slope gate 1476 = ReLU(0.257 + 1.145*slope:2 + 0.770*oldpeak - 0.374*C3464), with C3464 = ReLU(1.300 + 1.011*chol). Hypothesis: a one-sided risk detector that opens for patients with a flat peak-exercise ST segment (slope:2), its height scaled by ST depression (oldpeak) and reduced by cholesterol via an almost-always-active affine chol relay (3464 is zero only when z-chol < -1.29). Readout into output +0.225 (raises risk). EVIDENCE (train, n=242): 3464 is active for 93% (off only when z-chol < -1.29), so 1476 is nearly ReLU(-0.229 + 1.145*slope:2 + 0.770*oldpeak - 0.378*chol). Active for 98% of the 110 flat-slope patients vs 7% of the others, so it acts as a flat-slope indicator whose height rises with oldpeak (corr 0.69) and falls with cholesterol. Disease rate 67% when active vs 23% when inactive. Mean |sensitivity|: slope:2 0.55 > oldpeak 0.37 > chol 0.17. Removing its readout drops train AUC 0.943 -> 0.942 (logit SD 0.18). Supported. The negative chol term partly offsets chol's direct +0.613 edge for flat-slope patients; no clinical reading is offered for that.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.77 x_{oldpeak_a} \\
&\quad + 1.15 x_{slope:2_b} \\
&\quad - 0.374 \operatorname{relu}{\left(1.01 x_{chol_b} + 1.3 \right)} \\
&\quad + 0.257 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form; the chol relay is near-affine. (latex: y_0 = \operatorname{relu}(0.77 x_{oldpeak_a} + 1.15 x_{slope:2_b} - 0.374\,\operatorname{relu}(1.01 x_{chol_b} + 1.3) + 0.257); readout: logit += 0.225*1476; simplified: 3464 active 92.6% (zero only for z-chol < -1.29); where active, 1476 = relu(-0.229 + 0.770 oldpeak + 1.145 slope:2 - 0.378 chol). slope:2 in {-0.913, 1.095} so slope:2 shifts the pre-activation by 2.30 between categories.)
- *empirical_viz*: Gate is essentially a flat-ST-slope indicator whose magnitude grows with ST depression and shrinks with cholesterol. (n: 242; viz: sensitivity (max_samples 100000); split: train; outcome: disease rate 66.7% when active vs 23.2% when inactive; corr(1476, oldpeak)=0.69; ablation: removing the 1476 readout lowers train AUC 0.9434 -> 0.9418; readout logit SD 0.18; activity: 1476 active 48.3%; 98.2% of the 110 flat-slope (slope:2) patients vs 6.8% of the others)

## Worth checking (draft)

- Worth checking: going from 0 to 2 vessels coloured moves a typical patient's predicted risk from about 28% to about 94%. [C01]
- Worth checking: patients with typical angina get the lowest predicted risk of all chest pain types, below patients with no chest pain symptoms. [C06, C09, C17]
- Worth checking: the model treats fasting blood sugar above 120 mg/dL as lowering the risk of heart disease (for a typical patient from about 28% to about 14%). [C18]
- Worth checking: the model treats a fixed defect on the thallium scan as lowering the risk (for a typical patient from about 28% to about 11%), from only 15 training patients. [C15]
- Worth checking: across the whole model, older age raises the predicted risk only for patients without a reversible defect on the thallium scan, and through one part only up to about 56 years. [C33, C37]
