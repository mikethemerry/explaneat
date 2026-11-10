# Explanation vb-acd834

- Genome: `1f2cf7ea-5c30-452a-9277-0b4043e888d9`
- Dataset: Heart Disease (binary, prepared) (training split 165cebbd-6609-46ca-867e-c09e17b2d6ec, n = 242; test n = 61)
- Outcome: heart disease (more than 50% narrowing of a major coronary artery on angiography)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 19%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 1.7763568394002505e-15
- closed_form_max_abs_err: 0.021697190439301295
- n_leaf_annotations: 3
- n_composition_annotations: 2
- n_splits: 10
- n_identity_nodes: 0

**Summary.** Essentially logistic regression (18 direct edges; logit SD 3.18) plus one small nonlinear ischaemia correction (logit SD 0.78): a ReLU gate on oldpeak, thal:7 and restecg:0 that adds to risk directly and feeds a resting-BP-offset gate read out directly, via a high-threshold amplifier, and via an age term active only without ischaemia evidence. Nonlinear parts add ~+0.010 train AUC (0.932 linear-only vs 0.942).

**What resisted.** Nothing resisted coverage. Validator only treats direct-child nodes as internal, forcing a split of gate 204 (and its 3 inputs) because it reads out to both the output and gate 290. Nodes 290/1834/1987 take inputs from several annotations so live only in the ischaemia_pathway composition. thal:6 chain 3671->2116 is affine and near-inert (weight -0.013 into 290).

## Plain-language summary (draft)

For a typical patient (a 56-year-old man with no chest pain symptoms and median test results) the model predicts a risk of heart disease of about 19%; it adds up most of its inputs independently, and two parts, built on the same signal from ST depression on exercise, the thallium scan and the resting ECG, add to that sum for 203 and 117 of 242 training patients. [C33, C47, C56]

**Inputs that add up.** These add up independently, and these lower the risk: being a woman moves a typical patient's predicted risk from about 19% to about 2% and fasting blood sugar above 120 mg/dL to about 10%; a higher maximum heart rate lowers it steadily, from about 26% at 118 to about 16% at 175 beats per minute; and through this sum alone, compared with no chest pain symptoms, typical angina lowers it to about 2%, non-anginal chest pain to about 4% and atypical angina to about 11% (chest pain type also acts through the ischaemia pathway, below). [C12, C18, C17, C08, C13, C21] Each extra vessel coloured by fluoroscopy adds about the same, from about 19% at 0 to about 88% at 2 vessels, flattening out near the top because the predicted risk cannot go much higher; a flat rather than upsloping ST segment raises it to about 59%, higher cholesterol raises it steadily from about 13% at 193 mg/dL to about 34% at 308 mg/dL, and chest pain brought on by exercise raises it to about 30%. [C01, C02, C05, C16] Through this sum alone, a reversible defect on the thallium scan raises it to about 33% and a resting ECG other than normal to about 26%, while higher resting blood pressure (from about 12% at 110 mm Hg to about 32% at 155 mm Hg) and older age (from about 17% at 42 to about 21% at 66 years) raise it steadily; all four also act through the parts below. [C15, C24, C06, C26] Some rarer results also count (the model has little data on this, n = 2 to 17): through this sum alone a thallium result coded 4 raises the risk to about 36% and a fixed defect on the thallium scan lowers it to about 10%, and a downsloping ST segment raises it to about 22%. [C10, C19, C27]

**Ischaemia signs with blood pressure and age.** This part combines a copy of the ST depression and thallium signal (below) with resting blood pressure, chest pain type, age and the thallium scan, not as a simple sum, and acts on 203 of 242 training patients, 102 of whom had heart disease; for the typical patient it acts on, the predicted risk is about 61% without it and about 63% with it. [C48, C61, C47] Through this part, more ST depression on exercise raises the predicted risk slowly at first, then faster, adding risk mainly once it rises above about 1.3 mm: for a typical patient from about 19% at 0 mm to about 36% at 2.8 mm. [C34, C44] Across the whole model, that rise is larger with a reversible defect on the thallium scan, from about 35% to about 84% against about 19% to about 48% when the scan shows neither a fixed nor a reversible defect; with that defect it is slow at first, then faster for most patients, but for patients with a normal resting ECG it is steady. [C42] It is also larger when resting blood pressure is at or below 130 mm Hg (from about 16% to about 45%, against about 25% to about 52% above 130 mm Hg), slow at first, then faster for most patients, but steady for patients with a normal resting ECG or a reversible defect on the thallium scan. [C43]

**ST depression and thallium signal.** This part reads ST depression on exercise, a reversible defect on the thallium scan and a normal resting ECG, and acts on 117 of 242 training patients, 77 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 88% without it and about 92% with it, and a copy of it also feeds the part above. [C56, C61] Through this part, more ST depression adds nothing until about 0.9 mm (between 0.7 and 1.7 mm for different patients), then raises the predicted risk steadily, for a typical patient from about 19% at 0 mm to about 28% at 2.8 mm; this holds for most patients, but for patients with a normal resting ECG or a reversible defect on the thallium scan it rises steadily from the start. [C49, C55] ST depression counts more with a normal resting ECG (from about 19% to about 48%, against about 26% to about 37% otherwise), rising slowly at first, then faster for most patients, but steadily for patients with a reversible defect on the thallium scan. [C53] A reversible defect on the thallium scan also raises the risk more with a normal resting ECG, from about 19% to about 44% against about 26% to about 43% otherwise. [C54]

## Parts, from the root down

### The whole model (`ann_40`, `heart_disease_logit`)

- Kind: composition; parent: none; children: st_ischaemia_gate_output, ischaemia_pathway
- Reads: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate reached during the exercise test, exercise-induced angina, ST depression during exercise (compared with rest), slope of the ST segment at peak exercise, number of major vessels coloured by fluoroscopy, thallium stress test result
- Entry nodes: -2, -4, -7, -9, -10, -15, -16, -17, -19, -21, -22, -23, -1_a, -6_a, -8_a, -12_a, -25_a, -26_a, -24, -20, -14, -13, -11, -3
- Exit nodes: 0
- Subgraph nodes: -2, -4, -7, -9, -10, -15, -16, -17, -19, -21, -22, -23, -1_a, -6_a, -8_a, -12_a, -25_a, -26_a, -24, -20, -14, -13, -11, -3, 0

**Hypothesis (plain).** The model adds up its inputs into one score: the vessels coloured (each extra vessel adding about the same), the ST segment, sex and chest pain type each move the predicted risk on their own, and two parts built on the same ST depression and thallium signal add to that sum. [C33, C01, C02, C12, C09]

**Evidence (plain).** It consists of the inputs that feed the score directly and two parts, the ischaemia pathway and the ST depression and thallium signal (each explained below); the score is their sum, checked on all 242 training patients. [C33] When one input at a time is changed across the whole model, more ST depression raises a typical patient's predicted risk slowly at first, then faster, from about 19% at 0 mm to about 48% at 2.8 mm, and a reversible defect on the thallium scan takes it to about 44%, against about 33% through the sum alone. [C03, C04, C15] Looking at the whole model, higher resting blood pressure raises it in an uneven way, from about 13% at 110 mm Hg to about 32% at 155 mm Hg, and older age raises it in an uneven way, from about 16% at 42 to about 23% at 66 years. [C07, C23]

**Annotation hypothesis (technical).** Root: logit z = -0.496 + LINEAR CORE + 0.117*st_ischaemia_gate_output + ischaemia_pathway readouts (0.135*290 + 0.165*1834 + 0.097*1987); prediction = sigmoid(z) (the genome output node is ReLU but the evaluator overrides it with sigmoid for binary classification, so it is a logistic score). LINEAR CORE (18 direct input->output edges, z-scored inputs): ca +1.51, sex:0 -1.08, slope:1 -0.90, chol +0.53, cp:1 -0.53, trestbps +0.49, cp:3 -0.47, thal:3 -0.43, exang:1 +0.43, slope:3 -0.41, thal:6 -0.38, cp:4 +0.33, fbs:0 +0.28, thalach -0.25, restecg:0 -0.18, exang:0 +0.15, age +0.11, thal:7 -0.07. Signs match cardiology: more fluoroscopy-coloured vessels, male sex, high cholesterol/BP, exertional angina and asymptomatic chest pain raise risk; upsloping ST (slope:1), normal thallium (thal:3), typical/non-anginal pain and high max heart rate lower it. 6 inputs are unused (sex:1, fbs:1, restecg:1, restecg:2, slope:2, thal:4 - mostly redundant one-hot complements or rare categories). COMPOSITION: purely additive in the logit - the two children add to the linear core. Child 1 (st_ischaemia_gate_output) adds a ReLU ischaemia score (oldpeak, thal:7, restecg:0); child 2 (ischaemia_pathway) adds a BP-adjusted, convexly amplified version of the same score, plus an age term that is active only when the score is quiet. oldpeak and thal:7 influence the output ONLY through these gates (oldpeak has no direct edge). EVIDENCE (train, n=242): full model AUC 0.942 / acc 0.876; linear core alone AUC 0.932 (nonlinear parts add +0.010 train AUC); logit SD of linear core 3.18 vs 0.78 for all nonlinear readouts combined. Removing individual readouts: 290 -0.002, 1834 -0.0016, 204_a -0.0016, 1987 -0.0007, whole pathway -0.0055 AUC. Whole-model SHAP (prob scale): ca 0.139, sex:0 0.105, slope:1 0.099, thal:3 0.045, cp:3 0.041, exang:1 0.039, oldpeak 0.038, chol 0.033, cp:4 0.032. The model is essentially a logistic regression with one small nonlinear ischaemia correction.

**Evidence records.**

- *performance_metrics*: Complete structural and compositional coverage with 10 function-preserving splits; explained network computes the same function as the evolved one. (split: train; train_n: 242; split_id: 165cebbd-6609-46ca-867e-c09e17b2d6ec; train_acc: 0.876; train_auc: 0.942; function_preserved_check: train AUC/acc/MSE identical before and after all splits)
- *ablation_study*: The linear core carries most of the signal; the nonlinear ischaemia machinery adds ~+0.01 train AUC. Test split not inspected (protocol). (split: train; method: local recomputation of logit decomposition z = linear core + sum(readout weight * node activation); verified sigmoid(z) equals model output to 2e-16; readouts removed without re-fitting; corr_nonlinear_vs_linear: 0.39)
- *shap_importance*: Top drivers ca, sex:0, slope:1 are all direct linear-core edges; oldpeak (gate-only input) ranks 7th. (n: 242; scale: probability; split: train; base_value: 0.45169222239986206)
- *analytical_formula*: Closed-form root: an 18-term linear logit plus four additive nonlinear readouts from the two children. (note: formula shows the genome output activation (ReLU); evaluation/training for binary classification overrides it with sigmoid, so the effective model is p = sigmoid(argument); composition: additive: linear core + 0.117*st_ischaemia_gate_output + 0.135*290 + 0.165*1834 + 0.097*1987 - 0.496)


### Ischaemia signs with blood pressure and age (`ann_39`, `ischaemia_pathway`)

- Kind: composition; parent: heart_disease_logit; children: st_ischaemia_gate_bp, thal6_linear_passthrough
- Reads: age, chest pain type, resting blood pressure, resting ECG, ST depression during exercise (compared with rest), thallium stress test result
- Entry nodes: -1_b, -5, -6_b, -8_b, -25_b, -12_d, -18_b, -26_d, -25_c
- Exit nodes: 290, 1834, 1987
- Subgraph nodes: -1_b, -5, -6_b, -8_b, -25_b, 290, 1834, 1987

**Hypothesis (plain).** This part raises the predicted risk with ST depression on exercise, slowly at first, then faster, and more so with a reversible defect on the thallium scan, but for patients with a normal resting ECG steadily. [C34, C42] Age adds a little through it, nothing until about 53 years, then steadily. [C36, C45]

**Evidence (plain).** It consists of a copy of the ST depression and thallium signal (the copy computes exactly what that part computes; explained below) and the fixed thallium defect relay, together with age, chest pain type, resting blood pressure and the thallium scan themselves, combined in a way that is not a simple sum. [C48, C61] It is active for 203 of 242 training patients, 102 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 61% without it and about 63% with it. [C47] Through it, more ST depression takes a typical patient from about 19% at 0 mm to about 36% at 2.8 mm, slowly at first, then faster, adding risk mainly once it rises above about 1.3 mm. [C34, C44] Across the whole model the rise is from about 35% to about 84% with a reversible defect on the thallium scan, against about 19% to about 48% when the scan shows neither a fixed nor a reversible defect, slow at first, then faster for most patients, but steady for patients with a normal resting ECG. [C42] Age adds nothing until about 53 years (between 53 and 66 years for different patients), then raises the risk steadily, for a typical patient from about 19% at 42 to about 21% at 66 years. [C36, C45]

**Annotation hypothesis (technical).** Serial composition of the ischaemia gate into a BP-adjusted gate and its readouts. (1) Gate 290 = ReLU(1.088*st_ischaemia_gate_bp - 0.817*trestbps + 0.151*cp:2 + 0.017*cp:3 + 0.046*thal:6 - 0.013*thal6_linear_passthrough - 0.485): it passes on the ischaemia evidence of 204_b, offset downward by resting blood pressure (high-BP patients need more ST-depression evidence to open it). (2) 1834 = ReLU(0.941*290 - 1.672): high-threshold amplifier, fires only when 290 > 1.78, making the pathway convex. (3) 1987 = ReLU(0.880*age - 1.585*290 + 0.179): an age term switched OFF by the gate, so age adds risk only for patients in whom 290 is quiet. Output contribution 0.135*290 + 0.165*1834 + 0.097*1987. EVIDENCE (train, n=242): 290 active 56% (100/135 of its active cases have 204 on; 35 open from low BP alone, mean predicted risk 0.13); 1834 active 26%, only when 204 and 290 are both on, and 78% of those patients are diseased; 1987 active in 64% of patients with 290 quiet vs 13% with 290 active. The thal:6 chain is functionally inert. Interpretation: the network has exactly one nonlinear mechanism - an ischaemia pathway that amplifies strong ST-depression/defect evidence and uses age as a fallback signal when that evidence is absent.

**Evidence records.**

- *analytical_formula*: Closed form of the composed pathway; 290 is a serial composition over the child gate, and 1834/1987 are readouts of 290 that respectively amplify strong ischaemia evidence and switch off the age term. (note: get_formula without force returned null; with force the non-collapsed latex omits child terms (tool bug); latex_collapsed is correct; readout: logit += 0.135*290 + 0.165*1834 + 0.097*1987 (204_a, the other copy of the ischaemia gate, adds 0.117*204 via the root); slope_of_logit_in_290: 0.135 baseline; +0.155 extra when 1834 on; -0.154 when 1987 on (age gate cancels the direct readout))
- *empirical_viz*: Empirics agree with the formula: 1834 fires only in the regime where both 204 and 290 are on (26% of patients, 78% of them diseased) - an amplifier for strong ischaemia evidence. 1987 is on for 64% of patients with 290 quiet but only 13% when 290 is active, confirming that age acts as a fallback risk signal that the ischaemia pathway switches off. 290 is opened by 204 in 100/135 of its active cases; 35 cases open it from low resting BP alone and these patients have low predicted risk (mean 0.13). (n: 242; split: train; tool_note: composition-level function evaluation (sensitivity/line by annotation_id) ignores child subgraphs, so node_id evaluations were used)


### Fixed thallium defect relay (`ann_38`, `thal6_linear_passthrough`)

- Kind: leaf; parent: ischaemia_pathway; children: none
- Reads: thallium stress test result
- Entry nodes: -25_c
- Exit nodes: 2116
- Subgraph nodes: -25_c, 3671, 2116

**Hypothesis (plain).** This part passes the thallium scan on to the ischaemia pathway and hardly changes the predictions. [C59]

**Evidence (plain).** Switching it off barely changes the predictions. [C59] On all 242 training patients, its formula gives exactly what the part computes. [C60]

**Annotation hypothesis (technical).** Serial chain thal:6 -> 3671 -> 2116: 3671 = ReLU(1.000*thal:6 + 2.123), 2116 = ReLU(1.003*3671 + 0.036). Since thal:6 >= -0.257 on the data both ReLUs are always active (100% on train), so 2116 = 1.003*thal:6 + 2.165 exactly (r = 1.000 on train). It feeds gate 290 with weight -0.013, contributing -0.013*thal:6 - 0.029 to its pre-activation: an always-on affine relay that is functionally a vestige.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 \operatorname{relu}{\left(1.0 x_{thal:6_c} + 2.12 \right)} \\
&\quad + 0.0355 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form reduces to an affine map over the data domain; the two ReLUs never clip. (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 \operatorname{relu}{\left(1.0 x_{thal:6_c} + 2.12 \right)} \\
&\quad + 0.0355 \bigr)
\end{aligned}; plain: 2116 = relu(1.003*relu(0.9996 thal:6 + 2.123) + 0.036); readout: 2116 -> 290 weight -0.0132 => net -0.0132 thal:6 - 0.0286 into 290; simplified_on_data: 2116 = 1.003 thal:6 + 2.165 (both ReLUs always active since thal:6 >= -0.257))
- *empirical_viz*: Both nodes are 100% active on training data and 2116 is perfectly linear in thal:6 (r = 1.000). The chain is a linear pass-through whose downstream weight (-0.013) makes it practically inert. (n: 242; split: train; line_thal6_vs_2116_correlation: 1.0)


### ST depression and thallium signal (`ann_36`, `st_ischaemia_gate_output`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: resting ECG, ST depression during exercise (compared with rest), thallium stress test result
- Entry nodes: -12_c, -18_a, -26_c
- Exit nodes: 204_a
- Subgraph nodes: -12_c, -18_a, -26_c, 204_a

**Hypothesis (plain).** This part raises the predicted risk with ST depression on exercise: nothing until about 0.9 mm (between 0.7 mm and 1.7 mm for different patients), then steadily, but for patients with a normal resting ECG or a reversible defect on the thallium scan steadily from the start. [C49, C55]

**Evidence (plain).** It is active for 117 of 242 training patients, 77 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 88% without it and about 92% with it. [C56] Through this part, more ST depression adds nothing until about 0.9 mm (between 0.7 and 1.7 mm for different patients), then raises the risk steadily, for a typical patient from about 19% at 0 mm to about 28% at 2.8 mm, but for patients with a normal resting ECG or a reversible defect on the thallium scan it rises steadily from the start. [C49, C55] ST depression counts more with a normal resting ECG (from about 19% to about 48%, against about 26% to about 37% otherwise), rising slowly at first, then faster for most patients, but steadily for patients with a reversible defect on the thallium scan. [C53] It also counts more with a reversible defect on the thallium scan (from about 35% to about 84%, against about 19% to about 48% otherwise), rising slowly at first, then faster for most patients, but for patients with a normal resting ECG steadily; and a reversible defect raises the risk more with a normal resting ECG, from about 19% to about 44% against about 26% to about 43%. [C52, C54] Its formula reproduces the part exactly on all 242 training patients. [C57]

**Annotation hypothesis (technical).** Hidden gate 204 = ReLU(2.554*oldpeak + 1.133*thal:7 + 0.937*restecg:0 + 0.276): a one-sided ischaemia-evidence detector that switches on for patients with exercise-induced ST depression and/or a reversible thallium defect, normal resting ECG adding a little. EVIDENCE (train, n=242): hard-zero for 52%, active for 48%; mean |grad| oldpeak 1.23 > thal:7 0.55 > restecg:0 0.45; oldpeak alone correlates 0.88 with the gate. This is the output-facing copy 204_a (bit-identical to 204_b); it reads out directly to the output with +0.117, raising predicted risk.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.55 x_{oldpeak_a} \\
&\quad + 0.937 x_{restecg:0_b_c} \\
&\quad + 1.13 x_{thal:7_b_c} \\
&\quad + 0.276 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form of a single ReLU gate over three inputs; oldpeak weight is 2.3x/2.7x the others. (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.55 x_{oldpeak_a} \\
&\quad + 0.937 x_{restecg:0_b_c} \\
&\quad + 1.13 x_{thal:7_b_c} \\
&\quad + 0.276 \bigr)
\end{aligned}; plain: 204_a = relu(2.554 oldpeak + 1.133 thal:7 + 0.937 restecg:0 + 0.276); readout: 204_a -> 0 weight +0.117; interpretation: one-sided ischaemia-evidence detector dominated by ST depression (oldpeak); all weights positive so monotone increasing in each input)
- *empirical_viz*: Genuine gate on training data: hard-zero for 52% of patients, active for 48%. Mean |grad| ranks oldpeak >> thal:7 > restecg:0; oldpeak alone correlates 0.88 with the gate output. (n: 242; split: train; line_oldpeak_vs_gate_correlation: 0.8796682565541298)

## Worth checking (draft)

- Worth checking: the model treats fasting blood sugar above 120 mg/dL as lowering the risk of heart disease (for a typical patient from about 19% to about 10%). [C18]
- Worth checking: patients with typical angina get the lowest predicted risk of all chest pain types, below patients with no chest pain symptoms. [C09, C14, C22]
- Worth checking: the model treats a fixed defect on the thallium scan as lowering the risk compared with a normal scan, from only 15 training patients. [C20]
- Worth checking: the model gives a higher predicted risk to the 2 training patients whose thallium result is coded 4, a code the dataset documentation does not define. [C11]
- Worth checking: ST depression on exercise and a reversible thallium defect raise the predicted risk more when the resting ECG is normal than when it is not. [C53, C54]
- Worth checking: through one part, ST depression on exercise raises the predicted risk more when resting blood pressure is at or below 130 mm Hg than above it. [C43]
