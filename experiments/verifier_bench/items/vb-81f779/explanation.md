# Explanation vb-81f779

- Genome: `7c355ee7-77ad-426c-8a6d-9df81d1c65b3`
- Dataset: Heart Disease (binary, prepared) (training split 165cebbd-6609-46ca-867e-c09e17b2d6ec, n = 242; test n = 61)
- Outcome: heart disease (more than 50% narrowing of a major coronary artery on angiography)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 12%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 1.7763568394002505e-15
- closed_form_max_abs_err: 0.02693294224879228
- n_leaf_annotations: 3
- n_composition_annotations: 1
- n_splits: 6
- n_identity_nodes: 0

**Summary.** Essentially logistic regression: 19 direct edges (logit SD 4.90; ca +2.61, female -1.57) plus three one-sided ReLU corrections: an ischaemia gate on oldpeak and thal:7 (SD 0.78); a Boolean AND of female sex and exercise angina adding +2.07 logit, cancelling most of the female protective term (SD 0.56); a protective gate on high thalach mostly with non-anginal pain (SD 0.49). Linear core train AUC 0.930 vs 0.947 full; mean-ablating each gate gives 0.938, 0.940, 0.945.

**What resisted.** Nothing resisted coverage. 4 hidden nodes form 3 independent one-sided ReLU gates feeding the output. 6 input splits (fbs:0, thal:7, sex:0, exang:1, cp:3, thalach). Weak spots: thal:3->3818 branch nearly inert; in gate 557 slope:3 suppresses risk (clinically counter-intuitive; rare 7%, no direct edge) - reported as what the network computes. Unconnected inputs (cp:2, fbs:1, restecg:1, restecg:2) added to root entries.

## Plain-language summary (draft)

For a typical patient (a 56-year-old man with no chest pain symptoms and median test results) the model predicts a risk of heart disease of about 12%; it adds up most of its inputs independently, and three parts add to that sum: one acts on 116 of 242 training patients through ST depression on exercise, one on 74 through maximum heart rate, and one only on women whose chest pain is brought on by exercise (the model has little data on this, n = 19). [C36, C46, C60, C51]

**Inputs that add up.** These add up independently, and these lower the risk: through this sum alone, being a woman moves a typical patient's predicted risk from about 12% to below 1%, and compared with no chest pain symptoms typical angina lowers it to below 1%, non-anginal chest pain to about 1% and atypical angina to about 7%, and fasting blood sugar above 120 mg/dL to about 2%; a higher maximum heart rate lowers it steadily, from about 22% at 118 to about 8% at 175 beats per minute; sex, chest pain type, fasting blood sugar and maximum heart rate also act through the parts below. [C15, C17, C20, C25, C21, C13] Each extra vessel coloured by fluoroscopy adds about the same, from about 12% at 0 to about 98% at 2 vessels, flattening out near the top because the predicted risk cannot go much higher; higher cholesterol raises it steadily, from about 5% at 193 mg/dL to about 31% at 308 mg/dL, flattening out near the bottom because the predicted risk cannot go much lower; higher resting blood pressure (from about 7% at 110 mm Hg to about 22% at 155 mm Hg) and older age (from about 10% at 42 to about 13% at 66 years) raise it steadily, and a resting ECG other than normal raises it to about 14%. [C01, C08, C12, C27, C28] Through this sum alone, a flat rather than upsloping ST segment raises it to about 76% and a reversible defect on the thallium scan to about 27%; the ST segment and the thallium scan also act through the parts below. [C04, C11] Some rarer results also count (the model has little data on this, n = 2 to 17): through this sum alone a thallium result coded 4 raises the risk to about 90% and a downsloping ST segment to about 42%, and a fixed defect on the thallium scan lowers it to about 3%. [C02, C06, C22]

**ST depression gate.** This part reads ST depression on exercise, the slope of the ST segment, the thallium scan and fasting blood sugar, and acts on 116 of 242 training patients, 77 of whom had heart disease. [C46] Through it, more ST depression raises the predicted risk in an uneven way, mainly once it rises above about 1.2 mm (between 0.2 and 1.2 mm for different patients): for a typical patient from about 12% at 0 mm to about 32% at 2.8 mm. [C37, C45] Looking at the whole model, the rise is larger with a reversible defect on the thallium scan, from about 27% to about 76% against about 12% to about 32% otherwise, in an uneven way. [C41] Through this part, a reversible defect on the thallium scan and fasting blood sugar above 120 mg/dL each raise a typical patient's predicted risk from about 12% to about 18%. [C39, C38]

**Women with chest pain on exercise.** This part switches on only for women whose chest pain is brought on by exercise, 19 of 242 training patients, 12 of whom had heart disease, and raises the predicted risk of the typical such patient from about 9% to about 43% (the model has little data on this, n = 19). [C51] Looking at the whole model, being a woman lowers a typical patient's predicted risk from about 12% to below 1% when exercise does not bring on chest pain, and to about 2% when it does. [C50]

**High heart rate protective gate.** This part reads maximum heart rate, chest pain type and the thallium scan, and acts on 74 of 242 training patients, 13 of whom had heart disease; for the typical patient it acts on, the predicted risk is about 1% without it and below 1% with it. [C60] Through this part, a higher maximum heart rate changes nothing until about 147 beats per minute, then lowers the predicted risk steadily, for the typical patient among those it acts on from about 1% to below 1%, flattening out near the bottom because the predicted risk cannot go much lower. [C54] Looking at the whole model, a higher maximum heart rate lowers the risk slowly at first, then faster, more with non-anginal chest pain (from about 3% to below 1%, against about 22% to about 8% otherwise), flattening out near the bottom because the predicted risk cannot go much lower. [C58] Through this part, non-anginal chest pain moves a typical patient's predicted risk from about 12% to about 9%. [C53]

## Parts, from the root down

### The whole model (`ann_35`, `heart_disease_logit`)

- Kind: composition; parent: none; children: st_depression_gate, female_exang_and_gate, thalach_cp3_protective_gate
- Reads: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate reached during the exercise test, exercise-induced angina, ST depression during exercise (compared with rest), slope of the ST segment at peak exercise, number of major vessels coloured by fluoroscopy, thallium stress test result
- Entry nodes: -1, -2_a, -3, -4, -6_a, -7, -8, -9, -10_a, -12, -15_a, -16, -17_a, -19, -20, -22, -24, -25, -26_a, -5, -11, -13, -14, -10_b, -18, -21, -26_b, -2_b, -17_b, -6_b, -15_b, -23
- Exit nodes: 0
- Subgraph nodes: -1, -2_a, -3, -4, -6_a, -7, -8, -9, -10_a, -12, -15_a, -16, -17_a, -19, -20, -22, -24, -25, -26_a, -5, -11, -13, -14, 0

**Hypothesis (plain).** The model adds up its inputs into one score: the vessels coloured (each extra vessel adding about the same), the ST segment, sex and chest pain type each move the predicted risk on their own, and three parts add to that sum: one for ST depression on exercise, one for women with chest pain on exercise and one for maximum heart rate. [C36, C01, C05, C16, C18]

**Evidence (plain).** It consists of the inputs that feed the score directly and three parts, the ST depression gate, women with chest pain on exercise and the high heart rate protective gate (each explained below); the score is their sum, checked on all 242 training patients. [C36] ST depression on exercise does not feed the score directly, so it acts only through the ST depression gate, where it raises the risk in an uneven way. [C36, C37] When one input at a time is changed across the whole model, a reversible defect on the thallium scan moves a typical patient's predicted risk from about 12% to about 37%, against about 27% through the sum alone, and fasting blood sugar above 120 mg/dL lowers it to about 3%, against about 2% through the sum alone. [C09, C11, C24, C21] Looking at the whole model, being a woman lowers it to below 1%, and compared with no chest pain symptoms typical angina lowers it to below 1%, non-anginal chest pain to about 1% and atypical angina to about 7%. [C16, C18, C19, C26]

**Annotation hypothesis (technical).** Root hypothesis: an additive composition that is essentially logistic regression plus three one-sided ReLU corrections. Logit z = -0.680 + LINEAR CORE + 0.746*st_depression_gate + 1.730*female_exang_and_gate - 0.918*thalach_cp3_protective_gate, and the prediction is sigmoid(z) (get_formula renders the output as ReLU, but for this binary task the model applies a sigmoid to the output pre-activation). LINEAR CORE = 19 direct input->output edges, dominated by ca (+2.607, number of vessels), sex:0 (-1.573, women lower risk), chol (+0.870), slope:1 (-0.833, upsloping ST protective), cp:1 (-0.785), slope:2 (+0.743), cp:3 (-0.743), fbs:0 (+0.627), trestbps (+0.519), thal:7 (+0.490), thalach (-0.470), sex:1 (+0.390), thal:4 (+0.383), thal:6 (-0.351), cp:4 (+0.312); age, restecg:0, exang:0/1 near zero. cp:2, fbs:1, restecg:1, restecg:2 are unused. Corrections: (a) ischaemia evidence (ST depression, reversible thal defect) adds risk; (b) women with exercise angina lose most of their sex protection (AND gate, +2.07 logit); (c) high max heart rate, especially with non-anginal pain, removes extra risk. Evidence (train, n=242): accuracy 0.876, AUC 0.947. Linear core logit SD 4.90 vs gates 0.78 / 0.56 / 0.49. Linear core alone AUC 0.930; mean-ablating each gate gives AUC 0.938 / 0.940 / 0.945, so together the nonlinear parts add ~+0.017 train AUC. SHAP (probability scale): ca 0.170 >> sex:0 0.080 > slope:1 0.063 > cp:3 0.058 > slope:2 0.053 > thal:7 0.050 > chol 0.038 > thalach 0.034 > oldpeak 0.029. Function preservation of the refactored model 1.8e-15; all leaf closed forms exact up to 3-s.f. display rounding (max 0.027).

**Evidence records.**

- *analytical_formula*:  (source: get_formula force=true latex_collapsed; output shown as relu by tool but model applies sigmoid)
- *performance_metrics*:  (n: 242; rmse: 0.293; split: train; auc_roc: 0.9467; accuracy: 0.876)
- *ablation_study*:  (split: train; method: replace node by its train mean, local forward pass on in-memory copy; auc_full_logit: 0.9467; auc_linear_core_only: 0.9299; auc_ablate_st_depression_gate: 0.9382; auc_ablate_female_exang_and_gate: 0.9405; auc_ablate_thalach_cp3_protective_gate: 0.9446)
- *shap_importance*:  (n: 242; split: train; output: probability (base 0.441))


### ST depression gate (`ann_32`, `st_depression_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: fasting blood sugar, ST depression during exercise (compared with rest), slope of the ST segment at peak exercise, thallium stress test result
- Entry nodes: -10_b, -18, -21, -26_b
- Exit nodes: 557
- Subgraph nodes: -10_b, -18, -21, -26_b, 557

**Hypothesis (plain).** This part increases the predicted risk with ST depression on exercise, in an uneven way, and more so with a reversible defect on the thallium scan. [C37, C41]

**Evidence (plain).** It is active for 116 of 242 training patients, 77 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is above 99% both without it and with it. [C46] Through it, more ST depression raises the risk in an uneven way, mainly once it rises above about 1.2 mm (between 0.2 and 1.2 mm for different patients): for a typical patient from about 12% at 0 mm to about 32% at 2.8 mm. [C37, C45] Looking at the whole model, the rise is larger with a reversible defect on the thallium scan, from about 27% to about 76% against about 12% to about 32% otherwise, in an uneven way. [C41] More ST depression raises the risk in an uneven way only if the ST segment does not slope down (from about 12% to about 32%); with a downsloping segment the risk stays at about 42% (the model has little data on this, n = 17). [C43] On all 242 training patients, its formula gives exactly what the part computes. [C47]

**Annotation hypothesis (technical).** Hypothesis: hidden gate 557 = ReLU(1.283*oldpeak + 0.559*thal:7 - 1.076*slope:3 - 0.417*fbs:0 + 0.085) is a one-sided ischaemia-evidence detector. It switches on mainly with exercise-induced ST depression (oldpeak), more readily when there is a reversible thal defect (thal:7), is pushed down by a downsloping ST segment (slope:3, which is clinically counter-intuitive; it is rare, 7%, and the direct linear core has no slope:3 term, so this is probably how the network absorbs slope:3 at all), and slightly by normal fasting sugar (fbs:0). It enters the logit with weight +0.746, so it only ever adds risk. Evidence (train, n=242): closed form exact (single ReLU); active on 47.9% of patients (83% with thal:7, 27% without); max 4.72, logit contribution SD 0.78; sensitivity oldpeak 0.62 > slope:3 0.52 > thal:7 0.27 > fbs:0 0.20; replacing it by its mean drops train AUC of the logit from 0.947 to 0.938, the largest of the three gates.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.417 x_{fbs:0_b} \\
&\quad + 1.28 x_{oldpeak} \\
&\quad - 1.08 x_{slope:3} \\
&\quad + 0.559 x_{thal:7_b} \\
&\quad + 0.085 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: y = \operatorname{relu}(-0.417 x_{fbs:0} + 1.28 x_{oldpeak} - 1.08 x_{slope:3} + 0.559 x_{thal:7} + 0.085); source: get_formula force=true)
- *empirical_viz*:  (n: 242; max: 4.72; split: train; active_frac: 0.479; logit_contrib_sd: 0.775; active_frac_thal7_absent: 0.27; active_frac_thal7_present: 0.833)


### Women with chest pain on exercise (`ann_33`, `female_exang_and_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: sex, exercise-induced angina
- Entry nodes: -2_b, -17_b
- Exit nodes: 1239
- Subgraph nodes: -2_b, -17_b, 1239

**Hypothesis (plain).** This part increases the predicted risk only for women whose chest pain is brought on by exercise (the model has little data on this, n = 19). [C51]

**Evidence (plain).** It is active for the 19 of 242 training patients who are women with chest pain brought on by exercise, 12 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 9% without it and about 43% with it (the model has little data on this, n = 19). [C51] For the typical patient it acts on, being a woman and chest pain on exercise each take the predicted risk through it from about 9% to about 43%. [C48, C49] Looking at the whole model, being a woman lowers a typical patient's predicted risk from about 12% to below 1% when exercise does not bring on chest pain, and to about 2% when it does. [C50] On all 242 training patients, its formula gives exactly what the part computes. [C52]

**Annotation hypothesis (technical).** Hypothesis: hidden gate 1239 = ReLU(0.571*sex:0 + 0.903*exang:1 - 0.853) is a Boolean AND of female sex (sex:0) and exercise-induced angina (exang:1). Evidence (train, n=242): truth table of the standardised one-hot inputs gives activation 1.199 for women with exang (n=19, positive rate 63%) and exactly 0 for the other three cells (male/no exang n=96 pos 40%; male/exang n=63 pos 76%; female/no exang n=64 pos 14%; the male+exang pre-activation is -0.004, just below the threshold). Weighted +1.730 into the logit it adds +2.07, which cancels most of the direct female protective term (-1.573*sex:0 contributes -2.18 for women vs +1.14 for men): the network learns that exercise angina erases the female advantage. Active on 7.9% of patients, logit contribution SD 0.56; mean-ablation drops train AUC 0.947 -> 0.940. Closed form exact.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.903 x_{exang:1_b} \\
&\quad + 0.571 x_{sex:0_b} \\
&\quad -0.853 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: y = \operatorname{relu}(0.903 x_{exang:1} + 0.571 x_{sex:0} - 0.853); source: get_formula force=true)
- *empirical_viz*:  (n: 242; split: train; logit_contrib_sd: 0.558)


### High heart rate protective gate (`ann_34`, `thalach_cp3_protective_gate`)

- Kind: leaf; parent: heart_disease_logit; children: none
- Reads: chest pain type, maximum heart rate reached during the exercise test, thallium stress test result
- Entry nodes: -6_b, -15_b, -23
- Exit nodes: 2124
- Subgraph nodes: -6_b, -15_b, -23, 3818, 2124

**Hypothesis (plain).** This part reduces the predicted risk for patients with a high maximum heart rate: through it, maximum heart rate changes nothing until about 147 beats per minute, then lowers the risk steadily. [C54]

**Evidence (plain).** It is active for 74 of 242 training patients, 13 of whom had heart disease; for the typical patient among those it acts on, the predicted risk is about 1% without it and below 1% with it. [C60] Through this part, a higher maximum heart rate changes nothing until about 147 beats per minute, then lowers the risk steadily, for the typical patient among those it acts on from about 1% to below 1%, flattening out near the bottom because the predicted risk cannot go much lower. [C54] Across the whole model, a higher maximum heart rate lowers the risk slowly at first, then faster, more with non-anginal chest pain (from about 3% to below 1%, against about 22% to about 8% otherwise), flattening out near the bottom because the predicted risk cannot go much lower. [C58] A thallium result other than a normal scan raises the risk more for patients without non-anginal chest pain (from about 12% to about 37%, against about 1% to about 4% with it). [C56] On all 242 training patients, its formula gives exactly what the part computes. [C61]

**Annotation hypothesis (technical).** Hypothesis: a protective high-exercise-capacity gate. Serial chain thal:3 -> 3818 -> 2124. 3818 = ReLU(1.003*thal:3 - 0.192) is a normal-thallium indicator taking only the values 0 (abnormal thal) and 0.702 (normal thal, 56% of patients). 2124 = ReLU(1.338*thalach + 0.763*cp:3 - 0.108*3818 - 0.932) fires when maximum heart rate is high, much more readily with non-anginal chest pain (cp:3): active on 80% of cp:3 patients but 10.5% of others (30.6% overall, max 2.88). It enters the logit with weight -0.918, so it only ever lowers risk, adding a hinge to the direct linear thalach (-0.470) and cp:3 (-0.743) terms. The thal:3 branch only moves the threshold by -0.076 (sensitivity 0.025 vs thalach 0.41, cp:3 0.23) and is nearly inert. Evidence (train, n=242): closed form exact; logit contribution SD 0.49; mean-ablation drops train AUC only 0.947 -> 0.945, the weakest of the three gates.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.763 x_{cp:3_b} \\
&\quad + 1.34 x_{thalach_b} \\
&\quad - 0.108 \operatorname{relu}{\left(1.0 x_{thal:3} - 0.191 \right)} \\
&\quad -0.932 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: y = \operatorname{relu}(0.763 x_{cp:3} + 1.34 x_{thalach} - 0.108\,\operatorname{relu}(1.0 x_{thal:3} - 0.191) - 0.932); source: get_formula force=true)
- *empirical_viz*:  (n: 242; max: 2.88; split: train; logit_contrib_sd: 0.487; node2124_active_frac: 0.306; node3818_active_frac: 0.558; active_frac_cp3_absent: 0.105; active_frac_cp3_present: 0.8)

## Worth checking (draft)

- Worth checking: going from 0 to 2 vessels coloured moves a typical patient's predicted risk from about 12% to about 98%. [C01]
- Worth checking: patients with typical angina get the lowest predicted risk of all chest pain types, below patients with no chest pain symptoms. [C18, C19, C26]
- Worth checking: the model treats fasting blood sugar above 120 mg/dL as lowering the risk of heart disease (for a typical patient from about 12% to about 3%). [C24]
- Worth checking: the model treats a fixed defect on the thallium scan as lowering the risk compared with a normal scan (for a typical patient from about 12% to about 3%), from only 15 training patients. [C23]
- Worth checking: the model gives a higher predicted risk to the 2 training patients whose thallium result is coded 4, a code the dataset documentation does not define. [C03]
- Worth checking: through one part, more ST depression on exercise raises the predicted risk in an uneven way only if the ST segment does not slope down, from only 17 training patients with a downsloping segment. [C43]
