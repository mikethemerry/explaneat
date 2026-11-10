# Explanation vb-fd4da7

- Genome: `6a26c823-5e60-424d-a968-74f44d675efa`
- Dataset: churn_onehot (training split 45962ea6-3c78-49f9-a01e-3665f6809b1e, n = 4000; test n = 1000)
- Outcome: churn (the customer left)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 1%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 7.216449660063518e-16
- closed_form_max_abs_err: 0.0076642890955742615
- n_leaf_annotations: 5
- n_composition_annotations: 1
- n_splits: 16
- n_identity_nodes: 1

**Summary.** logit = -1.07 + sparse linear backbone over 41 inputs (day minutes +2.02, voicemail plan -0.99, international plan +0.75, ...) + a CSC x low-usage gate + a CSC>=4 hinge + voicemail chain and state offsets (exactly linear). The gate is a low-day-minutes hinge whose kink moves up with each service call from 3 onward; with the direct day-minutes edge the logit is V-shaped in day minutes, so frequent callers with low usage churn (empirical 0.88). Service calls enter only through hinges. Train AUC 0.901, acc 0.906.

**What resisted.** Nothing structural. Node 161 (state:4) split to separate its output and near-dead-node roles. The gate hinge also raises risk of low-CSC customers with very low day minutes (93 rows, predicted 0.17 vs empirical 0.04).

## Plain-language summary (draft)

For a typical customer (in West Virginia, area code 415, no international or voice-mail plan, 1 call to customer service and median usage) the model predicts a chance of leaving of about 1%; it adds up five parts: fixed-weight inputs, calls with low usage, a voice-mail discount, many service calls and state units. [C059]

**Fixed-weight inputs.** This part adds up most inputs with fixed weights, and daytime minutes count most: they raise a typical customer's chance of leaving steadily, from under 1% at 112 to about 17% at 249 minutes. [C060] An international plan takes the chance from about 1% to about 16%, a voice-mail plan lowers it to under 1%, and more voice-mail messages raise it steadily, from about 1% at 0 to about 6% at 32 messages. [C061, C069, C062] A higher international charge, more evening minutes (about 1% at 138 to about 3% at 266 minutes), a higher evening charge and more night-time minutes also raise it steadily, each by one to three points. [C065, C066, C068, C072] Through it, the state moves the chance from about 1% in most other states to about 6% in California and about 5% in Washington, and down to under 1% in Hawaii, Vermont and Colorado (41 to 80 training customers per state; the full list is in the technical detail). [C063, C064, C070, C071, C073]

**Calls with low usage.** This part combines customer-service calls with low daytime, evening and international use and acts on 3147 of 4000 training customers, 409 of whom left; for the typical customer it acts on, it changes the chance of leaving by less than a point. [C118] Through it, daytime minutes lower the chance only while they stay below about 169 minutes (between 154 and 214 minutes for different customers), in an uneven way: for a typical customer, from about 47% at 91 minutes to about 1% at 352 minutes. [C114, C102] A higher international charge lowers it only while it stays below about 2.96 US dollars (between 2.96 and 4.53 US dollars for different customers), in an uneven way, from about 4% to about 1%, and a higher evening charge lowers it in an uneven way, from about 3% to about 1%. [C116, C103, C105] Customer-service calls raise it in an uneven way, mostly for customers with an evening charge of 17.12 US dollars or less: from about 1% to about 6% between 0 and 3 calls, against no change above it. [C112]

**Voice-mail discount.** This part reads the voice-mail plan and a few states: through it, a voice-mail plan lowers a typical customer's chance by less than a point, and customers in states other than Alabama, North Carolina and West Virginia get about 2% instead of about 1%. [C120, C121]

**Many service calls.** Through this part, customer-service calls add nothing until about 4 calls, then raise the chance steadily; it acts on 306 of 4000 training customers, 154 of whom left; for the typical customer it acts on, the chance is about 16% without it and about 26% with it. [C126, C127] It is switched off for customers in West Virginia: for the typical customer it acts on, being in any other state takes the chance from about 16% to about 26%. [C125]

**State units.** This part reads only the state and moves the chance by less than a point: up in Tennessee and Ohio, down in Nevada and California. [C129, C131, C130, C132]

## Parts, from the root down

### The whole model (`ann_95`, `churn_logit`)

- Kind: composition; parent: none; children: linear_backbone, csc_usage_gate, csc_hinge_7208, voicemail_chain, state_offsets
- Reads: state, account length, area code, international calling plan, voice-mail plan, number of voice-mail messages, total daytime minutes, total daytime calls, total daytime charge, total evening minutes, total evening calls, total evening charge, total night-time minutes, total night-time calls, total night-time charge, total international minutes, total international calls, total international charge, number of calls to customer service
- Entry nodes: 10249, 1133, 11427, 1264, 14942, 161_a, 252, 3861, 4051, 6730, 7208, 807, 8896, identity_lin, -69, -67, -63, -61, -53, -51, -49, -39, -34, -29, -26, -19, -17, -16, -15, -11, -10, -8, -7, -6, -5, -3
- Exit nodes: 0
- Subgraph nodes: -69, -67, -63, -61, -53, -51, -49, -39, -34, -29, -26, -19, -17, -16, -15, -11, -10, -8, -7, -6, -5, -3, 0

**Hypothesis (plain).** The model adds up five parts into one score: fixed-weight inputs, calls with low usage, a voice-mail discount, many service calls and state units. [C059]

**Evidence (plain).** It is made up of these five parts (each explained below); the score is their sum, which was checked on all 4000 training customers. [C059] Changing one input at a time across the whole model, an international plan takes a typical customer's chance of leaving from about 1% to about 16%, and more voice-mail messages raise it steadily, from about 1% at 0 to about 6% at 32 messages. [C001, C005] The model ignores the daytime charge, evening calls, international calls and the night-time charge. [C055, C056, C057, C058]

**Annotation hypothesis (technical).** Root: logit P(churn) = -1.068 + linear_backbone + 1.59*807 + 1.81*1264 + 1.48*7208 - 0.88*1133 - 1.00*4051 + state offsets (+ negligible 252), purely additive in logit; the sigmoid is the only coupling. 22 inputs are unconnected (incl. totalDayCharge, totalNightCharge, totalEveCalls, totalIntlCalls, areaCode:408, 16 states) and have zero effect. Train AUC 0.901, accuracy 0.906. Performance sources (train): all children replaced by least-squares linear fits gives AUC 0.836; restoring csc_usage_gate alone gives 0.899, csc_hinge_7208 alone 0.851, voicemail_chain and state_offsets alone 0.836 (they are exactly linear on binary inputs). So the extra AUC over linear comes from the CSC x low-usage gate. Rare-category switches: backbone state:11/46/5 offsets of -2.2/-2.1/-1.4 logits on 73-80 rows (empirical -1.2/-0.8/-0.5), state:49 disabling the CSC>=4 hinge for 12 rows, state:33 offset of the wrong sign.

**Evidence records.**

- *analytical_formula*: Closed form from get_formula(force=true); check_explanation closed-form max abs err <= 0.0077 on train. (latex: y_0 = \frac{1}{2.91 e^{0.00893 x_{10249} + 0.884 x_{1133} - 0.0513 x_{11427} - 1.81 x_{1264} - 0.0406 x_{14942} + 0.0277 x_{161_a} + 2.23 \cdot 10^{-5} x_{252} + 0.15 x_{3861} + 1.0 x_{4051} - 0.343 x_{6730} - 1.48 x_{7208} - 1.59 x_{807} - 0.0306 x_{8896} - 1.0 x_{identity_lin}} + 1}; split: train)
- *ablation_study*: Least-squares linearisation of each child on its raw entry features (train). Nearly all of the gain over linear (0.836 -> 0.901) comes from csc_usage_gate. (n: 4000; split: train; lr_train_auc_refit: 0.8433)
- *sign_analysis*: Counterfactual logit change from switching each binary input from on to off for the rows where it is on, against the empirical log-odds (rows with vs without, +0.5 smoothing). (split: train)
- *shap_importance*: KernelSHAP on 300 train rows (output probability scale). (split: train; max_samples: 300)


### Fixed-weight inputs (`ann_90`, `linear_backbone`)

- Kind: leaf; parent: churn_logit; children: none
- Reads: state, account length, area code, international calling plan, voice-mail plan, number of voice-mail messages, total daytime minutes, total daytime calls, total evening minutes, total evening charge, total night-time minutes, total night-time calls, total international minutes, total international charge
- Entry nodes: -1, -13_b, -14, -18_b, -20, -21_b, -22, -23, -24_b, -25, -2_b, -31, -32, -33, -35_c, -36, -37_b, -4, -41, -42, -43, -44, -46, -47_b, -48, -50, -52, -54, -55, -56, -57_b, -58, -59_b, -60, -62, -64_b, -65, -66, -68, -70_c, -9
- Exit nodes: identity_lin
- Subgraph nodes: -1, -13_b, -14, -18_b, -20, -21_b, -22, -23, -24_b, -25, -2_b, -31, -32, -33, -35_c, -36, -37_b, -4, -41, -42, -43, -44, -46, -47_b, -48, -50, -52, -54, -55, -56, -57_b, -58, -59_b, -60, -62, -64_b, -65, -66, -68, -70_c, -9, identity_lin

**Hypothesis (plain).** This part adds up most inputs with fixed weights: daytime minutes, an international plan and voice-mail messages raise the chance of leaving, steadily for the minutes and messages, and a voice-mail plan lowers it. [C060, C061, C062, C069]

**Evidence (plain).** Through it, daytime minutes take a typical customer steadily from under 1% at 112 to about 17% at 249 minutes, and an international plan from about 1% to about 16%; a voice-mail plan lowers the chance to under 1%. [C060, C061, C069] The state moves it from about 1% in most other states to about 6% in California and about 5% in Washington, and to under 1% in Hawaii and Vermont. [C063, C064, C070, C071] Its formula reproduces the part exactly on all 4000 training customers. [C101]

**Annotation hypothesis (technical).** Hypothesis: a sparse logistic-regression term. identity_lin sums the 41 direct input->output edges: totalDayMinutes +2.02, voiceMailPlan -0.99, internationalPlan +0.75, totalIntlCharge +0.70, numberVmailMessages +0.61, totalEveMinutes +0.50 and totalEveCharge +0.35 (identical after z-scoring, so evening slope 0.85), totalNightMinutes +0.30, accountLength +0.13, totalDayCalls +0.12, and state offsets, the largest state:11 -0.30, state:46 -0.29, state:5 -0.20 (each one-hot z is about 7, so these are 1.4 to 2.2 logit offsets on 73-80 rows, roughly 2x the empirical log-odds). numberCustomerServiceCalls has no direct edge: CSC enters only through the hinges. Evidence: exact closed form; zeroing it drops train AUC to 0.54.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= 0.127 x_{accountLength} \\
&\quad + 0.0463 x_{areaCode:415} \\
&\quad - 0.0167 x_{areaCode:510} \\
&\quad + 0.751 x_{internationalPlan} \\
&\quad + 0.609 x_{numberVmailMessages} \\
&\quad - 0.101 x_{state:0} \\
&\quad - 0.297 x_{state:11} \\
&\quad + 0.00648 x_{state:16} \\
&\quad - 0.0282 x_{state:1_b} \\
&\quad + 0.0767 x_{state:20} \\
&\quad - 0.0756 x_{state:24_b} \\
&\quad + 0.0899 x_{state:26} \\
&\quad + 0.0255 x_{state:27_b} \\
&\quad - 0.0274 x_{state:28} \\
&\quad - 0.0523 x_{state:29} \\
&\quad - 0.0114 x_{state:2_b} \\
&\quad - 0.0485 x_{state:30} \\
&\quad - 0.02 x_{state:36} \\
&\quad + 0.0288 x_{state:37} \\
&\quad - 0.103 x_{state:38} \\
&\quad - 0.0016 x_{state:3_b} \\
&\quad + 0.0552 x_{state:40} \\
&\quad - 0.0991 x_{state:41_b} \\
&\quad - 0.113 x_{state:45} \\
&\quad - 0.291 x_{state:46} \\
&\quad + 0.158 x_{state:47} \\
&\quad - 0.0969 x_{state:48} \\
&\quad + 0.139 x_{state:4_c} \\
&\quad - 0.11 x_{state:50_b} \\
&\quad - 0.196 x_{state:5} \\
&\quad + 0.0339 x_{state:6} \\
&\quad - 0.0121 x_{state:8} \\
&\quad + 0.124 x_{totalDayCalls} \\
&\quad + 2.02 x_{totalDayMinutes_b} \\
&\quad + 0.351 x_{totalEveCharge_b} \\
&\quad + 0.503 x_{totalEveMinutes} \\
&\quad + 0.701 x_{totalIntlCharge_c} \\
&\quad - 0.0217 x_{totalIntlMinutes} \\
&\quad - 0.00876 x_{totalNightCalls} \\
&\quad + 0.3 x_{totalNightMinutes} \\
&\quad - 0.99 x_{voiceMailPlan_b}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula(force=true); check_explanation closed-form max abs err <= 0.0077 on train. (split: train)
- *performance_metrics*: Zeroing the backbone collapses AUC to 0.54: it is the dominant term in magnitude. (split: train; std_logit_contrib: 2.575; auc_backbone_zeroed: 0.539)


### Calls with low usage (`ann_91`, `csc_usage_gate`)

- Kind: leaf; parent: churn_logit; children: none
- Reads: state, total daytime minutes, total evening charge, total international charge, number of calls to customer service
- Entry nodes: -13_a, -27_b, -35_b, -37_a, -40, -59_a, -64_a, -70_ab, -71_abdef
- Exit nodes: 1264, 252, 807
- Subgraph nodes: -13_a, -27_b, -35_b, -37_a, -40, -59_a, -64_a, -70_ab, -71_abdef, 1264, 12926, 13061, 161_b, 252, 2627, 2796, 807, 9555

**Hypothesis (plain).** This part raises the chance of leaving for customers with low usage, more so when they call customer service: daytime minutes lower it only while they stay below about 169 minutes (between 154 and 214 minutes for different customers), and calls raise it, both in an uneven way. [C114, C104]

**Evidence (plain).** It acts on 3147 of 4000 training customers, 409 of whom left; for the typical customer it acts on, it changes the chance by less than a point. [C118] Through it, a typical customer's chance falls in an uneven way from about 47% at 91 minutes of daytime use to about 1% at 352 minutes, with no further fall past about 169 minutes (between 154 and 214 minutes for different customers). [C114, C102] A higher international charge lowers it only while it stays below about 2.96 US dollars (between 2.96 and 4.53 US dollars for different customers), in an uneven way, from about 4% to about 1%. [C116, C103] Customer-service calls raise it in an uneven way, from about 1% to about 3% between 0 and 3 calls, and from about 1% to about 6% with an evening charge of 17.12 US dollars or less, against no change above it. [C104, C112] Its formula reproduces the part exactly on all 4000 training customers. [C119]

**Annotation hypothesis (technical).** Hypothesis: a low-usage hinge whose threshold rises with customer service calls (CSC). 807 = relu(-1.36 dayMin + 0.05 CSC + 0.69*2627 + 0.61*2796 - 1.30), where 2627 = 1.01 + 0.32 relu(0.87 CSC - 0.09) (always positive, affine in a CSC hinge) and 2796 = relu(0.41 - 1.00 eveCharge + 0.37 CSC + 0.15 state:41 + 0.34*12926), with 12926/13061 a two-stage CSC hinge that turns on at 3 calls. 1264 = relu(0.077 - 0.32 intlCharge + 0.39*807) adds 0.39*807 and a convex low-intl-charge hinge. Readout 1.59*807 + 1.81*1264 (252 is near dead, readout -2e-5, and holds a state:2/state:44/state:4 path). Evidence (train, n=4000): sensitivity of 807 is dayMin 0.63, eveCharge 0.22, CSC 0.20. Mean gate logit rises 1.1 -> 7.4 from 0 to 7 calls. With the backbone +2.02 dayMin, the logit is V-shaped in day minutes; for CSC>=4 the low-usage arm dominates (empirical churn 0.88 low-day vs 0.24 high-day, which the network reproduces). The same hinge also lifts low-CSC, very-low-day customers (93 rows, predicted 0.17 vs empirical 0.04): a mild overextension. totalEveCharge equals totalEveMinutes after z-scoring, so the gate effectively uses evening usage. Ablation: linearising this gate drops train AUC 0.901 -> 0.851; restoring it alone on the fully linearised model gives 0.836 -> 0.899. It is the source of nearly all of the network s gain over a linear model.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.317 x_{totalIntlCharge_ab} \\
&\quad + 0.39 \operatorname{relu}{\left(0.049 x_{numberCustomerServiceCalls_abdef} - 1.36 x_{totalDayMinutes_a} + 0.687 \operatorname{relu}{\left(0.322 \operatorname{relu}{\left(0.87 x_{numberCustomerServiceCalls_abdef} - 0.0887 \right)} + 1.01 \right)} + 0.61 \operatorname{relu}{\left(0.366 x_{numberCustomerServiceCalls_abdef} + 0.152 x_{state:41_a} - 1.0 x_{totalEveCharge_a} + 0.341 \operatorname{relu}{\left(0.986 x_{numberCustomerServiceCalls_abdef} + 0.949 \operatorname{relu}{\left(0.994 x_{numberCustomerServiceCalls_abdef} - 0.655 \right)} - 0.64 \right)} + 0.405 \right)} - 1.3 \right)} \\
&\quad + 0.0768 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.76 x_{state:2_a} \\
&\quad + 0.432 x_{state:44} \\
&\quad - 0.851 x_{totalIntlCharge_ab} \\
&\quad - 0.378 \operatorname{relu}{\left(0.322 \operatorname{relu}{\left(0.87 x_{numberCustomerServiceCalls_abdef} - 0.0887 \right)} + 1.01 \right)} \\
&\quad + 0.791 \operatorname{relu}{\left(- 0.0107 x_{state:32_b} + 0.783 x_{state:4_b} - 0.14 \right)} \\
&\quad -2.22 \bigr)
\end{aligned} \\ \begin{aligned}
y_2 &= \operatorname{relu}\bigl( 0.049 x_{numberCustomerServiceCalls_abdef} \\
&\quad - 1.36 x_{totalDayMinutes_a} \\
&\quad + 0.687 \operatorname{relu}{\left(0.322 \operatorname{relu}{\left(0.87 x_{numberCustomerServiceCalls_abdef} - 0.0887 \right)} + 1.01 \right)} \\
&\quad + 0.61 \operatorname{relu}{\left(0.366 x_{numberCustomerServiceCalls_abdef} + 0.152 x_{state:41_a} - 1.0 x_{totalEveCharge_a} + 0.341 \operatorname{relu}{\left(0.986 x_{numberCustomerServiceCalls_abdef} + 0.949 \operatorname{relu}{\left(0.994 x_{numberCustomerServiceCalls_abdef} - 0.655 \right)} - 0.64 \right)} + 0.405 \right)} \\
&\quad -1.3 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula(force=true); check_explanation closed-form max abs err <= 0.0077 on train. (split: train)
- *empirical_viz*: 807 is driven by totalDayMinutes (0.63), totalEveCharge (0.22) and CSC (0.20); 1264 adds totalIntlCharge; 252 is negligible. (n: 4000; viz: sensitivity; split: train)
- *input_output_sampling*: Local forward passes on train. The gate is a hinge in low day usage whose kink moves right with each service call above 2 (13061/12926 turn on at 3 calls). With the backbone +2.02 dayMin the logit is V-shaped in day minutes; the low-usage arm reproduces the empirical CSC>=4 x low-day churn of 0.88. (n: 4000; split: train)
- *ablation_study*: Replacing this gate by its least-squares linear fit drops train AUC 0.901 -> 0.851; restoring it alone onto the fully linearised model lifts 0.836 -> 0.899. It carries almost all the network s gain over linear. (split: train; full_auc: 0.9009299028011302; auc_gate_zeroed: 0.7635; auc_all_linearised: 0.8363903060436993; auc_gate_linearised: 0.8504710739209445; auc_all_linearised_plus_gate: 0.8987679842604921)


### Voice-mail discount (`ann_93`, `voicemail_chain`)

- Kind: leaf; parent: churn_logit; children: none
- Reads: state, voice-mail plan
- Entry nodes: -21_a, -2_a, -45_a, -57_a
- Exit nodes: 1133, 4051
- Subgraph nodes: -21_a, -2_a, -45_a, -57_a, 1133, 4051, 5577

**Hypothesis (plain).** This part gives customers with a voice-mail plan a small discount on the chance of leaving. [C120]

**Evidence (plain).** Through it, a voice-mail plan lowers a typical customer's chance by less than a point, and customers in states other than Alabama, North Carolina and West Virginia get about 2% instead of about 1%. [C120, C121] Its formula reproduces the part exactly on all 4000 training customers. [C124]

**Annotation hypothesis (technical).** Hypothesis: a linear voicemail-plan discount. 5577 = relu(0.98 voiceMailPlan + 0.25 state:1 + 1.0) -> 1133 = relu(0.14*5577 + 0.02 state:27 + 0.69) -> 4051 = relu(1.15*1133 + 0.06 state:49 + 2.10); readouts -0.88*1133 - 1.00*4051. All three ReLUs are active on every train row, so the chain is exactly linear on the data: net -0.29 logit per z of voiceMailPlan (on top of the backbone -0.99), -0.07 per z of state:1, -0.06 per z of state:49. Linearising it changes nothing (AUC 0.9009).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.0211 x_{state:27_a} \\
&\quad + 0.144 \operatorname{relu}{\left(0.248 x_{state:1_a} + 0.982 x_{voiceMailPlan_a} + 0.999 \right)} \\
&\quad + 0.689 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.0557 x_{state:49_a} \\
&\quad + 1.15 \operatorname{relu}{\left(0.0211 x_{state:27_a} + 0.144 \operatorname{relu}{\left(0.248 x_{state:1_a} + 0.982 x_{voiceMailPlan_a} + 0.999 \right)} + 0.689 \right)} \\
&\quad + 2.1 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula(force=true); check_explanation closed-form max abs err <= 0.0077 on train. (split: train)
- *empirical_viz*: All three ReLUs are active on every train row, so the chain is exactly linear on the data: an extra -0.29 logit per z of voiceMailPlan, -0.07 state:1, -0.06 state:49. (viz: sensitivity; split: train; linear_fit_R2: 1.0; all_relus_active_frac: 1.0)


### Many service calls (`ann_92`, `csc_hinge_7208`)

- Kind: leaf; parent: churn_logit; children: none
- Reads: state, number of calls to customer service
- Entry nodes: -45_b, -71_c
- Exit nodes: 7208
- Subgraph nodes: -45_b, -71_c, 7208

**Hypothesis (plain).** This part raises the chance of leaving steadily with every call to customer service, from the first call on, except in West Virginia. [C126, C125]

**Evidence (plain).** It acts on 306 of 4000 training customers, 154 of whom left; for the typical customer it acts on, the chance is about 16% without it and about 26% with it. [C127] It is switched off in West Virginia: for the typical customer it acts on, being in any other state takes the chance from about 16% to about 26%. [C125] Its formula reproduces the part exactly on all 4000 training customers. [C128]

**Annotation hypothesis (technical).** Hypothesis: a service-call threshold switched off for one state. 7208 = relu(0.54 CSC - 0.37 state:49 - 0.48), readout +1.48. Active only when CSC >= 4 calls (7.6% of train rows), adding about 0.61 logit per extra call. For state:49 (one-hot z = 5.64) the pre-activation drops by 2.1, so the hinge is never active for state:49: 12 state:49 customers with CSC>=4 (5 churned, 42% vs 50% for other states) lose up to 2.75 logits. Evidence: closed form exact; linearising this unit costs only 0.002 AUC (0.901 -> 0.899) because csc_usage_gate already encodes CSC>=4. The state:49 exemption is a rare-category switch fitted to about 12 rows.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.542 x_{numberCustomerServiceCalls_c} \\
&\quad - 0.367 x_{state:49_b} \\
&\quad -0.479 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula(force=true); check_explanation closed-form max abs err <= 0.0077 on train. (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.542 x_{numberCustomerServiceCalls_c} \\
&\quad - 0.367 x_{state:49_b} \\
&\quad -0.479 \bigr)
\end{aligned}; split: train)
- *input_output_sampling*: Active on 7.6% of train rows (CSC>=4), never for state:49. Adds 1.48*0.54=0.80 logit per z of CSC. Removing its nonlinearity costs only 0.002 AUC because the gate already encodes CSC>=4. (split: train; threshold: CSC >= 4 calls (z>1.25); active_rows: 304; auc_linearised: 0.8987679842604921; auc_restore_alone: 0.8504710739209445; slope_logit_per_call: 0.61)


### State units (`ann_94`, `state_offsets`)

- Kind: leaf; parent: churn_logit; children: none
- Reads: state
- Entry nodes: -12, -18_a, -24_a, -27_a, -28, -30, -35_a, -38, -47_a
- Exit nodes: 10249, 11427, 14942, 161_a, 3861, 6730, 8896
- Subgraph nodes: -12, -18_a, -24_a, -27_a, -28, -30, -35_a, -38, -47_a, 10249, 11427, 11951, 14942, 161_a, 3861, 6730, 8896

**Hypothesis (plain).** This part adds small state offsets: up in Tennessee and Ohio, down in Nevada and California. [C129, C131, C130, C132]

**Evidence (plain).** Through it, a typical customer's chance moves by less than a point: about 2% in Tennessee and Ohio, and about 1% in Nevada and California. [C129, C131, C130, C132] Its formula reproduces the part exactly on all 4000 training customers. [C136]

**Annotation hypothesis (technical).** Hypothesis: per-state logit offsets. Every unit reads one (or a few) state one-hots, so it is an exact function of binary inputs and hence a constant offset per state: 10249 (state:3, -0.009 readout), 11427 (state:42, +0.05), 14942 (state:35, +0.04), 161_a (state:4, -0.03), 6730 (constant +0.43 logit except state:33, where it switches off: -0.45 logit for 77 rows), 8896 (state:19, <0.001), 3861 (state:50, dead on all rows). Evidence: sensitivity analysis on train; linear fit R2 = 1.0; zeroing all of them leaves AUC unchanged (0.9009). The state:33 switch has the opposite sign to the data (empirical log-odds +0.33).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.999 x_{state:3_a} + 0.17 \right)} \\ y_1 = \operatorname{relu}{\left(0.995 x_{state:42} + 0.194 \right)} \\ y_2 = \operatorname{relu}{\left(1.0 x_{state:35} - 1.74 \right)} \\ \begin{aligned}
y_3 &= \operatorname{relu}\bigl( - 0.0107 x_{state:32_a} \\
&\quad + 0.783 x_{state:4_a} \\
&\quad -0.14 \bigr)
\end{aligned} \\ y_4 = \operatorname{relu}{\left(- 0.771 x_{state:50_a} - 0.431 \right)} \\ \begin{aligned}
y_5 &= \operatorname{relu}\bigl( - 0.0184 x_{state:24_a} \\
&\quad - 0.414 x_{state:33} \\
&\quad + 0.00756 x_{state:35} \\
&\quad - 0.00151 \operatorname{relu}{\left(1.0 x_{state:35} - 1.09 \right)} \\
&\quad + 1.25 \bigr)
\end{aligned} \\ y_6 = \operatorname{relu}{\left(- 1.0 x_{state:19} - 0.134 \right)}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula(force=true); check_explanation closed-form max abs err <= 0.0077 on train. (split: train)
- *empirical_viz*: Each unit depends on one binary one-hot, so it is an exact per-state logit offset; 3861 (state:50) is dead on all rows; 8896 (state:19) and 161_a contribute <0.03 logit. (viz: sensitivity; split: train; linear_fit_R2: 1.0)

## Worth checking (draft)

- Worth checking: the state moves the chance by several points (for example Washington and California about 6%, Hawaii under 1%), based on 41 to 122 training customers per state. [C003, C004, C014]
- Worth checking: the model treats a voice-mail plan as lowering the chance of leaving but more voice-mail messages as raising it. [C012, C005]
- Worth checking: the many-service-calls part is switched off for customers in West Virginia, the state of the typical customer. [C125]
- Worth checking: daytime minutes, the evening charge and the international charge raise the chance through the fixed-weight inputs but lower it through calls with low usage. [C060, C102, C068, C105, C065, C103]
