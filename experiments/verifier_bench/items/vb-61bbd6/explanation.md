# Explanation vb-61bbd6

- Genome: `1f180718-4e8b-43bc-be1d-9a4276f234c3`
- Dataset: churn_onehot (training split 45962ea6-3c78-49f9-a01e-3665f6809b1e, n = 4000; test n = 1000)
- Outcome: churn (the customer left)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 3%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 5.551115123125783e-16
- closed_form_max_abs_err: 0.02795209712577318
- n_leaf_annotations: 7
- n_composition_annotations: 1
- n_splits: 6
- n_identity_nodes: 2

**Summary.** P(churn) = sigmoid(-2.18 + sparse linear backbone over 30 inputs + customer-service-calls threshold + high day/eve-minutes hinge + small day-charge hinges + single-state offsets + affine account-length term), additive. CSC unit cancels the direct slope for 0-2 calls then switches off from 3-4 calls (logit +1.25 per call after). Day/eve hinge adds +2.3 logit on average in the top decile. Train AUC 0.898 vs 0.843 refit LR, 0.837 all-linearised.

**What resisted.** Nothing structural. 3384 dead on train; 6468 and 3930 always on (affine). state:44 offset a near-cancelling pair (+0.98 hidden, -0.91 direct). voiceMailPlan and numberVmailMessages partly offset.

## Plain-language summary (draft)

For a typical customer (in West Virginia, area code 415, no international or voice-mail plan, 1 call to customer service and median usage) the model predicts a chance of leaving of about 3%; it adds up seven parts, each of which changes that chance for particular customers. [C051]

**Plans, usage and state.** An international plan raises a typical customer's chance of leaving from about 3% to about 24%, a voice-mail plan lowers it to under 1%, and more voice-mail messages raise it steadily, from about 3% at 0 to about 13% at 32 messages. [C052, C059, C053] More evening minutes (about 2% at 138 to about 3% at 266 minutes), night-time minutes (about 2% at 136 to about 3% at 264 minutes) and international minutes (about 2% at 7 to about 3% at 14 minutes) raise it steadily, while more international calls (about 3% at 2 to about 2% at 8 calls) and a higher night-time charge lower it steadily, by less than a point. [C074, C065, C063, C067, C079] The customer's state shifts the chance up or down: from about 3% in West Virginia to about 10% in Montana, about 9% in Oregon and about 8% in Washington, and down to about 1% in Vermont and Wyoming (each of these states has 72 to 122 training customers; the full list is in the technical detail). [C054, C055, C056, C062, C064]

**Heavy daytime use.** This part combines daytime and evening minutes and acts on 747 of 4000 training customers, 269 of whom left; for the typical customer it acts on, the chance of leaving is about 5% without it and about 23% with it. [C088] Through it, daytime minutes add nothing until about 234 minutes (between 199 and 263 minutes for different customers), then raise the chance steadily: for a typical customer from about 3% to about 17%. [C086, C084] Evening minutes add to it in an uneven way, from about 5% to about 45% for the typical customer it acts on. [C083]

**Customer-service calls.** This part combines customer-service calls, daytime minutes and the state: more calls raise the chance slowly at first, then faster, from about 2% at 0 to about 5% at 3 calls. [C090] For customers in Oregon the calls count much more, from under 1% to about 18%, against about 2% to about 5% elsewhere. [C093] For customers in Ohio, daytime minutes add to the chance only above about 85 minutes, in an uneven way, from about 2% to about 7%. [C095]

**State adjustments.** This part acts on 3845 of 4000 training customers, all but those in a few states, and lowers their chance: for the typical customer it acts on, from about 7% without it to about 3% with it. [C102] Compared with other states, it leaves the chance higher in South Dakota and Utah (about 7%) and Maine (about 5%), and lower in Hawaii (under 1%) and Alaska (about 1%). [C097, C098, C100, C099, C101]

**Daytime charge.** This part reads the daytime charge and two states: the charge adds nothing until about 39.72 US dollars, then raises the chance steadily, by less than a point for a typical customer. [C108, C105] Through it, a customer in Rhode Island has a lower chance, about 1% instead of about 3%. [C104]

**Account length.** This part reads the account length: a longer account raises the chance steadily, from about 2% at 48 to about 3% at 152 (the unit of account length is not documented in the data). [C110]

**Washington outside area code 415.** This part combines the state and the area code: for customers in Washington outside area code 415, the chance rises from about 3% to about 14%. [C114]

## Parts, from the root down

### The whole model (`ann_87`, `full_model`)

- Kind: composition; parent: none; children: linear_backbone, csc_threshold, day_eve_minutes_hinge, day_charge_hinges, state_switches, area415_state47, accountLength_affine
- Reads: state, account length, area code, international calling plan, voice-mail plan, number of voice-mail messages, total daytime minutes, total daytime calls, total daytime charge, total evening minutes, total evening calls, total evening charge, total night-time minutes, total night-time calls, total night-time charge, total international minutes, total international calls, total international charge, number of calls to customer service
- Entry nodes: 216, 595, 1255, 1529, 3227, 3384, 3930, 4311, 5685, 5853, 6468, 10761, identity_lin, identity_csc, -10, -12, -13, -17, -19, -21, -22, -23, -25, -27, -28, -29, -30, -31, -35, -38, -39, -46, -48, -5, -50, -53, -55, -6, -60, -63, -64, -66, -7, -70, -9
- Exit nodes: 0
- Subgraph nodes: 0, -10, -12, -13, -17, -19, -21, -22, -23, -25, -27, -28, -29, -30, -31, -35, -38, -39, -46, -48, -5, -50, -53, -55, -6, -60, -63, -64, -66, -7, -70, -9

**Hypothesis (plain).** The model adds up seven parts into one score: plans, usage and state; heavy daytime use; customer-service calls; state adjustments; the daytime charge; account length; and Washington outside area code 415. [C051]

**Evidence (plain).** It is made up of seven parts (each explained below); the score is their sum, which was checked on all 4000 training customers. [C051] Changing one input at a time across the whole model, an international plan takes a typical customer's chance of leaving from about 3% to about 24%, and more customer-service calls raise it slowly at first, then faster, from about 2% at 0 to about 5% at 3 calls. [C001, C009] Daytime minutes raise it in an uneven way across the whole model, from about 2% at 112 to about 9% at 249 minutes. [C006]

**Annotation hypothesis (technical).** P(churn) = sigmoid(-2.182 + linear_backbone + csc_threshold + day_eve_minutes_hinge + day_charge_hinges + state_switches + area415_state47 + accountLength_affine): the seven children add in the logit; 31 inputs are unconnected (zero effect), including every duplicate charge column except night charge and day charge.

Evidence (train, n=4000): full AUC 0.898 vs refit L2 logistic regression 0.843. Replacing every child by its least-squares linear fit in its own inputs gives AUC 0.837, so the network's gain over linear is about 0.061 AUC, and it comes from two hinges: csc_threshold (restoring it alone to the linearised model +0.032; linearising it alone -0.019) and day_eve_minutes_hinge (+0.041 / -0.025); day_charge_hinges adds +0.006; the state units are exact category offsets (linear in their one-hots) and add nothing beyond linear. Mean-ablation per child: linear_backbone -0.084, csc_threshold -0.079, day_eve_minutes_hinge -0.038, state_switches -0.003, accountLength -0.001, day_charge_hinges -0.0004, area415_state47 -0.0003. SHAP (300 rows, probability scale) top: day minutes 0.067, voice mail plan 0.063, customer service calls 0.057, international plan 0.050, vmail messages 0.036, eve minutes 0.032. Rare-category: states 11 (-2.08, 73 rows), 46 (-1.92, 80), 39 (-1.25, 82), 47 (+1.26, 72), 26 (+1.08, 75), 50 (-1.07, 89), 41 (+0.98, 66) carry large logit offsets on few rows; state 37 disables the CSC threshold (effect -2.0 to +0.97 depending on calls, 84 rows). Neutralising all states costs only 0.0065 train AUC.

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = \frac{1}{8.87 e^{- 0.101 x_{10761} - 2.6 x_{1255} + 1.19 x_{1529} - 1.25 x_{216} - 3.02 x_{3227} - 0.271 x_{3384} - 0.456 x_{3930} - 0.293 x_{4311} + 0.259 x_{5685} + 0.126 x_{5853} - 0.203 x_{595} + 0.922 x_{6468} - 1.0 x_{identity_csc} - 1.0 x_{identity_lin}} + 1})
- *shap*:  (split: train; max_samples: 300)
- *local_ablation*:  (full_auc_train: 0.8979; all_linearised_auc: 0.8367; lr_refit_train_auc: 0.8433; hidden_units_mean_ablated_auc: 0.7408; all_hidden_leaves_mean_ablated_auc: 0.7177)


### Plans, usage and state (`ann_80`, `linear_backbone`)

- Kind: leaf; parent: full_model; children: none
- Reads: state, international calling plan, voice-mail plan, number of voice-mail messages, total evening minutes, total night-time minutes, total night-time charge, total international minutes, total international calls
- Entry nodes: -11, -14, -18, -1_b, -2, -20, -24, -26, -3, -32_b, -33, -36, -40_b, -41, -42, -43_b, -44, -45, -47, -49, -51, -56, -57, -58, -62_c, -65, -67, -68, -69, -8
- Exit nodes: identity_lin
- Subgraph nodes: -11, -14, -18, -1_b, -2, -20, -24, -26, -3, -32_b, -33, -36, -40_b, -41, -42, -43_b, -44, -45, -47, -49, -51, -56, -57, -58, -62_c, -65, -67, -68, -69, -8, identity_lin

**Hypothesis (plain).** This part reads the customer's plans, usage and state: an international plan raises the chance of leaving, more voice-mail messages raise it steadily, a voice-mail plan lowers it, and the state shifts it up or down. [C052, C053, C059, C054, C062]

**Evidence (plain).** Through it, an international plan takes a typical customer's chance from about 3% to about 24%, and a voice-mail plan to under 1%, so having a voice-mail plan lowers the chance of leaving while an international plan raises it; more voice-mail messages raise it steadily, to about 13% at 32 messages. [C052, C059, C053] The state moves it from about 3% in West Virginia to about 10% in Montana and down to about 1% in Vermont and Wyoming. [C054, C062, C064] Its formula reproduces the part exactly on all 4000 training customers. [C082]

**Annotation hypothesis (technical).** Identity node summing the 30 direct input->output edges: an additive logit backbone. Large terms: voiceMailPlan -1.20 and numberVmailMessages +0.71 (together a near-cancelling pair, net effect of having a voicemail plan), internationalPlan +0.72, intl minutes +0.21 / intl calls -0.22, night minutes +0.20 and night charge -0.05 (identical after z-scoring, net +0.14), eve minutes +0.09, plus 22 small state offsets.

Evidence (train, n=4000): Linear backbone (identity_lin, logit term sd 1.09): exact sum of 30 direct edges (linear R2 1.0). In logit units per category (z-score span x weight): voiceMailPlan -2.73 (1,061 rows) partly offset by numberVmailMessages +0.71/sd; internationalPlan +2.45 (380 rows); state:46 -1.92 (80 rows), state:26 +1.08, state:50 -1.07, state:44 -0.91 (cancelled by 1529), state:45 -0.84, state:24 -0.80, state:47 +0.81, state:40 +0.69. Night minutes +0.197 and night charge -0.053 are the same column after z-scoring (max diff 0.002): net +0.144/sd. Intl minutes +0.21/sd, intl calls -0.22/sd, eve minutes +0.09/sd. Mean-ablation on train drops AUC 0.898 -> 0.814.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= 0.719 x_{internationalPlan} \\
&\quad + 0.706 x_{numberVmailMessages} \\
&\quad + 0.0294 x_{state:0_b} \\
&\quad - 0.0729 x_{state:10} \\
&\quad + 0.00795 x_{state:15} \\
&\quad + 0.0365 x_{state:18} \\
&\quad - 0.0512 x_{state:1} \\
&\quad + 0.00491 x_{state:20} \\
&\quad - 0.112 x_{state:24} \\
&\quad + 0.146 x_{state:26} \\
&\quad + 0.0559 x_{state:31} \\
&\quad + 0.139 x_{state:37_b} \\
&\quad - 0.0597 x_{state:38} \\
&\quad - 0.0181 x_{state:3} \\
&\quad + 0.0885 x_{state:40} \\
&\quad - 0.134 x_{state:44_b} \\
&\quad - 0.126 x_{state:45} \\
&\quad - 0.268 x_{state:46} \\
&\quad + 0.107 x_{state:47_b} \\
&\quad - 0.0266 x_{state:48} \\
&\quad - 0.0606 x_{state:49} \\
&\quad - 0.158 x_{state:50} \\
&\quad - 0.0703 x_{state:7} \\
&\quad - 0.0325 x_{state:9} \\
&\quad + 0.0922 x_{totalEveMinutes_c} \\
&\quad - 0.223 x_{totalIntlCalls} \\
&\quad + 0.208 x_{totalIntlMinutes} \\
&\quad - 0.0531 x_{totalNightCharge} \\
&\quad + 0.197 x_{totalNightMinutes} \\
&\quad - 1.2 x_{voiceMailPlan}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: sensitivity)


### Heavy daytime use (`ann_82`, `day_eve_minutes_hinge`)

- Kind: leaf; parent: full_model; children: none
- Reads: total daytime minutes, total evening minutes
- Entry nodes: -59_b, -62_ab
- Exit nodes: 3227
- Subgraph nodes: -59_b, -62_ab, 3227, 6038

**Hypothesis (plain).** This part raises the chance of leaving for heavy daytime users: daytime minutes add nothing until about 234 minutes (between 199 and 263 minutes for different customers), then raise it steadily. [C086, C084]

**Evidence (plain).** It acts on 747 of 4000 training customers, 269 of whom left; for the typical customer it acts on, the chance is about 5% without it and about 23% with it. [C088] For a typical customer, daytime minutes add nothing until about 234 minutes (between 199 and 263 minutes for different customers), then take the chance steadily from about 3% to about 17% through it; evening minutes add to it in an uneven way. [C086, C083] It switches on when daytime minutes plus about 0.40 × evening minutes exceed about 319 (for evening minutes up to about 179; beyond that the line tilts to about 0.55 × evening minutes and 346); 747 of 4000 training customers are on that side, and 36% of them left, against 9% of the others. [C117] Its formula reproduces the part exactly on all 4000 training customers. [C089]

**Annotation hypothesis (technical).** High-usage hinge: 3227 = relu(-1.044 + 0.972*dayMinutes + 0.363*eveMinutes + 0.371*relu(0.177+0.382*eveMinutes)), read out with weight 3.019. Zero for typical usage, rising steeply for the heaviest day (and evening) users, the well-known high-bill churn segment.

Evidence (train, n=4000): High-usage hinge 3.019*relu(-1.044 + 0.972*dayMin + 0.363*eveMin + 0.371*relu(0.177+0.382*eveMin)). Zero for the lower 50% of day minutes; active in 13% of the 6th-7th deciles, 48% of the 9th and 93% of the top decile, where its mean logit term is +2.32 and train churn is 0.49 (vs 0.06-0.17 elsewhere). The inner eve-minutes relu makes the eve slope 0.363 (low eve) or 0.505 (high eve): heavy day+eve users (high total bill) churn. Linear R2 of the term only 0.41. Train AUC: mean-ablation -> 0.860; linearised -> 0.873.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.972 x_{totalDayMinutes_b} \\
&\quad + 0.363 x_{totalEveMinutes_ab} \\
&\quad + 0.371 \operatorname{relu}{\left(0.382 x_{totalEveMinutes_ab} + 0.177 \right)} \\
&\quad -1.04 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.972 x_{totalDayMinutes_b} \\
&\quad + 0.363 x_{totalEveMinutes_ab} \\
&\quad + 0.371 \operatorname{relu}{\left(0.382 x_{totalEveMinutes_ab} + 0.177 \right)} \\
&\quad -1.04 \bigr)
\end{aligned}; source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: heatmap)


### Customer-service calls (`ann_81`, `csc_threshold`)

- Kind: leaf; parent: full_model; children: none
- Reads: state, total daytime minutes, number of calls to customer service
- Entry nodes: -71, -32_a, -59_a
- Exit nodes: 1255, identity_csc
- Subgraph nodes: -32_a, -59_a, -71, 1255, identity_csc

**Hypothesis (plain).** This part raises the chance of leaving with each call to customer service, slowly at first, then faster, and for customers in Oregon steadily and much more. [C090, C093]

**Evidence (plain).** Through it, the chance rises slowly at first, then faster, from about 2% with no calls to about 5% with 3; in Oregon it rises steadily from under 1% to about 18%. [C090, C093] For customers in Ohio, daytime minutes add to the chance only above about 85 minutes, in an uneven way, from about 2% to about 7%. [C095] Its unit that reads the calls is on while customer-service calls minus about 0.0053 × daytime minutes stay below about 1.55, which holds for 3080 of 4000 training customers (11% of whom left, against 23% of the others). [C118] Its formula reproduces the part exactly on all 4000 training customers. [C096]

**Annotation hypothesis (technical).** Customer-service-calls threshold. identity_csc = 1.635*CSC (direct edge); 1255 = relu(0.003 - 2.878*state37 + 0.129*dayMinutes - 0.595*CSC) read out with weight 2.598, cancelling the CSC slope while 1255 is on (0-2 calls) and releasing it from 3-4 calls on. Net: flat logit for 0-2 calls, then ~+1.25 logit per extra call. State 37 rows switch 1255 off.

Evidence (train, n=4000): Customer-service-calls threshold. Logit term = 1.635*CSC + 2.598*relu(0.0033 - 0.595*CSC + 0.129*dayMinutes - 2.878*state37). For CSC 0-2 the relu is on in 95-98% of rows and cancels the slope (net 1.635-2.598*0.595 = +0.09/sd): mean term 0.94, 1.04, 1.11 for 0, 1, 2 calls. From 3 calls the relu shuts off (on in 3% at 3 calls, 0% at 4+): mean term 1.79, 3.04, 4.30, 5.55 for 3-6 calls, about +1.25 logit per call. Train churn by calls: 0-3 ~0.11, 4: 0.44, 5: 0.57, 6: 0.68, so the network places the threshold at 3-4 calls. A linear model cannot do this (linear R2 of the term 0.73). State 37 (84 rows, z=6.83) drives the relu negative, so for those rows CSC is linear throughout: logit effect of being in state 37 runs from -2.0 (0 calls) to +0.97 (3+ calls). Train AUC: mean-ablation 0.898 -> 0.819; replacing the term by its best linear fit 0.898 -> 0.879.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.595 x_{numberCustomerServiceCalls} \\
&\quad - 2.88 x_{state:37_a} \\
&\quad + 0.129 x_{totalDayMinutes_a} \\
&\quad + 0.00331 \bigr)
\end{aligned} \\ y_1 = 1.64 x_{numberCustomerServiceCalls}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.595 x_{numberCustomerServiceCalls} \\
&\quad - 2.88 x_{state:37_a} \\
&\quad + 0.129 x_{totalDayMinutes_a} \\
&\quad + 0.00331 \bigr)
\end{aligned} \\ y_1 = 1.64 x_{numberCustomerServiceCalls}; source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: sensitivity)


### State adjustments (`ann_84`, `state_switches`)

- Kind: leaf; parent: full_model; children: none
- Reads: state
- Entry nodes: -4, -1_a, -15, -37, -40_a
- Exit nodes: 5685, 5853, 10761, 1529
- Subgraph nodes: -15, -1_a, -37, -4, -40_a, 10761, 1529, 5685, 5853

**Hypothesis (plain).** This part lowers the chance of leaving for customers in most states and leaves it higher in South Dakota, Utah and Maine and lower in Hawaii and Alaska. [C102, C097, C098, C100, C099, C101]

**Evidence (plain).** It acts on 3845 of 4000 training customers; for the typical customer it acts on, the chance is about 7% without it and about 3% with it. [C102] Compared with other states, the chance through it is about 7% in South Dakota and Utah, about 5% in Maine, under 1% in Hawaii and about 1% in Alaska. [C097, C098, C100, C099, C101] Its formula reproduces the part exactly on all 4000 training customers. [C103]

**Annotation hypothesis (technical).** Single-category state offsets. 5685 (state 11), 5853 (state 0) and 10761 (state 21) are relu units that fire only for their state's ~60-90 rows, giving a fixed logit offset; 1529 = relu(0.688-0.455*state41-0.521*state44) is on for everyone except states 41 and 44, so it is a shared negative offset removed for those two states.

Evidence (train, n=4000): Category offsets (exact linear in the one-hots, linear R2 1.0). 5685: state 11 only (73 rows, churn 0.041), logit -2.08. 5853: state 0 (56 rows, churn 0.071), -1.10 hidden (+0.25 direct) = net -0.85. 10761: state 21 (86 rows, churn 0.186), +0.56. 1529 = relu(0.688-0.455*state41-0.521*state44): -0.98 for everyone, switched off for state 41 (66 rows, net +0.98) and state 44 (89 rows; +0.98 cancels the direct -0.91, net +0.08). These are large effects learned from ~60-90 rows each. Train AUC: mean-ablation 0.898 -> 0.895.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(1.08 x_{state:11} + 0.106 \right)} \\ y_1 = \operatorname{relu}{\left(1.09 x_{state:0_a} - 0.379 \right)} \\ y_2 = \operatorname{relu}{\left(1.01 x_{state:21} - 1.32 \right)} \\ \begin{aligned}
y_3 &= \operatorname{relu}\bigl( - 0.455 x_{state:41} \\
&\quad - 0.521 x_{state:44_a} \\
&\quad + 0.688 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = \operatorname{relu}{\left(1.08 x_{state:11} + 0.106 \right)} \\ y_1 = \operatorname{relu}{\left(1.09 x_{state:0_a} - 0.379 \right)} \\ y_2 = \operatorname{relu}{\left(1.01 x_{state:21} - 1.32 \right)} \\ \begin{aligned}
y_3 &= \operatorname{relu}\bigl( - 0.455 x_{state:41} \\
&\quad - 0.521 x_{state:44_a} \\
&\quad + 0.688 \bigr)
\end{aligned}; source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: sensitivity)


### Daytime charge (`ann_83`, `day_charge_hinges`)

- Kind: leaf; parent: full_model; children: none
- Reads: state, total daytime charge
- Entry nodes: -61, -16, -34
- Exit nodes: 216, 4311, 6468, 3384
- Subgraph nodes: -16, -34, -61, 216, 3384, 4311, 6468

**Hypothesis (plain).** This part adds a little for a high daytime charge: nothing until about 39.72 US dollars, then steadily. [C108, C105]

**Evidence (plain).** For a typical customer the charge adds nothing until about 39.72 US dollars, then raises the chance steadily by less than a point; a customer in Rhode Island has about 1% instead of about 3% through it. [C105, C104] Its formula reproduces the part exactly on all 4000 training customers. [C109]

**Annotation hypothesis (technical).** Small day-charge hinges plus a dead gate. 4311 = relu(-0.283+0.295*dayCharge); 216 = relu(-0.355+0.251*dayCharge-0.744*state22) (switched off in state 22); 6468 = relu(2.068+0.191*state39) is always on, so it is an affine state-39 offset; 3384 = relu(-1.268-1.162*216-0.527*6468) is never positive on train (dead). dayCharge equals dayMinutes after z-scoring, so these extend the high-usage hinge.

Evidence (train, n=4000): Four small units, day charge equals day minutes after z-scoring (max diff 5e-4). 4311 = relu(0.295*dayCharge-0.283) (w 0.293) and 216 = relu(0.251*dayCharge-0.744*state22-0.355) (w 1.253) are weak extensions of the day-usage hinge (each active ~16% of rows, together logit sd ~0.1); 216 is forced off in state 22 (80 rows; effect 0 to -0.47 logit). 6468 = relu(2.068+0.191*state39) is always on (min pre-activation 2.04): exact affine state-39 offset, -1.25 logit on 82 rows (train churn 0.061). 3384 = relu(-1.268-1.162*216-0.527*6468) has max pre-activation -2.34 on train: dead. Train AUC: mean-ablation -0.0004; linearised -0.001.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.744 x_{state:22} \\
&\quad + 0.251 x_{totalDayCharge} \\
&\quad -0.355 \bigr)
\end{aligned} \\ y_1 = \operatorname{relu}{\left(0.295 x_{totalDayCharge} - 0.283 \right)} \\ y_2 = \operatorname{relu}{\left(0.191 x_{state:39} + 2.07 \right)} \\ \begin{aligned}
y_3 &= \operatorname{relu}\bigl( - 0.527 \operatorname{relu}{\left(0.191 x_{state:39} + 2.07 \right)} \\
&\quad - 1.16 \operatorname{relu}{\left(- 0.744 x_{state:22} + 0.251 x_{totalDayCharge} - 0.355 \right)} \\
&\quad -1.27 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: sensitivity)


### Account length (`ann_86`, `accountLength_affine`)

- Kind: leaf; parent: full_model; children: none
- Reads: account length
- Entry nodes: -52
- Exit nodes: 3930
- Subgraph nodes: -52, 3930

**Hypothesis (plain).** This part raises the chance of leaving steadily with a longer account. [C110]

**Evidence (plain).** Through it, a typical customer's chance goes from about 2% at an account length of 48 to about 3% at 152, steadily. [C110] Its formula reproduces the part exactly on all 4000 training customers. [C111]

**Annotation hypothesis (technical).** 3930 = relu(1.149+0.23*accountLength) is always positive on train, so it is an exact affine term in accountLength (slope 0.105 on the logit).

Evidence (train, n=4000): 3930 = relu(1.149+0.23*accountLength) is always positive (min pre-activation 0.58), so it is exactly affine: logit +0.105 per sd of account length (correlation with input 1.0). Train AUC -0.0009 under mean-ablation.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.23 x_{accountLength} + 1.15 \right)}
```

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = \operatorname{relu}{\left(0.23 x_{accountLength} + 1.15 \right)}; source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: line_plot)


### Washington outside area code 415 (`ann_85`, `area415_state47`)

- Kind: leaf; parent: full_model; children: none
- Reads: state, area code
- Entry nodes: -43_a, -54
- Exit nodes: 595
- Subgraph nodes: -43_a, -54, 595

**Hypothesis (plain).** This part raises the chance of leaving for customers in Washington who are outside area code 415. [C114]

**Evidence (plain).** For such a customer the chance rises from about 3% to about 14%. [C114] Its formula reproduces the part exactly on all 4000 training customers. [C116]

**Annotation hypothesis (technical).** 595 = relu(-0.924+0.41*state47-1.077*areaCode415): small positive offset (~0.15) for rows outside area code 415, larger (~+3) for state 47 rows outside area 415, zero in area 415.

Evidence (train, n=4000): 595 = relu(-0.924 - 1.077*areaCode415 + 0.41*state47), w 0.203. Zero in area 415 (2,016 rows); 0.16 for other areas (logit +0.03); 3.19 for state 47 rows outside area 415 (logit +0.65 on top of state 47's direct +0.81). Small: train AUC -0.0003 under mean-ablation.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.08 x_{areaCode:415} \\
&\quad + 0.41 x_{state:47_a} \\
&\quad -0.924 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.08 x_{areaCode:415} \\
&\quad + 0.41 x_{state:47_a} \\
&\quad -0.924 \bigr)
\end{aligned}; source: get_formula force=true)
- *empirical_viz*:  (n: 4000; split: train; viz_type: heatmap)

## Worth checking (draft)

- Worth checking: night-time minutes raise the chance of leaving and night-time charge lowers it; in this data the charge is an exact multiple of the minutes, so the two pull against each other on the same information. [C065, C079]
- Worth checking: the model treats a voice-mail plan as lowering the chance of leaving but more voice-mail messages as raising it. [C059, C053]
- Worth checking: the state moves the chance by several points (for example Montana and Oregon high, Vermont and Wyoming low), based on 75 to 122 training customers per state. [C054, C055, C062, C064]
