# Explanation vb-7a480c

- Genome: `4b0eb5a6-5859-4ee8-80fe-18ab72f22959`
- Dataset: churn_onehot (training split 45962ea6-3c78-49f9-a01e-3665f6809b1e, n = 4000; test n = 1000)
- Outcome: churn (the customer left)
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 2%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 4.44e-16
- closed_form_max_abs_err: 0.0224
- n_leaf_annotations: 4
- n_composition_annotations: 1
- n_splits: 8
- n_identity_nodes: 0

**Summary.** Logit = linear term + three hinge groups: (1) service-call hinges (>=3-4 calls raise churn, much more for low day usage: +3.8 logit, 95% empirical churn at 4 calls with day z<-1); (2) day-usage hinges + total-charge chain gated by voice-mail plan (no plan with high day and eve: +5.2 logit, 94% churn); (3) linear terms (international plan +2.67, voice-mail plan -2.37, per-state offsets). Rare-state hidden units and relay essentially inert. No international plan x intl minutes interaction learned.

**What resisted.** Always-on relay (2377, 12977->5868->7162) takes exits from three sibling leaves so is root-owned (4 hidden nodes); affine on train. rareStateUnits leaf (14 nodes) is an honest near-dead annotation with zero measurable logit effect.

## Plain-language summary (draft)

For a typical customer (in West Virginia, area code 415, no international or voice-mail plan, 1 call to customer service and median usage) the model predicts a chance of leaving of about 2%; it adds up most of its inputs directly, and four parts add to that sum, each changing the chance for particular customers. [C069]

**Inputs that add up.** These add up independently: through this sum alone, an international plan takes a typical customer's chance of leaving from about 2% to about 24%, a voice-mail plan lowers it to under 1%, and more voice-mail messages raise it steadily, from about 2% at 0 to about 6% at 32 messages; all three also act through the parts below. [C002, C013, C008] More international minutes (through this sum alone, about 1% at 7 to about 4% at 14 minutes; they also act through the total bill, below), more evening minutes (about 2% at 138 to about 3% at 266 minutes), a higher night-time charge (about 2% at 6.13 to about 3% at 11.88 US dollars) and a longer account (about 2% at 48 to about 3% at 152) raise it steadily. [C011, C020, C023, C034] Through this sum alone, the state shifts it up or down: from about 2% in West Virginia to about 9% in California, about 7% in Washington and about 5% in New Jersey, and down to under 1% in Vermont and about 1% in Alaska and Wyoming (each of these states has 41 to 90 training customers; the state also acts through the parts below, and the full list is in the technical detail). [C004, C006, C009, C015, C021, C024] The area code barely changes it, and the model ignores daytime calls, international calls, the international charge, night-time calls and night-time minutes. [C054, C064, C065, C066, C067, C068]

**Customer-service calls.** This part combines customer-service calls, the daytime charge, the international plan and the state, and acts on 1303 of 4000 training customers, 302 of whom left; for the typical customer it acts on, the chance of leaving is about 2% without it and about 3% with it. [C080] Through it, customer-service calls add to the chance only from about 3 calls (between 1 and 3 calls for different customers) upwards, in an uneven way: for a typical customer, going from 0 to 4 calls takes it from about 2% to about 13%. [C078, C071] The calls count more for customers with a daytime charge of 30.69 US dollars or less, rising slowly at first, then faster, from about 2% to about 6% at 3 calls, against about 2% for those above it. [C075] A higher daytime charge lowers the chance through it, steadily up to about 15.64 US dollars (between 7.93 and 31.04 US dollars for different customers), then no further: for the typical customer it acts on, from about 11% at 19.07 to about 2% at 42.40 US dollars. [C070, C079]

**Heavy daytime use.** This part reads daytime minutes, the daytime charge and the voice-mail plan, and acts on 1033 of 4000 training customers, all without a voice-mail plan, 283 of whom left; for the typical customer it acts on, the chance is about 2% without it and about 6% with it. [C090] Through it, daytime minutes add nothing until about 218 minutes (between 171 and 265 minutes for different customers), then raise the chance steadily for customers without a voice-mail plan: for a typical customer from about 2% to about 7%. [C088, C083] Daytime minutes raise it only without a voice-mail plan, adding nothing until about 217 minutes, then steadily: from 112 to 249 minutes the chance goes from about 2% to about 5% without one, and stays below 1% with one. [C086] Through it, a voice-mail plan lowers the chance for the typical customer it acts on from about 6% to about 2%. [C082]

**Total bill.** This part reads the daytime and evening charges, voice-mail messages, international minutes and the state: through it, the daytime charge adds nothing until about 41.59 US dollars (between 34.11 and 46.36 US dollars for different customers), then raises the chance in an uneven way, for a typical customer from about 2% to about 8%. [C101, C094] The evening charge raises it only for customers with a daytime charge above 30.69 US dollars: nothing until about 20.56 US dollars, then steadily, from about 2% to about 5%, while for the others it stays at about 2%. [C098] More voice-mail messages raise it steadily up to about 20 messages (between 9 and 28 messages for different customers), then no further: for a typical customer from about 2% to about 4%. [C092, C103] More international minutes lower it through this part in an uneven way, from about 3% at 7 to about 2% at 14 minutes. [C093]

**Small state units.** This part reads the state and voice-mail messages and moves the chance by about a point at most: a customer in Utah has about 3% instead of about 2%, one in North Dakota about 2%, and voice-mail messages barely change it. [C107, C108, C112]

## Parts, from the root down

### The whole model (`ann_83`, `churnModelRoot`)

- Kind: composition; parent: none; children: dayUsageHinges, totalChargeChain, serviceCallsHinges, rareStateUnits
- Reads: state, account length, area code, international calling plan, voice-mail plan, number of voice-mail messages, total daytime minutes, total daytime calls, total daytime charge, total evening minutes, total evening calls, total evening charge, total night-time minutes, total night-time calls, total night-time charge, total international minutes, total international calls, total international charge, number of calls to customer service
- Entry nodes: -1, -17_a, -18, -23, -24, -26, -29, -30, -32, -35, -39, -42, -43, -44, -47_a, -52, -54, -56_a, -57_a, -58_a, -58_d, -62, -63, -67, -68_a, -7, -8, -10, -11, -12, -14, -15, -19, -20, -21, -25, -28, -3, -31, -33, -34, -36, -38, -4, -41, -45, -46, -48, -49, -50, -51, -53, -55, -6, -60, -65, -66, -69, -70, -9, 1254, 3460, 4045, 9871, 387, 553, 5395, 9203, 1887, 3854, 4332, 7740, 15985
- Exit nodes: 0
- Subgraph nodes: -1, -17_a, -18, -23, -24, -26, -29, -30, -32, -35, -39, -42, -43, -44, -47_a, -52, -54, -56_a, -57_a, -58_a, -58_d, -62, -63, -67, -68_a, -7, -8, -10, -11, -12, -14, -15, -19, -20, -21, -25, -28, -3, -31, -33, -34, -36, -38, -4, -41, -45, -46, -48, -49, -50, -51, -53, -55, -6, -60, -65, -66, -69, -70, -9, 0, 2377, 12977, 5868, 7162

**Hypothesis (plain).** The model adds up its inputs into one score: the plans, voice-mail messages, the state and several usage figures each move the chance of leaving on their own, and four parts add to that sum: customer-service calls, heavy daytime use, the total bill and small state units. [C069]

**Evidence (plain).** It is made up of the inputs that feed the score directly and four parts (each explained below); the score is their sum, which was checked on all 4000 training customers. [C069] Changing one input at a time across the whole model, an international plan takes a typical customer's chance of leaving from about 2% to about 24%, and more voice-mail messages raise it quickly at first, then level off, from about 2% at 0 to about 10% at 32 messages. [C001, C003] Across the whole model, daytime minutes add nothing until about 217 minutes, then raise it steadily, from about 2% at 112 to about 5% at 249 minutes. [C012]

**Annotation hypothesis (technical).** HYPOTHESIS: churn logit = -2.00 + linear term (direct edges) + dayUsageHinges + totalChargeChain + serviceCallsHinges + rareStateUnits + an always-on affine relay; output = sigmoid. Additive (parallel) composition at the output, plus a serial path: 553 (serviceCalls), 9871 (charge) and 3854 (states) feed the relay 12977/2377 -> 5868 -> 7162 -> output, which is active on all 4000 training rows and so contributes a constant (about -1.28) plus small linear terms. Linear term: international_plan +0.77/SD (+2.67 logit for plan holders), voice_mail_plan -0.86/SD (-2.37 for holders), vmail messages +0.41, intl minutes +0.45, eve minutes +0.22, night charge +0.22, account_length +0.12, and 17 per-state offsets (largest: state:46 -1.89, state:4 +1.48, state:47 +1.22, state:31 +0.86, state:0 -0.82, state:50 -0.76 logit on their 41-90 rows). 33 inputs are unconnected (zero effect). EVIDENCE (train, local mean-ablation and linearisation): full AUC 0.910; all hidden parts mean-ablated (direct edges only) 0.717; all hidden parts replaced by their best joint linear fit 0.835 (cf. L2 logistic regression train 0.843). So the nonlinear gain over a linear model is ~0.07 AUC, from service-call hinges with the low-usage interaction (0.021 alone) and the day/total-charge hinges with the voice-mail-plan gate (0.025 jointly), plus synergy between them (joint linearisation 0.074). rareStateUnits and the relay contribute ~0.

**Evidence records.**

- *analytical_formula*: 
- *shap*:  (split: train; base_value: 0.14697212660093426; max_samples: 300)
- *empirical_viz*:  (n: 4000; split: train)


### Customer-service calls (`ann_81`, `serviceCallsHinges`)

- Kind: leaf; parent: churnModelRoot; children: none
- Reads: state, international calling plan, total daytime charge, number of calls to customer service
- Entry nodes: -37, -56_b, -61_d, -71
- Exit nodes: 553, 5395, 9203
- Subgraph nodes: -37, -56_b, -61_d, -71, 553, 5395, 9203

**Hypothesis (plain).** This part raises the chance of leaving, in an uneven way, once a customer makes about 3 calls to customer service, and the calls count more when the daytime charge is low, rising slowly at first, then faster. [C078, C075]

**Evidence (plain).** It acts on 1303 of 4000 training customers, 302 of whom left; for the typical customer it acts on, the chance is about 2% without it and about 3% with it. [C080] For a typical customer, the calls add only from about 3 calls (between 1 and 3 calls for different customers) upwards, in an uneven way, and going from 0 to 4 calls takes the chance from about 2% to about 13%. [C078, C071] With a daytime charge of 30.69 US dollars or less the calls raise it slowly at first, then faster, to about 6% at 3 calls, against about 2% above it; a higher daytime charge lowers it steadily up to about 15.64 US dollars (between 7.93 and 31.04 US dollars for different customers), then no further, from about 11% to about 2% for the typical customer it acts on. [C075, C070, C079] Its formula reproduces the part exactly on all 4000 training customers. [C081]

**Annotation hypothesis (technical).** HYPOTHESIS: customer-service-call hinges, with an interaction against day usage: many calls by low-usage customers = churn. 9203 = relu(0.66*cs - 0.67) is on at >=3 calls (848 rows, churn 25%), logit +1.41 per unit. 5395 = relu(0.71*cs - 0.65*day_charge - 0.75) is on at >=4 calls unless day z > ~0.9, or 3 calls with below-average day usage (810 rows), logit +1.93 per unit (up to +6.0). 553 = relu(0.51*cs + 0.28*international_plan - 0.68*state:41 - 0.66): >=1 call with an international plan or >=4 without (555 rows, churn 46%); reaches the logit only via the always-on relay (+0.14 per unit, max +0.45); state:41 rows switch it off (66 rows, effect <= 0.15 logit). EVIDENCE (train): table of this leaf's logit contribution / empirical churn by calls x day z-bin: 4 calls & day z<-1: +3.8 / 95%; 4 calls & day z>1: +0.8 / 33%; 5 calls & day z<-1: +5.6 / 100%; <=2 calls: ~0 / 5-9% (except the high-usage bin, handled by the usage leaves). number_customer_service_calls has no direct edge to the output, so this leaf is its only route: mean-ablating the leaf costs 0.113 train AUC; replacing it by its best linear fit costs 0.021, the largest nonlinear gain of any part.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.282 x_{international_plan_b} \\
&\quad + 0.515 x_{number_customer_service_calls} \\
&\quad - 0.683 x_{state:41} \\
&\quad -0.66 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.71 x_{number_customer_service_calls} \\
&\quad - 0.645 x_{total_day_charge_d} \\
&\quad -0.753 \bigr)
\end{aligned} \\ \begin{aligned}
y_2 &= \operatorname{relu}\bigl( 0.66 x_{number_customer_service_calls} \\
&\quad -0.672 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: 
- *empirical_viz*:  (n: 4000; split: train)


### Heavy daytime use (`ann_79`, `dayUsageHinges`)

- Kind: leaf; parent: churnModelRoot; children: none
- Reads: voice-mail plan, total daytime minutes, total daytime charge
- Entry nodes: -57_b, -59, -61_a, -61_b
- Exit nodes: 1254, 3460
- Subgraph nodes: -57_b, -59, -61_a, -61_b, 1254, 3460

**Hypothesis (plain).** This part raises the chance of leaving for heavy daytime users without a voice-mail plan: daytime minutes raise it steadily from the lightest users upwards. [C088, C086]

**Evidence (plain).** It acts on 1033 of 4000 training customers, all without a voice-mail plan, 283 of whom left; for the typical customer it acts on, the chance is about 2% without it and about 6% with it. [C090] For a typical customer, daytime minutes add nothing until about 218 minutes (between 171 and 265 minutes for different customers), then take the chance steadily from about 2% to about 7%. [C088] With a voice-mail plan they add nothing: the chance stays below 1%, while without one it adds nothing until about 217 minutes, then rises steadily from about 2% to about 5%. [C086] They count more when the daytime charge is above 30.69 US dollars: nothing until about 192 minutes, then steadily, from about 2% to about 7%, against about 2% to about 3% otherwise. [C085] Its formula reproduces the part exactly on all 4000 training customers. [C091]

**Annotation hypothesis (technical).** HYPOTHESIS: two ReLU hinges on daytime usage that raise churn for heavy day users, mainly those without a voice-mail plan. total_day_minutes and total_day_charge are identical after z-scoring (max diff 5e-4), so the leaf uses both interchangeably. 1254 = relu(0.42*day - 0.73): on for day z > 1.74 (164 rows, churn 62% vs 12%), logit +2.08 per unit (up to +1.26). 3460 = relu(0.86*day - 0.83*voice_mail_plan - 0.85): on for day z > 0.39 without a voice-mail plan, z > 2.5 with one (995 rows, churn 28% vs 9.5%), logit +2.25 per unit (up to +5.3). EVIDENCE (train, n=4000): closed form above (max err 0.002); sensitivity: 3460 driven by voice_mail_plan (0.21) and day (0.21 total). Combined usage logit (with totalChargeChain): no voice-mail plan, day z>1 and eve z>1 -> +5.2 logit, empirical churn 94%; same usage with a voice-mail plan -> churn 10%. Replacing this leaf by its best linear fit costs 0.008 train AUC; mean-ablating 3460 costs 0.016.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.323 x_{total_day_charge_a} \\
&\quad + 0.0974 x_{total_day_minutes} \\
&\quad -0.731 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.337 x_{total_day_charge_b} \\
&\quad + 0.52 x_{total_day_minutes} \\
&\quad - 0.831 x_{voice_mail_plan_b} \\
&\quad -0.849 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.323 x_{total_day_charge_a} \\
&\quad + 0.0974 x_{total_day_minutes} \\
&\quad -0.731 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.337 x_{total_day_charge_b} \\
&\quad + 0.52 x_{total_day_minutes} \\
&\quad - 0.831 x_{voice_mail_plan_b} \\
&\quad -0.849 \bigr)
\end{aligned})
- *empirical_viz*:  (n: 4000; split: train)


### Total bill (`ann_80`, `totalChargeChain`)

- Kind: leaf; parent: churnModelRoot; children: none
- Reads: state, number of voice-mail messages, total daytime charge, total evening charge, total international minutes
- Entry nodes: -2_a, -58_b, -61_c, -61_e, -64, -68_b
- Exit nodes: 4045, 9871, 387
- Subgraph nodes: -2_a, -58_b, -61_c, -61_e, -64, -68_b, 5604, 4045, 9871, 387

**Hypothesis (plain).** This part adds to the chance of leaving for a high total bill: the daytime charge adds nothing until about 41.59 US dollars (between 34.11 and 46.36 US dollars for different customers), then raises it in an uneven way, and the evening charge counts only when the daytime charge is high. [C101, C098]

**Evidence (plain).** Through it, the daytime charge takes a typical customer's chance from about 2% to about 8% past about 41.59 US dollars (between 34.11 and 46.36 US dollars for different customers), in an uneven way. [C101, C094] More voice-mail messages raise it steadily up to about 20 messages (between 9 and 28 messages for different customers), then no further, from about 2% to about 4%. [C092, C103] Their rise is larger with more than 10 international minutes, quickly at first, then levelling off: from about 3% to about 14%, against about 2% to about 8% with 10 minutes or fewer. [C097] Its formula reproduces the part exactly on all 4000 training customers. [C106]

**Annotation hypothesis (technical).** HYPOTHESIS: a hinge chain on the combined day+evening bill. 5604 = relu(1.14*day_charge + 1.26*eve_charge - 0.94) (eve_charge == eve_minutes after z-scoring) is on for 1143 rows (churn 26%); 4045 = relu(0.60*day_charge + 0.73*5604 - 1.02) sharpens it (626 rows, churn 41% vs 9%), logit +1.57 per unit (up to +7.0). 9871 = relu(0.49*5604 - 0.84) is a higher threshold (244 rows, churn 56%) with a small negative logit weight (-0.24) and a tiny relay path. 387 = relu(0.94 - 1.07*vmail_messages + 0.56*intl_minutes + 0.04*state:1 - 0.86*4045) is on for 76% of rows; its -0.36 output weight lowers churn for customers with few voice-mail messages and many international minutes, and a high bill switches it off (removing that negative term). EVIDENCE: closed form (max err 0.008), sensitivity (4045: eve 0.14, day 0.22; 387: vmail messages 0.81, intl minutes 0.42). Linearising the leaf costs 0.008 AUC; jointly with dayUsageHinges 0.025 (the two overlap: both are day-usage hinges).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.598 x_{total_day_charge_c} \\
&\quad + 0.726 \operatorname{relu}{\left(1.14 x_{total_day_charge_e} + 1.26 x_{total_eve_charge} - 0.943 \right)} \\
&\quad -1.02 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.494 \operatorname{relu}{\left(1.14 x_{total_day_charge_e} + 1.26 x_{total_eve_charge} - 0.943 \right)} \\
&\quad -0.842 \bigr)
\end{aligned} \\ \begin{aligned}
y_2 &= \operatorname{relu}\bigl( - 1.07 x_{number_vmail_messages_b} \\
&\quad + 0.0379 x_{state:1_a} \\
&\quad + 0.559 x_{total_intl_minutes_b} \\
&\quad - 0.862 \operatorname{relu}{\left(0.598 x_{total_day_charge_c} + 0.726 \operatorname{relu}{\left(1.14 x_{total_day_charge_e} + 1.26 x_{total_eve_charge} - 0.943 \right)} - 1.02 \right)} \\
&\quad + 0.937 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: 
- *empirical_viz*:  (n: 4000; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.323 x_{total_day_charge_a} \\
&\quad + 0.0974 x_{total_day_minutes} \\
&\quad -0.731 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.337 x_{total_day_charge_b} \\
&\quad + 0.52 x_{total_day_minutes} \\
&\quad - 0.831 x_{voice_mail_plan_b} \\
&\quad -0.849 \bigr)
\end{aligned}; split: train)


### Small state units (`ann_82`, `rareStateUnits`)

- Kind: leaf; parent: churnModelRoot; children: none
- Reads: state, number of voice-mail messages
- Entry nodes: -13, -16, -17_b, -22, -27, -2_b, -40, -47_b, -47_c, -47_d, -5, -58_c
- Exit nodes: 1887, 3854, 4332, 7740, 15985
- Subgraph nodes: -13, -16, -17_b, -22, -27, -2_b, -40, -47_b, -47_c, -47_d, -5, -58_c, 3854, 13212, 8290, 948, 8671, 13450, 13939, 11368, 14533, 15149, 1887, 4332, 7740, 15985

**Hypothesis (plain).** This part adds small state effects: a customer in Utah gets a higher chance, one in North Dakota a lower one. [C107, C108]

**Evidence (plain).** Through it, a customer in Utah has about 3% instead of about 2%, one in North Dakota about 2%; voice-mail messages barely change it. [C107, C108, C112] Its formula reproduces the part exactly on all 4000 training customers. [C114]

**Annotation hypothesis (technical).** HYPOTHESIS (revised): rare-state units that are almost inert. 14 hidden nodes driven only by state one-hots (states 50, 23, 22, 1, 28, 32, 44, 12, 2; ~55-105 rows each, large z-values) plus a tiny vmail-messages term. The cascade 3854->13212->8290->948->{13939,11368}->14533->1887 is on for <=105 rows and reaches the output through weights -0.09 then -0.015: zero measurable effect. 1887 (state:28) moves the logit by at most -0.23; 7740 (state:44, always on via bias) by at most +0.40 on state:44 rows; 4332 (state:12) +0.07; 15985 (state:2) +0.02; 3854 (state:32/44) feeds the relay with activation <=0.08. EVIDENCE: closed form (max err 0.022, sympy rounding through the deep cascade); sensitivities all <0.04 except 7740/state:44 (0.98 per unit of activation but output weight 0.05). Mean-ablating the whole leaf changes train AUC by -0.0002; its best linear fit has R2 1.000. The large per-state effects in this network are in the root's direct linear edges, not here.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.515 x_{number_vmail_messages_c} \\
&\quad + 1.08 x_{state:28} \\
&\quad + 0.998 \operatorname{relu}{\left(0.994 x_{state:28} + 0.652 \right)} \\
&\quad - 0.0934 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(1.01 \operatorname{relu}{\left(- 0.147 x_{state:22} + 1.24 x_{state:50_b} - 1.27 \operatorname{relu}{\left(1.01 x_{state:1_b} - 0.0674 \right)} - 1.54 \operatorname{relu}{\left(1.0 x_{state:23_b} + 1.07 \right)} + 1.24 \operatorname{relu}{\left(1.0 x_{state:50_c} + 1.0 \operatorname{relu}{\left(1.0 x_{state:50_d} - 1.07 \operatorname{relu}{\left(0.173 x_{state:32} - 0.685 x_{state:44} - 1.3 \right)} - 0.392 \right)} + 0.0774 \right)} + 0.0738 \right)} + 0.181 \right)} + 1.0 \operatorname{relu}{\left(- 0.147 x_{state:22} + 1.24 x_{state:50_b} - 1.27 \operatorname{relu}{\left(1.01 x_{state:1_b} - 0.0674 \right)} - 1.54 \operatorname{relu}{\left(1.0 x_{state:23_b} + 1.07 \right)} + 1.24 \operatorname{relu}{\left(1.0 x_{state:50_c} + 1.0 \operatorname{relu}{\left(1.0 x_{state:50_d} - 1.07 \operatorname{relu}{\left(0.173 x_{state:32} - 0.685 x_{state:44} - 1.3 \right)} - 0.392 \right)} + 0.0774 \right)} + 0.0738 \right)} + 1.05 \right)} + 0.733 \right)} \\
&\quad - 0.0931 \operatorname{relu}{\left(1.0 \operatorname{relu}{\left(1.01 \operatorname{relu}{\left(- 0.147 x_{state:22} + 1.24 x_{state:50_b} - 1.27 \operatorname{relu}{\left(1.01 x_{state:1_b} - 0.0674 \right)} - 1.54 \operatorname{relu}{\left(1.0 x_{state:23_b} + 1.07 \right)} + 1.24 \operatorname{relu}{\left(1.0 x_{state:50_c} + 1.0 \operatorname{relu}{\left(1.0 x_{state:50_d} - 1.07 \operatorname{relu}{\left(0.173 x_{state:32} - 0.685 x_{state:44} - 1.3 \right)} - 0.392 \right)} + 0.0774 \right)} + 0.0738 \right)} + 0.181 \right)} + 1.0 \operatorname{relu}{\left(- 0.147 x_{state:22} + 1.24 x_{state:50_b} - 1.27 \operatorname{relu}{\left(1.01 x_{state:1_b} - 0.0674 \right)} - 1.54 \operatorname{relu}{\left(1.0 x_{state:23_b} + 1.07 \right)} + 1.24 \operatorname{relu}{\left(1.0 x_{state:50_c} + 1.0 \operatorname{relu}{\left(1.0 x_{state:50_d} - 1.07 \operatorname{relu}{\left(0.173 x_{state:32} - 0.685 x_{state:44} - 1.3 \right)} - 0.392 \right)} + 0.0774 \right)} + 0.0738 \right)} + 1.05 \right)} \\
&\quad -0.979 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.173 x_{state:32} \\
&\quad - 0.685 x_{state:44} \\
&\quad -1.3 \bigr)
\end{aligned} \\ y_2 = \operatorname{relu}{\left(1.14 x_{state:12} - 0.144 \right)} \\ y_3 = \operatorname{relu}{\left(0.984 x_{state:44} + 1.13 \right)} \\ y_4 = \operatorname{relu}{\left(1.0 x_{state:2} + 0.00966 \right)}
```

**Evidence records.**

- *analytical_formula*: 
- *empirical_viz*:  (n: 4000; split: train)

## Worth checking (draft)

- Worth checking: the model treats a voice-mail plan as lowering the chance of leaving but more voice-mail messages as raising it. [C013, C008]
- Worth checking: the state moves the chance by several points (for example California and Washington high, Vermont low), based on 41 to 90 training customers per state. [C004, C006, C015]
- Worth checking: through the customer-service part, a higher daytime charge lowers the chance of leaving, so frequent callers with low daytime use get the higher chance. [C070]
- Worth checking: international minutes raise the chance through the sum but lower it through the total bill. [C011, C093]
