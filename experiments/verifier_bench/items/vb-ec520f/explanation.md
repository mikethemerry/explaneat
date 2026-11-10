# Explanation vb-ec520f

- Genome: `9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66`
- Dataset: adult_ohe_v2 (training split 1df50169-8095-40d1-b61a-30bb541cd313, n = 39073; test n = 9769)
- Outcome: an income of $50,000 a year or less
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 62%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 4.440892098500626e-16
- closed_form_max_abs_err: 0.0650125195875102
- n_leaf_annotations: 7
- n_composition_annotations: 1
- n_splits: 2
- n_identity_nodes: 4

**Summary.** P(<=50K) = sigmoid(0.776 + sparse linear backbone + additive child terms); backbone does most of the work (ablation AUC 0.908 -> 0.748). Two nonlinear terms matter: a concave piecewise-linear capital-gain penalty (kink ~5.2k USD; ablation AUC 0.890) and a youth hinge (+0.16 logit per year below ~35; ablation AUC 0.900). Five tiny categorical gates. Train acc 0.852, AUC 0.908.

**What resisted.** Nothing resisted coverage. Four hidden nodes never activate on train (race gate 9712/5394; education:15/Portugal/Holand-Netherlands chain 12643/2197), documented as dead code. Five categorical gates real but negligible (AUC change <=0.0003 each). Category names and USD/year scales inferred.

## Plain-language summary (draft)

For a typical person (a 37-year-old White married man born in the United States, a high-school graduate in a professional job in the private sector, working 40 hours a week, with no capital gains or losses) the model predicts a chance of a low income ($50,000 or less) of about 62%; it adds up most of the person's characteristics independently, and seven parts add to that sum. [C082]

**Inputs that add up.** These add up independently, and these lower the chance: a higher education level lowers the predicted chance of a low income ($50,000 or less) steadily, from about 89% at 7 to about 24% at 13 on the 1-16 scale, flattening out near the top because the predicted chance cannot go much higher, and higher capital losses (about 62% at 0 to about 29% at 1887 US dollars) and longer working hours (about 71% at 24 to about 52% at 55 hours a week) lower it steadily. [C001, C010, C031] Being a wife rather than a husband lowers it to about 27%; through this sum alone, a highest education of 1st-4th grade lowers it to about 10%, 7th-8th grade to about 19%, and being born in Ireland to about 26% (highest education and country of birth also act through the parts below). [C009, C003, C005, C007] Being a child of the household rather than a husband raises it to about 94%, a marital status other than married to a civilian to about 87% and being a woman to about 80%; a missing work class raises it to about 79%, and through this sum alone farming or fishing raises it to about 83%, other service work to about 81% and being born in Vietnam to about 87% (occupation also acts through the parts below). [C011, C017, C034, C035, C025, C029, C019] Through this sum alone, being born in Scotland raises it to about 88% and being born in Honduras lowers it to about 43% (the model has little data on this, n = 16 and 14). [C013, C032]

**Capital gains.** Through this part, higher capital gains lower a typical person's predicted chance of a low income ($50,000 or less) slowly at first, then faster: from about 62% at 0 to about 6% at 6849 US dollars, flattening out near the bottom because the predicted chance cannot go much lower. [C083]

**Youth.** Through this part, older age lowers a typical person's predicted chance of a low income ($50,000 or less) steadily up to about 35 years, then no further: from about 93% at 22 to about 62% at 58 years. [C085, C086]

**Handlers, doctorates and Cambodia.** Through this part, handling or cleaning raises a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 76%, while a doctorate lowers it to about 52% and being born in Cambodia to about 52% (the model has little data on this, n = 19). [C088, C089, C090]

**Private household service.** This part acts only on the 187 people in private household service, 185 of whom have an income of $50,000 or less; through it, such work raises a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 87%. [C093, C092]

**Technical support and two countries.** Through this part, a technical-support job lowers a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 52%, while being born in Guatemala or a missing country of birth raises it a little, to about 65%. [C095, C096, C097]

**Born in China.** This part acts only on the 96 people born in China, 68 of whom have an income of $50,000 or less; through it, being born in China raises a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 85%. [C101, C100]

**Asian or Pacific Islander.** Through this part, being Asian or Pacific Islander lowers a typical person's predicted chance of a low income ($50,000 or less) a little, from about 62% to about 58%. [C103]

## Parts, from the root down

### The whole model (`ann_120`, `income_logit`)

- Kind: composition; parent: none; children: youth_hinge, capital_gain_penalty, tech_support_gate, asian_pac_offset, handlers_doctorate_relay, priv_house_serv_gate, china_gate
- Reads: age, work class, census sampling weight, highest education, education level (1 = preschool to 16 = doctorate), marital status, occupation, relationship in the household, race, sex, capital gains, capital losses, hours worked per week, country of birth
- Entry nodes: identity_age, identity_capgain, identity_techsupport, identity_asianpac, 7165, 11966, 10191, -2, -3, -4, -6, -8, -9, -11, -20, -21, -23, -28, -31, -38, -39, -45, -46, -49, -51, -54, -56, -59, -62, -64, -65, -68, -72, -74, -75, -76, -77, -80, -82, -83, -94, -101, -103, -104, -107, -92_a, -73_a, -106, -105, -102, -100, -99, -98, -97, -96, -95, -93, -91, -90, -88, -87, -86, -85, -84, -81, -79, -78, -70, -69, -60, -57, -55, -53, -52, -48, -44, -43, -42, -40, -37, -36, -35, -34, -33, -32, -30, -29, -27, -26, -25, -24, -22, -18, -17, -16, -15, -13, -12, -10, -7, -5
- Exit nodes: 0
- Subgraph nodes: -2, -3, -4, -6, -8, -9, -11, -20, -21, -23, -28, -31, -38, -39, -45, -46, -49, -51, -54, -56, -59, -62, -64, -65, -68, -72, -74, -75, -76, -77, -80, -82, -83, -94, -101, -103, -104, -107, -92_a, -73_a, -106, -105, -102, -100, -99, -98, -97, -96, -95, -93, -91, -90, -88, -87, -86, -85, -84, -81, -79, -78, -70, -69, -60, -57, -55, -53, -52, -48, -44, -43, -42, -40, -37, -36, -35, -34, -33, -32, -30, -29, -27, -26, -25, -24, -22, -18, -17, -16, -15, -13, -12, -10, -7, -5, 0

**Hypothesis (plain).** The model adds up most of a person's characteristics on their own into one score, and seven parts add to that sum: capital gains, youth, and five single-category adjustments. [C082]

**Evidence (plain).** It is made up of the inputs that feed the score directly and seven parts (each explained below); the score is their sum, which was checked on all 39073 training people. [C082] Changing one input at a time across the whole model, a higher education level takes a typical person's chance steadily from about 89% at 7 to about 24% at 13 on the 1-16 scale, flattening out near the top because the predicted chance cannot go much higher, and higher capital gains lower it slowly at first, then faster, from about 62% at 0 to about 6% at 6849 US dollars, flattening out near the bottom because the predicted chance cannot go much lower. [C001, C002] Across the whole model, a highest education of 1st-4th grade lowers it to about 10% and 7th-8th grade to about 19%, and being born in Ireland to about 26%. [C004, C006, C008] Across the whole model, being a child of the household raises it to about 94% and private household service rather than a professional job to about 87%, while being a wife rather than a husband lowers it to about 27%. [C011, C018, C009]

**Annotation hypothesis (technical).** Hypothesis: the network is an additive logistic model. logit P(<=50K) = 0.776 + sparse linear backbone over 40 direct inputs + sum of 7 child contributions, then sigmoid. Backbone (direct edges, standardised inputs): education-num -1.41, marital-status:2 (Married-civ-spouse) -0.73, relationship:5 (Wife) -0.56, relationship:0 (Husband) -0.56, relationship:3 (Own-child) +0.43, sex (Male) -0.42, hours-per-week -0.33, capital-loss -0.30, plus small workclass/occupation/education/country offsets. Children: capital_gain_penalty (concave piecewise-linear, kink ~5.2k USD), youth_hinge (+0.16 logit per year below ~35), and five near-inert categorical gates (Handlers-cleaners/Doctorate relay, Tech-support, Priv-house-serv, China, Asian-Pac offset). The children have no multiplicative interactions; they meet only at the output sum. 54 one-hot inputs are unconnected (zero effect); native-country:15 is constant in train; four hidden nodes (9712, 5394, 12643, 2197) never activate. Evidence: closed form (get_formula); train acc 0.852, AUC 0.908 (n=39073); local mean-ablations: backbone AUC->0.748, capital gain ->0.890, age ->0.900, each gate costs <=0.0003 AUC; KernelSHAP (200 train samples) top mean|SHAP|: education-num 0.105, marital-status:2 0.068, relationship:0 0.054, identity_age 0.046, identity_capgain 0.040, sex 0.026. Supported.

**Evidence records.**

- *analytical_formula*:  (composition: additive in the logit: z = 0.776 + linear backbone + sum of 7 child exits (identity_* exits weight 1.0; 7165 x0.294, 11966 x0.1003, 10191 x0.0621); output = sigmoid(z))
- *performance*:  (n: 39073; mae: 0.18544741232716244; mse: 0.10290348354022202; rmse: 0.3207857283923679; split: train; auc_roc: 0.9078326749585034; accuracy: 0.8520461699895068; n_samples: 39073; correlation: 0.6645991166532825)
- *shap*:  (split: train; method: compute_shap (KernelSHAP) on root entry nodes, output-probability scale; base_value: 0.739943393309731; max_samples: 200)


### Capital gains (`ann_114`, `capital_gain_penalty`)

- Kind: leaf; parent: income_logit; children: none
- Reads: capital gains
- Entry nodes: -63
- Exit nodes: identity_capgain
- Subgraph nodes: -63, 9352, identity_capgain

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) as capital gains rise, slowly at first, then faster. [C083]

**Evidence (plain).** Through it, a typical person's chance goes from about 62% at 0 to about 6% at 6849 US dollars, slowly at first, then faster, flattening out near the bottom because the predicted chance cannot go much lower. [C083] Its formula reproduces the part exactly on all 39073 training people. [C084]

**Annotation hypothesis (technical).** Hypothesis: capital gain's whole effect, a concave piecewise-linear penalty on P(<=50K). Contribution = -2.787*cg - 3.581*relu(0.8175*cg-0.453): slope -2.79 per sd below z=0.554 (~5.2k USD), -5.71 per sd above it. +0.41 logit at cg=0 (95% of rows), -9.0 at the 99th pct, -73.7 at 99,999 USD (output saturated to ~0). Evidence: closed form; line plot on full train, corr -0.997; ablation AUC 0.9078->0.8896, acc 0.852->0.832. The largest non-linear effect in the network. Supported.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 2.79 x_{capital-gain} \\
&\quad - 3.58 \operatorname{relu}{\left(0.818 x_{capital-gain} - 0.453 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= - 2.79 x_{capital-gain} \\
&\quad - 3.58 \operatorname{relu}{\left(0.818 x_{capital-gain} - 0.453 \right)}
\end{aligned}; source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: line_plot; local_ablation: Local ablation: AUC 0.9078->0.8896, acc 0.8520->0.8319, mean |dlogit| 1.264. 9352 active on 4.50% of rows. Contribution +0.406 at cg=0 (95% of rows), -1.46 at the 95th pct, -8.99 at the 99th pct, -73.7 at the max (99,999 USD). Kink at z=0.554 (~5.2k USD, from max z 13.19 => sd ~7.5k USD).)


### Youth (`ann_113`, `youth_hinge`)

- Kind: leaf; parent: income_logit; children: none
- Reads: age
- Entry nodes: -1
- Exit nodes: identity_age
- Subgraph nodes: -1, 599, 8375, 3328, identity_age

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for younger people: it falls steadily as age rises up to about 35 years, then no further. [C085, C086]

**Evidence (plain).** It acts on 16819 of 39073 training people, those younger than about 35 years. [C086] Through it, a typical person's chance goes from about 93% at 22 to about 62% at 58 years, steadily up to about 35 years, then no further. [C085, C086] Its formula reproduces the part exactly on all 39073 training people. [C087]

**Annotation hypothesis (technical).** Hypothesis: a youth bonus to P(<=50K). Age enters only via the ReLU hinge 599=relu(-1.763*age-0.494), active below z=-0.280 (~35 years, using est. mean 38.6/sd 13.7 y). 8375 and 3328 are always active, so they are linear relays and the contribution collapses to 1.258*relu(-1.763*age-0.494)+0.463: flat (0.463) at 35+ years, +0.161 logit per year younger, 3.335 at age 17. Evidence: closed form from get_formula; line plot on full train (n=39073) shows the hinge, corr -0.78; local ablation AUC 0.9078->0.9002. Supported.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= 0.371 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} \\
&\quad + 1.3 \operatorname{relu}{\left(0.31 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} + 0.812 \operatorname{relu}{\left(0.456 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} + 0.0871 \right)} + 0.285 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= 0.371 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} \\
&\quad + 1.3 \operatorname{relu}{\left(0.31 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} + 0.812 \operatorname{relu}{\left(0.456 \operatorname{relu}{\left(- 1.76 x_{age} - 0.494 \right)} + 0.0871 \right)} + 0.285 \right)}
\end{aligned}; source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: line_plot; local_ablation: Local ablation (contribution replaced by train mean): AUC 0.9078->0.9002, acc 0.8520->0.8502, mean |dlogit| 0.739. Contribution by age z: 0.463 for z>=-0.264 (age>=35 y, est. mean 38.6 sd 13.7); 0.590 at 34 y ... 3.335 at 17 y (step +0.161/yr). 8375 and 3328 active on 100% of rows (pure linear relays); 599 active on 43.05%.)


### Handlers, doctorates and Cambodia (`ann_117`, `handlers_doctorate_relay`)

- Kind: leaf; parent: income_logit; children: none
- Reads: highest education, occupation, country of birth
- Entry nodes: -14, -47, -67
- Exit nodes: 7165
- Subgraph nodes: -14, -47, -67, 5993, 7165

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for handlers and cleaners and lowers it for people with a doctorate or born in Cambodia (the model has little data on this, n = 19). [C088, C089, C090]

**Evidence (plain).** Through it, handling or cleaning takes a typical person's chance from about 62% to about 76% (1683 training people). [C088] A doctorate lowers it through this part to about 52% (473 training people), and being born in Cambodia to about 52% (the model has little data on this, n = 19). [C089, C090] Its formula reproduces the part exactly on all 39073 training people. [C091]

**Annotation hypothesis (technical).** Hypothesis: a three-level categorical offset. Gate 5993=relu(1.861-0.453*edu10+0.605*occ6-0.729*nc1) is relayed linearly by 7165 (always active) into the output with weight 0.294. Contribution: baseline 0.683; occupation:6 (inferred Handlers-cleaners) 1.328 (+0.645, more likely <=50K); education:10 (Doctorate) or native-country:1 (Cambodia) clamp 5993 to 0, giving 0.294 (-0.389). Evidence: closed form; exhaustive value table on train (36898/1683/492 rows); ablation AUC 0.9078->0.9076. Supported.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.737 \operatorname{relu}{\left(- 0.453 x_{education:10} - 0.729 x_{native-country:1} + 0.604 x_{occupation:6} + 1.86 \right)} \\
&\quad + 0.999 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.737 \operatorname{relu}{\left(- 0.453 x_{education:10} - 0.729 x_{native-country:1} + 0.604 x_{occupation:6} + 1.86 \right)} \\
&\quad + 0.999 \bigr)
\end{aligned}; source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: sensitivity; local_ablation: Local ablation: AUC 0.9078->0.9076, acc 0.8512, mean |dlogit| 0.054. Contribution (0.294*7165) takes 3 values: 0.683 (n=36898, baseline), 1.328 (n=1683, occupation:6), 0.294 (n=492, education:10 or native-country:1 clamps 5993 to 0). 5993 active 98.74%, 7165 active 100% (linear relay).)


### Private household service (`ann_118`, `priv_house_serv_gate`)

- Kind: leaf; parent: income_logit; children: none
- Reads: highest education, occupation, country of birth
- Entry nodes: -19, -92_b, -73_b, -50
- Exit nodes: 11966
- Subgraph nodes: -19, -92_b, -73_b, -50, 12643, 2197, 11966

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people in private household service. [C092]

**Evidence (plain).** It acts only on the 187 training people in private household service, 185 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 99% without it and above 99% with it. [C093] Through it, such work takes a typical person's chance from about 62% to about 87%. [C092] Its formula reproduces the part exactly on all 39073 training people. [C094]

**Annotation hypothesis (technical).** Hypothesis: occupation:9 (inferred Priv-house-serv) switch. 11966=relu(1.0045*occ9-0.516+0.968*2197) is 13.97 for the 287 occupation:9 rows (+1.40 logit, more likely <=50K), else 0. The upstream chain (2197 <- education:15, native-country:32, 12643 <- native-country:15) never activates on train: native-country:15 is constant and 0.907*max(edu15)=1.70 < bias 1.72. So the leaf reduces to relu(1.0045*occ9-0.516). Evidence: closed form; sensitivity only on occ9; ablation AUC unchanged. Supported; dead branch noted.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 x_{occupation:9} \\
&\quad + 0.968 \operatorname{relu}{\left(0.907 x_{education:15} - 0.0292 x_{native-country:32_b} - 0.108 \operatorname{relu}{\left(1.0 x_{native-country:15_b} - 0.878 \right)} - 1.72 \right)} \\
&\quad -0.516 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.0 x_{occupation:9} \\
&\quad + 0.968 \operatorname{relu}{\left(0.907 x_{education:15} - 0.0292 x_{native-country:32_b} - 0.108 \operatorname{relu}{\left(1.0 x_{native-country:15_b} - 0.878 \right)} - 1.72 \right)} \\
&\quad -0.516 \bigr)
\end{aligned}; source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: sensitivity; local_ablation: Local ablation: AUC unchanged, mean |dlogit| 0.013. 11966 is 0 for 38886 rows and 13.97 (contribution +1.401 logit) for the 187 occupation:9 rows. 12643 and 2197 active 0.00% of rows (dead): native-country:15 is constant in train and 0.907*max(edu15)=1.70 < 1.72.)


### Technical support and two countries (`ann_115`, `tech_support_gate`)

- Kind: leaf; parent: income_logit; children: none
- Reads: occupation, country of birth
- Entry nodes: -41, -71, -66
- Exit nodes: identity_techsupport
- Subgraph nodes: -41, -71, -66, 2892, 8892, 12120, identity_techsupport

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) for people in technical support, and raises it a little for people born in Guatemala or with a missing country of birth. [C095, C096, C097]

**Evidence (plain).** Through it, a technical-support job takes a typical person's chance from about 62% to about 52% (1173 training people). [C095] Being born in Guatemala raises it to about 65% (72 training people), and a missing country of birth also to about 65% (677 training people). [C096, C097] Its formula reproduces the part exactly on all 39073 training people. [C099]

**Annotation hypothesis (technical).** Hypothesis: small per-category offsets for occupation:13 (inferred Tech-support) and two rare countries. Three ReLU gates (2892, 8892, 12120) plus the direct occupation:13 edge, all off unless one of the one-hot inputs is set. Train table: baseline -0.003; Tech-support -0.39 (or -0.16 with nc13); native-country:0 ('?') alone +0.13; native-country:13 alone +0.15. Evidence: closed form; sensitivity on full train (occ13 dominant); ablation leaves AUC unchanged (mean |dlogit| 0.023). Supported but practically negligible.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= 0.015 x_{occupation:13} \\
&\quad - 0.179 \operatorname{relu}{\left(1.02 x_{occupation:13} - 0.197 \right)} \\
&\quad + 0.0205 \operatorname{relu}{\left(1.05 x_{native-country:0} + 1.44 \operatorname{relu}{\left(0.0943 x_{native-country:13} + 0.963 x_{occupation:13} - 0.334 \right)} - 1.25 \right)} \\
&\quad + 0.0781 \operatorname{relu}{\left(0.0943 x_{native-country:13} + 0.963 x_{occupation:13} - 0.334 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: sensitivity; local_ablation: Local ablation: AUC unchanged to 4 d.p. (0.9077), acc 0.8515, mean |dlogit| 0.023. Exhaustive table on train: neither (n=37244) -0.003 or +0.151 (nc13); nc0 only (656) +0.133; occ13 only (1152) -0.392 or -0.156; both (21) -0.228. 2892 active 3.18%, 8892 4.86%, 12120 3.00%.)


### Born in China (`ann_119`, `china_gate`)

- Kind: leaf; parent: income_logit; children: none
- Reads: country of birth
- Entry nodes: -89
- Exit nodes: 10191
- Subgraph nodes: -89, 10191

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people born in China. [C100]

**Evidence (plain).** It acts only on the 96 training people born in China, 68 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 22% without it and about 49% with it. [C101] Through it, being born in China takes a typical person's chance from about 62% to about 85%. [C100] Its formula reproduces the part exactly on all 39073 training people. [C102]

**Annotation hypothesis (technical).** Hypothesis: a native-country:3 (inferred China) switch. 10191=relu(1.0132*nc3-0.419) is 20.0 for the 96 China rows (+1.24 logit after weight 0.0621), else 0. Evidence: closed form; line plot corr 1.0 over the live range; ablation AUC 0.9078->0.9077. Supported.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(1.01 x_{native-country:3} - 0.419 \right)}
```

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = \operatorname{relu}{\left(1.01 x_{native-country:3} - 0.419 \right)}; source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: line_plot; local_ablation: Local ablation: AUC 0.9077, mean |dlogit| 0.006. 10191 is 0 for 38977 rows and 20.0 (contribution 0.0621*20.0 = +1.242 logit) for the 96 native-country:3 rows.)


### Asian or Pacific Islander (`ann_116`, `asian_pac_offset`)

- Kind: leaf; parent: income_logit; children: none
- Reads: race
- Entry nodes: -61, -58
- Exit nodes: identity_asianpac
- Subgraph nodes: -61, -58, 9712, 5394, identity_asianpac

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) a little for Asian or Pacific Islander people. [C103]

**Evidence (plain).** Through it, being Asian or Pacific Islander rather than White takes a typical person's chance from about 62% to about 58%; 1201 training people are Asian or Pacific Islander. [C103] Its formula reproduces the part exactly on all 39073 training people. [C105]

**Annotation hypothesis (technical).** Hypothesis: a small negative offset for race:1 (inferred Asian-Pac-Islander), -0.149 logit relative to others, via the direct edge. The race:4 (White) gate 9712 and the gate 5394 are dead on train (0% active; bias outweighs every reachable input), contributing a constant 0. Evidence: closed form; sensitivity zero for race:4; ablation AUC unchanged. Supported; the gates are dead code.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 0.0256 x_{race:1} \\
&\quad + 0.0413 \operatorname{relu}{\left(- 1.65 x_{race:1} + 1.0 \operatorname{relu}{\left(0.882 x_{race:4} - 1.06 \right)} - 0.866 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= - 0.0256 x_{race:1} \\
&\quad + 0.0413 \operatorname{relu}{\left(- 1.65 x_{race:1} + 1.0 \operatorname{relu}{\left(0.882 x_{race:4} - 1.06 \right)} - 0.866 \right)}
\end{aligned}; source: get_formula force=true; tractable: True)
- *empirical_viz*:  (n: 39073; split: train; viz_type: sensitivity; local_ablation: Local ablation: AUC 0.9078 unchanged, mean |dlogit| 0.009. Values: -0.144 for race:1 (1201 rows), +0.005 otherwise. 9712 and 5394 are active on 0.00% of train rows (dead).)

## Worth checking (draft)

- Worth checking: the model treats capital losses (0 to 1887 US dollars) as lowering a typical person's predicted chance of a low income ($50,000 or less), from about 62% to about 29%. [C010]
- Worth checking: being born in Scotland raises the predicted chance of a low income ($50,000 or less) to about 88% and in Vietnam to about 87%, while being born in Honduras lowers it to about 43%, from groups of 16, 62 and 14 people. [C014, C020, C033]
- Worth checking: being a wife rather than a husband lowers the predicted chance of a low income ($50,000 or less) to about 27%, while being a woman rather than a man raises it to about 80%. [C009, C034]
- Worth checking: a highest education of 1st-4th grade lowers the predicted chance of a low income ($50,000 or less) to about 10% and 7th-8th grade to about 19%, although a higher education level lowers it too. [C004, C006, C001]
- Worth checking: the census sampling weight, which describes the survey rather than the person, lowers the predicted chance of a low income ($50,000 or less) from about 64% to about 59%. [C062]
