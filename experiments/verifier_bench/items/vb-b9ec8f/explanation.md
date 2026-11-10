# Explanation vb-b9ec8f

- Genome: `7a8237f3-74cb-40bd-8268-b605f64dac2b`
- Dataset: adult_ohe_v2 (training split 1df50169-8095-40d1-b61a-30bb541cd313, n = 39073; test n = 9769)
- Outcome: an income of $50,000 a year or less
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 63%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 5.551115123125783e-16
- closed_form_max_abs_err: 0.04082495714157375
- n_leaf_annotations: 9
- n_composition_annotations: 2
- n_splits: 6
- n_identity_nodes: 7

**Summary.** GAM: P(income<=50K) = sigmoid(3.385 + additive terms): sparse linear backbone (ablation AUC 0.912 -> 0.790); concave capital-gain penalty (steepening ~$4.8k and $5.2k, with country/education switches; ablation AUC 0.894); a V-shaped age term (vertex at 46); a short-hours hinge; small category offsets. Train acc 0.858, AUC 0.912.

**What resisted.** Nothing resisted coverage. Degenerate gates: 4262 always on (Holand-Netherlands never occurs in train), 3843 and 7420 always on (linear). Thailand gate's -4.0 direct effect and age-hinge release nearly cancel (net +0.10 over 26 rows). Rare standardised one-hots act as extreme switches (Married-AF-spouse -59, Never-worked +30 logit). Category names and kinks inferred.

## Plain-language summary (draft)

For a typical person (a 37-year-old White married man born in the United States, a high-school graduate in a professional job in the private sector, working 40 hours a week, with no capital gains or losses) the model predicts a chance of a low income ($50,000 or less) of about 63%; it adds up eight parts: capital gains, most characteristics added up on their own, age, working hours, country of birth, self-employment, and two small sets of single-category adjustments. [C072]

**Capital gains.** Through this part, higher capital gains lower a typical person's predicted chance of a low income ($50,000 or less) slowly at first, then faster: from about 63% at 0 to about 9% at 6849 US dollars, flattening out near the bottom because the predicted chance cannot go much lower. [C073] For people whose highest education is 9th grade the fall is smaller, again slowly at first, then faster, from about 44% to about 6%, and for people born in Jamaica it ends higher, from about 71% to about 16%. [C081, C083] It is made of two parts: the main capital-gains curve, and a second one that adds nothing until about 4883 US dollars, then lowers the chance steadily (about 63% to about 56% through it alone). [C084, C126]

**Characteristics that add up.** This part adds up most characteristics, each on its own: a higher education level lowers a typical person's predicted chance of a low income ($50,000 or less) steadily, from about 84% at 7 to about 35% at 13 on the 1-16 scale, and higher capital losses lower it steadily, from about 63% at 0 to about 31% at 1887 US dollars. [C091, C095] Being married to a member of the armed forces lowers it to below 1%, while a marital status other than married, divorced or widowed raises it to about 97%, being divorced to about 95% and widowed to about 91%; being a wife lowers it to about 41% and being a woman raises it to about 74%. [C090, C093, C094, C096, C099, C109] Compared with a professional job, farming or fishing raises it to about 87%, other service work to about 85% and handling or cleaning to about 83%, while an executive or managerial job lowers it to about 56%; never having worked raises it to above 99% (the model has little data on this, n = 9) and federal government work lowers it to about 52%. [C098, C100, C101, C115, C092, C106] A highest education of 7th-8th grade lowers it to about 35%, 9th grade to about 44% and 10th grade to about 49%. [C097, C102, C105]

**Age.** Through this part, a typical person's predicted chance of a low income ($50,000 or less) is lowest at about 47 years and higher either side: from about 84% at 22 to about 55% at 58 years. [C137] For people born in Thailand it goes the other way, rising with age from about 66% to about 85%. [C139] Through it, being born in Ireland lowers the chance from about 63% to about 26%. [C136]

**Short working hours.** This part acts on the 31364 people who work less than about 50 hours a week, 25290 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance of a low income ($50,000 or less) is about 62% without it and about 73% with it. [C143] Through it, longer working hours lower a typical person's chance steadily up to about 50 hours a week, then no further: from about 79% at 24 to about 51% at 55 hours. [C141, C142]

**Country of birth.** Compared with the United States, being born in Colombia raises a typical person's predicted chance of a low income ($50,000 or less) to above 99%, in Scotland to about 94% (the model has little data on this, n = 16), in 'South' (as recorded) to about 89%, in Peru to about 87% and in China to about 84%. [C145, C146, C147, C149, C150] Being born in Portugal lowers it to about 38% and in England to about 52%. [C148, C152]

**Self-employed, not incorporated.** This part acts on the 3034 people who are self-employed without incorporation, 2215 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance of a low income ($50,000 or less) is about 60% without it and about 74% with it. [C163]

**Other relatives.** Through this part, being another relative in the household raises a typical person's predicted chance of a low income ($50,000 or less) from about 63% to about 82%, while some college as the highest education barely changes it (about 64%). [C165, C166]

**Rare categories.** Through this part, being born in Trinidad and Tobago raises a typical person's predicted chance of a low income ($50,000 or less) to about 92% and preschool as the highest education to about 88%, while being born in Yugoslavia lowers it to about 33%. [C169, C170, C168]

## Parts, from the root down

### The whole model (`ann_131`, `income_logit`)

- Kind: composition; parent: none; children: age_profile, capital_gain_effect, self_emp_not_inc_offset, short_hours_hinge, always_on_linear_gates, rare_category_indicators, country_linear_offsets, core_linear_backbone
- Reads: age, work class, census sampling weight, highest education, education level (1 = preschool to 16 = doctorate), marital status, occupation, relationship in the household, race, sex, capital gains, capital losses, hours worked per week, country of birth
- Entry nodes: 2451, 5559, identity_age, identity_cg, identity_core, identity_country, identity_lin_gates, identity_rare, -106, -103, -101, -96, -93, -87, -85, -84, -83, -81, -77, -75, -74, -71, -70, -68, -66, -61, -60, -59, -52, -51, -50, -44, -43, -42, -37, -34, -33, -32, -27, -26, -22, -21, -20, -16, -15, -14, -6, -4, -2
- Exit nodes: 0
- Subgraph nodes: 0, -106, -103, -101, -96, -93, -87, -85, -84, -83, -81, -77, -75, -74, -71, -70, -68, -66, -61, -60, -59, -52, -51, -50, -44, -43, -42, -37, -34, -33, -32, -27, -26, -22, -21, -20, -16, -15, -14, -6, -4, -2

**Hypothesis (plain).** The model adds up eight parts into one score: capital gains, most characteristics added up on their own, age, working hours, country of birth, self-employment, and two small sets of single-category adjustments. [C072]

**Evidence (plain).** It is made up of eight parts (each explained below); the score is their sum, which was checked on all 39073 training people. [C072] Changing one input at a time across the whole model, higher capital gains take a typical person's chance slowly at first, then faster, from about 63% at 0 to about 9% at 6849 US dollars, flattening out near the bottom because the predicted chance cannot go much lower, and a higher education level lowers it steadily from about 84% at 7 to about 35% at 13 on the 1-16 scale. [C002, C003] Across the whole model, the chance is lowest at about 47 years of age and higher either side, from about 84% at 22 to about 55% at 58 years. [C014] Across the whole model, being married to a member of the armed forces lowers it to below 1%, while a marital status other than married, divorced or widowed raises it to about 97% and being divorced to about 95%. [C001, C006, C008]

**Annotation hypothesis (technical).** Root. P(income<=50K) = sigmoid(3.385 + core_linear_backbone + country_linear_offsets + capital_gain_effect + age_profile + 0.814*short_hours_hinge + 0.433*self_emp_not_inc_offset + always_on_linear_gates + rare_category_indicators). The logit is purely additive (GAM-like): no child interacts with another; all nonlinearity is univariate relu hinges in age, capital gain and hours, plus category indicators. 41 one-hot inputs have no edges and zero effect (they are listed as entries, sensitivity 0). EVIDENCE: Function preservation 5.6e-16; local reconstruction 3.3853 + sum of child contributions matches the pre-sigmoid logit to 3e-14 on all 39,073 train rows. Sensitivity of 41 unconnected inputs = 0. Whole-model KernelSHAP on 200 train rows (base 0.753): top features marital-status:2 Married-civ-spouse 0.139, education-num 0.075, age 0.047, capital-gain 0.040, hours 0.030, occupation:4 Exec-managerial 0.018, sex 0.016, capital-loss 0.014. Train acc 0.858, AUC 0.912.

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = \frac{e^{0.814 x_{2451} + 0.433 x_{5559} + 1.0 x_{identity_age} + 1.0 x_{identity_cg} + 1.0 x_{identity_core} + 1.0 x_{identity_country} + 1.0 x_{identity_lin_gates} + 1.0 x_{identity_rare}}}{e^{0.814 x_{2451} + 0.433 x_{5559} + 1.0 x_{identity_age} + 1.0 x_{identity_cg} + 1.0 x_{identity_core} + 1.0 x_{identity_country} + 1.0 x_{identity_lin_gates} + 1.0 x_{identity_rare}} + 0.0339}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: sensitivity; split: train)


### Capital gains (`ann_130`, `capital_gain_effect`)

- Kind: composition; parent: income_logit; children: capital_gain_hinge, capital_gain_country_gate
- Reads: highest education, capital gains, country of birth
- Entry nodes: identity_cg_main, 262
- Exit nodes: identity_cg
- Subgraph nodes: identity_cg

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) as capital gains rise, slowly at first, then faster, a little less for people whose highest education is 9th grade and from a higher start for people born in Jamaica. [C073, C081, C083]

**Evidence (plain).** It is made of two parts, the main capital-gains curve and an extra capital-gains bend (both below), and adds them up. [C084] Through it, a typical person's chance goes from about 63% at 0 to about 9% at 6849 US dollars, slowly at first, then faster, flattening out near the bottom because the predicted chance cannot go much lower. [C073] For people whose highest education is 9th grade the fall, again slowly at first, then faster, is from about 44% to about 6%, and for people born in Jamaica from about 71% to about 16%. [C081, C083]

**Annotation hypothesis (technical).** Additive composition: identity_cg = capital_gain_hinge - 1.842*capital_gain_country_gate. Together a concave piecewise-linear penalty on the logit of P(<=50K): -0.337 per $1000 everywhere, steepening to -0.485 per $1000 above ~$4.8k and -0.644 per $1000 above ~$5.2k, plus a constant -2.44. Ecuador removes the $5.2k kink; 9th grade, Jamaica, Nicaragua, Philippines and Scotland remove the $4.8k kink. EVIDENCE: Train evidence: mean contribution by gain band: $1-3k -2.74, $3-4.5k -3.29, $4.5-5k -3.66, $5-5.5k -3.85, $5.5-8k -5.28, $8-15k -7.76, $15-30k -11.4, >$30k -63 (saturated). Observed P(<=50K) falls from 0.71 to 0.10 across $4.5k-8k, where the kinks sit. Ablating the whole composition: AUC 0.912->0.894, acc ->0.837.

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = - 1.84 x_{262} + 1.0 x_{identity_cg_main}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: sensitivity; note: Train evidence: mean contribution by gain band: $1-3k -2.74, $3-4.5k -3.29, $4.5-5k -3.66, $5-5.5k -3.85, $5.5-8k -5.28, $8-15k -7.76, $15-30k -11.4, >$30k -63 (saturated). Observed P(<=50K) falls from 0.71 to 0.10 across $4.5k-8k, where the kinks sit. Ablating the whole composition: AUC 0.912->0.894, acc ->0.837.; split: train)


### Main capital-gains curve (`ann_122`, `capital_gain_hinge`)

- Kind: leaf; parent: capital_gain_effect; children: none
- Reads: capital gains, country of birth
- Entry nodes: -63_bc, -105, -73
- Exit nodes: identity_cg_main
- Subgraph nodes: -63_bc, -105, -73, 1485, 4262, identity_cg_main

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) as capital gains rise, slowly at first, then faster. [C085]

**Evidence (plain).** Through it, a typical person's chance goes from about 63% at 0 to about 12% at 6849 US dollars, slowly at first, then faster, flattening out near the bottom because the predicted chance cannot go much lower. [C085] For people born in Ecuador it starts and ends higher, slowly at first, then faster, from about 71% to about 15%. [C088] Its formula reproduces the part exactly on all 39073 training people. [C089]

**Annotation hypothesis (technical).** Main capital-gain penalty (more gain -> less likely <=50K). Linear -2.524/SD (-0.337 per $1000, SD~$7.5k) everywhere, plus hinge 1485=relu(0.245 cg-0.2-2.168 Ecuador) that switches on above ~$5.2k and adds -1.19/SD (-0.159 per $1000) via its direct edge (-2.728) and via 4262 (0.77*-2.786). 4262=relu(0.875+0.77*1485-1.379 Holand-Netherlands) is always on (that country never occurs in train), so it is linear and contributes a constant -2.44. Ecuador (native-country:7) disables the hinge. EVIDENCE: Train evidence: the line plot is almost linear (corr -0.999) from -2.07 (no gain) to -50.8 (max gain). Kink at $5.16k computed analytically. Mean-ablation: AUC 0.9118->0.9028, acc ->0.8463 (the largest single leaf after the backbone).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 2.52 x_{capital-gain_bc} \\
&\quad - 2.73 \operatorname{relu}{\left(0.245 x_{capital-gain_bc} - 2.17 x_{native-country:7} - 0.2 \right)} \\
&\quad - 2.79 \operatorname{relu}{\left(- 1.38 x_{native-country:15} + 0.77 \operatorname{relu}{\left(0.245 x_{capital-gain_bc} - 2.17 x_{native-country:7} - 0.2 \right)} + 0.875 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= - 2.52 x_{capital-gain_bc} \\
&\quad - 2.73 \operatorname{relu}{\left(0.245 x_{capital-gain_bc} - 2.17 x_{native-country:7} - 0.2 \right)} \\
&\quad - 2.79 \operatorname{relu}{\left(- 1.38 x_{native-country:15} + 0.77 \operatorname{relu}{\left(0.245 x_{capital-gain_bc} - 2.17 x_{native-country:7} - 0.2 \right)} + 0.875 \right)}
\end{aligned}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: line; note: Train evidence: the line plot is almost linear (corr -0.999) from -2.07 (no gain) to -50.8 (max gain). Kink at $5.16k computed analytically. Mean-ablation: AUC 0.9118->0.9028, acc ->0.8463 (the largest single leaf after the backbone).; split: train)


### Extra capital-gains bend (`ann_123`, `capital_gain_country_gate`)

- Kind: leaf; parent: capital_gain_effect; children: none
- Reads: highest education, capital gains, country of birth
- Entry nodes: -63_a, -24_a, -82, -86, -90, -94_a
- Exit nodes: 262
- Subgraph nodes: -63_a, -24_a, -82, -86, -90, -94_a, 2391, 262

**Hypothesis (plain).** This part adds nothing until capital gains reach about 4883 US dollars, then lowers the predicted chance of a low income ($50,000 or less) steadily; for people born in Jamaica the chance, falling slowly at first, then faster, starts and ends higher. [C126, C132]

**Evidence (plain).** It acts on 1993 of 39073 training people, 234 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 1% with or without it. [C134] Through it alone, a typical person's chance goes from about 63% at 0 to about 56% at 6849 US dollars, nothing until about 4883 US dollars, then steadily. [C126] Changing capital gains (slowly at first, then faster), people born in Jamaica go from about 71% to about 16%, against about 63% to about 9% for people born outside Jamaica, Nicaragua, the Philippines and Scotland. [C132] Its formula reproduces the part exactly on all 39073 training people. [C135]

**Annotation hypothesis (technical).** Second capital-gain hinge 262=relu(0.603 cg-0.5-0.984 9th-grade-1.274 Jamaica-0.254 Philippines-1.285*2391), weight -1.842 into the output: above ~$4.8k gain it adds a further -1.11/SD (-0.148 per $1000). It is switched off for education=9th, and for Jamaica, Nicaragua, Philippines and Scotland via 2391 (a relu OR-detector of those four countries, native-country 23/27/30/34). So those groups get a shallower capital-gain slope. EVIDENCE: Train evidence: 262 is active for 5.1% of rows (gains above ~$4.8k), exit range 0..7.65 (contribution down to -14.1). Mean-ablation: AUC 0.9118->0.9114 (small, because it is collinear with the main capital-gain terms).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.603 x_{capital-gain_a} \\
&\quad - 0.984 x_{education:6_a} \\
&\quad - 1.27 x_{native-country:23} \\
&\quad - 0.254 x_{native-country:30} \\
&\quad - 1.28 \operatorname{relu}{\left(1.01 x_{native-country:23} + 0.347 x_{native-country:27} + 0.964 x_{native-country:30} + 1.02 x_{native-country:34_a} - 0.913 \right)} \\
&\quad -0.5 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: line; note: Train evidence: 262 is active for 5.1% of rows (gains above ~$4.8k), exit range 0..7.65 (contribution down to -14.1). Mean-ablation: AUC 0.9118->0.9114 (small, because it is collinear with the main capital-gain terms).; split: train)


### Characteristics that add up (`ann_129`, `core_linear_backbone`)

- Kind: leaf; parent: income_logit; children: none
- Reads: work class, census sampling weight, highest education, education level (1 = preschool to 16 = doctorate), marital status, occupation, relationship in the household, race, sex, capital losses
- Entry nodes: -10, -11, -12, -13, -18, -19_b, -23, -24_b, -25, -28, -29, -3, -30, -31, -35, -36, -38, -39_b, -40, -41, -45, -46, -47, -48, -49, -5, -54, -55, -56, -57, -58, -62, -64, -7, -9_b
- Exit nodes: identity_core
- Subgraph nodes: -10, -11, -12, -13, -18, -19_b, -23, -24_b, -25, -28, -29, -3, -30, -31, -35, -36, -38, -39_b, -40, -41, -45, -46, -47, -48, -49, -5, -54, -55, -56, -57, -58, -62, -64, -7, -9_b, identity_core

**Hypothesis (plain).** This part adds up most characteristics, each on its own: more education and capital losses lower the predicted chance of a low income ($50,000 or less) steadily, being unmarried, divorced, widowed, a woman or in a farming or service job raises it, and being a wife or married to a member of the armed forces lowers it. [C091, C095, C093, C094, C096, C109, C098, C100, C099, C090]

**Evidence (plain).** Through it, a typical person's chance goes steadily from about 84% at 7 to about 35% at 13 on the 1-16 education scale, from about 63% at 0 to about 31% at 1887 US dollars of capital losses, and from about 66% to about 60% as the census sampling weight rises. [C091, C095, C116] Being married to a member of the armed forces lowers it to below 1%, while a marital status other than married, divorced or widowed raises it to about 97%, being divorced to about 95% and widowed to about 91%; being a wife lowers it to about 41%, being a child of the household raises it to about 74% and being a woman to about 74%. [C090, C093, C094, C096, C099, C107, C109] Compared with a professional job, farming or fishing raises it to about 87%, other service work to about 85%, handling or cleaning to about 83% and a missing occupation to about 78%, while an executive or managerial job lowers it to about 56%; never having worked raises it to above 99% (the model has little data on this, n = 9) and federal government work lowers it to about 52%. [C098, C100, C101, C104, C115, C092, C106] A highest education of 7th-8th grade lowers it to about 35%, 9th grade to about 44% and 10th grade to about 49%, while an academic associate degree raises it to about 72%. [C097, C102, C105, C110] Its formula reproduces the part exactly on all 39073 training people. [C125]

**Annotation hypothesis (technical).** Sparse logistic-regression backbone on the remaining direct edges (logit of P(<=50K)). Main effects: education-num -0.98/SD (-0.38 per education year), Married-civ-spouse -2.87, male -0.50, capital-loss -0.28/SD, Exec-managerial -0.67, Other-service +0.85, Wife -0.91, Own-child +0.51, Widowed -1.05, fnlwgt -0.10/SD. Two rare one-hots act as extreme switches: Married-AF-spouse -59 (29 rows) and Never-worked +30 (9 rows). EVIDENCE: Train evidence: ablating the backbone to its mean drops acc 0.858->0.801 and AUC 0.912->0.790. Whole-model SHAP (200 train rows) ranks Married-civ-spouse (0.139), education-num (0.075), age, capital-gain, hours, Exec-managerial, sex, capital-loss highest, so the backbone holds the dominant features. Backbone+country alone (others at their mean): acc 0.833, AUC 0.878.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 0.282 x_{capital-loss} \\
&\quad - 0.986 x_{education-num} \\
&\quad - 0.0948 x_{education:0} \\
&\quad - 0.00674 x_{education:14} \\
&\quad - 0.0578 x_{education:15_b} \\
&\quad - 0.0318 x_{education:1} \\
&\quad - 0.155 x_{education:5} \\
&\quad - 0.0957 x_{education:6_b} \\
&\quad + 0.0714 x_{education:7} \\
&\quad - 0.102 x_{fnlwgt} \\
&\quad - 0.142 x_{marital-status:0} \\
&\quad - 1.62 x_{marital-status:1} \\
&\quad - 1.43 x_{marital-status:2} \\
&\quad - 0.18 x_{marital-status:6} \\
&\quad + 0.0864 x_{occupation:0} \\
&\quad - 0.122 x_{occupation:10} \\
&\quad - 0.0791 x_{occupation:11_b} \\
&\quad - 0.0643 x_{occupation:12} \\
&\quad - 0.0895 x_{occupation:13} \\
&\quad - 0.219 x_{occupation:4} \\
&\quad + 0.164 x_{occupation:5} \\
&\quad + 0.136 x_{occupation:6} \\
&\quad + 0.0913 x_{occupation:7} \\
&\quad + 0.255 x_{occupation:8} \\
&\quad + 0.0318 x_{race:0} \\
&\quad - 0.0612 x_{race:1} \\
&\quad + 0.186 x_{relationship:3} \\
&\quad - 0.0149 x_{relationship:4} \\
&\quad - 0.195 x_{relationship:5} \\
&\quad - 0.237 x_{sex} \\
&\quad - 0.0758 x_{workclass:1} \\
&\quad + 0.45 x_{workclass:3} \\
&\quad - 0.033 x_{workclass:5} \\
&\quad + 0.0735 x_{workclass:7_b} \\
&\quad + 0.0106 x_{workclass:8}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: sensitivity; note: Train evidence: ablating the backbone to its mean drops acc 0.858->0.801 and AUC 0.912->0.790. Whole-model SHAP (200 train rows) ranks Married-civ-spouse (0.139), education-num (0.075), age, capital-gain, hours, Exec-managerial, sex, capital-loss highest, so the backbone holds the dominant features. Backbone+country alone (others at their mean): acc 0.833, AUC 0.878.; split: train)


### Age (`ann_121`, `age_profile`)

- Kind: leaf; parent: income_logit; children: none
- Reads: age, country of birth
- Entry nodes: -1, -80, -97
- Exit nodes: identity_age
- Subgraph nodes: -1, -80, -97, 2134, 2990, 5070, identity_age

**Hypothesis (plain).** This part makes the predicted chance of a low income ($50,000 or less) lowest at about 47 years and higher either side, except for people born in Thailand, for whom it rises with age; being born in Ireland lowers it. [C137, C139, C136]

**Evidence (plain).** Through it, a typical person's chance goes from about 84% at 22 to about 55% at 58 years, lowest at about 47 years and higher either side. [C137] For people born in Thailand it goes the other way, from about 66% to about 85%. [C139] Being born in Ireland lowers it through this part from about 63% to about 26%. [C136] Its formula reproduces the part exactly on all 39073 training people. [C140]

**Annotation hypothesis (technical).** Piecewise-linear age term in the logit of P(income<=50K), with a Thailand override. Direct age edge gives -1.03 per SD (-0.075/yr). 2990=relu(1.446 age-0.48) and 5070=relu(1.031*2990-0.32) form a stacked hinge that switches on at age ~46 (z=0.545) and adds +1.45/SD, so the net slope above 46 is +0.42/SD (+0.031/yr): the probability of earning >50K rises with age to a peak at ~46 and then declines. 2134 is a Thailand indicator (native-country:37); for Thai respondents (27 rows) it adds -4.0 directly and pushes 2990 permanently on (removing the hinge). Ireland (native-country:21) only adds a direct -1.57 and keeps 2134 off. EVIDENCE: Train evidence (39,073 rows): mean contribution by age (non-Thai) +1.63 at 17, +0.65 at 30, -0.10 at 40, minimum -0.55 at exactly 46, -0.29 at 55, +0.17 at 70, +0.79 at 90, which confirms a V shape with its vertex at 46. Thailand (26 rows): the -4.0 direct term and the released hinge almost cancel, so being Thai changes the logit by only +0.10 on average. Mean-ablation of the leaf: acc 0.8578->0.8534, AUC 0.9118->0.9051.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 1.03 x_{age} \\
&\quad - 0.0464 x_{native-country:21} \\
&\quad + 0.0192 x_{native-country:37} \\
&\quad + 0.975 \operatorname{relu}{\left(1.03 \operatorname{relu}{\left(1.45 x_{age} + 0.137 \operatorname{relu}{\left(- 0.673 x_{native-country:21} + 0.854 x_{native-country:37} - 0.705 \right)} - 0.481 \right)} - 0.317 \right)} \\
&\quad - 0.124 \operatorname{relu}{\left(- 0.673 x_{native-country:21} + 0.854 x_{native-country:37} - 0.705 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: line; split: train)


### Short working hours (`ann_125`, `short_hours_hinge`)

- Kind: leaf; parent: income_logit; children: none
- Reads: hours worked per week
- Entry nodes: -65
- Exit nodes: 2451
- Subgraph nodes: -65, 2451

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people who work fewer hours: it falls steadily as hours rise up to about 50 hours a week, then no further. [C141, C142]

**Evidence (plain).** It acts on 31364 of 39073 training people, those working less than about 50 hours a week, 25290 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 62% without it and about 73% with it. [C143] Through it, a typical person's chance goes from about 79% at 24 to about 51% at 55 hours a week, steadily up to about 50 hours, then no further. [C141, C142] Its formula reproduces the part exactly on all 39073 training people. [C144]

**Annotation hypothesis (technical).** Hours-per-week hinge 2451=relu(0.598-0.777 hours_z), weight +0.814: below ~50 h/week each fewer hour adds +0.051 to the logit of P(<=50K) (part-time workers are more likely <=50K); flat (0) above ~50 h/week. Hours has no other path. EVIDENCE: Train evidence: mean contribution by hours: 10h +2.04, 20h +1.53, 30h +1.02, 40h +0.51, 45h +0.25, >=50h 0. Active for 80% of rows. Mean-ablation: acc ->0.8556, AUC ->0.9076.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.598 - 0.777 x_{hours-per-week} \right)}
```

**Evidence records.**

- *analytical_formula*:  (latex: y_0 = \operatorname{relu}{\left(0.598 - 0.777 x_{hours-per-week} \right)}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: line; note: Train evidence: mean contribution by hours: 10h +2.04, 20h +1.53, 30h +1.02, 40h +0.51, 45h +0.25, >=50h 0. Active for 80% of rows. Mean-ablation: acc ->0.8556, AUC ->0.9076.; split: train)


### Country of birth (`ann_128`, `country_linear_offsets`)

- Kind: leaf; parent: income_logit; children: none
- Reads: country of birth
- Entry nodes: -100, -104, -107, -67, -69, -72, -76, -78, -79, -88, -89, -91, -92, -94_b, -95, -99
- Exit nodes: identity_country
- Subgraph nodes: -100, -104, -107, -67, -69, -72, -76, -78, -79, -88, -89, -91, -92, -94_b, -95, -99, identity_country

**Hypothesis (plain).** This part adds a fixed amount for each country of birth: being born in Colombia, Scotland, 'South' (as recorded), Peru or China raises the predicted chance of a low income ($50,000 or less), and being born in Portugal or England lowers it (for Scotland the model has little data on this, n = 16). [C145, C146, C147, C149, C150, C148, C152]

**Evidence (plain).** Through it, compared with the United States, being born in Colombia takes a typical person's chance to above 99%, in Scotland to about 94% (the model has little data on this, n = 16), in 'South' (as recorded) to about 89%, in Peru to about 87% and in China to about 84%. [C145, C146, C147, C149, C150] Being born in Portugal lowers it to about 38%, in England to about 52% and in Canada to about 58%, while countries outside the named ones together raise it to about 71%. [C148, C152, C157, C154] Its formula reproduces the part exactly on all 39073 training people. [C161]

**Annotation hypothesis (technical).** Linear per-country offsets from the direct native-country edges (standardised one-hot, so each acts as a switch). Largest: Columbia +5.8 (rare, 64 rows), Scotland +1.77, Portugal -1.40, South +1.16, England -0.83, United-States -0.36. Population SD of the term 0.29 logit. EVIDENCE: Train evidence: per-switch effects w*(hi-lo): Columbia +5.80, Scotland +1.77, Portugal -1.40, South +1.16, Peru +0.97, England -0.83, China +0.77, US -0.36. Contribution SD 0.29. Mean-ablation: AUC ->0.9111.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 0.0192 x_{native-country:11} \\
&\quad + 0.00525 x_{native-country:14} \\
&\quad - 0.00301 x_{native-country:18} \\
&\quad - 0.0146 x_{native-country:1} \\
&\quad - 0.00105 x_{native-country:20} \\
&\quad + 0.0301 x_{native-country:29} \\
&\quad - 0.0366 x_{native-country:2} \\
&\quad - 0.00654 x_{native-country:31} \\
&\quad - 0.0494 x_{native-country:32} \\
&\quad + 0.0359 x_{native-country:34_b} \\
&\quad + 0.0561 x_{native-country:35} \\
&\quad - 0.108 x_{native-country:39} \\
&\quad + 0.0381 x_{native-country:3} \\
&\quad + 0.234 x_{native-country:4} \\
&\quad + 0.0259 x_{native-country:6} \\
&\quad - 0.0402 x_{native-country:9}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: sensitivity; note: Train evidence: per-switch effects w*(hi-lo): Columbia +5.80, Scotland +1.77, Portugal -1.40, South +1.16, Peru +0.97, England -0.83, China +0.77, US -0.36. Contribution SD 0.29. Mean-ablation: AUC ->0.9111.; split: train)


### Self-employed, not incorporated (`ann_124`, `self_emp_not_inc_offset`)

- Kind: leaf; parent: income_logit; children: none
- Reads: work class
- Entry nodes: -8
- Exit nodes: 5559
- Subgraph nodes: -8, 8292, 5559

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people who are self-employed without incorporation. [C162]

**Evidence (plain).** It acts on the 3034 self-employed people without incorporation among 39073 training people, 2215 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 60% without it and about 74% with it. [C163] Through it, such work takes a typical person's chance from about 63% to about 77%. [C162] Its formula reproduces the part exactly on all 39073 training people. [C164]

**Annotation hypothesis (technical).** Serial relu chain 8292->5559 that is an indicator of workclass=Self-emp-not-inc (workclass:6): 8292 = 1.822 when self-employed-not-inc, else 0; 5559=relu(0.94*8292-0.22). Adds +0.646 to the logit of P(<=50K) for that group (7.8% of rows), 0 otherwise. EVIDENCE: Train evidence: exit is 1.49 for all 7.8% Self-emp-not-inc rows and 0 otherwise (line corr 1.0); contribution +0.646. Mean-ablation: acc ->0.8557, AUC ->0.9111.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.94 \operatorname{relu}{\left(0.827 x_{workclass:6} - 1.03 \right)} \\
&\quad -0.222 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.94 \operatorname{relu}{\left(0.827 x_{workclass:6} - 1.03 \right)} \\
&\quad -0.222 \bigr)
\end{aligned}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: line; note: Train evidence: exit is 1.49 for all 7.8% Self-emp-not-inc rows and 0 otherwise (line corr 1.0); contribution +0.646. Mean-ablation: acc ->0.8557, AUC ->0.9111.; split: train)


### Other relatives (`ann_126`, `always_on_linear_gates`)

- Kind: leaf; parent: income_logit; children: none
- Reads: highest education, relationship in the household
- Entry nodes: -53, -19_a
- Exit nodes: identity_lin_gates
- Subgraph nodes: -53, -19_a, 3843, 7420, identity_lin_gates

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people who are another relative in the household. [C165]

**Evidence (plain).** Through it, being another relative takes a typical person's chance from about 63% to about 82%; 1180 training people are other relatives. [C165] Some college as the highest education barely changes it (about 63% to about 64%). [C166] Its formula reproduces the part exactly on all 39073 training people. [C167]

**Annotation hypothesis (technical).** Two relu gates that never switch off on train, so they are exactly linear: 3843=1.738+0.533 Other-relative (relationship:2) -> +0.95 logit for Other-relative; 7420=1.439-0.188 Some-college (education:15) -> +0.017 for Some-college. Constant part about +0.48. EVIDENCE: Train evidence: both relus are active for 100% of rows (min pre-activations 1.64 and 1.09), so the leaf is exactly linear; sensitivity 0.163 (Other-relative) vs 0.007 (Some-college). Mean-ablation: no measurable change (AUC 0.9117).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 0.037 \operatorname{relu}{\left(1.44 - 0.188 x_{education:15_a} \right)} \\
&\quad + 0.306 \operatorname{relu}{\left(0.533 x_{relationship:2} + 1.74 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= - 0.037 \operatorname{relu}{\left(1.44 - 0.188 x_{education:15_a} \right)} \\
&\quad + 0.306 \operatorname{relu}{\left(0.533 x_{relationship:2} + 1.74 \right)}
\end{aligned}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: sensitivity; note: Train evidence: both relus are active for 100% of rows (min pre-activations 1.64 and 1.09), so the leaf is exactly linear; sensitivity 0.163 (Other-relative) vs 0.007 (Some-college). Mean-ablation: no measurable change (AUC 0.9117).; split: train)


### Rare categories (`ann_127`, `rare_category_indicators`)

- Kind: leaf; parent: income_logit; children: none
- Reads: work class, highest education, occupation, country of birth
- Entry nodes: -9_a, -102, -39_a, -98, -17
- Exit nodes: identity_rare
- Subgraph nodes: -9_a, -102, -39_a, -98, -17, 1624, 3389, 16072, identity_rare

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people born in Trinidad and Tobago and for people whose highest education is preschool, and lowers it for people born in Yugoslavia. [C169, C170, C168]

**Evidence (plain).** Through it, being born in Trinidad and Tobago takes a typical person's chance from about 63% to about 92% (22 training people) and preschool as the highest education to about 88% (64 training people). [C169, C170] Being born in Yugoslavia lowers it to about 33% (21 training people); state government work and a protective-service job barely change it through this part. [C168, C171, C172] Its formula reproduces the part exactly on all 39073 training people. [C175]

**Annotation hypothesis (technical).** Three relu indicators of rare categories (zero for everyone else): 1624 = Yugoslavia (native-country:41) -> -1.22 (State-gov adds only -0.005); 3389 = Trinadad&Tobago (native-country:38) -> +1.87 (Protective-serv only modulates within it); 16072 = Preschool education (education:13) -> +1.43. Affects ~0.3% of rows. EVIDENCE: Train evidence: 1624 is >0.2 only for Yugoslavia (State-gov gives 0.155, contribution -0.005); 3389 is active for 0.06% of rows (Trinadad&Tobago); 16072 for 0.16% (Preschool). Mean-ablation: no measurable change in acc/AUC (rare groups).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= 0.0604 \operatorname{relu}{\left(0.999 x_{education:13} - 0.908 \right)} \\
&\quad + 0.0467 \operatorname{relu}{\left(0.998 x_{native-country:38} - 0.0079 x_{occupation:11_a} - 2.13 \right)} \\
&\quad - 0.0353 \operatorname{relu}{\left(0.812 x_{native-country:41} + 0.0583 x_{workclass:7_a} - 0.127 \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (latex: \begin{aligned}
y_0 &= 0.0604 \operatorname{relu}{\left(0.999 x_{education:13} - 0.908 \right)} \\
&\quad + 0.0467 \operatorname{relu}{\left(0.998 x_{native-country:38} - 0.0079 x_{occupation:11_a} - 2.13 \right)} \\
&\quad - 0.0353 \operatorname{relu}{\left(0.812 x_{native-country:41} + 0.0583 x_{workclass:7_a} - 0.127 \right)}
\end{aligned}; source: get_formula(force=True))
- *empirical_viz*:  (n: 39073; viz: sensitivity; note: Train evidence: 1624 is >0.2 only for Yugoslavia (State-gov gives 0.155, contribution -0.005); 3389 is active for 0.06% of rows (Trinadad&Tobago); 16072 for 0.16% (Preschool). Mean-ablation: no measurable change in acc/AUC (rare groups).; split: train)

## Worth checking (draft)

- Worth checking: the model treats capital losses (0 to 1887 US dollars) as lowering a typical person's predicted chance of a low income ($50,000 or less), from about 63% to about 31%. [C009]
- Worth checking: being born in Colombia raises the predicted chance of a low income ($50,000 or less) to above 99%, and in Scotland or Trinidad and Tobago to about 94%, from groups of 64, 16 and 22 people. [C005, C011, C010]
- Worth checking: being married to a member of the armed forces lowers the predicted chance of a low income ($50,000 or less) to below 1%, from 29 people, and never having worked raises it to above 99%, from only 9 people. [C001, C004]
- Worth checking: being a wife rather than a husband or someone not in a family lowers the predicted chance of a low income ($50,000 or less) to about 41%, while being a woman rather than a man raises it to about 74%. [C022, C039]
- Worth checking: a highest education of 7th-8th grade lowers the predicted chance of a low income ($50,000 or less) to about 35%, 9th grade to about 44% and 10th grade to about 49%. [C016, C027, C032]
- Worth checking: an academic associate degree raises the predicted chance of a low income ($50,000 or less) to about 72%, compared with most other highest education levels. [C041]
- Worth checking: across the whole model, the predicted chance of a low income ($50,000 or less) is lowest at about 47 years of age and rises for older people. [C014]
- Worth checking: the census sampling weight, which describes the survey rather than the person, lowers the predicted chance of a low income ($50,000 or less) from about 66% to about 60%. [C054]
