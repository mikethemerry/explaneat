# Explanation vb-dce983

- Genome: `c32e8fc8-410f-4126-ab7e-974c658ef3fc`
- Dataset: adult_ohe_v2 (training split 1df50169-8095-40d1-b61a-30bb541cd313, n = 39073; test n = 9769)
- Outcome: an income of $50,000 a year or less
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 62%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 6.661338147750939e-16
- closed_form_max_abs_err: 0.012146997277238114
- n_leaf_annotations: 9
- n_composition_annotations: 1
- n_splits: 8
- n_identity_nodes: 0

**Summary.** Almost a sparse logistic regression on 55 of 107 inputs (direct edges carry logit SD 3.37 of 3.85; mean-ablating every hidden unit: acc 0.856 -> 0.838, AUC 0.910 -> 0.895). One consequential nonlinearity: an age hinge turning the age slope into a U-shape (high-income share peaks mid-career; keeping only that unit recovers AUC 0.904). Remaining hidden units are affine relays, category lookups, or dead structure.

**What resisted.** Nothing resisted coverage. Most hidden structure trivial on the domain: five units never clip (affine relays); 2114 and the 4-deep chain 613->2245->8678->170 are identically 0 for every valid one-hot input (shown analytically); three readouts <0.02. Only the age hinge 4647 is consequential. native-country:4 and :32 are memorised rare categories. No semantic category names claimed.

## Plain-language summary (draft)

For a typical person (a 37-year-old White married man born in the United States, a high-school graduate in a professional job in the private sector, working 40 hours a week, with no capital gains or losses) the model predicts a chance of a low income ($50,000 or less) of about 62%; it adds up most of the person's characteristics independently, and nine parts add to that sum. [C089]

**Inputs that add up.** These add up independently, and these lower the chance steadily: a higher education level lowers the predicted chance of a low income ($50,000 or less) from about 84% at 7 to about 34% at 13 on the 1-16 scale, and, through this sum alone, older age from about 81% at 22 to about 30% at 58 years and capital gains from about 62% at 0 to about 23% at 6849 US dollars (both also act through parts below); higher capital losses (about 62% at 0 to about 31% at 1887 US dollars) and longer working hours (about 70% at 24 to about 54% at 55 hours a week) lower it too. [C002, C001, C004, C008, C031] Being a wife rather than a husband lowers it, through this sum alone, to about 39%; a highest education of 5th-6th grade lowers it to about 30% and 7th-8th grade to about 41%, an executive or managerial job to about 52% and working for the federal government to about 51%. [C017, C007, C023, C047, C040] Any marital status other than married (to a civilian, to a member of the armed forces, or with the spouse absent) raises it to about 87% and being a woman to about 73%; through this sum alone, a relationship in the household other than husband, unmarried or wife raises it to about 90%, being unmarried to about 82%, farming or fishing to about 85% and handling or cleaning to about 80% (relationship, occupation and country of birth also act through the parts below). [C016, C041, C012, C024, C019, C030] Through this sum alone, being born in Trinidad and Tobago raises it to about 92%, in Scotland to about 81% (the model has little data on this, n = 16) and in Puerto Rico to about 74%, while being born in Ireland lowers it to about 37% and in England to about 52%. [C009, C026, C038, C014, C049]

**Late-career age.** This part acts on 8609 of 39073 training people, 5807 of whom have an income of $50,000 or less; for the typical person it acts on (aged 57), the predicted chance is about 31% without it and about 50% with it. [C094] Through it, age adds nothing until about 50 years, then raises a typical person's predicted chance of a low income ($50,000 or less) steadily, from about 62% at 22 to about 81% at 58 years. [C091, C093] Being born in Colombia raises it through this part to above 99%, and for people born in Colombia age works the other way, steadily, though their chance stays above 99%. [C090, C092] Across the whole model, the chance is lowest at about 50 years and higher either side, from about 81% at 22 to about 52% at 58 years. [C011]

**Capital gains.** Through this part, higher capital gains lower a typical person's predicted chance of a low income ($50,000 or less) steadily, from about 62% at 0 to about 46% at 6849 US dollars; across the whole model the fall is steadily from about 62% to about 14%, flattening out near the bottom because the predicted chance cannot go much lower. [C096, C003]

**Not in a family.** This part acts on the 10067 people not living in a family, 9086 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 98% without it and about 94% with it. [C099] Through it, not living in a family lowers a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 37%. [C098]

**Missing occupation.** This part acts on the 2197 people whose occupation is missing, 1992 of whom have an income of $50,000 or less; through it, a missing occupation raises a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 80%. [C104, C101] A missing country of birth raises it from about 62% to about 72% for people whose occupation is known, but lowers it from about 80% to about 72% when the occupation is missing too. [C103]

**Other service work.** Through this part, other service work raises a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 72%. [C106]

**Administrative or clerical work.** Through this part, administrative or clerical work raises a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 70%. [C108]

**Born in Portugal.** Through this part, being born in Portugal lowers a typical person's predicted chance of a low income ($50,000 or less) from about 62% to about 18%. [C110]

**Handlers and cleaners.** Through this part, handling or cleaning raises a typical person's predicted chance of a low income ($50,000 or less) only from about 62% to about 64%, and not living in a family barely changes it. [C112, C113]

**Relationship and country (no effect).** One small part, reading relationship and country of birth, barely changes the predictions. [C118]

## Parts, from the root down

### The whole model (`ann_124`, `income_logit_root`)

- Kind: composition; parent: none; children: capital_gain_affine_relay, age_late_career_hinge, occupation8_relays, occupation1_relays, occupation6_relationship1_gate, relationship1_indicator, occupation0_country0_gate, country32_affine_relay, dead_relationship_chain
- Reads: age, work class, census sampling weight, highest education, education level (1 = preschool to 16 = doctorate), marital status, occupation, relationship in the household, race, sex, capital gains, capital losses, hours worked per week, country of birth
- Entry nodes: -2, -3, -4, -8, -11, -12, -16, -22, -23, -25, -27, -28, -30, -31, -32, -39, -41, -42, -44, -45, -46, -48, -57, -60, -62, -64, -65, -68, -69, -71, -73, -74, -78, -80, -90, -93, -94, -98, -99, -103, -107, -63_a, -1_a, -49_a, -47_a, -51_a, -55_a, -56_a, -106, -105, -104, -102, -101, -97, -96, -95, -91, -89, -88, -87, -86, -85, -84, -83, -82, -81, -79, -76, -75, -72, -70, -67, -61, -59, -58, -54, -53, -50, -43, -40, -38, -35, -34, -33, -29, -26, -24, -21, -20, -19, -18, -17, -15, -14, -13, -10, -9, -7, -6, -5
- Exit nodes: 0
- Subgraph nodes: -2, -3, -4, -8, -11, -12, -16, -22, -23, -25, -27, -28, -30, -31, -32, -39, -41, -42, -44, -45, -46, -48, -57, -60, -62, -64, -65, -68, -69, -71, -73, -74, -78, -80, -90, -93, -94, -98, -99, -103, -107, -63_a, -1_a, -49_a, -47_a, -51_a, -55_a, -56_a, -106, -105, -104, -102, -101, -97, -96, -95, -91, -89, -88, -87, -86, -85, -84, -83, -82, -81, -79, -76, -75, -72, -70, -67, -61, -59, -58, -54, -53, -50, -43, -40, -38, -35, -34, -33, -29, -26, -24, -21, -20, -19, -18, -17, -15, -14, -13, -10, -9, -7, -6, -5, 0

**Hypothesis (plain).** The model adds up most of a person's characteristics on their own into one score, and nine parts add to that sum: late-career age, capital gains, not living in a family, a missing occupation, other service work, administrative or clerical work, being born in Portugal, handlers and cleaners, and one that barely changes anything. [C089]

**Evidence (plain).** It is made up of the inputs that feed the score directly and nine parts (each explained below); the score is their sum, which was checked on all 39073 training people. [C089] Changing one input at a time across the whole model, a higher education level takes a typical person's chance steadily from about 84% at 7 to about 34% at 13 on the 1-16 scale, and capital gains lower it steadily from about 62% at 0 to about 14% at 6849 US dollars, flattening out near the bottom because the predicted chance cannot go much lower. [C002, C003] Across the whole model, the chance is lowest at about 50 years of age and higher either side, from about 81% at 22 to about 52% at 58 years. [C011] Across the whole model, being born in Colombia raises it to above 99%, while being born in Portugal lowers it to about 27%. [C005, C006] Being a wife rather than a husband lowers it to about 39% across the whole model, while a relationship other than husband, not in a family, unmarried or wife raises it to about 90%. [C018, C013]

**Annotation hypothesis (technical).** ROOT (supported). Output 0 = sigmoid(z), z = 1.399 + 48 direct input terms + 12 hidden readouts (9 child annotations); 52 of the 107 one-hot/standardized inputs are unconnected and listed here as zero-effect entries. Class 1 is the 76% majority (presumably <=50K), so negative weights push toward high income. The model is essentially a sparse logistic regression with one consequential nonlinearity. Direct edges carry logit SD 3.37 of 3.85 (mean-ablating them drops train AUC 0.910->0.518); all hidden readouts together carry SD 1.32 and mean-ablating every hidden unit costs only acc 0.856->0.838, AUC 0.910->0.895. The only nonlinearity that matters is age_late_career_hinge (4647): keeping it alone recovers acc 0.850 / AUC 0.904. It turns the direct age slope (-0.88) into a U-shaped age effect with its minimum at age z~0.8. The other children are lookups or affine relays: capital_gain_affine_relay steepens the capital-gain slope to -2.54/SD; relationship1_indicator (-1.03 for relationship:1), occupation0_country0_gate (+0.92 for occupation:0 unless native-country:0), occupation8_relays (+0.45) and occupation1_relays (+0.37) are category offsets; country32_affine_relay and the native-country:4 part of the age hinge memorise rare countries; occupation6_relationship1_gate is near-inert (<=0.10 logit); dead_relationship_chain (613->2245->8678->170) and node 2114 are identically 0 on every valid input. Children combine ADDITIVELY in the logit (each child exit has one readout edge into node 0; no child feeds another). Largest exact logit-space attributions (mean |phi|, full train): relationship:0 0.82, education-num 0.72, marital-status:2 0.70, capital-gain 0.68, age 0.51, relationship:1 0.39, relationship:5 0.24, sex 0.23, occupation:8 0.21, hours-per-week 0.18.

**Evidence records.**

- *analytical_formula*:  (note: h_k are the child-annotation exits. On the train domain 1870, 1303, 9258, 4973, 13173 are always active (affine), 170 and 2114 are identically 0, 7530/3119 are category indicators, so z is affine in every input except age (hinge 4647) and the 2-input gates 392 and 4249.)
- *ablation_study*:  (n: 39073; split: train; method: local forward passes on in-memory copy; ablation = replace hidden readout by its train mean)
- *shap_importance*:  (note: compute_shap(max_samples=500) was started but killed after ~20 min (KernelExplainer, 107 features, machine load ~90); replaced by this exact logit-space attribution, which is exact for the additive part and uses 2,000 background rows for the 2-input gates (additivity error 0.0014).; method: exact interventional Shapley in logit space, full train split (39,073 rows); 2-input gates use 2,000-row background)
- *performance_metrics*:  (n: 39073; split: train; auc_roc: 0.9099; accuracy: 0.8558; class1_rate: 0.7612)


### Late-career age (`ann_116`, `age_late_career_hinge`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: age, country of birth
- Entry nodes: -1_b, -100
- Exit nodes: 4647
- Subgraph nodes: -1_b, -100, 4647

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for older people: age adds nothing until about 50 years, then raises it steadily; it also raises it to almost certain for people born in Colombia. [C091, C093, C090]

**Evidence (plain).** It acts on 8609 of 39073 training people, 5807 of whom have an income of $50,000 or less; for the typical person it acts on (aged 57), the predicted chance is about 31% without it and about 50% with it. [C094] Through it, age adds nothing until about 50 years, then raises a typical person's chance steadily, from about 62% at 22 to about 81% at 58 years. [C091, C093] Being born in Colombia raises the chance through it to above 99%, and for people born in Colombia age works the other way, steadily, though their chance stays above 99%. [C090, C092] Its formula reproduces the part exactly on all 39073 training people. [C095]

**Annotation hypothesis (technical).** SUPPORTED: a single-knot age hinge giving a U-shaped (in class-1 logit) age effect. Below age z~0.80 the direct age term dominates (class-1 logit falls with age, i.e. the high-income share rises); above the knot 4647 reverses the slope to +0.65 per SD, so older workers drift back toward class 1, matching the empirical rate (minimum P(y=1) ~0.60 near z 0.8). native-country:4 (64 rows, standardized value 24.7) rides on the same unit as a large class-1 offset (memorisation of a rare category). EVIDENCE: Knot (for rows not in native-country:4, where nc4 = -0.041) at age z = 0.798. Active for 21.9% of those rows. Net age effect on logit = -0.881*age + 1.224*ReLU(1.249*age + 0.711*nc4 - 0.968): slope -0.881 below the knot and +0.648 above it. Logit contribution range [0, 24.36], SD 1.00 (the largest hidden-node contribution). The 64 native-country:4 rows (value 24.69) have 4647 ~ 16.6 (+20 logit), i.e. they are forced to class 1. line_plot on train (39,073 rows): exit 0 below z~0.8 then linear; scatter exit range [0, 19.9] (the top from nc4 rows). Local binned net age effect (non-nc4 rows): z=-1.5: +1.31, z=-1: +0.88, z=0: -0.02, z=0.5: -0.44, z=0.8: -0.67 (minimum), z=1.5: -0.26, z=2: +0.07, z=3: +0.73. Empirical P(y=1) follows the same U: 0.999, 0.944, 0.664, 0.606, 0.597, 0.662, 0.764, 0.816. Mean-ablation of 4647: acc 0.8558->0.8436, AUC 0.9099->0.9009 (largest hidden-node effect).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.25 x_{age_b} \\
&\quad + 0.711 x_{native-country:4} \\
&\quad -0.968 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.25 x_{age_b} \\
&\quad + 0.711 x_{native-country:4} \\
&\quad -0.968 \bigr)
\end{aligned}; split: train; readout: readout 4647->0 = +1.224)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000)


### Capital gains (`ann_115`, `capital_gain_affine_relay`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: capital gains
- Entry nodes: -63_b
- Exit nodes: 1870
- Subgraph nodes: -63_b, 1870

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) steadily as capital gains rise, on top of the sum. [C096]

**Evidence (plain).** Through it, a typical person's chance goes steadily from about 62% at 0 to about 46% at 6849 US dollars of capital gains. [C096] Its formula reproduces the part exactly on all 39073 training people. [C097]

**Annotation hypothesis (technical).** SUPPORTED: affine relay. 1870 = ReLU(0.439*capital-gain + 0.850) never clips on train, so it is a linear rescaling of capital-gain that adds -0.702*cg - 1.359 to the logit and steepens the direct capital-gain term (-1.841) to a total slope of -2.54 per SD. No nonlinearity is used on the data domain (the knot sits at cg z = -1.94, below the data minimum -0.147). EVIDENCE: Always active on train (pre-activation in [0.786, 6.636], 39,073/39,073 rows), so on the data domain 1870 = 0.439*cg + 0.850 exactly and its logit contribution is -0.702*cg - 1.359 (range [-10.61, -1.26], SD 0.70). Combined with the direct edge (-1.841), the total capital-gain slope is -2.54 logit per SD, pushing toward class 0 (the 24% minority, presumably >50K). line_plot on train (39,073 rows): exit vs capital-gain correlation 1.000 (exactly linear over the observed range); 91.6% of rows sit at the zero-gain floor (cg z = -0.147). Mean-ablation of 1870 (replace by train mean): acc 0.8558->0.8553, AUC 0.9099->0.9093 (small, because the direct edge carries the same signal).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.439 x_{capital-gain_b} + 0.85 \right)}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: y_0 = \operatorname{relu}{\left(0.439 x_{capital-gain_b} + 0.85 \right)}; split: train; readout: readout 1870->0 = -1.599; analysis: Always active on train (pre-activation in [0.786, 6.636], 39,073/39,073 rows), so on the data domain 1870 = 0.439*cg + 0.850 exactly and its logit contribution is -0.702*cg - 1.359 (range [-10.61, -1.26], SD 0.70). Combined with the direct edge (-1.841), the total capital-gain slope is -2.54 logit per SD, pushing toward class 0 (the 24% minority, presumably >50K).)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): exit vs capital-gain correlation 1.000 (exactly linear over the observed range); 91.6% of rows sit at the zero-gain floor (cg z = -0.147). Mean-ablation of 1870 (replace by train mean): acc 0.8558->0.8553, AUC 0.9099->0.9093 (small, because the direct edge carries the same signal).)


### Not in a family (`ann_120`, `relationship1_indicator`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: relationship in the household
- Entry nodes: -52_b
- Exit nodes: 7530
- Subgraph nodes: -52_b, 7530

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) for people not living in a family. [C098]

**Evidence (plain).** It acts on the 10067 people not living in a family among 39073 training people, 9086 of whom have an income of $50,000 or less; for the typical person it acts on, the predicted chance is about 98% without it and about 94% with it. [C099] Through it, not living in a family takes a typical person's chance from about 62% to about 37%. [C098] Its formula reproduces the part exactly on all 39073 training people. [C100]

**Annotation hypothesis (technical).** SUPPORTED: a category indicator for relationship:1 (-1.03 logit). Because the other relationship one-hots enter linearly and standardized one-hots shift every row, this indicator places relationship:1 between relationship:0 and relationship:4 on the class-1 logit scale. It is a lookup, not an interaction. EVIDENCE: relationship:1 binary (-0.589 / 1.697). 7530 = 0 for the 29,006 non-rel1 rows and 2.466 for the 10,067 rel1 rows: a pure indicator, -1.030 logit for relationship:1. relationship:1 has no direct edge to the output, so the net relationship effect for rel1 rows (sum of all relationship terms) is -0.14, vs -0.81 for relationship:0, +0.90 for 2 and 3, +0.18 for 4, -1.75 for 5. line_plot on train (39,073 rows): two-level step, correlation 1.000. Mean-ablation: acc 0.8558->0.8533, AUC 0.9099->0.9073 (second-largest hidden effect).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(1.56 x_{relationship:1_b} - 0.18 \right)}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: y_0 = \operatorname{relu}{\left(1.56 x_{relationship:1_b} - 0.18 \right)}; split: train; readout: readout 7530->0 = -0.418; analysis: relationship:1 binary (-0.589 / 1.697). 7530 = 0 for the 29,006 non-rel1 rows and 2.466 for the 10,067 rel1 rows: a pure indicator, -1.030 logit for relationship:1. relationship:1 has no direct edge to the output, so the net relationship effect for rel1 rows (sum of all relationship terms) is -0.14, vs -0.81 for relationship:0, +0.90 for 2 and 3, +0.18 for 4, -1.75 for 5.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): two-level step, correlation 1.000. Mean-ablation: acc 0.8558->0.8533, AUC 0.9099->0.9073 (second-largest hidden effect).)


### Missing occupation (`ann_121`, `occupation0_country0_gate`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: occupation, country of birth
- Entry nodes: -66, -36
- Exit nodes: 392
- Subgraph nodes: -66, -36, 392

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people whose occupation is missing. [C101]

**Evidence (plain).** It acts on the 2197 people among 39073 training people whose occupation is missing, 1992 of whom have an income of $50,000 or less. [C104] Through it, a missing occupation takes a typical person's chance from about 62% to about 80%. [C101] A missing country of birth raises it from about 62% to about 72% for people whose occupation is known, but lowers it from about 80% to about 72% when the occupation is missing too. [C103] Its formula reproduces the part exactly on all 39073 training people. [C105]

**Annotation hypothesis (technical).** SUPPORTED: an AND-NOT gate: +0.92 class-1 logit for occupation:0 workers, cancelled when native-country:0 is set (34 rows). occupation:0 has no direct edge, so this is its only route; the native-country:0 veto touches very few rows and is probably fitted noise. EVIDENCE: Train cells: occupation:0 & not native-country:0: 2,197 rows, 392 = 2.749 (+0.919 logit); occupation:0 & native-country:0: 34 rows, 0; not occupation:0: 0 regardless (36,842 rows). P(y=1) in those cells: 0.907, 0.853, ~0.75. line_plot on train (39,073 rows) along native-country:0 is near-flat (correlation -0.03) because the gate depends on occupation:0; the cell table above is the operative evidence. Mean-ablation: acc 0.8558->0.8547, AUC 0.9099->0.9090.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.58 x_{native-country:0} \\
&\quad + 0.741 x_{occupation:0} \\
&\quad -0.471 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.58 x_{native-country:0} \\
&\quad + 0.741 x_{occupation:0} \\
&\quad -0.471 \bigr)
\end{aligned}; split: train; readout: readout 392->0 = +0.334; analysis: Train cells: occupation:0 & not native-country:0: 2,197 rows, 392 = 2.749 (+0.919 logit); occupation:0 & native-country:0: 34 rows, 0; not occupation:0: 0 regardless (36,842 rows). P(y=1) in those cells: 0.907, 0.853, ~0.75.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows) along native-country:0 is near-flat (correlation -0.03) because the gate depends on occupation:0; the cell table above is the operative evidence. Mean-ablation: acc 0.8558->0.8547, AUC 0.9099->0.9090.)


### Other service work (`ann_117`, `occupation8_relays`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: occupation
- Entry nodes: -49_b, -49_c
- Exit nodes: 1303, 2114, 9258
- Subgraph nodes: -49_b, -49_c, 1303, 2114, 9258

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people in administrative or clerical work. [C106]

**Evidence (plain).** Through it, other service work takes a typical person's chance from about 62% to about 72%; 3916 training people do such work. [C106] Its formula reproduces the part exactly on all 39073 training people. [C107]

**Annotation hypothesis (technical).** SUPPORTED: a category offset for occupation:8 implemented by redundant relays. 1303 and 9258 never clip (affine); 2114 is dead on the whole input domain. Net effect: +0.45 logit for occupation:8 rows (via 1303) on top of the direct occupation:8 term; 9258 and 2114 are inert (excess structure). EVIDENCE: occupation:8 is binary (-0.334 / 2.996). 1303 takes only {1.711, 2.120} and 9258 only {0.240, 3.140} (always active, affine). 2114 pre-activation is 0.224*1303 - 0.886 in [-0.503, -0.412] so 2114 is dead on every possible input value of occupation:8, not just on train. Net logit: constant +1.878 plus +0.449 for occupation:8 rows (via 1303), +0.004 via 9258. line_plot on train (39,073 rows): exit-1303 vs occupation:8 correlation 1.000. Mean-ablation of 1303: acc 0.8558->0.8554, AUC 0.9099->0.9097; 2114: no change; 9258: max logit change 0.004.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.123 x_{occupation:8_b} + 1.75 \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.224 \operatorname{relu}{\left(0.123 x_{occupation:8_b} + 1.75 \right)} \\
&\quad -0.886 \bigr)
\end{aligned} \\ y_2 = \operatorname{relu}{\left(0.871 x_{occupation:8_c} + 0.53 \right)}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: y_0 = \operatorname{relu}{\left(0.123 x_{occupation:8_b} + 1.75 \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.224 \operatorname{relu}{\left(0.123 x_{occupation:8_b} + 1.75 \right)} \\
&\quad -0.886 \bigr)
\end{aligned} \\ y_2 = \operatorname{relu}{\left(0.871 x_{occupation:8_c} + 0.53 \right)}; split: train; readout: readouts: 1303->0 = +1.098, 2114->0 = +0.542, 9258->0 = +0.0015; also 1303->2114 = 0.224; analysis: occupation:8 is binary (-0.334 / 2.996). 1303 takes only {1.711, 2.120} and 9258 only {0.240, 3.140} (always active, affine). 2114 pre-activation is 0.224*1303 - 0.886 in [-0.503, -0.412] so 2114 is dead on every possible input value of occupation:8, not just on train. Net logit: constant +1.878 plus +0.449 for occupation:8 rows (via 1303), +0.004 via 9258.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): exit-1303 vs occupation:8 correlation 1.000. Mean-ablation of 1303: acc 0.8558->0.8554, AUC 0.9099->0.9097; 2114: no change; 9258: max logit change 0.004.)


### Administrative or clerical work (`ann_118`, `occupation1_relays`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: occupation
- Entry nodes: -37
- Exit nodes: 3119, 4973
- Subgraph nodes: -37, 3119, 4973

**Hypothesis (plain).** This part raises the predicted chance of a low income ($50,000 or less) for people in other service work. [C108]

**Evidence (plain).** Through it, administrative or clerical work takes a typical person's chance from about 62% to about 70%; 4481 training people do such work. [C108] Its formula reproduces the part exactly on all 39073 training people. [C109]

**Annotation hypothesis (technical).** SUPPORTED: occupation:1 receives a small positive class-1 offset (+0.37 logit) via the affine relay 4973; the parallel indicator 3119 is nearly inert (readout -0.0047). occupation:1 has no direct edge to the output, so this leaf is its only route. EVIDENCE: occupation:1 binary (-0.360 / 2.778). 4973 in {0.555, 2.400}, always active (affine). 3119 is 0 for non-occ1 rows and 2.349 for occ1 rows (a true indicator) but its readout is -0.0047 (max logit change 0.011). Net logit: +0.111 baseline and +0.368 for occupation:1 rows. line_plot on train (39,073 rows): correlation 1.000 (two-level step). Mean-ablation: 4973 acc 0.8558->0.8552, AUC 0.9099->0.9095; 3119 negligible (AUC unchanged).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(1.04 x_{occupation:1} - 0.543 \right)} \\ y_1 = \operatorname{relu}{\left(0.588 x_{occupation:1} + 0.766 \right)}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: y_0 = \operatorname{relu}{\left(1.04 x_{occupation:1} - 0.543 \right)} \\ y_1 = \operatorname{relu}{\left(0.588 x_{occupation:1} + 0.766 \right)}; split: train; readout: readouts: 3119->0 = -0.0047, 4973->0 = +0.200; analysis: occupation:1 binary (-0.360 / 2.778). 4973 in {0.555, 2.400}, always active (affine). 3119 is 0 for non-occ1 rows and 2.349 for occ1 rows (a true indicator) but its readout is -0.0047 (max logit change 0.011). Net logit: +0.111 baseline and +0.368 for occupation:1 rows.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): correlation 1.000 (two-level step). Mean-ablation: 4973 acc 0.8558->0.8552, AUC 0.9099->0.9095; 3119 negligible (AUC unchanged).)


### Born in Portugal (`ann_122`, `country32_affine_relay`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: country of birth
- Entry nodes: -92
- Exit nodes: 13173
- Subgraph nodes: -92, 13173

**Hypothesis (plain).** This part lowers the predicted chance of a low income ($50,000 or less) for people born in Portugal. [C110]

**Evidence (plain).** Through it, being born in Portugal takes a typical person's chance from about 62% to about 31%; 49 training people were born in Portugal. [C110] Its formula reproduces the part exactly on all 39073 training people. [C111]

**Annotation hypothesis (technical).** SUPPORTED: affine relay giving a country-specific offset (-1.98 logit) to the 49 native-country:32 rows; no nonlinearity used. Rare-category memorisation with negligible global effect. EVIDENCE: native-country:32 binary (-0.035 / 28.229). 13173 in {0.882, 27.229}, always active (affine). The 49 native-country:32 rows get -1.98 logit; everyone else a constant -0.066. line_plot on train (39,073 rows): correlation 1.000. Mean-ablation: no change in acc/AUC at 4 d.p. (max logit change 1.98 on 49 rows).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.932 x_{native-country:32} + 0.915 \right)}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: y_0 = \operatorname{relu}{\left(0.932 x_{native-country:32} + 0.915 \right)}; split: train; readout: readout 13173->0 = -0.075; analysis: native-country:32 binary (-0.035 / 28.229). 13173 in {0.882, 27.229}, always active (affine). The 49 native-country:32 rows get -1.98 logit; everyone else a constant -0.066.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): correlation 1.000. Mean-ablation: no change in acc/AUC at 4 d.p. (max logit change 1.98 on 49 rows).)


### Handlers and cleaners (`ann_119`, `occupation6_relationship1_gate`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: occupation, relationship in the household
- Entry nodes: -47_b, -52_a
- Exit nodes: 4249
- Subgraph nodes: -47_b, -52_a, 4249

**Hypothesis (plain).** This part reads occupation and relationship in the household together but changes the chance by only a few points. [C112]

**Evidence (plain).** Through it, handling or cleaning raises a typical person's chance from about 62% to about 64%, and not living in a family barely changes it (about 62% either way). [C112, C113] Its formula reproduces the part exactly on all 39073 training people. [C117]

**Annotation hypothesis (technical).** SUPPORTED but near-inert: formally an occupation:6 x relationship:1 interaction gate (off only for relationship:1 rows outside occupation:6), but readout 0.0186 makes it worth at most +0.10 logit. It is a vestigial interaction, not a meaningful part of the decision. EVIDENCE: Four train levels: -1.011 (relationship:1 rows outside occupation:6, 9,693 rows, gate off), 0.074 (neither, 27,697), 4.164 (occ6 and rel1, 374), 5.249 (occ6 not rel1, 1,309). Logit contribution range [0, 0.098], SD 0.019. line_plot on train (39,073 rows): exit vs first entry correlation 0.995. Mean-ablation: acc 0.8558->0.8557, AUC unchanged at 4 d.p. (max logit change 0.093).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.05 x_{occupation:6_b} \\
&\quad - 0.475 x_{relationship:1_a} \\
&\quad + 0.0173 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.05 x_{occupation:6_b} \\
&\quad - 0.475 x_{relationship:1_a} \\
&\quad + 0.0173 \bigr)
\end{aligned}; split: train; readout: readout 4249->0 = +0.0186; analysis: Four train levels: -1.011 (relationship:1 rows outside occupation:6, 9,693 rows, gate off), 0.074 (neither, 27,697), 4.164 (occ6 and rel1, 374), 5.249 (occ6 not rel1, 1,309). Logit contribution range [0, 0.098], SD 0.019.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): exit vs first entry correlation 0.995. Mean-ablation: acc 0.8558->0.8557, AUC unchanged at 4 d.p. (max logit change 0.093).)


### Relationship and country (no effect) (`ann_123`, `dead_relationship_chain`)

- Kind: leaf; parent: income_logit_root; children: none
- Reads: relationship in the household, country of birth
- Entry nodes: -56_b, -77, -55_b, -51_b
- Exit nodes: 170
- Subgraph nodes: -56_b, -77, -55_b, -51_b, 613, 2245, 8678, 170

**Hypothesis (plain).** This part reads relationship in the household and country of birth but barely changes the predictions. [C118]

**Evidence (plain).** Switched off, it barely changes the predictions. [C118] Its formula reproduces the part exactly on all 39073 training people. [C119]

**Annotation hypothesis (technical).** SUPPORTED: dead chain. relationship:5 and native-country:19 -> 613 -> 2245 -> 8678 (+relationship:4) -> 170 (+relationship:0) never activates for any valid one-hot input, so the 4-deep structure contributes exactly 0 to the logit. Excess structure left over from evolution. EVIDENCE: 170 pre-activation takes only {-1.720, -1.275, -0.123} on train. Analytically, with one-hot relationship (exactly one of rel0/rel4/rel5 can be set): if relationship:0 is set, 613>=0.88 so 2245 = 0, 8678 = 0 and 170 pre = -0.123; otherwise 0.783*rel0 <= -0.644 and 8678 <= 0.576, so 170 pre < 0. Hence 170 = 0 on the entire valid input domain, not only on train. line_plot on train (39,073 rows): exit identically 0 (y range [0, 0]). Mean-ablation: no change.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.783 x_{relationship:0_b} \\
&\quad + 1.43 \operatorname{relu}{\left(0.259 x_{relationship:4_b} + 0.971 \operatorname{relu}{\left(0.273 - 0.992 \operatorname{relu}{\left(0.493 x_{native-country:19} - 2.66 x_{relationship:5_b} + 0.309 \right)} \right)} - 0.45 \right)} \\
&\quad -1.08 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*:  (n: 39073; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.783 x_{relationship:0_b} \\
&\quad + 1.43 \operatorname{relu}{\left(0.259 x_{relationship:4_b} + 0.971 \operatorname{relu}{\left(0.273 - 0.992 \operatorname{relu}{\left(0.493 x_{native-country:19} - 2.66 x_{relationship:5_b} + 0.309 \right)} \right)} - 0.45 \right)} \\
&\quad -1.08 \bigr)
\end{aligned}; split: train; readout: readout 170->0 = +0.0076; analysis: 170 pre-activation takes only {-1.720, -1.275, -0.123} on train. Analytically, with one-hot relationship (exactly one of rel0/rel4/rel5 can be set): if relationship:0 is set, 613>=0.88 so 2245 = 0, 8678 = 0 and 170 pre = -0.123; otherwise 0.783*rel0 <= -0.644 and 8678 <= 0.576, so 170 pre < 0. Hence 170 = 0 on the entire valid input domain, not only on train.)
- *empirical_viz*:  (split: train; viz_type: line_plot; max_samples: 100000; local_analysis: line_plot on train (39,073 rows): exit identically 0 (y range [0, 0]). Mean-ablation: no change.)

## Worth checking (draft)

- Worth checking: the model treats capital losses (0 to 1887 US dollars) as lowering a typical person's predicted chance of a low income ($50,000 or less), from about 62% to about 31%. [C008]
- Worth checking: being born in Colombia raises the predicted chance of a low income ($50,000 or less) to above 99% and being born in Portugal lowers it to about 27%, from groups of 64 and 49 people. [C005, C006]
- Worth checking: being a wife rather than a husband lowers the predicted chance of a low income ($50,000 or less) to about 39%, while being a woman rather than a man raises it to about 73%. [C018, C041]
- Worth checking: a highest education of 5th-6th grade lowers the predicted chance of a low income ($50,000 or less) to about 30% and 7th-8th grade to about 41%. [C007, C023]
- Worth checking: a master's degree raises the predicted chance of a low income ($50,000 or less) to about 71% and a bachelor's degree to about 69%, compared with most other highest education levels. [C057, C060]
- Worth checking: across the whole model, the predicted chance of a low income ($50,000 or less) is lowest at about 50 years of age and rises for older people. [C011]
- Worth checking: the census sampling weight, which describes the survey rather than the person, lowers the predicted chance of a low income ($50,000 or less) from about 64% to about 59%. [C072]
