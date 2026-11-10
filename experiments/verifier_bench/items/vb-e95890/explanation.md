# Explanation vb-e95890

- Genome: `e324239d-f591-4c8a-b149-7699ec2d0346`
- Dataset: monk2_onehot (training split 63257a45-49c9-42a6-8b28-8e346eab2bee, n = 345; test n = 87)
- Outcome: the 'yes' class
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 0%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 8.33e-16
- closed_form_max_abs_err: 0.065
- n_leaf_annotations: 4
- n_composition_annotations: 1
- n_splits: 6
- n_identity_nodes: 1

**Summary.** Additive logit under a saturating sigmoid, dominated by ReLU node 263 (about 19 with no first values, dropping 6.8-9.3 per first value of a2, a3, a4, a6). Direct linear term penalises a1=1 and a5=1 (about -15 each) and gives a1=3 +24.9. Nodes 1855 and 3471 are minor.

**What resisted.** Nothing structural. Closed-form error 0.065 is 3-s.f. rounding on a linear leaf with range -33..+32. The a1=1/count-3 boundary is pattern-specific (74% positive), described via per-indicator weights rather than a clean rule. One root evidence payload says a1=3 '108/108 positive' (correct: 107/107; narrative corrected, payload not).

## Plain-language summary (draft)

For the typical case (the most common value of each attribute: a1 = 1, a2 = 3, a3 = 1, a4 = 2, a5 = 1, a6 = 1) the model gives a predicted probability of 'yes' below 1%, and a1 = 2 instead of 1 raises it to above 99%; it adds up four parts: a ramp that falls as more attributes take their first value, a direct sum over all six attributes, a small unit on a5 and a3, and a tiny unit on a6. [C12, C16]

**Ramp on few first values.** This ramp reads a1, a2, a3, a4 and a6. It acts on 206 of 345 training cases, 78 of which are actually 'yes'; for the typical case it acts on (no attribute at its first value: a1 = 2, a2 = 3, a3 = 2, a4 = 2, a5 = 4, a6 = 2), it takes the predicted probability of 'yes' from about 72% without it to above 99% with it. [C27] Through it, a1 = 2, a3 = 2 or a6 = 2 instead of the first value takes the typical case's predicted probability of 'yes' from below 1% to above 99%. [C17, C18, C19] With this part in place, a1 = 3 instead of 1 takes the typical case from below 1% to above 99% when a3 = 1 or a6 = 1; when a3 = 2 or a6 = 2 the prediction is above 99% either way. [C23, C24] Through it, a1 = 3 instead of 1 and giving a2 or a4 its first value lower the probability, though for the typical case it stays below 1%. [C20, C21, C22]

**Direct sum.** This part adds up all six attributes with fixed weights. Through it, a1 = 3 instead of 1 raises the typical case's predicted probability of 'yes' from below 1% to above 99% and a1 = 2 to about 75%, and a5 = 2 instead of 1 raises it to about 93%, a5 = 3 to about 55% and a5 = 4 to about 3%. [C29, C31, C30, C32, C33] Through it, a4 = 1, a3 = 2 and a6 = 2 lower the probability and a2 = 1 or 2 instead of 3 raises it, though for the typical case it stays below 1%. [C34, C35, C37, C36, C38] Across the whole model, a5 = 2 instead of 1 takes the typical case to about 93%, a5 = 3 to about 55% and a5 = 4 to about 1%; a4 = 1 lowers it and a2 = 2 instead of 3 raises it, below 1%. [C06, C07, C08, C09, C11]

**Unit on a5 and a3.** This small unit reads a5 and a3. It acts on 259 of 345 training cases, 82 of which are actually 'yes'; for the typical case it acts on, the predicted probability of 'yes' stays below 1% with or without it. [C43] Through it, a5 = 4 and a3 = 2 lower the probability, though for the typical case it stays below 1%. [C40, C41]

**Tiny unit on a6.** A tiny unit on a6 barely changes the predictions. [C47]

## Parts, from the root down

### The whole model (`ann_28`, `upper_edge_count_classifier`)

- Kind: composition; parent: none; children: few_first_values_ramp, a5_not4_unit, a6_tiny_unit, direct_linear
- Reads: attribute a1, attribute a2, attribute a3, attribute a4, attribute a5, attribute a6
- Entry nodes: 263, 1855, 3471, id_linear, -10, -11, -14
- Exit nodes: 0
- Subgraph nodes: -10, -11, -14, 0

**Hypothesis (plain).** The model adds up four parts into one score: a ramp that gives 'yes' while few of a1, a2, a3, a4 and a6 take their first value, a direct sum over all six attributes in which a1 and a5 count most, and two small units. [C16, C27, C29, C30]

**Evidence (plain).** It is made up of four parts (each explained below); the score is their sum, which was checked on all 345 training cases. [C16] Changing one attribute at a time across the whole model, a1 = 2 instead of 1 raises the typical case's predicted probability of 'yes' from below 1% to above 99%. [C02, C12] Across the whole model, a5 = 2 instead of 1 takes it to about 93%, a5 = 3 to about 55% and a5 = 4 to about 1%. [C06, C07, C08, C13] Across the whole model, a4 = 1 lowers it and a2 = 2 instead of 3 raises it, though the typical case stays below 1%. [C09, C11, C14, C15]

**Annotation hypothesis (technical).** Hypothesis: additive composition under the output sigmoid, logit = 2.58 + Linear + 5.14*Ramp + 1.53*A5not4 + 0.105*A6. a4:2, a4:3, a5:3 have no enabled connections (zero-effect entries). The model is an upper-edge-only count detector: for a1 in {1,2} it is positive while few attributes take their first value and negative once about three of {a2,a3,a4,a6} (a1=1) or four (a1=2) do; a5=1 counts only through Linear's -14.8, which is not enough to flip a1=1 inputs with one other first value. There is no lower edge: inputs with zero or one first value get the largest logits. a1=3 inputs are all positive (107/107 on train) via Linear's +24.9. Evidence (train, n=345): positive rate by (a1, count) - a1=1: count<=2 100%, 3 74%, >=4 0%; a1=2: count<=3 100%, 4 29%, 5 0%; a1=3: 100% at every count. Predicts positive on 85.5% of train rows (accuracy 0.472 at 0.5). Logits reach |~130| so the sigmoid saturates; the sigmoid-output AUC 0.807 is inflated by ties at 1.0, the logit ranking AUC is 0.734. Compared with the MONK-2 target (exactly two first values): the network captures only the upper edge (>=3-4 first values -> negative) and none of the lower edge (0-1 -> negative), and ignores the count entirely when a1=3.

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (note: inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative; latex: y_0 = \frac{e^{1.53 x_{1855} + 5.14 x_{263} + 0.105 x_{3471} + 1.0 x_{id_linear}}}{e^{1.53 x_{1855} + 5.14 x_{263} + 0.105 x_{3471} + 1.0 x_{id_linear}} + 0.0758})
- *empirical_viz*: get_viz_summary sensitivity on the full training split (345 rows). (n: 345; split: train; viz_type: sensitivity; dataset_split_id: 63257a45-49c9-42a6-8b28-8e346eab2bee)
- *input_output_sampling*: Local forward pass (read-only repo weights) on the training split: positive unless many attributes take their first value; a1=3 always positive. (n: 345; note: sigmoid saturates (|logit| up to ~130) so many outputs tie at 1.0; the sigmoid-output AUC is inflated by ties relative to the logit ranking AUC; split: train; train_auc_logit: 0.7335; train_auc_sigmoid_output: 0.8065; train_predicted_positive_rate: 0.855)


### Ramp on few first values (`ann_24`, `few_first_values_ramp`)

- Kind: leaf; parent: upper_edge_count_classifier; children: none
- Reads: attribute a1, attribute a2, attribute a3, attribute a4, attribute a6
- Entry nodes: -2, -3_b, -4_b, -8_b, -9_b, -17_b
- Exit nodes: 263
- Subgraph nodes: -2, -3_b, -4_b, -8_b, -9_b, -17_b, 263

**Hypothesis (plain).** This ramp is the model's core rule: a weighted sum of a1, a2, a3, a4 and a6 plus a constant, kept only where it is above zero and passed on to the score with a fixed weight. [C17, C18, C19]

**Evidence (plain).** It acts on 206 of 345 training cases, 78 of which are actually 'yes'; for the typical case it acts on (no attribute at its first value: a1 = 2, a2 = 3, a3 = 2, a4 = 2, a5 = 4, a6 = 2), it takes the predicted probability of 'yes' from about 72% without it to above 99% with it. [C27] Through it, a1 = 2, a3 = 2 or a6 = 2 instead of the first value takes the typical case's predicted probability of 'yes' from below 1% to above 99%. [C17, C18, C19] With this part in place, a1 = 3 instead of 1 takes the typical case from below 1% to above 99% when a3 = 1 or a6 = 1; when a3 = 2 or a6 = 2 the prediction is above 99% either way. [C23, C24] Through it, a1 = 3 instead of 1 and giving a2 or a4 its first value lower the probability, though for the typical case it stays below 1%; a1 = 3 also counts more when a4 = 1 or a2 = 1. [C20, C21, C22, C25, C26] Its formula reproduces the part exactly on all 345 training cases. [C28]

**Annotation hypothesis (technical).** Hypothesis: a falling count ramp over first-value indicators. In indicator terms node 263 = ReLU(18.98 - 8.46[a2=1] - 9.29[a3=1] - 6.75[a4=1] - 8.23[a6=1] + 5.81[a1=2] - 24.11[a1=3]); a3:2 and a6:2 enter as complements of a3=1, a6=1. a1=1: positive iff at most two of {a2,a3,a4,a6}=1 (all pairs >= 1.23, all triples <= -4.46). a1=2: +5.8 shift, three indicators leave it at 0.3-1.4 except a2+a3+a6 (-1.19); four switch it off. a1=3: always 0 (max pre -5.13). a5 is not consulted. Evidence (train, n=345): closed form from get_formula (z-scored inputs; closed-form max err 0.041 from 3-s.f. rounding); sensitivity largest for a1:3 (6.66); values 0..24.79, zero on 40.3% of rows; ablation flips 112/345 decisions and drops logit AUC 0.734 -> 0.537. Readout 5.14: the dominant decision-bearing term.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.74 x_{a1:2} \\
&\quad - 11.2 x_{a1:3_b} \\
&\quad - 4.0 x_{a2:1_b} \\
&\quad + 4.65 x_{a3:2_b} \\
&\quad - 3.12 x_{a4:1_b} \\
&\quad + 4.11 x_{a6:2_b} \\
&\quad -0.47 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (note: inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 2.74 x_{a1:2} \\
&\quad - 11.2 x_{a1:3_b} \\
&\quad - 4.0 x_{a2:1_b} \\
&\quad + 4.65 x_{a3:2_b} \\
&\quad - 3.12 x_{a4:1_b} \\
&\quad + 4.11 x_{a6:2_b} \\
&\quad -0.47 \bigr)
\end{aligned})
- *empirical_viz*: get_viz_summary sensitivity on the full training split (345 rows). (n: 345; split: train; viz_type: sensitivity; dataset_split_id: 63257a45-49c9-42a6-8b28-8e346eab2bee)
- *ablation_study*: Removing the ramp flips 112/345 training decisions and drops logit AUC 0.734 -> 0.537: it is the main decision-bearing unit. (n: 345; split: train; values_on_train: 0 to 24.79; zero on 40.3% of rows)


### Direct sum (`ann_27`, `direct_linear`)

- Kind: leaf; parent: upper_edge_count_classifier; children: none
- Reads: attribute a1, attribute a2, attribute a3, attribute a4, attribute a5, attribute a6
- Entry nodes: -1, -3_a, -4_a, -5, -6, -8_a, -9_a, -12, -13, -15_a, -17_a
- Exit nodes: id_linear
- Subgraph nodes: -1, -3_a, -4_a, -5, -6, -8_a, -9_a, -12, -13, -15_a, -17_a, id_linear

**Hypothesis (plain).** This part adds up all six attributes with fixed weights, and a1 and a5 count most: a1 = 3 instead of 1 takes the typical case's predicted probability of 'yes' from below 1% to above 99%, and a5 = 2 instead of 1 to about 93%. [C29, C30]

**Evidence (plain).** Through it, a1 = 3 instead of 1 raises the typical case's predicted probability of 'yes' from below 1% to above 99% and a1 = 2 to about 75%, and a5 = 2 instead of 1 raises it to about 93%, a5 = 3 to about 55% and a5 = 4 to about 3%. [C29, C31, C30, C32, C33] Through it, a4 = 1, a3 = 2 and a6 = 2 raise the probability and a2 = 1 or 2 instead of 3 lowers it, though for the typical case it stays below 1%. [C34, C35, C37, C36, C38] Its formula reproduces the part exactly on all 345 training cases. [C39]

**Annotation hypothesis (technical).** Hypothesis: the direct linear part. Per attribute in indicator terms (relative to the reference value): a1=1 -15.69, a1=3 +24.94 (a1=2: 0); a5=1 -14.80, a5=2 +2.38, a5=4 -3.58 (a5=3: 0); a4=1 -4.90; a3=2 -2.19; a2=1 +1.20, a2=2 +0.73, a2=3 +0.61; a6=2 -0.24. It penalises the two first values the ramp does not see (a1=1, a5=1) by ~15 each, and gives a1=3 a +24.9 bonus that keeps every a1=3 input positive although the ramp is off there. Evidence (train, n=345): closed form (linear; err 0.065 from 3-s.f. rounding on values -33.4..+32.2); sensitivities equal the z-scale weights (a1:3 11.5, a1:1 7.5, a5:1 6.4).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 7.5 x_{a1:1} \\
&\quad + 11.5 x_{a1:3_a} \\
&\quad + 0.566 x_{a2:1_a} \\
&\quad + 0.342 x_{a2:2} \\
&\quad + 0.291 x_{a2:3} \\
&\quad - 1.09 x_{a3:2_a} \\
&\quad - 2.27 x_{a4:1_a} \\
&\quad - 6.43 x_{a5:1} \\
&\quad + 1.03 x_{a5:2} \\
&\quad - 1.55 x_{a5:4_a} \\
&\quad - 0.12 x_{a6:2_a}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (note: inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative; latex: \begin{aligned}
y_0 &= - 7.5 x_{a1:1} \\
&\quad + 11.5 x_{a1:3_a} \\
&\quad + 0.566 x_{a2:1_a} \\
&\quad + 0.342 x_{a2:2} \\
&\quad + 0.291 x_{a2:3} \\
&\quad - 1.09 x_{a3:2_a} \\
&\quad - 2.27 x_{a4:1_a} \\
&\quad - 6.43 x_{a5:1} \\
&\quad + 1.03 x_{a5:2} \\
&\quad - 1.55 x_{a5:4_a} \\
&\quad - 0.12 x_{a6:2_a}
\end{aligned})
- *empirical_viz*: get_viz_summary sensitivity on the full training split (345 rows). (n: 345; split: train; viz_type: sensitivity; dataset_split_id: 63257a45-49c9-42a6-8b28-8e346eab2bee)


### Unit on a5 and a3 (`ann_25`, `a5_not4_unit`)

- Kind: leaf; parent: upper_edge_count_classifier; children: none
- Reads: attribute a3, attribute a5
- Entry nodes: -7, -15_b
- Exit nodes: 1855
- Subgraph nodes: -7, -15_b, 1855

**Hypothesis (plain).** This small unit reads a5 and a3: through it, a5 = 4 and a3 = 2 lower the probability of 'yes', though for the typical case it stays below 1%. [C40, C41]

**Evidence (plain).** It acts on 259 of 345 training cases, 82 of which are actually 'yes'; for the typical case it acts on, the predicted probability of 'yes' stays below 1% with or without it. [C43] Through it, a5 = 4 and a3 = 2 lower the probability, though for the typical case it stays below 1%. [C40, C41] With this part in place, a3 = 2 instead of 1 takes the typical case from about 1% to above 99% when a5 = 4, and from below 1% to above 99% otherwise. [C42] Its formula reproduces the part exactly on all 345 training cases. [C44]

**Annotation hypothesis (technical).** Hypothesis: an a5-not-4 indicator weighted by a3. Node 1855 = ReLU(0.295 + 0.682[a3=1] - 3.271[a5=4]) takes values 0 (a5=4), 0.295 (a3=2) or 0.977 (a3=1). Evidence (train, n=345): closed form (err 0.0026 from rounding); sensitivity a5:4 1.06 vs a3:1 0.26; readout 1.53 adds at most 1.49 to the logit; ablation flips 3/345 decisions, logit AUC 0.7335 -> 0.7329. A minor correction term.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.341 x_{a3:1} \\
&\quad - 1.42 x_{a5:4_b} \\
&\quad -0.171 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (note: inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative; latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.341 x_{a3:1} \\
&\quad - 1.42 x_{a5:4_b} \\
&\quad -0.171 \bigr)
\end{aligned})
- *empirical_viz*: get_viz_summary sensitivity on the full training split (345 rows). (n: 345; split: train; viz_type: sensitivity; dataset_split_id: 63257a45-49c9-42a6-8b28-8e346eab2bee)
- *ablation_study*: Ablating flips 3/345 decisions; AUC essentially unchanged. (n: 345; split: train)


### Tiny unit on a6 (`ann_26`, `a6_tiny_unit`)

- Kind: leaf; parent: upper_edge_count_classifier; children: none
- Reads: attribute a6
- Entry nodes: -16
- Exit nodes: 3471
- Subgraph nodes: -16, 3471

**Hypothesis (plain).** This tiny unit on a6 barely changes the predictions. [C47]

**Evidence (plain).** It acts on the 168 of 345 training cases with a6 = 2, 58 of which are actually 'yes'; for the typical case it acts on, the predicted probability of 'yes' is above 99% with or without it. [C46] Through it, a6 = 2 instead of 1 raises the probability, though for the typical case it stays below 1%. [C45] Switching it off barely changes the predictions. [C47] Its formula reproduces the part exactly on all 345 training cases. [C48]

**Annotation hypothesis (technical).** Hypothesis: a near-inert a6 indicator. Node 3471 = ReLU(0.743 - 1.412[a6=1]) = 0.743[a6=2], linear in binary a6. Evidence (train, n=345): closed form (err 0.0003); readout 0.105 adds at most 0.078 to the logit; ablation flips no decisions. Functionally negligible, unearned structure.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(0.0186 - 0.706 x_{a6:1} \right)}
```

**Evidence records.**

- *analytical_formula*: Closed form from get_formula (force=true). (note: inputs are train-z-scored one-hot columns; indicator-scale coefficients in narrative; latex: y_0 = \operatorname{relu}{\left(0.0186 - 0.706 x_{a6:1} \right)})
- *empirical_viz*: get_viz_summary sensitivity on the full training split (345 rows). (n: 345; split: train; viz_type: sensitivity; dataset_split_id: 63257a45-49c9-42a6-8b28-8e346eab2bee)
- *ablation_study*: Ablating flips no decisions; contribution at most 0.078 to the logit. (n: 345; split: train)

## Worth checking (draft)

- Worth checking: the model gives 'yes' to cases with few attributes at their first value: the typical case the ramp acts on has no attribute at its first value and gets above 99%, and a1 = 2 instead of 1 takes the typical case (four attributes at their first value) from below 1% to above 99%, which the exactly-two rule does not do. [C27, C02]
- Worth checking: the model separates the values of a5 other than the first: for the typical case, a5 = 2 gives about 93%, a5 = 3 about 55% and a5 = 4 about 1%, which the exactly-two rule does not do. [C06, C07, C08]
