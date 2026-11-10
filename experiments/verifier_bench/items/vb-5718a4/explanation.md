# Explanation vb-5718a4

- Genome: `60ee493c-bce5-4a2b-b7af-d06843835ae6`
- Dataset: monk2_onehot (training split 63257a45-49c9-42a6-8b28-8e346eab2bee, n = 345; test n = 87)
- Outcome: the 'yes' class
- Typical case: the typical patient: the median training patient for every number and the most common training category for every category (not a real patient); predicted 0%

Machine-readable artefacts: `record.json` (the explaining agent's record and the automatic checks), `operations.json` (the refactoring operations and annotations, with hypotheses and evidence), `plain_language.json` (claims, parts and the plain-language draft). Claim ids in brackets refer to `plain_language.json` claims.

## Record

- structural_coverage: 1.0
- compositional_coverage: 1.0
- full_coverage: True
- hierarchy_complete: True
- function_preservation_max_abs_err: 4.440892098500626e-16
- closed_form_max_abs_err: 0.16991625616246608
- n_leaf_annotations: 5
- n_composition_annotations: 1
- n_splits: 8
- n_identity_nodes: 2

**Summary.** sigmoid(-2.36 + Lin + Veto + 2.55 Gate - 0.60 A5): Lin +4..+12 per first value of a1,a2,a3,a4,a6; Veto a ReLU count unit, 0 for <=2 and down to -70 for >=3; Gate +9..+11 when a5!=1 (bonus with a3=1). Predicts 'exactly two of {a1,a2,a3,a4,a6}, preferably a5!=1'. Train acc 0.887, AUC 0.925.

**What resisted.** a5_a3_bonus_gate is not a clean logical unit (additive a5!=1 / a3=1 bonus with one clip interaction, plus near-inert sub-unit 1717). Nodes 1948, 2786 affine on the domain; node 927 inert. Closed-form error 0.17 is 3-s.f. rounding at magnitudes up to 269 (numeric 2.8e-14).

## Plain-language summary (draft)

For the typical case (the most common value of each attribute: a1 = 1, a2 = 3, a3 = 1, a4 = 2, a5 = 1, a6 = 1) the model gives a predicted probability of 'yes' below 1%; it adds up five parts: a sum over all six attributes, a veto on a1, a2, a3, a4 and a6, a gate on a1, a3, a4, a5 and a6, a part on a5 alone, and a unit that is switched off. [C16, C06, C52]

**Veto on a1, a2, a3, a4 and a6.** This part reads a1, a2, a3, a4 and a6: through it, moving a1, a3 or a6 away from its first value raises the typical case's predicted probability of 'yes' from below 1% to above 99%. [C17, C18, C19] Giving a2 or a4 its first value pulls it back down: with a3 = 2, a2 = 1 or a4 = 1 takes it from above 99% to below 1%, and with a6 = 2, a2 = 1 takes it from about 81% to below 1%, while with a3 = 1 or a6 = 1 it is below 1% either way. [C22, C23, C24]

**Attribute sum.** This part adds up all six attributes with fixed weights: through it, a2 = 1 or 2 instead of 3 and a4 = 1 instead of 2 raise the predicted probability of 'yes', and moving a1, a3, a5 or a6 away from its first value or a4 from 2 to 3 lowers it, though for the typical case it stays below 1%. [C37, C27, C29, C28, C30, C31, C34, C35, C36, C32, C33] Across the whole model, a5 = 2, 3 or 4 instead of 1 and a2 = 2 instead of 3 raise it, and a4 = 3 instead of 2 lowers it, while the typical case stays below 1%. [C06, C07, C08, C09, C12]

**Gate on a5 and a3.** This gate acts on 305 of 345 training cases, 99 of which are actually 'yes'; for the typical case it acts on, the predicted probability of 'yes' stays below 1% with or without it. [C47] Through it, a5 other than 1, a4 = 3 and a1 = 3 raise the probability and a3 = 2 and a6 = 2 lower it, all below 1% for the typical case it acts on. [C39, C38, C42, C40, C41] How much a6 = 2 instead of 1 moves it depends on a5: from below 1% to above 99% when a5 is not 1, and to about 81% when a5 = 1. [C43]

**Part on a5.** This part reads only a5 and acts on 173 of 345 training cases, 65 of which are actually 'yes'; through it, a5 = 2 or 4 instead of 1 lowers the predicted probability of 'yes', though for the typical case it stays below 1%. [C50, C49]

**Switched-off unit on a3.** This unit reads a3 and is switched off: it barely changes the predictions. [C52]

## Parts, from the root down

### The whole model (`ann_32`, `two_of_five_peak`)

- Kind: composition; parent: none; children: linear_first_value_score, atleast3_of_five_veto, a5_a3_bonus_gate, a5_in_2_4_penalty, inert_a32_unit
- Reads: attribute a1, attribute a2, attribute a3, attribute a4, attribute a5, attribute a6
- Entry nodes: id_linear, id_suppress, 1223, 1751, 927, -10, -13, -17
- Exit nodes: 0
- Subgraph nodes: 0, -10, -13, -17

**Hypothesis (plain).** The model adds up five parts into one score: a sum over all six attributes, a veto on a1, a2, a3, a4 and a6, a gate on a1, a3, a4, a5 and a6, a part on a5 alone, and a unit that is switched off. [C16, C52]

**Evidence (plain).** It is made up of five parts, the attribute sum, the veto, the gate on a5 and a3, the part on a5 and the switched-off unit (each explained below); the score is their sum, which was checked on all 345 training cases. [C16] Changing one attribute at a time across the whole model, a5 = 2, 3 or 4 instead of 1 and a2 = 2 instead of 3 raise the predicted probability of 'yes', and a4 = 3 instead of 2 lowers it, but the typical case stays below 1% in each case. [C06, C07, C08, C09, C12]

**Annotation hypothesis (technical).** Two-of-five peak with an a5 handicap. The output is sigmoid(s) with s = -2.363 + Lin + Veto + 2.547*Gate - 0.599*A5 + 0.735*Inert, where Inert is identically 0. Lin (linear_first_value_score) rises by +4..+12 for each first value among a1, a2, a3, a4, a6 and is the rising edge: with at most one such first value s < 0 on all 432 configurations (max -0.42). Veto (atleast3_of_five_veto) is exactly 0 for at most two first values and <= -70 for three or more, so it is the falling edge: s < 0 for every configuration with three or more. Gate (a5_a3_bonus_gate) adds +8.5..+11 when a5 != 1 (more when a3 = 1) and nothing when a5 = 1 and a3 = 2; A5 is a -0.72 penalty for a5 in {2,4}. Net: with exactly two first values among {a1,a2,a3,a4,a6} the model predicts positive in 113 of 114 configurations when a5 != 1 but only 23 of 38 (61%) when a5 = 1 (the a5_a3 gate is weak or off there; pairs (2,3) and (3,4) always fail). So the network computes approximately 'exactly two of the five attributes a1,a2,a3,a4,a6 take their first value, and preferably a5 != 1': it agrees with [count5 == 2] on 96.3% of all 432 configurations and 95.9% of training rows, and reaches 88.7% train accuracy (AUC 0.925). Inputs a4:2, a5:2, a6:2 have no enabled connections and are zero-effect entries. Evidence: on train, ablating the veto drops accuracy 0.887 -> 0.562 and ablating the gate 0.887 -> 0.690; the score sign is negative for every configuration with <=1 or >=3 such first values. Comparison with the target rule (exactly two of all six attributes take their first value): the network has replaced a5=1 as a rule indicator with a5 != 1 as a bonus, so it is correct on two-of-six cases where a5 != 1 (113/114) but misses all 28 two-of-six cases in which a5 = 1 is one of the two, and wrongly accepts 23 of 38 three-of-six cases that include a5 = 1 (agreement with the rule 88.0% over all 432 configurations).

**Evidence records.**

- *analytical_formula*: Additive composition of five children under the output sigmoid. (latex: y_0 = \frac{e^{2.55 x_{1223} - 0.599 x_{1751} + 0.735 x_{927} + 1.0 x_{id_linear} + 1.0 x_{id_suppress}}}{e^{2.55 x_{1223} - 0.599 x_{1751} + 0.735 x_{927} + 1.0 x_{id_linear} + 1.0 x_{id_suppress}} + 10.6}; score: s = -2.363 + Lin + Veto + 2.547*Gate - 0.599*A5 + 0.735*Inert)
- *empirical_viz*: Score is negative for <=1 and >=3 first-values among a1,a2,a3,a4,a6 on every configuration; positive for exactly two except mostly when a5=1. (n: 345; split: train; viz_type: sensitivity; train_agree_with_count5_eq_2: 0.959)


### Veto on a1, a2, a3, a4 and a6 (`ann_28`, `atleast3_of_five_veto`)

- Kind: leaf; parent: two_of_five_peak; children: none
- Reads: attribute a1, attribute a2, attribute a3, attribute a4, attribute a6
- Entry nodes: -1_b, -4_b, -7_b, -9_b, -16_b
- Exit nodes: id_suppress
- Subgraph nodes: -1_b, -4_b, -7_b, -9_b, -16_b, 274, 2786, id_suppress

**Hypothesis (plain).** This part holds the probability of 'yes' down for the typical case: moving a1, a3 or a6 away from its first value lifts it to above 99% through it, while giving a2 or a4 its first value brings it back below 1%. [C17, C18, C19, C22, C23]

**Evidence (plain).** Through it, moving a1, a3 or a6 away from its first value takes the typical case's predicted probability of 'yes' from below 1% to above 99%. [C17, C18, C19] With a3 = 2, a2 = 1 or a4 = 1 takes it from above 99% to below 1%, and with a6 = 2, a2 = 1 takes it from about 81% to below 1%, while with a3 = 1 or a6 = 1 it is below 1% either way. [C22, C23, C24] For the typical case itself, a2 = 1 or a4 = 1 lowers it through this part, and a4 = 1 counts more when a2 = 1, though it stays below 1%. [C20, C21, C25] Its formula reproduces the part exactly on all 345 training cases. [C26]

**Annotation hypothesis (technical).** At-least-three-of-five veto. Node 274 = ReLU(-1.34 + 3.46 z(a1:1) + 4.56 z(a2:1) + 4.67 z(a3:1) + 3.98 z(a4:1) + 4.48 z(a6:1)) counts first-values of a1, a2, a3, a4 and a6 (a5 is NOT included). Per-indicator increments are 7.2 to 9.7 on a base of -19.2: with two or fewer indicators the pre-activation is at most -0.2 (exactly 0 output), with three it is at least +5.6. Node 2786 = ReLU(0.605*274 - 0.609) never clips while 274 is on, so the exit id_suppress = -7.14*274 - 6.56*2786 = -11.11*274 + 4.00 whenever 274 fires, i.e. a penalty of -70 or more (-269 with all five), which forces the output to 0. Exactly zero otherwise. Forms the falling edge of the response peak at three-of-five. Evidence: on all 345 training rows 274 > 0 exactly when three or more of a1,a2,a3,a4,a6 take their first value (112 rows); removing the veto drops train accuracy from 0.887 to 0.562.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= - 6.56 \operatorname{relu}{\left(0.605 \operatorname{relu}{\left(3.46 x_{a1:1_b} + 4.56 x_{a2:1_b} + 4.67 x_{a3:1_b} + 3.98 x_{a4:1_b} + 4.48 x_{a6:1_b} - 1.34 \right)} - 0.609 \right)} \\
&\quad - 7.14 \operatorname{relu}{\left(3.46 x_{a1:1_b} + 4.56 x_{a2:1_b} + 4.67 x_{a3:1_b} + 3.98 x_{a4:1_b} + 4.48 x_{a6:1_b} - 1.34 \right)}
\end{aligned}
```

**Evidence records.**

- *empirical_viz*: Count threshold at three-of-five confirmed on every training row; the veto is essential. (n: 345; split: train; viz_type: sensitivity; train_check: 274>0 iff count of first-values among {a1,a2,a3,a4,a6} >=3: true on all 345 train rows (112 rows fire); min firing pre-activation 5.6; max non-firing -0.2 (exhaustive over 432 configs); ablation_train: removing the veto drops train accuracy 0.887 -> 0.562)
- *analytical_formula*: Serial relu chain 274 -> 2786; 2786 never clips while 274>0. (latex: \begin{aligned}
y_0 &= - 6.56 \operatorname{relu}{\left(0.605 \operatorname{relu}{\left(3.46 x_{a1:1_b} + 4.56 x_{a2:1_b} + 4.67 x_{a3:1_b} + 3.98 x_{a4:1_b} + 4.48 x_{a6:1_b} - 1.34 \right)} - 0.609 \right)} \\
&\quad - 7.14 \operatorname{relu}{\left(3.46 x_{a1:1_b} + 4.56 x_{a2:1_b} + 4.67 x_{a3:1_b} + 3.98 x_{a4:1_b} + 4.48 x_{a6:1_b} - 1.34 \right)}
\end{aligned}; simplified: id_suppress = 0 if 274=0 else -11.109*274 + 3.995)


### Attribute sum (`ann_27`, `linear_first_value_score`)

- Kind: leaf; parent: two_of_five_peak; children: none
- Reads: attribute a1, attribute a2, attribute a3, attribute a4, attribute a5, attribute a6
- Entry nodes: -1_a, -2, -3_a, -4_a, -5, -6, -7_a, -9_a, -11_a, -12_a, -15, -16_a
- Exit nodes: id_linear
- Subgraph nodes: -1_a, -2, -3_a, -4_a, -5, -6, -7_a, -9_a, -11_a, -12_a, -15, -16_a, 1948, id_linear

**Hypothesis (plain).** This part adds up all six attributes with fixed weights: a first value of a1, a2, a3, a4, a5 or a6 raises the probability of 'yes', and a2 = 3 and a4 = 3 lower it. [C27, C28, C29, C30, C31, C32, C33, C34, C35, C36]

**Evidence (plain).** Through it, a2 = 1 or 2 instead of 3 and a4 = 1 instead of 2 raise the typical case's predicted probability of 'yes', and moving a1, a3, a5 or a6 away from its first value or a4 from 2 to 3 lowers it, though it stays below 1%. [C27, C29, C28, C30, C31, C34, C35, C36, C32, C33] Its formula reproduces the part exactly on all 345 training cases. [C37]

**Annotation hypothesis (technical).** Additive per-attribute score (the linear backbone). Measured per attribute value relative to the all-low baseline, the direct edges give large positive credit when an attribute takes its first value: a6=1 +11.94, a1=1 +10.22, a2=1 +6.63, a4=1 +6.22, a3=1 +4.12, but only +2.63 for a5=1; non-first values add 0 or a penalty (a4=3 -4.60, a2=3 -3.07, a1=2 -3.00, a1=3 -1.71, a2=2 -1.91, a5=4 -0.26). Node 1948 = ReLU(0.827 - 0.090 z(a1:2)) never clips (it takes only 0.70 or 0.89), so it is an affine relay of a1:2 folded into the a1=2 penalty. The score rises roughly linearly with the number of first-values among a1, a2, a3, a4, a6 and is the rising edge of the model's response peak. Evidence: closed form is linear (sensitivities on train equal the direct weights exactly; 1948 takes only 0.70/0.89 on train, never clipping).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= 4.89 x_{a1:1_a} \\
&\quad - 1.34 x_{a1:2} \\
&\quad - 0.789 x_{a1:3_a} \\
&\quad + 3.13 x_{a2:1_a} \\
&\quad - 0.892 x_{a2:2} \\
&\quad - 1.46 x_{a2:3} \\
&\quad + 2.06 x_{a3:1_a} \\
&\quad + 2.88 x_{a4:1_a} \\
&\quad - 2.13 x_{a4:3_a} \\
&\quad + 1.14 x_{a5:1_a} \\
&\quad - 0.112 x_{a5:4} \\
&\quad + 5.97 x_{a6:1_a} \\
&\quad + 0.899 \operatorname{relu}{\left(0.827 - 0.0898 x_{a1:2} \right)}
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Closed form is linear in the 12 entries plus one never-clipping relu relay of a1:2. (note: Affine on domain: 1948 relu never clips (values 0.70/0.89 on train).)
- *empirical_viz*: Sensitivities equal the direct weights (constant gradient = linear). First values of a1,a2,a3,a4,a6 earn +4.1..+11.9; a5=1 only +2.6. (n: 345; split: train; viz_type: sensitivity)


### Gate on a5 and a3 (`ann_29`, `a5_a3_bonus_gate`)

- Kind: leaf; parent: two_of_five_peak; children: none
- Reads: attribute a1, attribute a3, attribute a4, attribute a5, attribute a6
- Entry nodes: -3_b, -12_b, -12_c, -16_c, -7_c, -11_b
- Exit nodes: 1223
- Subgraph nodes: -3_b, -12_b, -12_c, -16_c, -7_c, -11_b, 1717, 1223

**Hypothesis (plain).** This gate raises the probability of 'yes' when a5 is not 1, and lowers it for a3 = 2 and a6 = 2. [C39, C40, C41]

**Evidence (plain).** It acts on 305 of 345 training cases, 99 of which are actually 'yes'; for the typical case it acts on, the predicted probability of 'yes' stays below 1% with or without it. [C47] Through it, a5 other than 1, a4 = 3 and a1 = 3 raise the probability and a3 = 2 and a6 = 2 lower it, all below 1% for the typical case it acts on. [C39, C38, C42, C40, C41] How much a6 = 2 instead of 1 moves it depends on a5: from below 1% to above 99% when a5 is not 1, and to about 81% when a5 = 1. [C43] With a3 = 2, a4 = 3 takes it from above 99% to about 97%. [C44] Its formula reproduces the part exactly on all 345 training cases. [C48]

**Annotation hypothesis (technical).** A clipped bonus that rewards a5 != 1 and a3 = 1. Node 1223 = ReLU(2.47 + 1.41 z(a3:1) + 0.30 z(a4:3) - 1.86 z(a5:1) - 0.69*1717). Readout to the output (x2.547): a5 != 1 gives +12.0..13.7 when a3=1 and +4.7..6.5 when a3=2; a5 = 1 gives +1.1..2.75 when a3=1 and exactly 0 when a3=2 (clipped). So it is mostly an additive a5 != 1 bonus of ~+8.5 to +11 plus an a3=1 bonus, with one clip interaction (a5=1 and a3=2). Sub-unit 1717 = ReLU(-0.52 - 0.03 z(a1:3) - 0.05 z(a5:1) - 0.56 z(a6:1)) is near-inert: it is nonzero only when a6 != 1 and never exceeds 0.10, moving 1223 by at most 0.07. Net effect: a5 = 1 is treated as evidence AGAINST the positive class, cancelling the small direct +2.63 credit for a5=1. Evidence: on all 345 training rows 1223 = 0 exactly when a5 = 1 and a3 = 2 (40 rows); train sensitivities are dominated by a5:1 (1.65) and a3:1 (1.25), with the 1717 path < 0.15; removing the gate drops train accuracy from 0.887 to 0.690.

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.41 x_{a3:1_c} \\
&\quad + 0.298 x_{a4:3_b} \\
&\quad - 1.86 x_{a5:1_b} \\
&\quad - 0.691 \operatorname{relu}{\left(- 0.0341 x_{a1:3_b} - 0.0515 x_{a5:1_c} - 0.559 x_{a6:1_c} - 0.525 \right)} \\
&\quad + 2.47 \bigr)
\end{aligned}
```

**Evidence records.**

- *empirical_viz*: a5:1 and a3:1 dominate; 1717 path effect <=0.07. (n: 345; split: train; viz_type: sensitivity; train_check: 1223==0 iff a5=1 and a3=2 on all 345 rows (40 rows clipped); ablation_train: removing it drops train accuracy 0.887 -> 0.690)
- *analytical_formula*: Single clipped relu over a3:1, a4:3, a5:1 with near-inert 1717 sub-unit. (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.41 x_{a3:1_c} \\
&\quad + 0.298 x_{a4:3_b} \\
&\quad - 1.86 x_{a5:1_b} \\
&\quad - 0.691 \operatorname{relu}{\left(- 0.0341 x_{a1:3_b} - 0.0515 x_{a5:1_c} - 0.559 x_{a6:1_c} - 0.525 \right)} \\
&\quad + 2.47 \bigr)
\end{aligned})


### Part on a5 (`ann_30`, `a5_in_2_4_penalty`)

- Kind: leaf; parent: two_of_five_peak; children: none
- Reads: attribute a5
- Entry nodes: -12_d, -14
- Exit nodes: 1751
- Subgraph nodes: -12_d, -14, 1751

**Hypothesis (plain).** This part lowers the probability of 'yes' when a5 = 2 or 4. [C49]

**Evidence (plain).** It acts on 173 of 345 training cases, 65 of which are actually 'yes'; for the typical case it acts on, the predicted probability of 'yes' stays below 1% with or without it. [C50] Through it, a5 = 2 or 4 instead of 1 lowers the probability, though for the typical case it stays below 1%. [C49] Its formula reproduces the part exactly on all 345 training cases. [C51]

**Annotation hypothesis (technical).** Exact indicator of a5 in {2,4}. Node 1751 = ReLU(-0.29 - 1.32 z(a5:1) - 1.26 z(a5:3)) is 1.198 when a5 is 2 or 4 and exactly 0 when a5 is 1 or 3. With the -0.599 readout it is a constant -0.72 penalty for a5 in {2,4}: a small per-attribute lookup on a5, not an interaction. Evidence: on train the exit takes only 0 and 1.198, and is positive exactly when a5 in {2,4}; removing it changes train accuracy 0.887 -> 0.890 (negligible).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.32 x_{a5:1_d} \\
&\quad - 1.26 x_{a5:3} \\
&\quad -0.286 \bigr)
\end{aligned}
```

**Evidence records.**

- *analytical_formula*: Two-input relu over a5 one-hot columns. (latex: \begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 1.32 x_{a5:1_d} \\
&\quad - 1.26 x_{a5:3} \\
&\quad -0.286 \bigr)
\end{aligned})
- *empirical_viz*: Exact a5 in {2,4} indicator; negligible for accuracy. (n: 345; split: train; viz_type: line; train_check: 1751>0 iff a5 in {2,4} on all 345 rows; ablation_train: removing it changes train accuracy 0.887 -> 0.890)


### Switched-off unit on a3 (`ann_31`, `inert_a32_unit`)

- Kind: leaf; parent: two_of_five_peak; children: none
- Reads: attribute a3
- Entry nodes: -8
- Exit nodes: 927
- Subgraph nodes: -8, 927

**Hypothesis (plain).** This unit is switched off and barely changes the predictions. [C52]

**Evidence (plain).** It reads a3, but switching it off barely changes the predictions. [C52] Its formula reproduces the part exactly on all 345 training cases. [C53]

**Annotation hypothesis (technical).** Inert unit. Node 927 = ReLU(-2.43 - 1.93 z(a3:2)); z(a3:2) takes only -0.974 or +1.026, so the pre-activation is at most -0.55 and the node outputs exactly 0 on every input in the domain. Its +0.73 readout contributes nothing. Evidence: the exit is 0 on all 345 training rows (line plot y range [0,0]).

**Closed form** (weights at 3 significant figures; inputs are standardised):

```latex
y_0 = \operatorname{relu}{\left(- 1.93 x_{a3:2} - 2.43 \right)}
```

**Evidence records.**

- *analytical_formula*: Pre-activation negative for both values of a3:2. (latex: y_0 = \operatorname{relu}{\left(- 1.93 x_{a3:2} - 2.43 \right)}; max_preactivation: -0.55)
- *empirical_viz*: Exit is identically 0 on all 345 training rows. (n: 345; split: train; viz_type: line)

## Worth checking (draft)

- Worth checking: the veto reads a1, a2, a3, a4 and a6 but not a5, and another part raises the probability of 'yes' when a5 is not 1, so a5 = 1 does not count like the first value of the other attributes, as it does in the exactly-two rule. [C17, C39]
- Worth checking: the model separates values of a2 and a4 other than the first: a2 = 2 instead of 3 raises the predicted probability of 'yes' and a4 = 3 instead of 2 lowers it, which the exactly-two rule does not do. [C09, C12]
