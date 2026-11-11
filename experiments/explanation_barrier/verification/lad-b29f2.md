# Verdict: lad-b29f2 (genome 7130d166)

**Overall: refuted.** The rule-list reading is sound and well supported. The stated rule matches the model on 99.5% of training rows, and its thresholds fit best. One statement in the root hypothesis is wrong. The record also has four smaller imprecisions.

## Refuted and weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.risk_rules_root.hypothesis | L2, refuted | Switching off the bmi, under-age and few-visits hinges together lowers the training AUC to 0.500 | AUC 0.623 with those three off. The cited query (q_a5bd54a192) switched off all seven leaves, which makes the prediction constant (AUC 0.5) | `verify ablate --part high_bmi_hinges --part under_age_hinge --part few_visits_hinges` |
| operations.sex_smoking_age_detectors.hypothesis | L3, weakened | Units 7 and 9 are large for young women, larger still for non-smoking ones | They detect young non-smokers and are boosted by being female. At age 25, unit 7 is 1.90 for a male non-smoker and 1.53 for a female smoker. Unit 9 is 1.65 and 0.53 | node values in the four smoker-by-sex cells; weights |
| operations.older_age_hinges.hypothesis | L3, weakened | Risk drifts from 11% at age 20 to 4% at 70, and these units give that slope | The part contributes nothing up to age 50, where the risk falls from 10.8% to 8.0% anyway. It lowers the risk only above about 55, most at 70 (4.4% vs 7.1% without it) | contribution over an age sweep |
| record.summary | L3, weakened | Income, urban and noise shift the risk near the bmi step | Income does (50% to 15%) and urban does (44% to 48%). Noise from -2 to 2 gives 44%, 48%, 43% at bmi 32, a few points only | probe at bmi 32 |
| record.leaves.score_insurance_hinges.meaning | L4, weakened | meaningful | The unit-6 veto is confirmed. The record's own resisted text says unit 14 is read only partly, so "partial" fits better | contribution by insurance and rule |

## Upheld
- Record: coverage 1.0/1.0, hierarchy complete, function preservation 0, closed-form error 0.0188, operation counts. The resisted numbers 0.54, -0.28 and 0.896 reproduce, as does "three dead units".
- Summary rule: 5709 rows at 90% and 18,291 at 8%; 109 and 13 on the other side of even odds; 92 on the bmi step. These thresholds give 122 disagreements. Every alternative threshold tested does worse, except bmi 32.1 (52) and score 81 (121), and both are within rounding.
- high_bmi_hinges (ann_20): every number reproduces. The step at bmi 32 to 32.5 holds across seven backgrounds.
- under_age_hinge (ann_21): the unit falls linearly with age, and is never on at 62 or older. The risk jumps between 60 and 61 for 3 or more visits only.
- few_visits_hinges (ann_22): every number reproduces. The jump at 2 to 3 visits happens at age 70 and not at 45.
- sex_smoking_age_detectors (ann_24): unit 23 is on for 347 of 350 young female smokers. Every probe and subgroup number reproduces.
- score_insurance_hinges (ann_25): the veto numbers reproduce. The typical-case crossing is at a score of 81 to 82, within rounding of 80.
- mixed_hinges (ann_26): all contributions reproduce. The effect is mainly for insured people (0.64 vs -0.31), as stated.
- risk_rules_root (ann_27): the output weights, default-on units, AND gate (unit 29), OR by saturation, veto units, dead units, small-input effects and node-group ablations all reproduce, except V1.

## Flags
None. The data are synthetic.

Note: `verify recheck` gave no target_hash for this record-file target.
