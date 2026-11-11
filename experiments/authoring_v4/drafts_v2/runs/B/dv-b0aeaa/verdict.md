# Verdict dv-b0aeaa (genome b19b9536, Heart Disease)

**Overall: weakened.** Every number in the plain-language draft reproduces (78 of 78 claims, 244 prose numbers, no citation gaps). Every threshold, subgroup count and interaction I probed holds in the model. Two wording problems remain: the sex-by-exercise-angina sentence says "even more so" where that is true only on the odds scale, and it uses whole-model numbers for one part. The thallium code 4 sentence also contradicts itself.

Started 2026-11-11T03:24:37Z, finished 03:29:22Z (5 minutes). Scope: the plain-language layer and the claims it cites.

## Weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.blocks[2], drafted.readdown.ann_32.evidence[1] (C45) | L2 | Being a woman lowers risk "even more so" without exercise angina: 20% to below 1% without it, 27% to about 1% with it | The numbers are right (19.7% to 0.2%, 27.4% to 1.0%). The drop is larger without angina only on the odds scale (-4.89 against -3.65 log-odds). In risk points it is larger *with* angina (26 points against 20). The numbers are also whole-model: of the -4.89 logit, -3.65 comes from the direct sex terms and only -1.24 from this gate. The part's own effect (C43, 20% to 7%) is left out of the summary block. | `probe --hold typical --sweep sex=1,0 --sweep exang=0,1` |
| drafted.blocks[4], drafted.worth_checking[2] (C53, C02) | L4 | "A thallium result the data does not decode (... coded 4 (meaning not recorded))" raises risk from 20% to 93%, n = 2 | 92.9% reproduces (unit +3.55 logit). The wording contradicts itself: it says the code is not decoded, then gives it a meaning. The large push comes from z-scoring a one-hot column that is set on only 2 rows, and the text does not say so. | `probe --hold typical --sweep thal=3,6,7,4 --sweep fbs=0,1` |

## Upheld parts

- ann_37 (Heart disease risk): the typical patient is at 19.7%; six parts are added; resting ECG has no effect (L3).
- ann_36 (Inputs that add up): the directions and sizes reproduce, and the "biggest first" order holds. Note: the test-split checks of the flat-slope and cholesterol claims disagree, and the draft does not mention this (L3).
- ann_34 (ST depression hinge): it switches on between 1.4 and 1.5 mm, then rises in equal steps. Risk goes from 19.7% to 61.6% at 3.5 mm. It acts on 70 of 70 patients at 1.5 mm or more (50 had heart disease) and on no one else (L3).
- ann_32 (Protection without exercise angina): the hypothesis holds. The part acts on 160 of 160 patients (47 had heart disease). The gate gives -1.67 logit for a woman and -0.43 for a man. Exercise angina takes risk from 20% to 27%. Weakened only for the sentence above (L2).
- ann_33 (Fixed thallium defect): "only if" fasting sugar is normal holds. With normal sugar risk goes from 19.7% to 1.0%; with high sugar the gate gives only -0.017 logit. It acts on 15 patients (9 had heart disease). The typical acted-on patient is at 89.9% (L3).
- ann_31 (Vessels and thallium code 4): the unit is always on and rises steadily with vessels (20% to 45%). Weakened only for the wording above (L4).
- ann_35 (Atypical angina flag): the largest change is 1.03%, AUC goes from 0.9437 to 0.9434, and no prediction flips (L3).

## Flags (L4, both already raised by the draft)

- A fixed thallium defect lowers the predicted risk, although the observed rate in those patients is 60% (9 of 15). This looks fitted to 15 patients.
- High fasting blood sugar lowers the predicted risk (typical patient 3.4% against 19.7%), against clinical expectation.
