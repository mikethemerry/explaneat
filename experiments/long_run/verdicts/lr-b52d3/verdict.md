# Verdict: lr-b52d3 (genome 6e4d57e3)

**Overall: weakened.** Every stated number, threshold, direction, subgroup count, ablation AUC and concept score reproduces, coverage is complete and the decomposition matches the evolved network to 1e-15. What is weakened is three descriptions of shape around serum sodium.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.sodium_U.hypothesis | L3 | the low arm of the sodium U is the steeper one | Once open, the low arm is steeper per mmol/L (1.8 against 0.79 logit). But the high arm opens right at the bottom (140) and the low arm only below 133. So 6 mmol/L either side of 139, risk is 20.6% (133) against 79.6% (145). Within the 1st-99th percentile range the U is about symmetric: 96.2% at 129, 96.3% at 148 | probe, sodium 126:154 |
| record.summary (also overall_function, risk_model) | L3 | "a steeper low arm" | same as above | same |
| operations.sodium_U.hypothesis | L3 | the direct sodium edge tilts the U "slightly" | The direct term moves the logit by about 2.9 across 129 to 148. Between 133 and 139 it is the only thing that moves the risk (20.6% to 9.3%) | probe, direct_terms contribution |
| operations.low_sodium_smoker_gate.hypothesis | L3 | smoking raises the opening sodium **and** makes the gate open wider | One parallel shift: the pre-activation is +0.775 higher at every sodium, with the same slope. "Wider" is the same shift, not a second effect | probe, sodium x smoker, node 228 |

Upheld:
- **low_sodium_smoker_gate:** thresholds 133/132 and 135/134, second exit at 131/132, contributions 5.77 and 9.02, 124 open (67.7%), creatinine edge negligible.
- **high_sodium_hinges:** on at 140, 1.57 and 2.74 at 150, risk 9%/80%/99%, noise 62% against 90%, 635 on, AUC 0.7631 when off.
- **bmi_hinge:** turns on near BMI 27, weight 0.048, no AUC change.
- **low_hr_hinge:** turns on below about 77 beats/min, weights -0.75 and -0.78, 772 on.
- **direct_terms:** all eight weights match. Age 2/8/25%, creatinine 6/26%, diabetes 9/15%.
- **sbp_chain:** turn-on points at 117 and 131 mmHg, values at 180 mmHg, readout weights -0.83/-0.78/-0.97.
- **bp_hr_gate:** opening pressure rises with heart rate. Values 1.23, 11.59, 1.70, 6.58, 0 and 4.04 match. 729 open. Concept scores 0.91 (shock index) and 0.97 (sbp minus heart rate).
- **bp_hr_interaction:** the crossing holds: a faster heart is worse below about 130-140 mmHg and better above. All the probe risks and the 40/63/32 corner subgroups match. U-shaped in pressure at 76 beats/min, lowest at 130.
- **risk_model:** sex is not connected (9.3% both). Removing both compositions gives AUC 0.5781. The interaction composition is the largest effect.

Flags (L4, for the domain reviewer):
- A fast heart with high pressure is predicted almost no risk (0.8% mean, 32 patients). This matches the data, not clinical expectation.
- The noise column moves risk by up to 28 points at sodium 145 (stated in the explanation).

Tool notes:
- `verify linearise --core` fails on this model ("not a DAG").
- The recheck report has no target_hash for a record target.
