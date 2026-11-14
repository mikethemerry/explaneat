# Verdict: shape-C (genome a1cbdd4a)

**Overall: weakened.** Nothing is refuted. Coverage (1.0), function preservation (3e-16), closed forms (max error 0.057) and every probed number, threshold, direction and shape reproduce, including the sodium V and the pulse-pressure saddle. Four minor weakenings remain.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.high_sbp_and_hr_penalty.hypothesis | L2 | the chain switches on when sbp and heart rate are "jointly high" | It is a threshold on about sbp + 1.5 x heart rate. It acts on all 334 both-high patients, but also on 249 of 394 low-pressure/fast-pulse and 150 of 404 high-pressure/slow-pulse patients. It opens at heart rate 92 when sbp is 100 mmHg. | `verify subgroup` by quadrant; `probe --nodes 3040,3544` |
| operations.young_high_sodium_tachy.hypothesis | L3 | at age 40 the gate is "on from sodium 138" | It is already on at 137 (0.10) and opens at about 136.5 mmol/L | `probe --set age=40 --sweep sodium=136,137,138 --nodes 1269` |
| operations.linear_terms.hypothesis | L3 | creatinine 40 to 250 µmol/L takes the risk from 6% to 87% | The numbers reproduce, but 250 is outside the data (max 207, where the risk is 71%) | `probe --sweep creatinine=40,85,150,207,250` |
| operations.hypertension_low_hr_gate.hypothesis | L1 | switching it off gives AUC 0.876 | 0.87655, which rounds to 0.877. The stored query held 0.8765, so it was rounded twice. | `ablate --part ann_24` |

**Upheld parts:**
- risk_logit (ann_31): five added terms; male has no edge; AUC 0.896; direct edges off give 0.652; affine switching terms give 0.704.
- bp_hr_saddle (ann_30): the risk grid, the quadrant risks (61/51/26/24% against observed 59/52/29/23%), the 14 patients with both gates open, and the ablation and linearisation AUCs all reproduce.
- sodium_V (ann_22): minimum at about 138.7 mmol/L, with the high side about 2.6 times steeper. Exit values, 713 rows with the hinge on, and AUCs 0.855 and 0.808 reproduce.
- tachycardia_low_sbp_gate (ann_23): opens at heart rate about 85 (sbp 132) and about 74.5 (sbp 100). Opens earlier at low sodium. 454 rows on, AUC 0.834 when off.
- sbp_hinge (ann_26): off at 123 mmHg, on from 124, and no effect on AUC.
- diabetes_step (ann_28): on for diabetics only; typical risk goes from 15% to 23%.
- Weakened parts whose other statements were upheld: ann_24 (thresholds, sodium effect, 369 and 60 rows), ann_25 (thresholds, -3.06 / -4.40, AUC 0.783 is the largest loss of any part), ann_27 (age brackets, heart-rate partner, the hinged age response), ann_29 (weights, monotone sweeps).

**Flag (L4):** The model predicts the lowest risks when pressure and pulse are both high (below 1% at sbp 170 with heart rate 100) or both low (0.2% at sbp 90 with heart rate 45). Clinically that is odd. The training data agree with the model (both-high quadrant observed 29%, against 59% and 52% in the discordant quadrants), so the flag is on the data-generating process, not on the model.

Leaf meaning: sodium_V, the two mismatch gates, sbp_hinge, young_high_sodium_tachy, diabetes_step and linear_terms are meaningful. high_sbp_and_hr_penalty is partial. About 118 statements were checked.
