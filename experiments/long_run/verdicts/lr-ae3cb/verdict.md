# Verdict: lr-ae3cb (genome 466cca51)

**Overall: weakened.** Every stated number reproduces (coverage 1.0, function preservation 1.8e-15, all 84 templated texts and 57 parsed statements, every self-probe, subgroup and AUC checked by hand), and nothing is refuted. Seven statements are weakened: shape and scope words that hold in the typical case but not across the data. 0 refuted, 7 weakened, 17 upheld.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.root.hypothesis (1); record.summary; clinical_terms | L3 | a fast heart rate with low **or normal** pressure is high risk | only below about 125 mmHg. At sbp 130: 25% at 80 bpm, 12% at 90, 17% at 100. In training, heart_rate > 90 with sbp 120 to 145 (n 115): mean risk 37.8%, vs 41.4% at normal rates | probe sbp x heart_rate grid; subgroup |
| operations.root.hypothesis (3); record.summary; clinical_terms | L3 | risk rises monotonically (straight line) with age | reverses when sodium is raised: at sodium 142, 81% at age 75, 67% at 85; at 143, 89.5% at 75, 73.8% at 90. Age hinge (about 74 y) shrinks the high-sodium arm. 64 patients have sodium >= 141 and age >= 75 | probe --set sodium=142 --sweep age |
| operations.root.hypothesis (3); overall_function | L3 | diabetes acts inside the gate, mostly when it is open | also through the brake: with the gate shut (sbp 160, 70 bpm) diabetes takes risk from 56% to 63%. Over 428 diabetics, the brake edge changes 266 (mean abs 0.015) and the gate edge 142 (0.020) | probe; ablate each diabetes edge |
| operations.rate_pressure_brake.hypothesis | L3 | the brake is "what makes" a fast rate protective at sbp 160 (73% to below 1%) | the fall from 60 to 80 bpm (73% to 8%) is the high-pressure arm closing (+5.02 to 0); the brake only moves -2.63 to -3.12 there. It drives the fall above 80 bpm | probe --set sbp=160 with parts |
| operations.high_sodium_arm.hypothesis | L4 | off through the normal range, opens above it; "hypernatraemia" | opens at about 139.2 mmol/L, mid-normal. Risk 20% at 139 and 63% at 142. Acts on 588 patients (39%) | probe sodium sweep --nodes 144 |
| operations.sodium_u_with_pressure.hypothesis | L3 | high pressure with a slow rate penalises "mild hyponatraemia" | at sbp 170 / 60 bpm the low-sodium hinge is open at normal sodium too (1.62 at 136, 0.54 at 138), so the flat bottom shrinks to about 139 to 141 | probe --set heart_rate=60 --set sbp=170 --sweep sodium |
| operations.smoker_unit.hypothesis | L4 | copy of smoking, a small negative push on the gate | not meaningful: restates the closed form. Switching it off leaves the AUC at 0.9007; smoking acts through its direct term (20% to 39%) | ablate --part smoker_unit |

Upheld:
- **root**: the saddle self-probe and four training corners (87.4%, 74.4%, 12.8%, 1.2%), the non-monotone single-input sweeps, and BMI/noise being unused.
- **shock_gate**: thresholds, the sodium-128 relay case, diabetes 42% to 67%, 404 open, AUC 0.805 (gate unit) and 0.650 (whole).
- **sodium_u_with_pressure**: the flat bottom from 134 to 139 in the typical case, a steeper low side, the training bins, AUC 0.730 and 0.888.
- **rate_pressure_brake**: self-probe values, readout -1.85, 921 patients at -3.49, AUC 0.787, concept 0.92.
- **high_sbp_low_hr_arm**: opening pressures at 55, 75 and 95 bpm, 326 patients at +3.27, AUC 0.829, concepts 0.96. The ratio reading is loose; the probe values are right.
- **high_hr_low_sbp_detector**: opens about 74 bpm at sbp 120 and about 100 at sbp 160, the age hinge, AUC 0.806, concept 0.97.
- **high_sodium_arm**: all probe values, the age effect, 207 and 588 patients, AUC 0.827.
- **direct_terms**: weights, creatinine 10% to 64%, age 3% to 71%, AUC 0.764.
- **male_unit**: 22% to 42% with the gate open, no effect with it shut, AUC 0.899.

Flags: none. Note: this recheck version writes no target_hash, so the field is null.
