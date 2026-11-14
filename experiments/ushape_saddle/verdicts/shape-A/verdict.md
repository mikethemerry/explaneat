# Verdict: shape-A (genome 2aac39c4)

**Refuted.** Every number reproduces, and so do the closed forms, coverage (1.0 / 1.0) and function preservation (3e-16). Three statements are wrong about the shape: the record says normal sodium leaves all three hinges off, but the model's floor is only 135-137 mmol/L. It says a higher BMI raises risk steadily, but for low-sodium patients higher BMI lowers it. And one ablation sentence gets the direction of a small effect backwards.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.overall_function | L3 refuted | "normal sodium leaves all three off"; gentle hinge "from the middle", steep one "from the top" of the normal range | Only 135-137 leaves all off. Hinge 219 is on from 138 (812 of 812 rows with sodium >= 139). The steep hinge is on from 142. Typical risk is 10% at 137, 21% at 140 and 80% at 144 | probe --sweep sodium x male --nodes 219 |
| record.clinical_terms | L3 refuted | higher BMI "raises the risk steadily" | At sodium 130 risk falls from 87% (BMI 16) to 58% (BMI 42). On 163 of 1500 rows (sodium 126-135), +5 BMI lowers risk (mean -4.7 points) | probe --sweep bmi x sodium=130,133,139 |
| operations.low_creatinine_noise_chain.hypothesis | L3 refuted | switching off changes mean risk "by -0.007" | Switching off **raises** mean risk by 0.007. The -0.007 is the with-minus-without difference | ablate --part ann_20 |
| record.clinical_terms | L4 weakened | lowest for normal sodium; sharp rise in hypernatraemia; hypernatraemia has the heaviest edge | Most of the high-side rise is inside the normal range. The heaviest edge (3.22) turns on at 141-142 | probe sodium sweep |
| operations.high_sodium_steep_hinge.hypothesis | L4 weakened | rises "above the upper end of the normal range" | Turns on at 141-142, inside 135-145. The reading "sodium above about 142" holds | probe sodium sweep |
| operations.sodium_U.hypothesis | L4 weakened | first high hinge "just above the middle of the normal range" | Turns on at 137-138, below the middle (140) | probe sodium x male |
| operations.sodium_U.hypothesis | L3 weakened | "high side is steeper" | Gentler for the first 4-5 mmol/L (score slope 0.30 vs 0.84 per mmol/L). Steeper (1.01) only above 142 | probe sodium sweep |
| record.summary | L3 weakened | chains feed "small negative terms" | Small on average (0.007), but up to 0.49 risk on one row. 14 predictions flip | ablate --part ann_19 --part ann_20 |
| record.resisted | L3 weakened | "(mean risk change -0.0072)" | Switching off raises mean risk by 0.0072. The sign is unexplained | ablate both chains |
| low_sodium_hinge / arm / sodium_U / risk_score | L3 weakened | values at sodium 125, creatinine 300, sbp 200 | Outside the training range (126-152, 40-221, 85-186): extrapolation | probe support notes |
| operations.nondiabetic_noise_chain.hypothesis | L4 weakened | "described, not explained" | Restates the closed form (disclosed) | — |

Upheld numbers (every count, rate, AUC, weight, node value and concept score in each hypothesis):
- ann_23 risk_score: direct weights, age, BMI, creatinine, smoker/diabetes and noise sweeps, sbp and heart rate inert, AUC 0.857 to 0.767 and 0.697, accuracy 0.775.
- ann_22 sodium_U: output weights, hinge 219 values by sex, the 945, 812, 133 and 334 counts, the arms never on together, sodium bands, ablations.
- ann_21 low_sodium_arm: the 187, 0 and 12 counts, switch-on at 134/135, smoker and diabetes effects, the noise pull-down, ablations.
- ann_17 low_sodium_hinge: typical exits, BMI shifts at 20, 30 and 40, the 495, 406 and 187 counts, AUC 0.821, concept score 1.00.
- ann_18 high_sodium_steep_hinge: exits by sodium and noise, the 334, 0 and 63 counts, AUC 0.839.
- ann_19 and ann_20 (the two noise chains): all counts and the constructed cases.
- Structure, coverage and closed forms: all leaves' closed forms are identical to the recomputed ones.

Flag (domain, about the model): the model's lowest-risk sodium is 135-137 mmol/L, with risk doubled at 140. Low-sodium patients with a higher BMI get a lower predicted risk.
