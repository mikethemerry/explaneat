# Verdict: cox-whas500-s0 (genome f96ebc8d-5ab6-4b78-866f-5c82f1640771)

**Overall: weakened.** Every stated number reproduces (the nine hazard ratios, the sweeps, the subgroups, the ablations, the concept scores, the survival metrics, coverage and function preservation); one sentence in the age-and-body-mass hypothesis attaches the body-mass spread to the percentiles of age rather than of body-mass index (a wording slip, the number itself is right); two protective terms are flagged for the domain reviewer, and the data agree with both.

Verified 2026-11-12 14:05 to 14:17 NZDT (12 minutes), blind: the record JSON, the stored explanation and evidence log through the verify tools, the training split only. Read only. 39 findings: 36 upheld, 1 weakened, 0 refuted, 2 flags. `verify recheck` gives no `target_hash` for a record-JSON target; the text verified is the record file plus the stored explanation at operation seq 23.

## Weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `operations.age_and_body_mass.hypothesis` (ann_18) | L2 | "between the patients at the 5th and the 95th percentile **of age** the age term changes the hazard 3.19-fold, and the body-mass term 2.29-fold" | 3.19 is right (age 45 to 90 years, exp(0.02575 x 45) = 3.186). 2.29 is the body-mass term between the 5th and 95th percentile **of body-mass index** (19.1 to 38.1 kg/m2, exp(0.04357 x 19.0) = 2.288), not of age. The population of the second number is misstated; the number is right. | weights / training sd and percentiles (scratch script) |

## Upheld, by part

- **ann_22 cox_log_risk** (L1-L4): formula exact (sum of four identity relays minus 0.832; max err 2.5e-5); identity output, nine input edges, no hidden node; the five unconnected inputs leave the constructed patient at 29.5% / log-risk -0.911 in every cell (woman + cardiovascular history + recurrent infarction, man with neither, atrial fibrillation, complete heart block); all four parts at their means give log-risk -0.832 and AUC 0.50, one at a time 0.781 / 0.807 / 0.822 / 0.832; Harrell 0.770, Uno 0.753, td-AUC 3 y 0.854, IBS 0.145, E/O 1.01, p95/p5 hazard ratio 27.0, mean risk 37% vs 43% died (320 patients, 138 deaths); two-way sweeps show plain addition on the log-risk, no interaction; PH caveat present everywhere.
- **ann_18 age_and_body_mass** (apart from the wording slip above): 0.372 age - 0.242 bmi exact; 1.026 per year, 0.957 per kg/m2 (1.45 / 0.79 per SD) re-derived from sweeps; age 45 to 95: 17.1% to 49.3%; bmi 20 to 40: 38.4% to 18.3%; age > 72: n 156, 67% died, risk 53%, contribution +0.41 vs 164 / 20% / 22% / -0.39; off: AUC 0.833 to 0.781, the largest loss; age r2 0.82; corr(age, bmi) -0.40.
- **ann_19 admission_haemodynamics**: 0.286 hr - 0.199 diasbp - 0.0289 sysbp exact; 1.0121 per beat, 0.9908 per mm Hg diastolic, 0.9991 per mm Hg systolic; spreads 2.44 / 1.82 / 1.10; hr 50 to 150: 21.8% to 56.0%; diastolic 50 to 120: 37.9% to 22.1%; systolic 90 to 200: 32.5% to 30.0%; hr > 85: 154 / 59% / 49% / +0.25 vs 166 / 28% / 26% / -0.23; diastolic <= 76: 161, 48% vs 27%; off: 0.833 to 0.822; hr r2 0.62, shock index 0.73, hr - diasbp 0.99.
- **ann_20 in_hospital_course**: 0.333 chf + 0.0416 los + 0.121 sho; 2.08 / 1.94 / 1.010 per day (1.40 / 1.13 / 1.04 per SD, stay 1.12-fold p5 to p95); constructed patient 29.5% (-0.911) with neither, 51.7% (-0.177) with heart complications, 75.7% (0.487) with shock added, stay 1 to 21 days 30.3% to 35.6%; chf: 93 / 73% / 64% / +0.56 vs 227 / 31% / 26% / -0.23; shock n 11, 64% died, mean risk 64%; both: 7 of 7 died; off: 0.833 to 0.807; chf score 0.99; the rare-regime n is carried.
- **ann_21 index_mi_type**: -0.219 mitype exact; 0.617 (0.80 per SD); constructed patient 19.4% (-1.394) Q-wave vs 29.5% (-0.911); Q-wave: 92 / 26% / 22% / -0.34 vs 228 / 50% / 44% / +0.14; corr(age, Q-wave) -0.28; off: 0.833 to 0.832, the smallest loss; score 1.00; honestly "described, not explained".
- **Record**: coverage 1.0 / 1.0 with the five unconnected inputs as Null, hierarchy complete, function preserved to 4.4e-16, closed-form error 0.00141, operation counts (14 renames, 4 identity nodes, 5 annotations, 0 splits, 0 hidden nodes), the summary's hazard ratios, metrics and meaning scores, the resisted paragraph's numbers (shock index 0.73, hr - diasbp 0.99, systolic 1.10-fold, 11 shock patients), and the provenance: all 53 referenced queries on the train split, unsampled, computed after the last operation, with populations matching their sentences.

## Flags (L4, domain reviewer; the data do not contradict them)

- **ann_18, body-mass index protective** (0.957 per kg/m2): conflicts with obesity as a risk factor in general, but the "obesity paradox" after a heart attack is reported, and here patients with BMI >= 30 died at 29% vs 48% of the rest, and those under 22 at 71%.
- **ann_21, Q-wave infarction protective** (0.617): a transmural infarction is often expected to carry the worse prognosis, but non-Q-wave patients in registries are older with more comorbidity; here 26% of Q-wave patients died vs 50% of non-Q-wave, and Q-wave patients are younger (corr -0.28).

No part is refuted. Nothing was written to the database or the target.
