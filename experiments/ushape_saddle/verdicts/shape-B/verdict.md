# Verdict: shape-B (genome 25f5830e)

**Overall: refuted.** Coverage, function preservation, every weight, threshold and count, and the
pressure-rate saddle reproduce. Two shape statements do not: the rate-bmi amplifier's heart-rate
direction and the clinical summary's "bmi only above a turn in the overweight range".

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.rate_bmi_amplifier.hypothesis (clinical) | L3 refuted | a faster heart rate raises risk through this part, more so when pressure is high for the rate | where pressure is high for the rate, heart rate lowers the part (net -0.33/SD into node 2391); at sbp 150 the exit falls 2.59 -> 1.82 from 45 to 75 bpm | probe --set sbp=150 --sweep heart_rate=45:75:7 --nodes 440,2391,1686 |
| record.clinical_terms | L3 refuted | bmi raises risk only above a turn in the overweight range | 41% of rows: bmi raises risk from bmi 15 up (sbp 150, hr 60: 56.7% at 15, 63.7% at 19); other turns span bmi 16 to 38, median 24 | weights per row; probe bmi sweeps |
| record.summary; risk_score hypothesis; amplifier hypothesis | L3 weakened | bmi flat then rising above a turn (about 26) | typical case only; node 1686 is almost linear (R2 0.996, linearising costs 0.0005 AUC) | linearise --level nodes; probe |
| risk_score hypothesis; age_ceiling; load evidence[2]; amplifier | L3 weakened | end points at age 20, hr 130, sbp 200, bmi 46, sodium 120, sbp 180 with hr 40 | outside the training range or with no rows nearby (sbp>=175 & hr<=50: 0 rows); shapes hold inside the support | probe ranges; subgroup |
| amplifier hypothesis | L3 weakened | the side signal matters little | AUC -0.005, but one risk moves by 0.37 and 63 predictions flip | ablate --edge identity_560->1686 |
| direct_terms hypothesis; record.leaves | L4 weakened | graded meaningful | restates the linear closed form (the record says so itself) | concept |

Upheld: age_ceiling (hinge on age, turn at 80.4, 0 rows under 80, 36/36 aged 85+, creatinine
shift); pressure_rate_load (weights, stage counts, all eight turning points, convex fall,
ablations); pressure_over_rate (weights, turning points, hr-55 sweep, largest AUC loss, sbp minus
heart rate concept); direct_terms numbers; root saddle sweeps, four training corners, part
ranking, sodium and noise disconnected, linearisation 0.7450.

Flag (L4): the load chain lowers risk when pressure and heart rate are both high (a high
rate-pressure product); clinically odd, but the training data agree (38 rows, 5.3% outcome).
