# Verdict: lad-6e63b (genome 8fc79414)

**Overall: weakened.** Every stated number reproduces: recheck found 0 mismatches, and I re-ran by hand all the probe, subgroup and ablation numbers in the four hypotheses. But two leaf readings miss part of what their units do, and the record's summary stretches two points.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.unit11_smoker_bmi_lowincome.hypothesis (ann_67) | L3 | The gate switches on for two groups: high-bmi smokers and young people on low income | It also opens for older smokers with ordinary bmi and middle or higher income. Typical smoker at bmi 25: +0.84 at age 75, +1.28 at age 80 (income 44.6). It is open for 325 smokers aged 60+ with bmi under 28 (mean +0.98). About 650 of the 6580 open rows fall outside the two named groups | probe smoker=1 bmi=25, age x income; subgroup node_11>0 and smoker==1 and bmi<28 and age>=60 |
| record.resisted | L3 | The gate does two unrelated jobs | It does three (see above) | same |
| operations.unit16_highincome_brake.hypothesis (ann_68) | L3 | For smokers the brake pulls the score down unless bmi is high | It works in a middle bmi band. For the typical smoker it is 0 at bmi 15 to 20, -0.17 at 22, -0.49 at 26 and 0 again by 32. It acts on 41% of smokers with bmi under 21, against 87% at bmi 21 to 30 | probe smoker=1, bmi 15..24; two subgroups |
| record.summary | L2 | The crossover part, shown with "99% at age 25, income 15, against below 1% at income 100" | These are whole-model risks. Of the 15.7-logit drop, the crossover gives 9.0, the gate 4.35 and the brake 2.33. For young people on low income the gate's mean is +2.47 against the crossover's +0.76, so the gate drives that high risk | probe age x income; subgroup age<40 and income<30 |
| record.summary | L3 | For a non-smoker, bmi barely matters | The typical non-smoker's risk falls from 21.5% at bmi 18 to 12.3% at bmi 35. This is small next to smokers, but a real effect about the size of the minor inputs | probe smoker x bmi |

**Upheld:**
- **Record:** coverage 1.0/1.0, function preservation 0, closed-form errors 0.065/0.014/0.013, and the operation counts all reproduce. The root has 0 direct edges.
- **ann_66 (age_income_units):** upheld through L4. The probe values at ages 25/55/75 are exact, and the income slope changes sign between ages 55 and 60. The four subgroup means and counts, the dead node 14 (0 rows), the smoker by bmi values and AUC 0.883 to 0.725 all reproduce.
- **ann_69 (risk_score_root):** upheld at L3. The whole-model grid, the four cell risks and outcome rates, and the smoker by bmi risks all reproduce. Holding the six minor inputs at their means gives AUC 0.8831 to 0.8812 and a mean change of 2.2%. The minor-input sweeps and the part ablations (0.725, 0.856, 0.874) also reproduce.
- **ann_67 and ann_68:** all their numbers reproduce (6580, 70.0% vs 29.9%, 4233/2347/2256, 2022 of 2032 at +3.21; 7452, 5435 of 7242 at -1.03, 2017 at -0.59, 1759). Their AUC drops reproduce too.

**Leaf meaning:** all three are partial. The crossover also carries a smoker by bmi effect and has a dead unit. The gate has an unnamed third regime. The smoker brake works in a middle bmi band and has no domain reading.

**Flag (L4):** in the typical case the model gives the noise input an effect as large as score (13.6% to 18.4% over -2.5 to 2.5). The explanation reports this correctly.
