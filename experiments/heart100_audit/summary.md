## Counts

| level | total | generalises (TEST CI > 0) | no measurable test effect | worse on test | recurring | seed-specific | near-linear | small support (n<30) | train gain ≥ 0.02 | train gain < 0.001 | matches a 40-gen signal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| units | 62 | 1 | 61 | 0 | 19 | 28 | 15 | 16 | 2 | 24 | 19 |
| parts | 45 | 1 | 44 | 0 | 9 | 27 | 9 | 6 | 8 | 11 | 11 |

Binary one-step units excluded: 11. Units exactly affine in the inputs on the data (kept, counted near-linear): 5.

Generalising units / parts (TEST):

- s2 unit 3138 (thal_age_gate, thal × age): train +0.0022, TEST +0.034 [+0.006, +0.075]; generalises; near-linear
- s2 part 3138 (thal_age_gate, thal × age): train +0.0022, TEST +0.034 [+0.006, +0.075]; generalises; near-linear

## Every non-linear unit (compact; train unless TEST)

| seed | part (title) | unit | family | where it switches (train n in regime) | being on | mean / max \|Δrisk\| | half the effect on n | train gain | TEST gain [95% CI] | other seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | angina_sex_gate (Exercise angina gate) | 665 | exang × resting ECG × sex | on for 38; mostly exang ≠ no chest pain on exercise and sex ≠ male (19/19) | raises | 0.099 / 0.78 | 12 | +0.0163 | +0.004 [-0.010, +0.023] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s0 | exertional_angina_hinge (Exercise angina and ST depression add-on) | 5884 | exang | on for 64; mostly exang = chest pain brought on by exercise (64/82) | raises | 0.000 / 0.00 | 27 | +0.0001 | +0.000 [+0.000, +0.000] | 0 | no measurable test effect; seed-specific |
| s0 | flat_slope_hinge (Flat ST segment step) | 1650 | ST slope × chest pain | on for 71; mostly slope = a flat ST segment and cp ≠ no chest pain (asymptomatic) (46/46) | raises | 0.020 / 0.24 | 22 | +0.0087 | +0.017 [-0.001, +0.044] | 0 | no measurable test effect; seed-specific |
| s0 | flat_slope_hinge (Flat ST segment step) | 3598 | ST slope | on for 112; mostly slope = a flat ST segment (110/110) | raises | 0.018 / 0.61 | 2 | +0.0031 | +0.009 [+0.000, +0.030] | 0 | no measurable test effect; seed-specific |
| s0 | lvh_ramp (Left ventricular hypertrophy signal) | 3280 | resting ECG | on for 119; mostly restecg = signs of left ventricular hypertrophy on the resting ECG (119/119) | lowers | 0.000 / 0.00 | 0 | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear |
| s0 | lvh_ramp (Left ventricular hypertrophy signal) | 4331 | resting ECG | on for 119; mostly restecg = signs of left ventricular hypertrophy on the resting ECG (119/119) | lowers | 0.000 / 0.00 | 1 | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear |
| s0 | thal_ecg_unit (Thallium scan and resting ECG check) | 2011 | thal | on for 2 | raises | 0.006 / 0.01 | 22 | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s0 | thal_ecg_unit (Thallium scan and resting ECG check) | 2349 | thal | off for 2 | lowers | 0.020 / 0.04 | 2 | -0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s1 | normal_ecg_ischaemia_gate (Normal-ECG gate) | 1801 | resting ECG | on for 86; mostly restecg = a normal resting ECG (85/121) | raises | 0.053 / 0.32 | 31 | +0.0092 | -0.001 [-0.027, +0.025] | 0 | no measurable test effect; seed-specific |
| s1 | normal_ecg_ischaemia_gate (Normal-ECG gate) | 4375 | ST depression | on for 66; mostly oldpeak ≥ 1.55 mm (66/66) | raises | 0.014 / 0.17 | 11 | +0.0024 | +0.002 [-0.010, +0.016] | 2 | no measurable test effect; recurring |
| s1 | vessel_count_age_brake (Vessel count by age) | 3196 | vessels × age | off for 3; mostly ca ≥ 2.5 vessels and age < 62.5 years (3/5) | raises | 0.007 / 0.05 | 23 | +0.0003 | +0.005 [-0.004, +0.020] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s1 | vessel_count_age_brake (Vessel count by age) | 5446 | vessels × age | on for 17; mostly ca ≥ 2.5 vessels (12/12) | lowers | 0.055 / 0.47 | 31 | +0.0086 | +0.013 [-0.030, +0.061] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s2 | age_cholesterol_gate (Older-age brake) | 3464 | cholesterol | off for 29; mostly chol < 196.5 mg/dL (29/29) | lowers | 0.003 / 0.05 | 9 | +0.0001 | +0.003 [+0.000, +0.014] | – | no measurable test effect; near-linear; small support (n<30) |
| s2 | age_cholesterol_gate (Older-age brake) | 3880 | age | on for 84; mostly age ≥ 63.5 years (44/46) | lowers | 0.036 / 0.22 | 26 | +0.0040 | +0.009 [-0.009, +0.035] | 0 | no measurable test effect; seed-specific |
| s2 | heart_rate_age_gate (Youth and heart rate gate) | 1718 | thal × age | on for 103; mostly thal ≠ a reversible defect on the thallium scan and age < 56.5 years (85/85) | lowers | 0.035 / 0.38 | 31 | +0.0093 | +0.014 [-0.008, +0.041] | 1 | no measurable test effect; recurring |
| s2 | heart_rate_age_gate (Youth and heart rate gate) | 5718 | max HR | on for 121; mostly thalach ≥ 152.5 beats per minute (121/121) | lowers | 0.012 / 0.12 | 14 | +0.0003 | +0.003 [-0.008, +0.017] | 2 | no measurable test effect; recurring |
| s2 | high_heart_rate_units (Very high heart rates) | 7245 | max HR | on for 10; mostly thalach ≥ 184.5 beats per minute (10/10) | lowers | 0.000 / 0.00 | 24 | +0.0000 | +0.000 [+0.000, +0.000] | 2 | no measurable test effect; recurring; small support (n<30) |
| s2 | high_heart_rate_units (Very high heart rates) | 7595 | max HR | on for 49; mostly thalach ≥ 169.5 beats per minute (49/49) | lowers | 0.001 / 0.01 | 30 | +0.0001 | +0.000 [+0.000, +0.000] | 4 | no measurable test effect; recurring |
| s2 | slope_gate (Flat ST segment and cholesterol) | 1476 | ST slope | on for 108; mostly slope = a flat ST segment (108/110) | raises | 0.008 / 0.09 | 28 | +0.0008 | -0.003 [-0.013, +0.003] | 0 | no measurable test effect; seed-specific |
| s2 | st_depression_gate (ST depression and the thallium scan) | 1946 | thal × ST depression | on for 116; mostly thal = a reversible defect on the thallium scan and oldpeak < 1.95 mm (55/62) | raises | 0.056 / 0.52 | 24 | +0.0121 | -0.001 [-0.012, +0.010] | 1 | no measurable test effect; recurring |
| s2 | st_depression_gate (ST depression and the thallium scan) | 3284 | thal × ST depression | on for 83; mostly thal ≠ a normal thallium scan and oldpeak ≥ 1.45 mm (43/45) | raises | 0.027 / 0.16 | 33 | +0.0063 | +0.001 [-0.011, +0.015] | 1 | no measurable test effect; recurring |
| s2 | thal_age_gate (Thallium defect detector) | 3138 | thal × age | on for 107; mostly thal ≠ a normal thallium scan and thal = a reversible defect on the thallium scan (90/90) | raises | 0.025 / 0.30 | 9 | +0.0022 | +0.034 [+0.006, +0.075] | – | generalises; near-linear |
| s3 | bp_slope1_damper (Blood-pressure brake) | 3392 | BP × ST slope | off for 78; mostly slope ≠ an upsloping ST segment and trestbps < 131 mm Hg (63/63) | lowers | 0.024 / 0.16 | 34 | +0.0030 | -0.001 [-0.019, +0.017] | 1 | no measurable test effect; recurring |
| s3 | exercise_reserve_age_chain (Exercise test and age chain) | 245 | ST depression × ST slope | off for 89; mostly slope ≠ an upsloping ST segment and oldpeak ≥ 0.85 mm (82/88) | raises | 0.018 / 0.22 | 32 | +0.0039 | +0.010 [-0.010, +0.033] | 0 | no measurable test effect; seed-specific |
| s3 | exercise_reserve_age_chain (Exercise test and age chain) | 4376 | ST slope × age | off for 62; mostly slope = a flat ST segment and age < 58.5 years (55/62) | raises | 0.025 / 0.17 | 29 | +0.0012 | +0.003 [-0.011, +0.018] | 0 | no measurable test effect; seed-specific |
| s3 | exercise_reserve_age_chain (Exercise test and age chain) | 5302 | max HR | on for 9; mostly thalach ≥ 185.5 beats per minute (9/9) | raises | 0.001 / 0.01 | 16 | +0.0000 | +0.000 [+0.000, +0.000] | 2 | no measurable test effect; recurring; small support (n<30) |
| s3 | old_age_relief (Older-age relief) | 1499 | age | on for 68; mostly age ≥ 60.5 years (68/68) | lowers | 0.047 / 0.33 | 35 | +0.0048 | +0.004 [-0.022, +0.032] | 2 | no measurable test effect; recurring |
| s3 | st_depression_hinge (ST-depression switch) | 4072 | ST depression | off for 100; mostly oldpeak < 0.35 mm (100/100) | raises | 0.010 / 0.05 | 40 | +0.0008 | +0.008 [-0.004, +0.024] | – | no measurable test effect; near-linear |
| s3 | vessels_sex_gate (Combined-findings gate) | 3683 | chest pain | on for 59; mostly cp = atypical angina (42/42) | raises | 0.037 / 0.24 | 32 | +0.0036 | -0.008 [-0.030, +0.010] | 0 | no measurable test effect; seed-specific |
| s4 | bp_hinge (Blood pressure threshold) | 8900 | BP | off for 94; mostly trestbps < 127 mm Hg (94/94) | raises | 0.003 / 0.02 | 35 | +0.0002 | +0.002 [+0.000, +0.009] | 2 | no measurable test effect; recurring |
| s4 | bp_thalach_chain (Blood pressure and heart rate signal for the gate) | 1754 | BP × max HR | off for 40; mostly trestbps < 113.5 mm Hg and thalach < 157 beats per minute (26/27) | raises | 0.000 / 0.00 | 17 | -0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s4 | bp_thalach_chain (Blood pressure and heart rate signal for the gate) | 6137 | BP × max HR | off for 39; mostly trestbps < 113.5 mm Hg and thalach < 157 beats per minute (25/27) | raises | 0.001 / 0.03 | 12 | +0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s4 | exang_cp2_chain (Chest pain signal for the gate) | 4652 | exang × chest pain | on for 111; mostly exang = chest pain brought on by exercise and cp ≠ non-anginal chest pain (63/71) | lowers | 0.010 / 0.10 | 16 | +0.0015 | -0.001 [-0.006, +0.000] | 0 | no measurable test effect; seed-specific |
| s4 | exang_cp2_chain (Chest pain signal for the gate) | 6931 | BP × exang | off for 110; mostly exang = no chest pain on exercise and trestbps ≥ 121 mm Hg (109/109) | lowers | 0.003 / 0.12 | 12 | -0.0001 | -0.001 [-0.006, +0.000] | 1 | no measurable test effect; recurring |
| s4 | oldpeak_hinge (ST depression threshold) | 8065 | ST depression | off for 107; mostly oldpeak < 0.45 mm (107/107) | raises | 0.006 / 0.03 | 41 | +0.0015 | +0.003 [+0.000, +0.013] | – | no measurable test effect; near-linear |
| s4 | protective_gate (Protective gate) | 2065 | BP × thal × exang × chest pain × sex | on for 102; mostly sex ≠ female and exang = chest pain brought on by exercise (54/63) | lowers | 0.060 / 0.61 | 35 | +0.0145 | -0.003 [-0.027, +0.021] | 0 | no measurable test effect; seed-specific |
| s5 | bp_suppressed_ischaemia (Blood-pressure gate on the exercise score) | 290 | thal × ST depression | on for 71; mostly thal = a reversible defect on the thallium scan and oldpeak ≥ 1.7 mm (32/33) | raises | 0.047 / 0.74 | 13 | +0.0046 | +0.004 [-0.005, +0.018] | 0 | no measurable test effect; seed-specific |
| s5 | high_bp_hinge (High blood pressure switch) | 5320 | BP | on for 84; mostly trestbps ≥ 139 mm Hg (84/84) | lowers | 0.023 / 0.68 | 7 | +0.0082 | +0.004 [-0.005, +0.021] | 1 | no measurable test effect; recurring |
| s5 | ischaemia_amplifier (Exercise-test pathway) | 1834 | thal × ST depression | on for 78; mostly thal ≠ a fixed defect on the thallium scan and oldpeak ≥ 1.85 mm (40/45) | raises | 0.021 / 0.17 | 24 | +0.0033 | +0.001 [-0.013, +0.014] | 0 | no measurable test effect; seed-specific |
| s5 | ischaemia_amplifier (Exercise-test pathway) | 2335 | BP | off for 27; mostly 151 ≤ trestbps < 162 mm Hg (14/16) | raises | 0.002 / 0.07 | 6 | +0.0004 | +0.001 [+0.000, +0.006] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s5 | st_exercise_score (Exercise-test score) | 204 | thal × ST depression | on for 108; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.75 mm (68/70) | raises | 0.044 / 0.49 | 9 | +0.0062 | +0.012 [-0.006, +0.036] | 0 | no measurable test effect; seed-specific |
| s6 | fbsDownslopeUnit (Downsloping ST and blood sugar unit) | 4252 | fasting sugar | on for 46; mostly fbs = fasting blood sugar above 120 mg/dL (33/33) | lowers | 0.001 / 0.01 | 13 | +0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear |
| s6 | femaleNoAnginaGate (Gate for women without chest pain on exercise) | 1213 | exang | off for 101; mostly exang = chest pain brought on by exercise (77/82) | lowers | 0.047 / 0.41 | 29 | +0.0152 | -0.015 [-0.048, +0.018] | 1 | no measurable test effect; recurring |
| s6 | lowHeartRateUnit (Low heart rate unit) | 7133 | max HR | off for 61; mostly thalach ≥ 166.5 beats per minute (61/61) | raises | 0.005 / 0.07 | 20 | +0.0007 | -0.001 [-0.009, +0.005] | – | no measurable test effect; near-linear |
| s6 | noAnginaRiskHub (Hub for sex, exercise chest pain and heart rate) | 47 | exang × max HR × sex | off for 98; mostly sex ≠ male and thalach < 173 beats per minute (59/77) | raises | 0.031 / 0.25 | 30 | +0.0073 | -0.017 [-0.043, +0.001] | 0 | no measurable test effect; seed-specific |
| s6 | protectiveRelay (Relay for sex and chest pain type) | 5800 | exang × sex | on for 93; mostly exang ≠ chest pain brought on by exercise and sex = female (51/64) | lowers | 0.021 / 0.46 | 28 | +0.0072 | -0.006 [-0.027, +0.011] | 0 | no measurable test effect; seed-specific |
| s7 | bp_thal_veto (Blood pressure brake) | 2961 | BP × thal | off for 116; mostly thal ≠ a normal thallium scan and trestbps ≥ 113.5 mm Hg (86/95) | lowers | 0.027 / 0.34 | 9 | +0.0053 | -0.003 [-0.019, +0.009] | 0 | no measurable test effect; seed-specific |
| s7 | max_heart_rate_gate (Peak heart rate gate) | 2124 | max HR | on for 66; mostly 160.5 ≤ thalach < 174.5 beats per minute (40/61) | lowers | 0.023 / 0.18 | 31 | +0.0025 | +0.000 [-0.015, +0.014] | 2 | no measurable test effect; recurring |
| s7 | sex_sugar_angina_gate (Sex, blood sugar and exercise angina gate) | 1239 | sex | on for 120; mostly sex ≠ male (83/83) | raises | 0.041 / 0.31 | 38 | +0.0095 | +0.003 [-0.016, +0.022] | 0 | no measurable test effect; seed-specific |
| s7 | st_depression_gate (ST depression gate) | 557 | thal × ST depression × ST slope | on for 84; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.95 mm (55/64) | raises | 0.056 / 0.56 | 28 | +0.0182 | +0.002 [-0.025, +0.030] | 0 | no measurable test effect; seed-specific |
| s7 | st_depression_gate (ST depression gate) | 9147 | thal × ST depression × ST slope | on for 40; mostly thal = a reversible defect on the thallium scan and oldpeak ≥ 1.85 mm (25/31) | raises | 0.001 / 0.04 | 5 | +0.0001 | +0.000 [+0.000, +0.000] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s7 | thal_chain (Thallium offset) | 3860 | thal | off for 2 | lowers | 0.001 / 0.01 | 8 | -0.0002 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s7 | thal_chain (Thallium offset) | 8833 | thal | off for 2 | lowers | 0.000 / 0.00 | 3 | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s8 | fitness_pathway (Age and fitness pathway) | 2116 | BP | on for 31; mostly 104.5 ≤ trestbps < 113.5 mm Hg (22/32) | lowers | 0.035 / 0.15 | 27 | +0.0041 | -0.005 [-0.022, +0.006] | 1 | no measurable test effect; recurring; small support (n<30) |
| s8 | fitness_pathway (Age and fitness pathway) | 4037 | age | on for 67; mostly age < 44.5 years (34/42) | lowers | 0.028 / 0.15 | 30 | +0.0035 | -0.006 [-0.021, +0.002] | 0 | no measurable test effect; seed-specific |
| s8 | fitness_score (Fitness score) | 2445 | age | on for 106; mostly age < 49.5 years (63/68) | lowers | 0.039 / 0.43 | 13 | +0.0072 | +0.009 [-0.016, +0.035] | 0 | no measurable test effect; seed-specific |
| s8 | fitness_score (Fitness score) | 5869 | max HR | off for 24; mostly thalach ≥ 176 beats per minute (24/24) | raises | 0.009 / 0.09 | 6 | +0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s8 | fitness_score (Fitness score) | 9765 | age | off for 99; mostly age < 53.5 years (99/99) | raises | 0.014 / 0.18 | 10 | +0.0023 | +0.012 [-0.002, +0.031] | 2 | no measurable test effect; recurring |
| s8 | ischaemia_gate (Exercise-test warning signs) | 855 | thal × ST depression × ST slope | on for 93; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.45 mm (64/77) | raises | 0.071 / 0.82 | 30 | +0.0336 | +0.005 [-0.028, +0.038] | 0 | no measurable test effect; seed-specific |
| s9 | oldpeak_gate (ST depression switch) | 2985 | thal | on for 110; mostly thal ≠ a normal thallium scan and thal ≠ a fixed defect on the thallium scan (86/92) | raises | 0.056 / 0.48 | 30 | +0.0220 | +0.009 [-0.015, +0.033] | 1 | no measurable test effect; recurring |
| s9 | sex_bp_exang_gate (Blood pressure with exercise chest pain) | 791 | exang | off for 81; mostly exang = chest pain brought on by exercise (77/82) | lowers | 0.046 / 0.32 | 26 | +0.0046 | -0.011 [-0.032, +0.007] | 1 | no measurable test effect; recurring |
| s9 | thal_ca_age_gate (Vessels, age, sugar and scan) | 2034 | age | on for 82; mostly age ≥ 62.5 years (45/54) | lowers | 0.041 / 0.52 | 25 | +0.0095 | -0.001 [-0.028, +0.022] | 0 | no measurable test effect; seed-specific |

## Every non-linear part (compact; train unless TEST)

| seed | part (title) | units | family (dominant unit) | regime n (union) | max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|
| s0 | angina_sex_gate (Exercise angina gate) | 2011 2349 3280 4331 665 | exang × resting ECG × sex | 146 | 0.78 | +0.0161 | +0.004 [-0.010, +0.023] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s0 | exertional_angina_hinge (Exercise angina and ST depression add-on) | 5884 | exang | 64 | 0.00 | +0.0001 | +0.000 [+0.000, +0.000] | 0 | no measurable test effect; seed-specific |
| s0 | flat_slope_hinge (Flat ST segment step) | 1650 2011 2349 3280 3598 4331 665 | exang × resting ECG × sex | 172 | 0.75 | +0.0217 | +0.004 [-0.027, +0.038] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s0 | lvh_ramp (Left ventricular hypertrophy signal) | 3280 4331 | resting ECG | 119 | 0.00 | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear |
| s0 | thal_ecg_unit (Thallium scan and resting ECG check) | 2011 2349 | thal | 2 | 0.03 | -0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s1 | normal_ecg_ischaemia_gate (Normal-ECG gate) | 1801 4375 | resting ECG | 132 | 0.37 | +0.0103 | +0.002 [-0.024, +0.029] | 0 | no measurable test effect; seed-specific |
| s1 | vessel_count_age_brake (Vessel count by age) | 3196 5446 | vessels × age | 17 | 0.48 | +0.0084 | +0.011 [-0.031, +0.058] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s2 | age_cholesterol_gate (Older-age brake) | 3464 3880 | age | 108 | 0.23 | +0.0044 | +0.005 [-0.012, +0.029] | 0 | no measurable test effect; seed-specific |
| s2 | heart_rate_age_gate (Youth and heart rate gate) | 1718 5718 | thal × age | 147 | 0.26 | +0.0078 | +0.017 [-0.003, +0.047] | 1 | no measurable test effect; recurring |
| s2 | high_heart_rate_units (Very high heart rates) | 7245 7595 | max HR | 49 | 0.01 | +0.0001 | +0.000 [+0.000, +0.000] | 2 | no measurable test effect; recurring |
| s2 | slope_gate (Flat ST segment and cholesterol) | 1476 1718 5718 | thal × age | 208 | 0.27 | +0.0089 | +0.022 [-0.001, +0.055] | 0 | no measurable test effect; seed-specific |
| s2 | st_depression_gate (ST depression and the thallium scan) | 1476 1718 1946 3138 3284 5718 | thal × ST depression | 230 | 0.61 | +0.0325 | +0.020 [-0.010, +0.057] | 0 | no measurable test effect; seed-specific |
| s2 | thal_age_gate (Thallium defect detector) | 3138 | thal × age | 107 | 0.30 | +0.0022 | +0.034 [+0.006, +0.075] | – | generalises; near-linear |
| s3 | bp_slope1_damper (Blood-pressure brake) | 3392 | BP × ST slope | 78 | 0.21 | +0.0044 | -0.001 [-0.019, +0.017] | 1 | no measurable test effect; recurring |
| s3 | exercise_reserve_age_chain (Exercise test and age chain) | 245 4376 5302 | ST depression × ST slope | 116 | 0.31 | +0.0069 | +0.004 [-0.013, +0.024] | 0 | no measurable test effect; seed-specific |
| s3 | hidden_unit_terms (Six switch-on parts) | 1499 245 3683 4072 4376 | age | 218 | 0.40 | +0.0162 | +0.004 [-0.030, +0.039] | 0 | no measurable test effect; seed-specific |
| s3 | old_age_relief (Older-age relief) | 1499 | age | 68 | 0.33 | +0.0048 | +0.004 [-0.022, +0.032] | 2 | no measurable test effect; recurring |
| s3 | st_depression_hinge (ST-depression switch) | 4072 | ST depression | 100 | 0.05 | +0.0008 | +0.008 [-0.004, +0.024] | – | no measurable test effect; near-linear |
| s3 | vessels_sex_gate (Combined-findings gate) | 3683 | chest pain | 59 | 0.24 | +0.0036 | -0.008 [-0.030, +0.010] | 0 | no measurable test effect; seed-specific |
| s4 | bp_hinge (Blood pressure threshold) | 8900 | BP | 94 | 0.02 | +0.0002 | +0.002 [+0.000, +0.009] | 2 | no measurable test effect; recurring |
| s4 | bp_thalach_chain (Blood pressure and heart rate signal for the gate) | 1754 6137 | BP × max HR | 40 | 0.06 | +0.0001 | +0.002 [+0.000, +0.009] | – | no measurable test effect; near-linear; small support (n<30) |
| s4 | exang_cp2_chain (Chest pain signal for the gate) | 4652 6931 | exang × chest pain | 198 | 0.14 | +0.0009 | -0.001 [-0.009, +0.005] | 0 | no measurable test effect; seed-specific |
| s4 | oldpeak_hinge (ST depression threshold) | 8065 | ST depression | 107 | 0.03 | +0.0015 | +0.003 [+0.000, +0.013] | – | no measurable test effect; near-linear |
| s4 | protective_gate (Protective gate) | 1754 2065 4652 6137 6931 | BP × thal × exang × chest pain × sex | 222 | 0.59 | +0.0165 | -0.002 [-0.027, +0.024] | 0 | no measurable test effect; seed-specific |
| s5 | bp_suppressed_ischaemia (Blood-pressure gate on the exercise score) | 204 290 5320 | BP | 146 | 0.73 | +0.0190 | +0.008 [-0.022, +0.040] | 0 | no measurable test effect; seed-specific |
| s5 | high_bp_hinge (High blood pressure switch) | 5320 | BP | 84 | 0.68 | +0.0082 | +0.004 [-0.005, +0.021] | 1 | no measurable test effect; recurring |
| s5 | ischaemia_amplifier (Exercise-test pathway) | 1834 204 2335 290 5320 | BP | 158 | 0.77 | +0.0321 | +0.018 [-0.023, +0.063] | 0 | no measurable test effect; seed-specific |
| s5 | st_exercise_score (Exercise-test score) | 204 | thal × ST depression | 108 | 0.49 | +0.0062 | +0.012 [-0.006, +0.036] | 0 | no measurable test effect; seed-specific |
| s6 | fbsDownslopeUnit (Downsloping ST and blood sugar unit) | 4252 | fasting sugar | 46 | 0.01 | +0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear |
| s6 | femaleNoAnginaGate (Gate for women without chest pain on exercise) | 1213 4252 | exang | 133 | 0.41 | +0.0152 | -0.015 [-0.048, +0.018] | 0 | no measurable test effect; seed-specific |
| s6 | lowHeartRateUnit (Low heart rate unit) | 7133 | max HR | 61 | 0.07 | +0.0007 | -0.001 [-0.009, +0.005] | – | no measurable test effect; near-linear |
| s6 | noAnginaRiskHub (Hub for sex, exercise chest pain and heart rate) | 1213 4252 47 5800 7133 | exang | 220 | 0.51 | +0.0269 | -0.012 [-0.047, +0.024] | 0 | no measurable test effect; seed-specific |
| s6 | protectiveRelay (Relay for sex and chest pain type) | 1213 4252 5800 | exang | 198 | 0.60 | +0.0391 | -0.023 [-0.064, +0.014] | 0 | no measurable test effect; seed-specific |
| s7 | bp_thal_veto (Blood pressure brake) | 2961 | BP × thal | 116 | 0.34 | +0.0053 | -0.003 [-0.019, +0.009] | 0 | no measurable test effect; seed-specific |
| s7 | max_heart_rate_gate (Peak heart rate gate) | 2124 | max HR | 66 | 0.18 | +0.0025 | +0.000 [-0.015, +0.014] | 2 | no measurable test effect; recurring |
| s7 | sex_sugar_angina_gate (Sex, blood sugar and exercise angina gate) | 1239 | sex | 120 | 0.31 | +0.0095 | +0.003 [-0.016, +0.022] | 0 | no measurable test effect; seed-specific |
| s7 | st_depression_gate (ST depression gate) | 557 9147 | thal × ST depression × ST slope | 84 | 0.56 | +0.0186 | +0.002 [-0.025, +0.030] | 0 | no measurable test effect; seed-specific |
| s7 | st_pathway (ST depression pathway) | 1239 2961 557 9147 | thal × ST depression × ST slope | 186 | 0.60 | +0.0319 | +0.003 [-0.026, +0.032] | 0 | no measurable test effect; seed-specific |
| s7 | thal_chain (Thallium offset) | 3860 8833 | thal | 2 | 0.00 | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s8 | fitness_pathway (Age and fitness pathway) | 2116 2445 4037 5869 9765 | age | 138 | 0.46 | +0.0141 | -0.013 [-0.048, +0.015] | 0 | no measurable test effect; seed-specific |
| s8 | fitness_score (Fitness score) | 2445 5869 9765 | age | 121 | 0.42 | +0.0060 | -0.008 [-0.037, +0.013] | 0 | no measurable test effect; seed-specific |
| s8 | ischaemia_gate (Exercise-test warning signs) | 855 | thal × ST depression × ST slope | 93 | 0.82 | +0.0336 | +0.005 [-0.028, +0.038] | 0 | no measurable test effect; seed-specific |
| s9 | oldpeak_gate (ST depression switch) | 2985 | thal | 110 | 0.48 | +0.0220 | +0.009 [-0.015, +0.033] | 1 | no measurable test effect; recurring |
| s9 | sex_bp_exang_gate (Blood pressure with exercise chest pain) | 791 | exang | 81 | 0.32 | +0.0046 | -0.011 [-0.032, +0.007] | 1 | no measurable test effect; recurring |
| s9 | thal_ca_age_gate (Vessels, age, sugar and scan) | 2034 | age | 82 | 0.52 | +0.0093 | -0.001 [-0.028, +0.022] | 0 | no measurable test effect; seed-specific |

## Families (units; train unless TEST)

| family group | seeds (units) | being on: raises / lowers risk | regime n (median, range) | age thresholds | BP thresholds | mean \|Δrisk\| on regime (median, max) | train gain: median, max | TEST gain CI > 0 / < 0 | recurring | small support |
|---|---|---|---|---|---|---|---|---|---|---|
| thal | 6: s0 s2 s5 s7 s8 s9 (13) | 10 / 3 | 78 (2–116) | – | – | 0.027, 0.82 | +0.0046, +0.0336 | 0 / 0 | 3/13 | 5 |
| max HR | 5: s2 s3 s6 s7 s8 (7) | 3 / 4 | 49 (9–121) | – | – | 0.005, 0.18 | +0.0001, +0.0025 | 0 / 0 | 5/7 | 3 |
| BP | 4: s3 s4 s5 s8 (7) | 4 / 3 | 40 (27–94) | – | 104.5 113.5 113.5 113.5 127 131 139 151 162 | 0.003, 0.68 | +0.0004, +0.0082 | 0 / 0 | 4/7 | 4 |
| age | 4: s2 s3 s8 s9 (6) | 1 / 5 | 83 (67–106) | 44.5 49.5 52.5 53.5 53.5 54.5 55.5 56.5 57.5 59.5 60.5 60.5 62.5 63.5 | – | 0.037, 0.52 | +0.0044, +0.0095 | 0 / 0 | 2/6 | 0 |
| exang | 4: s0 s4 s6 s9 (7) | 3 / 4 | 93 (38–111) | – | – | 0.031, 0.78 | +0.0072, +0.0163 | 0 / 0 | 2/7 | 1 |
| ST depression | 3: s1 s3 s4 (4) | 4 / 0 | 94 (66–107) | – | – | 0.012, 0.22 | +0.0019, +0.0039 | 0 / 0 | 1/4 | 0 |
| ST slope | 2: s0 s2 (3) | 3 / 0 | 108 (71–112) | – | – | 0.018, 0.61 | +0.0031, +0.0087 | 0 / 0 | 0/3 | 0 |
| resting ECG | 2: s0 s1 (3) | 1 / 2 | 119 (86–119) | – | – | 0.000, 0.32 | +0.0000, +0.0092 | 0 / 0 | 0/3 | 0 |
| fasting sugar | 1: s6 (1) | 0 / 1 | 46 (46–46) | – | – | 0.001, 0.01 | +0.0001, +0.0001 | 0 / 0 | 0/1 | 0 |
| thal × age | 1: s2 (2) | 1 / 1 | 105 (103–107) | 56.5 | – | 0.030, 0.38 | +0.0057, +0.0093 | 1 / 0 | 1/2 | 0 |
| sex | 1: s7 (1) | 1 / 0 | 120 (120–120) | – | – | 0.041, 0.31 | +0.0095, +0.0095 | 0 / 0 | 0/1 | 0 |
| cholesterol | 1: s2 (1) | 0 / 1 | 29 (29–29) | – | – | 0.003, 0.05 | +0.0001, +0.0001 | 0 / 0 | 0/1 | 1 |
| BP × exang | 1: s4 (1) | 0 / 1 | 110 (110–110) | – | 121 172 | 0.003, 0.12 | -0.0001, -0.0001 | 0 / 0 | 1/1 | 0 |
| chest pain | 1: s3 (1) | 1 / 0 | 59 (59–59) | – | – | 0.037, 0.24 | +0.0036, +0.0036 | 0 / 0 | 0/1 | 0 |
| ST slope × age | 1: s3 (1) | 1 / 0 | 62 (62–62) | 37.5 58.5 | – | 0.025, 0.17 | +0.0012, +0.0012 | 0 / 0 | 0/1 | 0 |
| BP × thal × exang | 1: s4 (1) | 0 / 1 | 102 (102–102) | – | 129 | 0.060, 0.61 | +0.0145, +0.0145 | 0 / 0 | 0/1 | 0 |
| BP × thal | 1: s7 (1) | 0 / 1 | 116 (116–116) | – | 113.5 144 | 0.027, 0.34 | +0.0053, +0.0053 | 0 / 0 | 0/1 | 0 |
| vessels × age | 1: s1 (2) | 1 / 1 | 10 (3–17) | 62.5 | – | 0.031, 0.47 | +0.0045, +0.0086 | 0 / 0 | 0/2 | 2 |

### thal

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | thal_ecg_unit (Thallium scan and resting ECG check) | 2011 | thal /  | on for 2 | raises (+0.17 logit) | 0.006 / 0.01 | 22: thal: 16/22 a reversible defect on the thallium scan | – | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s0 | thal_ecg_unit (Thallium scan and resting ECG check) | 2349 | thal / restecg | off for 2 | lowers (-1.02 logit) | 0.020 / 0.04 | 2: thal: 1/2 thallium stress test result coded 4 (meaning not recorded); restecg: 2/2 a normal resting ECG | – | -0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s2 | st_depression_gate (ST depression and the thallium scan) | 1946 | thal oldpeak /  | on for 116; mostly thal = a reversible defect on the thallium scan and oldpeak < 1.95 mm (55/62) | raises (+2.68 logit) | 0.056 / 0.52 | 24: thal: 16/24 a reversible defect on the thallium scan; oldpeak 0–4.2 (median 1.1) | – | +0.0121 | -0.001 [-0.012, +0.010] | 1 | no measurable test effect; recurring |
| s2 | st_depression_gate (ST depression and the thallium scan) | 3284 | thal oldpeak /  | on for 83; mostly thal ≠ a normal thallium scan and oldpeak ≥ 1.45 mm (43/45) | raises (+2.60 logit) | 0.027 / 0.16 | 33: thal: 19/33 a reversible defect on the thallium scan; oldpeak 0–4.2 (median 1) | – | +0.0063 | +0.001 [-0.011, +0.015] | 1 | no measurable test effect; recurring |
| s5 | ischaemia_amplifier (Exercise-test pathway) | 1834 | oldpeak thal /  | on for 78; mostly thal ≠ a fixed defect on the thallium scan and oldpeak ≥ 1.85 mm (40/45) | raises (+2.86 logit) | 0.021 / 0.17 | 24: oldpeak 0–4.2 (median 1); thal: 12/24 a normal thallium scan | – | +0.0033 | +0.001 [-0.013, +0.014] | 0 | no measurable test effect; seed-specific |
| s5 | st_exercise_score (Exercise-test score) | 204 | oldpeak thal /  | on for 108; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.75 mm (68/70) | raises (+2.41 logit) | 0.044 / 0.49 | 9: oldpeak 0.3–3.5 (median 1.9); thal: 5/9 a normal thallium scan | – | +0.0062 | +0.012 [-0.006, +0.036] | 0 | no measurable test effect; seed-specific |
| s5 | bp_suppressed_ischaemia (Blood-pressure gate on the exercise score) | 290 | oldpeak thal /  | on for 71; mostly thal = a reversible defect on the thallium scan and oldpeak ≥ 1.7 mm (32/33) | raises (+3.91 logit) | 0.047 / 0.74 | 13: oldpeak 0–4.2 (median 1.8); thal: 6/13 a normal thallium scan | – | +0.0046 | +0.004 [-0.005, +0.018] | 0 | no measurable test effect; seed-specific |
| s7 | thal_chain (Thallium offset) | 3860 | thal /  | off for 2 | lowers (-0.19 logit) | 0.001 / 0.01 | 8: thal: 8/8 a fixed defect on the thallium scan | – | -0.0002 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s7 | st_depression_gate (ST depression gate) | 557 | slope oldpeak thal /  | on for 84; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.95 mm (55/64) | raises (+3.14 logit) | 0.056 / 0.56 | 28: slope: 15/28 a flat ST segment; oldpeak 0–5.6 (median 1.45); thal: 13/28 a normal thallium scan | – | +0.0182 | +0.002 [-0.025, +0.030] | 0 | no measurable test effect; seed-specific |
| s7 | thal_chain (Thallium offset) | 8833 | thal /  | off for 2 | lowers (-0.19 logit) | 0.000 / 0.00 | 3: thal: 3/3 a fixed defect on the thallium scan | – | +0.0000 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s7 | st_depression_gate (ST depression gate) | 9147 | oldpeak slope thal /  | on for 40; mostly thal = a reversible defect on the thallium scan and oldpeak ≥ 1.85 mm (25/31) | raises (+0.28 logit) | 0.001 / 0.04 | 5: oldpeak 0.6–1.4 (median 1); slope: 5/5 a flat ST segment; thal: 2/5 a normal thallium scan | – | +0.0001 | +0.000 [+0.000, +0.000] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s8 | ischaemia_gate (Exercise-test warning signs) | 855 | slope thal oldpeak /  | on for 93; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.45 mm (64/77) | raises (+4.78 logit) | 0.071 / 0.82 | 30: slope: 18/30 an upsloping ST segment; thal: 15/30 a normal thallium scan; oldpeak 0–4.2 (median 0.9) | – | +0.0336 | +0.005 [-0.028, +0.038] | 0 | no measurable test effect; seed-specific |
| s9 | oldpeak_gate (ST depression switch) | 2985 | thal / oldpeak | on for 110; mostly thal ≠ a normal thallium scan and thal ≠ a fixed defect on the thallium scan (86/92) | raises (+3.11 logit) | 0.056 / 0.48 | 30: thal: 19/30 a normal thallium scan; oldpeak 0–4.2 (median 1) | – | +0.0220 | +0.009 [-0.015, +0.033] | 1 | no measurable test effect; recurring |

### BP

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 | bp_slope1_damper (Blood-pressure brake) | 3392 | trestbps slope /  | off for 78; mostly slope ≠ an upsloping ST segment and trestbps < 131 mm Hg (63/63) | lowers (-1.35 logit) | 0.024 / 0.16 | 34: trestbps 94–192 (median 130); slope: 23/34 a flat ST segment | – | +0.0030 | -0.001 [-0.019, +0.017] | 1 | no measurable test effect; recurring |
| s4 | bp_thalach_chain (Blood pressure and heart rate signal for the gate) | 1754 | trestbps thalach /  | off for 40; mostly trestbps < 113.5 mm Hg and thalach < 157 beats per minute (26/27) | raises (+0.23 logit) | 0.000 / 0.00 | 17: trestbps 104–132 (median 112); thalach 105–168 (median 144) | – | -0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s4 | bp_thalach_chain (Blood pressure and heart rate signal for the gate) | 6137 | trestbps thalach /  | off for 39; mostly trestbps < 113.5 mm Hg and thalach < 157 beats per minute (25/27) | raises (+0.23 logit) | 0.001 / 0.03 | 12: trestbps 110–192 (median 122.5); thalach 105–195 (median 135.5) | – | +0.0001 | +0.000 [+0.000, +0.000] | – | no measurable test effect; near-linear; small support (n<30) |
| s4 | bp_hinge (Blood pressure threshold) | 8900 | trestbps /  | off for 94; mostly trestbps < 127 mm Hg (94/94) | raises (+0.11 logit) | 0.003 / 0.02 | 35: trestbps 94–192 (median 128) | – | +0.0002 | +0.002 [+0.000, +0.009] | 2 | no measurable test effect; recurring |
| s5 | ischaemia_amplifier (Exercise-test pathway) | 2335 | trestbps / oldpeak | off for 27; mostly 151 ≤ trestbps < 162 mm Hg (14/16) | raises (+0.53 logit) | 0.002 / 0.07 | 6: trestbps 110–160 (median 150); oldpeak 0.1–3.6 (median 1) | – | +0.0004 | +0.001 [+0.000, +0.006] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s5 | high_bp_hinge (High blood pressure switch) | 5320 | trestbps /  | on for 84; mostly trestbps ≥ 139 mm Hg (84/84) | lowers (-0.94 logit) | 0.023 / 0.68 | 7: trestbps 101–178 (median 110) | – | +0.0082 | +0.004 [-0.005, +0.021] | 1 | no measurable test effect; recurring |
| s8 | fitness_pathway (Age and fitness pathway) | 2116 | trestbps /  | on for 31; mostly 104.5 ≤ trestbps < 113.5 mm Hg (22/32) | lowers (-0.63 logit) | 0.035 / 0.15 | 27: trestbps 94–192 (median 112) | – | +0.0041 | -0.005 [-0.022, +0.006] | 1 | no measurable test effect; recurring; small support (n<30) |

### age

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2 | age_cholesterol_gate (Older-age brake) | 3880 | age / slope | on for 84; mostly age ≥ 63.5 years (44/46) | lowers (-1.12 logit) | 0.036 / 0.22 | 26: age 38–70 (median 58.5); slope: 15/26 a flat ST segment | age < 48: 0 (0.00); 48 ≤ age < 56: 7 (0.06); 56 ≤ age < 61.75: 23 (0.03); age ≥ 61.75: 54 (0.03) | +0.0040 | +0.009 [-0.009, +0.035] | 0 | no measurable test effect; seed-specific |
| s3 | old_age_relief (Older-age relief) | 1499 | age /  | on for 68; mostly age ≥ 60.5 years (68/68) | lowers (-1.27 logit) | 0.047 / 0.33 | 35: age 38–76 (median 59) | age < 48: 0 (0.00); 48 ≤ age < 56: 0 (0.00); 56 ≤ age < 61.75: 7 (0.06); age ≥ 61.75: 61 (0.05) | +0.0048 | +0.004 [-0.022, +0.032] | 2 | no measurable test effect; recurring |
| s8 | fitness_score (Fitness score) | 2445 | age / trestbps | on for 106; mostly age < 49.5 years (63/68) | lowers (-1.14 logit) | 0.039 / 0.43 | 13: age 35–54 (median 46); trestbps 110–192 (median 140) | age < 48: 55 (0.04); 48 ≤ age < 56: 37 (0.05); 56 ≤ age < 61.75: 14 (0.02); age ≥ 61.75: 0 (0.00) | +0.0072 | +0.009 [-0.016, +0.035] | 0 | no measurable test effect; seed-specific |
| s8 | fitness_pathway (Age and fitness pathway) | 4037 | age / trestbps | on for 67; mostly age < 44.5 years (34/42) | lowers (-1.67 logit) | 0.028 / 0.15 | 30: age 38–67 (median 51.5); trestbps 110–192 (median 132) | age < 48: 39 (0.02); 48 ≤ age < 56: 22 (0.03); 56 ≤ age < 61.75: 6 (0.06); age ≥ 61.75: 0 (0.00) | +0.0035 | -0.006 [-0.021, +0.002] | 0 | no measurable test effect; seed-specific |
| s8 | fitness_score (Fitness score) | 9765 | age /  | off for 99; mostly age < 53.5 years (99/99) | raises (+0.07 logit) | 0.014 / 0.18 | 10: age 35–59 (median 52) | age < 48: 59 (0.01); 48 ≤ age < 56: 40 (0.02); 56 ≤ age < 61.75: 0 (0.00); age ≥ 61.75: 0 (0.00) | +0.0023 | +0.012 [-0.002, +0.031] | 2 | no measurable test effect; recurring |
| s9 | thal_ca_age_gate (Vessels, age, sugar and scan) | 2034 | age / thal | on for 82; mostly age ≥ 62.5 years (45/54) | lowers (-1.40 logit) | 0.041 / 0.52 | 25: age 48–74 (median 59); thal: 9/25 a normal thallium scan | age < 48: 2 (0.02); 48 ≤ age < 56: 2 (0.01); 56 ≤ age < 61.75: 29 (0.04); age ≥ 61.75: 49 (0.04) | +0.0095 | -0.001 [-0.028, +0.022] | 0 | no measurable test effect; seed-specific |

### exang

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | exertional_angina_hinge (Exercise angina and ST depression add-on) | 5884 | exang / oldpeak | on for 64; mostly exang = chest pain brought on by exercise (64/82) | raises (+0.02 logit) | 0.000 / 0.00 | 27: exang: 15/27 chest pain brought on by exercise; oldpeak 0–5.6 (median 0.6) | – | +0.0001 | +0.000 [+0.000, +0.000] | 0 | no measurable test effect; seed-specific |
| s0 | angina_sex_gate (Exercise angina gate) | 665 | exang sex restecg /  | on for 38; mostly exang ≠ no chest pain on exercise and sex ≠ male (19/19) | raises (+2.63 logit) | 0.099 / 0.78 | 12: exang: 8/12 chest pain brought on by exercise; sex: 7/12 male; restecg: 10/12 a normal resting ECG | – | +0.0163 | +0.004 [-0.010, +0.023] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s4 | exang_cp2_chain (Chest pain signal for the gate) | 4652 | exang cp /  | on for 111; mostly exang = chest pain brought on by exercise and cp ≠ non-anginal chest pain (63/71) | lowers (-1.01 logit) | 0.010 / 0.10 | 16: exang: 14/16 chest pain brought on by exercise; cp: 12/16 no chest pain (asymptomatic) | – | +0.0015 | -0.001 [-0.006, +0.000] | 0 | no measurable test effect; seed-specific |
| s6 | femaleNoAnginaGate (Gate for women without chest pain on exercise) | 1213 | exang / sex | off for 101; mostly exang = chest pain brought on by exercise (77/82) | lowers (-3.36 logit) | 0.047 / 0.41 | 29: exang: 22/29 no chest pain on exercise; sex: 19/29 male | – | +0.0152 | -0.015 [-0.048, +0.018] | 1 | no measurable test effect; recurring |
| s6 | noAnginaRiskHub (Hub for sex, exercise chest pain and heart rate) | 47 | thalach sex exang / cp | off for 98; mostly sex ≠ male and thalach < 173 beats per minute (59/77) | raises (+1.47 logit) | 0.031 / 0.25 | 30: thalach 90–178 (median 149); sex: 25/30 male; exang: 27/30 no chest pain on exercise; cp: 10/30 non-anginal chest pain | – | +0.0073 | -0.017 [-0.043, +0.001] | 0 | no measurable test effect; seed-specific |
| s6 | protectiveRelay (Relay for sex and chest pain type) | 5800 | exang sex /  | on for 93; mostly exang ≠ chest pain brought on by exercise and sex = female (51/64) | lowers (-2.05 logit) | 0.021 / 0.46 | 28: exang: 28/28 no chest pain on exercise; sex: 23/28 male | – | +0.0072 | -0.006 [-0.027, +0.011] | 0 | no measurable test effect; seed-specific |
| s9 | sex_bp_exang_gate (Blood pressure with exercise chest pain) | 791 | exang / trestbps sex | off for 81; mostly exang = chest pain brought on by exercise (77/82) | lowers (-1.80 logit) | 0.046 / 0.32 | 26: exang: 13/26 no chest pain on exercise; trestbps 94–192 (median 119); sex: 19/26 male | – | +0.0046 | -0.011 [-0.032, +0.007] | 1 | no measurable test effect; recurring |

### thal × age

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2 | heart_rate_age_gate (Youth and heart rate gate) | 1718 | age thal /  | on for 103; mostly thal ≠ a reversible defect on the thallium scan and age < 56.5 years (85/85) | lowers (-2.19 logit) | 0.035 / 0.38 | 31: age 38–70 (median 57); thal: 16/31 a reversible defect on the thallium scan | age < 48: 53 (0.02); 48 ≤ age < 56: 34 (0.02); 56 ≤ age < 61.75: 15 (0.09); age ≥ 61.75: 1 (0.08) | +0.0093 | +0.014 [-0.008, +0.041] | 1 | no measurable test effect; recurring |
| s2 | thal_age_gate (Thallium defect detector) | 3138 | thal / age | on for 107; mostly thal ≠ a normal thallium scan and thal = a reversible defect on the thallium scan (90/90) | raises (+2.19 logit) | 0.025 / 0.30 | 9: thal: 7/9 a reversible defect on the thallium scan; age 38–70 (median 62) | age < 48: 16 (0.05); 48 ≤ age < 56: 27 (0.01); 56 ≤ age < 61.75: 38 (0.01); age ≥ 61.75: 26 (0.04) | +0.0022 | +0.034 [+0.006, +0.075] | – | generalises; near-linear |

### BP × exang

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4 | exang_cp2_chain (Chest pain signal for the gate) | 6931 | exang trestbps /  | off for 110; mostly exang = no chest pain on exercise and trestbps ≥ 121 mm Hg (109/109) | lowers (-0.51 logit) | 0.003 / 0.12 | 12: exang: 9/12 chest pain brought on by exercise; trestbps 94–156 (median 111) | – | -0.0001 | -0.001 [-0.006, +0.000] | 1 | no measurable test effect; recurring |

### ST slope × age

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 | exercise_reserve_age_chain (Exercise test and age chain) | 4376 | age slope / oldpeak | off for 62; mostly slope = a flat ST segment and age < 58.5 years (55/62) | raises (+1.25 logit) | 0.025 / 0.17 | 29: age 35–68 (median 57); slope: 19/29 a flat ST segment; oldpeak 0–3.8 (median 1.4) | age < 48: 25 (0.05); 48 ≤ age < 56: 21 (0.01); 56 ≤ age < 61.75: 16 (0.02); age ≥ 61.75: 0 (0.00) | +0.0012 | +0.003 [-0.011, +0.018] | 0 | no measurable test effect; seed-specific |

### BP × thal × exang

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4 | protective_gate (Protective gate) | 2065 | sex exang trestbps cp thal /  | on for 102; mostly sex ≠ female and exang = chest pain brought on by exercise (54/63) | lowers (-2.32 logit) | 0.060 / 0.61 | 35: sex: 24/35 male; exang: 20/35 chest pain brought on by exercise; trestbps 94–160 (median 128); cp: 19/35 no chest pain (asymptomatic); thal: 17/35 a normal thallium scan | – | +0.0145 | -0.003 [-0.027, +0.021] | 0 | no measurable test effect; seed-specific |

### BP × thal

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s7 | bp_thal_veto (Blood pressure brake) | 2961 | thal trestbps / slope | off for 116; mostly thal ≠ a normal thallium scan and trestbps ≥ 113.5 mm Hg (86/95) | lowers (-0.27 logit) | 0.027 / 0.34 | 9: thal: 5/9 a reversible defect on the thallium scan; trestbps 104–170 (median 150); slope: 9/9 a flat ST segment | – | +0.0053 | -0.003 [-0.019, +0.009] | 0 | no measurable test effect; seed-specific |

### vessels × age

| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \|Δrisk\| | half the effect on | age bands in regime: n (mean \|Δrisk\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | vessel_count_age_brake (Vessel count by age) | 3196 | age ca /  | off for 3; mostly ca ≥ 2.5 vessels and age < 62.5 years (3/5) | raises (+2.16 logit) | 0.007 / 0.05 | 23: age 42–69 (median 59); ca 1–3 (median 1) | age < 48: 1 (0.01); 48 ≤ age < 56: 2 (0.00); 56 ≤ age < 61.75: 0 (0.00); age ≥ 61.75: 0 (0.00) | +0.0003 | +0.005 [-0.004, +0.020] | 0 | no measurable test effect; seed-specific; small support (n<30) |
| s1 | vessel_count_age_brake (Vessel count by age) | 5446 | ca / age | on for 17; mostly ca ≥ 2.5 vessels (12/12) | lowers (-3.09 logit) | 0.055 / 0.47 | 31: ca 0–3 (median 1); age 42–71 (median 59) | age < 48: 2 (0.14); 48 ≤ age < 56: 6 (0.09); 56 ≤ age < 61.75: 1 (0.01); age ≥ 61.75: 8 (0.01) | +0.0086 | +0.013 [-0.030, +0.061] | 0 | no measurable test effect; seed-specific; small support (n<30) |

## Family tables (condensed, for AUDIT.md)

**thal** (6 seeds, 13 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s0 | 2011 (Thallium scan and resting ECG check) | on for 2; raises risk | 22 | 0.006 / 0.01 | +0.0000 | +0.000 [+0.000, +0.000] | – |
| s0 | 2349 (Thallium scan and resting ECG check) | off for 2 (strength: restecg); lowers risk | 2 | 0.020 / 0.04 | -0.0001 | +0.000 [+0.000, +0.000] | – |
| s2 | 1946 (ST depression and the thallium scan) | on for 116; mostly thal = a reversible defect on the thallium scan and oldpeak < 1.95 mm (55/62); raises risk | 24 | 0.056 / 0.52 | +0.0121 | -0.001 [-0.012, +0.010] | 1 |
| s2 | 3284 (ST depression and the thallium scan) | on for 83; mostly thal ≠ a normal thallium scan and oldpeak ≥ 1.45 mm (43/45); raises risk | 33 | 0.027 / 0.16 | +0.0063 | +0.001 [-0.011, +0.015] | 1 |
| s5 | 1834 (Exercise-test pathway) | on for 78; mostly thal ≠ a fixed defect on the thallium scan and oldpeak ≥ 1.85 mm (40/45); raises risk | 24 | 0.021 / 0.17 | +0.0033 | +0.001 [-0.013, +0.014] | 0 |
| s5 | 204 (Exercise-test score) | on for 108; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.75 mm (68/70); raises risk | 9 | 0.044 / 0.49 | +0.0062 | +0.012 [-0.006, +0.036] | 0 |
| s5 | 290 (Blood-pressure gate on the exercise score) | on for 71; mostly thal = a reversible defect on the thallium scan and oldpeak ≥ 1.7 mm (32/33); raises risk | 13 | 0.047 / 0.74 | +0.0046 | +0.004 [-0.005, +0.018] | 0 |
| s7 | 3860 (Thallium offset) | off for 2; lowers risk | 8 | 0.001 / 0.01 | -0.0002 | +0.000 [+0.000, +0.000] | – |
| s7 | 557 (ST depression gate) | on for 84; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.95 mm (55/64); raises risk | 28 | 0.056 / 0.56 | +0.0182 | +0.002 [-0.025, +0.030] | 0 |
| s7 | 8833 (Thallium offset) | off for 2; lowers risk | 3 | 0.000 / 0.00 | +0.0000 | +0.000 [+0.000, +0.000] | – |
| s7 | 9147 (ST depression gate) | on for 40; mostly thal = a reversible defect on the thallium scan and oldpeak ≥ 1.85 mm (25/31); raises risk | 5 | 0.001 / 0.04 | +0.0001 | +0.000 [+0.000, +0.000] | 0 |
| s8 | 855 (Exercise-test warning signs) | on for 93; mostly thal ≠ a normal thallium scan and oldpeak ≥ 0.45 mm (64/77); raises risk | 30 | 0.071 / 0.82 | +0.0336 | +0.005 [-0.028, +0.038] | 0 |
| s9 | 2985 (ST depression switch) | on for 110; mostly thal ≠ a normal thallium scan and thal ≠ a fixed defect on the thallium scan (86/92) (strength: oldpeak); raises risk | 30 | 0.056 / 0.48 | +0.0220 | +0.009 [-0.015, +0.033] | 1 |

**BP** (4 seeds, 7 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s3 | 3392 (Blood-pressure brake) | off for 78; mostly slope ≠ an upsloping ST segment and trestbps < 131 mm Hg (63/63); lowers risk | 34 | 0.024 / 0.16 | +0.0030 | -0.001 [-0.019, +0.017] | 1 |
| s4 | 1754 (Blood pressure and heart rate signal for the gate) | off for 40; mostly trestbps < 113.5 mm Hg and thalach < 157 beats per minute (26/27); raises risk | 17 | 0.000 / 0.00 | -0.0001 | +0.000 [+0.000, +0.000] | – |
| s4 | 6137 (Blood pressure and heart rate signal for the gate) | off for 39; mostly trestbps < 113.5 mm Hg and thalach < 157 beats per minute (25/27); raises risk | 12 | 0.001 / 0.03 | +0.0001 | +0.000 [+0.000, +0.000] | – |
| s4 | 8900 (Blood pressure threshold) | off for 94; mostly trestbps < 127 mm Hg (94/94); raises risk | 35 | 0.003 / 0.02 | +0.0002 | +0.002 [+0.000, +0.009] | 2 |
| s5 | 2335 (Exercise-test pathway) | off for 27; mostly 151 ≤ trestbps < 162 mm Hg (14/16) (strength: oldpeak); raises risk | 6 | 0.002 / 0.07 | +0.0004 | +0.001 [+0.000, +0.006] | 0 |
| s5 | 5320 (High blood pressure switch) | on for 84; mostly trestbps ≥ 139 mm Hg (84/84); lowers risk | 7 | 0.023 / 0.68 | +0.0082 | +0.004 [-0.005, +0.021] | 1 |
| s8 | 2116 (Age and fitness pathway) | on for 31; mostly 104.5 ≤ trestbps < 113.5 mm Hg (22/32); lowers risk | 27 | 0.035 / 0.15 | +0.0041 | -0.005 [-0.022, +0.006] | 1 |

**age** (4 seeds, 6 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s2 | 3880 (Older-age brake) | on for 84; mostly age ≥ 63.5 years (44/46) (strength: slope); lowers risk | 26 | 0.036 / 0.22 | +0.0040 | +0.009 [-0.009, +0.035] | 0 |
| s3 | 1499 (Older-age relief) | on for 68; mostly age ≥ 60.5 years (68/68); lowers risk | 35 | 0.047 / 0.33 | +0.0048 | +0.004 [-0.022, +0.032] | 2 |
| s8 | 2445 (Fitness score) | on for 106; mostly age < 49.5 years (63/68) (strength: trestbps); lowers risk | 13 | 0.039 / 0.43 | +0.0072 | +0.009 [-0.016, +0.035] | 0 |
| s8 | 4037 (Age and fitness pathway) | on for 67; mostly age < 44.5 years (34/42) (strength: trestbps); lowers risk | 30 | 0.028 / 0.15 | +0.0035 | -0.006 [-0.021, +0.002] | 0 |
| s8 | 9765 (Fitness score) | off for 99; mostly age < 53.5 years (99/99); raises risk | 10 | 0.014 / 0.18 | +0.0023 | +0.012 [-0.002, +0.031] | 2 |
| s9 | 2034 (Vessels, age, sugar and scan) | on for 82; mostly age ≥ 62.5 years (45/54) (strength: thal); lowers risk | 25 | 0.041 / 0.52 | +0.0095 | -0.001 [-0.028, +0.022] | 0 |

**exang** (4 seeds, 7 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s0 | 5884 (Exercise angina and ST depression add-on) | on for 64; mostly exang = chest pain brought on by exercise (64/82) (strength: oldpeak); raises risk | 27 | 0.000 / 0.00 | +0.0001 | +0.000 [+0.000, +0.000] | 0 |
| s0 | 665 (Exercise angina gate) | on for 38; mostly exang ≠ no chest pain on exercise and sex ≠ male (19/19); raises risk | 12 | 0.099 / 0.78 | +0.0163 | +0.004 [-0.010, +0.023] | 0 |
| s4 | 4652 (Chest pain signal for the gate) | on for 111; mostly exang = chest pain brought on by exercise and cp ≠ non-anginal chest pain (63/71); lowers risk | 16 | 0.010 / 0.10 | +0.0015 | -0.001 [-0.006, +0.000] | 0 |
| s6 | 1213 (Gate for women without chest pain on exercise) | off for 101; mostly exang = chest pain brought on by exercise (77/82) (strength: sex); lowers risk | 29 | 0.047 / 0.41 | +0.0152 | -0.015 [-0.048, +0.018] | 1 |
| s6 | 47 (Hub for sex, exercise chest pain and heart rate) | off for 98; mostly sex ≠ male and thalach < 173 beats per minute (59/77) (strength: cp); raises risk | 30 | 0.031 / 0.25 | +0.0073 | -0.017 [-0.043, +0.001] | 0 |
| s6 | 5800 (Relay for sex and chest pain type) | on for 93; mostly exang ≠ chest pain brought on by exercise and sex = female (51/64); lowers risk | 28 | 0.021 / 0.46 | +0.0072 | -0.006 [-0.027, +0.011] | 0 |
| s9 | 791 (Blood pressure with exercise chest pain) | off for 81; mostly exang = chest pain brought on by exercise (77/82) (strength: trestbps, sex); lowers risk | 26 | 0.046 / 0.32 | +0.0046 | -0.011 [-0.032, +0.007] | 1 |

**thal × age** (1 seeds, 2 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s2 | 1718 (Youth and heart rate gate) | on for 103; mostly thal ≠ a reversible defect on the thallium scan and age < 56.5 years (85/85); lowers risk | 31 | 0.035 / 0.38 | +0.0093 | +0.014 [-0.008, +0.041] | 1 |
| s2 | 3138 (Thallium defect detector) | on for 107; mostly thal ≠ a normal thallium scan and thal = a reversible defect on the thallium scan (90/90) (strength: age); raises risk | 9 | 0.025 / 0.30 | +0.0022 | +0.034 [+0.006, +0.075] | – |

**BP × exang** (1 seeds, 1 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s4 | 6931 (Chest pain signal for the gate) | off for 110; mostly exang = no chest pain on exercise and trestbps ≥ 121 mm Hg (109/109); lowers risk | 12 | 0.003 / 0.12 | -0.0001 | -0.001 [-0.006, +0.000] | 1 |

**ST slope × age** (1 seeds, 1 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s3 | 4376 (Exercise test and age chain) | off for 62; mostly slope = a flat ST segment and age < 58.5 years (55/62) (strength: oldpeak); raises risk | 29 | 0.025 / 0.17 | +0.0012 | +0.003 [-0.011, +0.018] | 0 |

**BP × thal × exang** (1 seeds, 1 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s4 | 2065 (Protective gate) | on for 102; mostly sex ≠ female and exang = chest pain brought on by exercise (54/63); lowers risk | 35 | 0.060 / 0.61 | +0.0145 | -0.003 [-0.027, +0.021] | 0 |

**BP × thal** (1 seeds, 1 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s7 | 2961 (Blood pressure brake) | off for 116; mostly thal ≠ a normal thallium scan and trestbps ≥ 113.5 mm Hg (86/95) (strength: slope); lowers risk | 9 | 0.027 / 0.34 | +0.0053 | -0.003 [-0.019, +0.009] | 0 |

**vessels × age** (1 seeds, 2 units)

| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \|Δrisk\| | train gain | TEST gain [95% CI] | other seeds |
|---|---|---|---|---|---|---|---|
| s1 | 3196 (Vessel count by age) | off for 3; mostly ca ≥ 2.5 vessels and age < 62.5 years (3/5); raises risk | 23 | 0.007 / 0.05 | +0.0003 | +0.005 [-0.004, +0.020] | 0 |
| s1 | 5446 (Vessel count by age) | on for 17; mostly ca ≥ 2.5 vessels (12/12) (strength: age); lowers risk | 31 | 0.055 / 0.47 | +0.0086 | +0.013 [-0.030, +0.061] | 0 |

## Key inputs: units where BP / thal / exang / age drive the switch or set its strength

| input | units it drives (seeds) | units it modifies (seeds) | seeds with either | of these, units also involving age | being on raises / lowers risk | regime ≥ 30 patients | TEST CI > 0 | max train gain |
|---|---|---|---|---|---|---|---|---|
| BP | 10 (s3 s4 s5 s7 s8) | 3 (s8 s9) | 6 | 2 | 4 / 9 | 12 | 0 | +0.0145 |
| thal | 17 (s0 s2 s4 s5 s7 s8 s9) | 1 (s9) | 7 | 3 | 11 / 7 | 14 | 1 | +0.0336 |
| exang | 9 (s0 s4 s6 s9) | 1 (s0) | 4 | 0 | 4 / 6 | 10 | 0 | +0.0163 |
| age | 9 (s1 s2 s3 s8 s9) | 2 (s1 s2) | 5 | – | 4 / 7 | 9 | 1 | +0.0095 |

## Recurring signals (strict residual match)

| network | level | unit | family | seeds matched (which) | 40-gen seeds reproducing | best 40-gen signal (R²) | train gain | TEST gain |
|---|---|---|---|---|---|---|---|---|
| s3 | unit | 3392 | BP × ST slope | 1 (s4) | 1 | heart-s3:bp_slope1_gate (0.94) | +0.0030 | -0.001 [-0.019, +0.017] |
| s3 | part | 3392 | BP × ST slope | 1 (s4) | 1 | heart-s3:bp_slope1_gate (0.94) | +0.0044 | -0.001 [-0.019, +0.017] |
| s4 | unit | 8900 | BP | 2 (s5 s8) | 0 | – (0.28) | +0.0002 | +0.002 [+0.000, +0.009] |
| s4 | part | 8900 | BP | 2 (s5 s8) | 0 | – (0.28) | +0.0002 | +0.002 [+0.000, +0.009] |
| s5 | unit | 5320 | BP | 1 (s4) | 0 | – (0.30) | +0.0082 | +0.004 [-0.005, +0.021] |
| s5 | part | 5320 | BP | 1 (s4) | 0 | – (0.30) | +0.0082 | +0.004 [-0.005, +0.021] |
| s8 | unit | 2116 | BP | 1 (s4) | 0 | – (0.22) | +0.0041 | -0.005 [-0.022, +0.006] |
| s4 | unit | 6931 | BP × exang | 1 (s9) | 0 | – (0.23) | -0.0001 | -0.001 [-0.006, +0.000] |
| s1 | unit | 4375 | ST depression | 2 (s3 s4) | 2 | heart-s2:oldpeak_hinges (1.00) | +0.0024 | +0.002 [-0.010, +0.016] |
| s3 | unit | 1499 | age | 2 (s2 s8) | 2 | heart-s3:age_gate (0.98) | +0.0048 | +0.004 [-0.022, +0.032] |
| s3 | part | 1499 | age | 2 (s2 s8) | 2 | heart-s3:age_gate (0.98) | +0.0048 | +0.004 [-0.022, +0.032] |
| s8 | unit | 9765 | age | 2 (s2 s3) | 2 | heart-s3:age_gate (0.75) | +0.0023 | +0.012 [-0.002, +0.031] |
| s6 | unit | 1213 | exang | 1 (s4) | 1 | heart-s6:sex_exang_interaction (0.79) | +0.0152 | -0.015 [-0.048, +0.018] |
| s9 | unit | 791 | exang | 1 (s4) | 0 | – (0.45) | +0.0046 | -0.011 [-0.032, +0.007] |
| s9 | part | 791 | exang | 1 (s4) | 0 | – (0.45) | +0.0046 | -0.011 [-0.032, +0.007] |
| s2 | unit | 5718 | max HR | 2 (s6 s7) | 1 | – (0.34) | +0.0003 | +0.003 [-0.008, +0.017] |
| s2 | unit | 7245 | max HR | 2 (s3 s8) | 0 | – (0.17) | +0.0000 | +0.000 [+0.000, +0.000] |
| s2 | unit | 7595 | max HR | 4 (s3 s6 s7 s8) | 0 | – (0.27) | +0.0001 | +0.000 [+0.000, +0.000] |
| s2 | part | 7245 7595 | max HR | 2 (s3 s8) | 0 | – (0.17) | +0.0001 | +0.000 [+0.000, +0.000] |
| s3 | unit | 5302 | max HR | 2 (s2 s8) | 0 | – (0.15) | +0.0000 | +0.000 [+0.000, +0.000] |
| s7 | unit | 2124 | max HR | 2 (s2 s6) | 0 | – (0.27) | +0.0025 | +0.000 [-0.015, +0.014] |
| s7 | part | 2124 | max HR | 2 (s2 s6) | 0 | – (0.27) | +0.0025 | +0.000 [-0.015, +0.014] |
| s2 | unit | 1946 | thal × ST depression | 1 (s9) | 0 | – (0.21) | +0.0121 | -0.001 [-0.012, +0.010] |
| s2 | unit | 3284 | thal × ST depression | 1 (s9) | 0 | – (0.27) | +0.0063 | +0.001 [-0.011, +0.015] |
| s9 | unit | 2985 | thal | 1 (s2) | 0 | – (0.34) | +0.0220 | +0.009 [-0.015, +0.033] |
| s9 | part | 2985 | thal | 1 (s2) | 0 | – (0.34) | +0.0220 | +0.009 [-0.015, +0.033] |
| s2 | unit | 1718 | thal × age | 1 (s8) | 1 | heart-s2:age_hr_reserve_gate (0.85) | +0.0093 | +0.014 [-0.008, +0.041] |
| s2 | part | 1718 5718 | thal × age | 1 (s8) | 1 | heart-s2:age_hr_reserve_gate (0.85) | +0.0078 | +0.017 [-0.003, +0.047] |

Unit-to-unit residual matches across Heart-100 seeds (R² ≥ 0.5): 8
- s2:1946 (thal × ST depression, being on raises risk) ~ s9:2985 (thal, raises) (R² 0.58)
- s2:3284 (thal × ST depression, being on raises risk) ~ s9:2985 (thal, raises) (R² 0.62)
- s2:5718 (max HR, being on lowers risk) ~ s7:2124 (max HR, lowers) (R² 0.56)
- s2:7245 (max HR, being on lowers risk) ~ s3:5302 (max HR, raises) (R² 0.98)
- s2:7595 (max HR, being on lowers risk) ~ s7:2124 (max HR, lowers) (R² 0.56)
- s3:1499 (age, being on lowers risk) ~ s8:9765 (age, raises) (R² 0.68)
- s4:6931 (BP × exang, being on lowers risk) ~ s9:791 (exang, lowers) (R² 0.52)
- s4:8900 (BP, being on raises risk) ~ s5:5320 (BP, lowers) (R² 0.73)

Units / parts whose residual matches a 40-generation Heart inventory signal (R² ≥ 0.5): 30
- s0 unit 1650 (ST slope × chest pain) ~ heart-s0:female_exertional_angina_gate (R² 0.58) [same seed]
- s0 unit 3598 (ST slope) ~ heart-s0:female_exertional_angina_gate (R² 0.65) [same seed]
- s0 unit 665 (exang × resting ECG × sex) ~ heart-s0:female_exertional_angina_gate (R² 0.71) [same seed]
- s0 part 1650 2011 2349 3280 3598 4331 665 (exang × resting ECG × sex) ~ heart-s0:female_exertional_angina_gate (R² 0.58) [same seed]
- s0 part 2011 2349 3280 4331 665 (exang × resting ECG × sex) ~ heart-s0:female_exertional_angina_gate (R² 0.71) [same seed]
- s1 unit 4375 (ST depression) ~ heart-s2:oldpeak_hinges (R² 1.00) [other seed]
- s2 unit 1718 (thal × age) ~ heart-s2:age_hr_reserve_gate (R² 0.85) [same seed]
- s2 part 1718 5718 (thal × age) ~ heart-s2:age_hr_reserve_gate (R² 0.85) [same seed]
- s3 unit 1499 (age) ~ heart-s3:age_gate (R² 0.98) [same seed]
- s3 unit 245 (ST depression × ST slope) ~ heart-s2:st_slope_chol_gate (R² 0.60) [other seed]
- s3 unit 3392 (BP × ST slope) ~ heart-s3:bp_slope1_gate (R² 0.94) [same seed]
- s3 unit 4376 (ST slope × age) ~ heart-s2:st_slope_chol_gate (R² 0.52) [other seed]
- s3 part 245 4376 5302 (ST depression × ST slope) ~ heart-s2:st_slope_chol_gate (R² 0.52) [other seed]
- s3 part 3392 (BP × ST slope) ~ heart-s3:bp_slope1_gate (R² 0.94) [same seed]
- s3 part 1499 (age) ~ heart-s3:age_gate (R² 0.98) [same seed]
- s5 unit 204 (thal × ST depression) ~ heart-s5:st_ischaemia_gate_output (R² 0.75) [same seed]
- s5 unit 290 (thal × ST depression) ~ heart-s5:st_ischaemia_gate_output (R² 0.52) [same seed]
- s5 part 204 (thal × ST depression) ~ heart-s5:st_ischaemia_gate_output (R² 0.75) [same seed]
- s6 unit 1213 (exang) ~ heart-s6:sex_exang_interaction (R² 0.79) [same seed]
- s6 unit 5800 (exang × sex) ~ heart-s6:sex_exang_interaction (R² 0.70) [same seed]
- s7 unit 1239 (sex) ~ heart-s6:sex_exang_interaction (R² 0.57) [other seed]
- s7 unit 557 (thal × ST depression × ST slope) ~ heart-s7:st_depression_gate (R² 0.69) [same seed]
- s7 unit 9147 (thal × ST depression × ST slope) ~ heart-s7:st_depression_gate (R² 0.56) [same seed]
- s7 part 557 9147 (thal × ST depression × ST slope) ~ heart-s7:st_depression_gate (R² 0.56) [same seed]
- s7 part 1239 (sex) ~ heart-s6:sex_exang_interaction (R² 0.57) [other seed]
- s8 unit 855 (thal × ST depression × ST slope) ~ heart-s8:exertional_ischaemia_hinge (R² 0.66) [same seed]
- s8 unit 9765 (age) ~ heart-s3:age_gate (R² 0.75) [other seed]
- s8 part 855 (thal × ST depression × ST slope) ~ heart-s8:exertional_ischaemia_hinge (R² 0.66) [same seed]
- s9 unit 2034 (age) ~ heart-s4:thal6_fixed_defect_gate (R² 0.58) [other seed]
- s9 part 2034 (age) ~ heart-s4:thal6_fixed_defect_gate (R² 0.58) [other seed]

## Linear core: linear-path coefficient across the 10 networks (log-odds per real unit; categories vs the most common level)

| term | per | networks with a linear path | sign + / − | median | IQR | min – max | LR on train (reference) | shared? | average effect, all 10: + / −, median (min – max) |
|---|---|---|---|---|---|---|---|---|---|
| age (age) | years | 9 | 8 / 1 | +0.0286 | +0.0129 – +0.0415 | -0.0194 – +0.0619 | +0.0135 | mostly | 10 / 0, +0.0236 (+0.0019 – +0.0598) |
| sex=0 (female vs male) | level vs reference | 9 | 0 / 9 | -2.51 | -3.07 – -2.33 | -3.41 – -1.5 | -1.92 | yes | 0 / 10, -2.22 (-2.61 – -1.34) |
| cp=1 (typical angina vs no chest pain (asymptomatic)) | level vs reference | 10 | 0 / 10 | -2.29 | -2.47 – -1.73 | -2.85 – -0.51 | -2.1 | yes | 0 / 10, -2.46 (-3.76 – -0.949) |
| cp=2 (atypical angina vs no chest pain (asymptomatic)) | level vs reference | 10 | 1 / 9 | -1.15 | -1.26 – -0.666 | -3 – +0.526 | -0.8 | mostly | 1 / 9, -0.894 (-1.62 – +0.00238) |
| cp=3 (non-anginal chest pain vs no chest pain (asymptomatic)) | level vs reference | 10 | 0 / 10 | -1.75 | -1.95 – -1.66 | -2.22 – -1.54 | -1.58 | yes | 0 / 10, -1.75 (-2.22 – -1.3) |
| trestbps (resting blood pressure) | mm Hg | 5 | 5 / 0 | +0.0344 | +0.0268 – +0.0461 | +0.0231 – +0.0713 | +0.0202 | rarely used | 8 / 0, +0.018 (+0 – +0.0411) |
| chol (serum cholesterol) | mg/dL | 10 | 10 / 0 | +0.013 | +0.012 – +0.0143 | +0.011 – +0.0166 | +0.00967 | yes | 10 / 0, +0.0127 (+0.0101 – +0.0147) |
| fbs=1 (fasting blood sugar above 120 mg/dL vs fasting blood sugar of 120 mg/dL or less) | level vs reference | 8 | 1 / 7 | -1.59 | -2.01 – -0.929 | -3.75 – +0.442 | -0.608 | mostly | 1 / 7, -0.667 (-1.32 – +0.0773) |
| restecg=1 (an ST-T wave abnormality on the resting ECG vs a normal resting ECG) | level vs reference | 5 | 4 / 1 | +0.175 | +0.161 – +0.595 | -0.269 – +2.24 | +0.42 | rarely used | 4 / 2, +0 (-0.269 – +1.02) |
| restecg=2 (signs of left ventricular hypertrophy on the resting ECG vs a normal resting ECG) | level vs reference | 7 | 7 / 0 | +0.356 | +0.271 – +0.696 | +0.175 – +1.3 | +0.205 | mostly | 7 / 1, +0.21 (-0.00342 – +0.356) |
| thalach (maximum heart rate) | beats minute | 6 | 1 / 5 | -0.0109 | -0.0145 – -0.00676 | -0.0158 – +0.00104 | -0.00773 | mostly | 0 / 10, -0.013 (-0.0222 – -0.00397) |
| exang=1 (chest pain brought on by exercise vs no chest pain on exercise) | level vs reference | 6 | 4 / 2 | +0.366 | -0.197 – +0.61 | -0.852 – +2.23 | +0.623 | no | 8 / 0, +0.741 (+0 – +1.27) |
| oldpeak (ST depression on exercise) | mm | 3 | 3 / 0 | +0.452 | +0.376 – +0.548 | +0.299 – +0.643 | +0.382 | rarely used | 9 / 1, +0.69 (-0.121 – +0.986) |
| slope=2 (a flat ST segment vs an upsloping ST segment) | level vs reference | 10 | 10 / 0 | +1.74 | +1.05 – +1.84 | +0.857 – +2.56 | +1.65 | yes | 10 / 0, +1.96 (+1.71 – +2.32) |
| slope=3 (a downsloping ST segment vs an upsloping ST segment) | level vs reference | 9 | 5 / 4 | +0.233 | -0.116 – +0.892 | -0.588 – +1.22 | +0.282 | no | 7 / 3, +0.411 (-0.588 – +1.65) |
| ca (number of vessels coloured by fluoroscopy) | vessels | 10 | 10 / 0 | +1.54 | +1.38 – +1.65 | +1.24 – +1.83 | +1.35 | yes | 10 / 0, +1.54 (+1.25 – +1.74) |
| thal=4 (thallium stress test result coded 4 (meaning not recorded) vs a normal thallium scan) | level vs reference | 7 | 7 / 0 | +2.35 | +1.68 – +2.54 | +1.19 – +4.29 | +2.26 | mostly | 9 / 0, +2.36 (+0 – +7.52) |
| thal=6 (a fixed defect on the thallium scan vs a normal thallium scan) | level vs reference | 7 | 1 / 6 | -0.712 | -0.874 – -0.555 | -1.97 – +1.19 | -0.577 | mostly | 0 / 9, -0.825 (-2 – +0) |
| thal=7 (a reversible defect on the thallium scan vs a normal thallium scan) | level vs reference | 8 | 7 / 1 | +1.06 | +0.656 – +1.2 | -1.33 – +1.8 | +0.963 | mostly | 10 / 0, +1.38 (+1.01 – +1.88) |

Variable classes across the 10 networks (after the binary one-step rule):

| variable | exact_linear | mixed | nonlinear | not used | support_linear |
|---|---|---|---|---|---|
| age | 5 | 4 | 1 | 0 | 0 |
| ca | 7 | 3 | 0 | 0 | 0 |
| chol | 9 | 1 | 0 | 0 | 0 |
| cp | 0 | 8 | 0 | 0 | 2 |
| exang | 2 | 4 | 2 | 2 | 0 |
| fbs | 2 | 6 | 0 | 2 | 0 |
| oldpeak | 1 | 2 | 7 | 0 | 0 |
| restecg | 5 | 3 | 1 | 1 | 0 |
| sex | 2 | 7 | 1 | 0 | 0 |
| slope | 3 | 7 | 0 | 0 | 0 |
| thal | 1 | 9 | 0 | 0 | 0 |
| thalach | 2 | 4 | 4 | 0 | 0 |
| trestbps | 2 | 3 | 3 | 2 | 0 |
