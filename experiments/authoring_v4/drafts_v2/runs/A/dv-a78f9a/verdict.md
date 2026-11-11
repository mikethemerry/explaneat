# Verdict dv-a78f9a (genome 1f2cf7ea, Heart Disease)

Started 2026-11-11T03:18:49Z, finished 03:22:57Z (4 minutes). Scope: the plain-language draft and the claims it cites.

**Overall: weakened.** Every claim (88 of 88) and every prose number reproduces, and no direction is wrong. Four wordings overreach.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_34.evidence[2] (also blocks[1], annotations.ann_34[4], hypothesis_revisions.ann_34) | L2 | ST depression counts above about 0.9 mm, and "for 88 of 242 patients it already acts below that" | 88 counts swept curves that move below 0.9 mm, not real patients: of the 132 with ST depression below 0.9 mm the part acts on 23. The threshold runs 0.7 to 1.7 mm across patients, about 1.8 mm with hypertrophy, and 0 mm with a reversible defect | subgroup `oldpeak < 0.9`; probe oldpeak x restecg, x thal |
| drafted.annotations.ann_34[4] (also annotations.ann_37[5]) | L4 | "for a typical patient with a fold", 26% to 53% | Numbers are right, but the phrase is jargon, and this case (hypertrophy) only opens at about 1.8 mm, so it does not show the 0.9 mm threshold | probe oldpeak x restecg |
| drafted.blocks[1] (also the matching readdown and annotation sentences) | L3 | 19% to 48% "with any other thallium result" | That is the normal scan. A fixed defect gives 10.5% to 31.9%, coded 4 gives 36.5% to 68.9% | probe thal=3,4,6,7 x oldpeak 0, 2.8 |
| drafted.lead[1] (also blocks[1] sentence 1) | L2 | the pathway combines ST depression with thallium, ECG, blood pressure and age | It also reads chest pain type (vars; C57) | verify vars |

Upheld:
- ann_38 (whole model): typical risk 19.4%. The two parts add up. The ST-depression effect accelerates.
- ann_36 (inputs that add up): all effect sizes reproduce, and their directions match the closed-form signs.
- ann_37 (pathway): acts on 206 patients. The rise is steeper with a reversible defect and smaller at higher blood pressure. The age effect starts at about 53 years and stays small. Weakened only by V1, V3 and V4.
- ann_34 (ST-depression score): ST depression rises more with a normal resting ECG. Weakened by V1 to V3.
- ann_35 (fixed-defect relay): has a negligible effect (mean +0.0001 on the score).

Flag (L4): the model gives a fixed thallium defect and high fasting blood sugar a lower risk. The draft already marks both as worth checking.

Note: the recheck report for this run has no `target_hash`, so `target_hash` is null.
