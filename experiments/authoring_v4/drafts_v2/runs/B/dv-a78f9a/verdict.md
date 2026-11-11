# Verdict dv-a78f9a (plain-language draft)

Genome 1f2cf7ea-5c30-452a-9277-0b4043e888d9, Heart Disease. Started 2026-11-11T03:24:48Z, finished 03:32Z (7 min).

**Overall: weakened.** Every number in the draft reproduces (88 of 88 claims; 244 prose numbers, none off), and nothing is refuted. The five weakenings are about wording and scope.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_34.hypothesis[0] (also blocks[1] s3, annotations ann_34[4], revision ann_34); C64 | L3 | ST depression adds risk "once it passes about 0.9 mm" | True for a normal resting ECG (unit opens between 0.8 and 1.0 mm). With hypertrophy and no reversible defect (74 patients) it opens near 1.75 mm. C64's own reference case has hypertrophy | probe oldpeak 0:2 by restecg, `--nodes 204_a`; subgroup |
| drafted.blocks[1] s1 (and ann_34/ann_37 evidence, annotations); C61, C51 | L2 | "(... for some patients with a normal resting ECG, 17 of the 44 sampled, it rises steadily)" after the "any other thallium result" clause | The 44 are the patients with a reversible defect and a normal ECG, so the exception belongs to the reversible-defect group | subgroup `thal == '7' and restecg == '0'` gives n = 44 |
| drafted.blocks[1] s2 (and ann_34 evidence[4], annotations ann_34[3]); C62 | L2 | The same note after the "any other resting ECG" clause | The exception sits inside the normal-ECG group | same subgroup |
| drafted.lead[1], blocks[1] s0; C24, C57 | L2 | The pathway combines ST depression with thallium, ECG, blood pressure and age | C57 also lists chest pain type. Atypical angina raises the pathway's addition from 0.628 to 0.744 at 2 mm | probe `--set oldpeak=2 --sweep cp=1,2,3,4` |
| drafted.hypothesis_revisions.ann_37; C52 | L3 | Blood pressure makes the rise "a little smaller, not a large discount" | From 110 to 170 mm Hg the pathway's addition at 2.8 mm falls from 1.64 to 0.85 on the score, about half, and the risk rise falls from 29 to 20 points | probe `--sweep oldpeak=0,2.8 --sweep trestbps=110,130,150,170` |

Upheld:
- ann_38 (whole model): the lead's typical patient (19.4%), the additive structure, vessels 19% to 88%, ST slope, chest pain and sex effects, and the uneven blood-pressure curve.
- ann_36 (inputs that add up): every block 0 number. In the whole model blood pressure gives 12.6% to 32.1% for 110 to 155 mm Hg.
- ann_37 (ST-depression pathway): hypothesis and evidence (206 of 242, 60% to 62%), the BP split at 130 mm Hg, and the age term of about 2.7 points. Weakened only by the rows above.
- ann_34 (ST-depression score): 117 of 242, 78% to 92%, the formula, and the reversible-defect and normal-ECG amplification. Weakened only by the rows above.
- ann_35 (fixed-defect relay): inert (largest risk change 0.0037).

Flags (L4, domain only):
- Fasting blood sugar above 120 mg/dL lowers the predicted risk (C13).
- A fixed thallium defect gets a lower risk than a normal scan, from 15 patients (C14).
- Thallium coded 4 raises the risk only because the normal-scan term drops out (2 patients, C08).

Note: the recheck report from this tool version has no target_hash, so verdict.json leaves it null.
