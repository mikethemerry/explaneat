# Verdict vb-2b8edc (genome 0057df7a, Heart Disease)

**Refuted.** The ST depression hinges are said to add risk only above about 1.1 mm, but the model starts adding at about 0.8 mm. The flat-ST-segment part is said to lower risk and make ST depression count less, but it does the opposite. Everything else holds: the record, the structure, the closed forms, the technical hypotheses and 71 of 72 claims.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| claims.C41 (ann_33) | L1 | threshold 1.1 mm (1.07) | recomputed 0.777 mm; part off at 0.6 mm, on at 0.8 mm; kinks at 0.78 and 1.67 mm | recheck; `probe --sweep oldpeak=0:2.4:13` |
| drafted.readdown.ann_33.hypothesis[0], .evidence[1], drafted.blocks[2] | L1 | "adding risk only once it rises above about 1.1 mm" | adds risk from about 0.8 mm | as above |
| drafted.readdown.ann_32.hypothesis[0] | L2 | "lowers the predicted risk mainly for flat ST segment, makes ST depression count less" | raises risk: score contribution +0.09 to +0.68 with a flat segment, rising with ST depression (log-odds gain 0 to 2.8 mm: +1.20 with a flat segment vs +0.79 upsloping); ablation mean risk change +0.012; the cited claims C44 and C48 say "raises" and "amplified" | `probe --sweep oldpeak=0,0.8,1.6,2.8,4 --sweep slope=1,2`; `ablate --part ann_32` |
| drafted.readdown.ann_31.evidence[1] | L3 (weakened) | "for patients with a reversible defect ... it rises in a jump" | with a reversible defect the age effect through the part is near flat (38.9% to 39.6% at 175 bpm, then flat) | `probe --set thalach=175 --sweep age=40:70:7 --sweep thal=3,7` |

Upheld:
- **Record**: coverage 1.0/1.0. Function error 2.2e-16. Linear core AUC 0.930 vs 0.943 for the full model. Logit SD 3.28 vs 0.91. The affine reductions and the counts of active relays all reproduce.
- **ann_34 (root)**: additive composition; the per-readout AUC drops reproduce (-0.0056, -0.0016, -0.0007, -0.0004; -0.0064 for the oldpeak terms together). The SHAP ranking was not checked.
- **ann_31**: the switch-on rule reproduces (on at 162 bpm, off at 161 bpm for the typical patient). The gate is active for 80 patients (50.7% of those without a reversible defect, 3.3% of those with one), with outcome rates of 11% vs 60%. AUC 0.9434 to 0.9378; the gate is protective.
- **ann_33**: the technical hypothesis is correct (kinks at z = -0.218 and 0.539, slopes 0.289 and 0.605; active 121 and 58, with outcome rates of 64% and 79%). Only the plain threshold is wrong.
- **ann_32**: the technical hypothesis is correct (readout +0.225; active for 98% of flat-segment patients vs 7% of the others; corr with oldpeak 0.69; AUC 0.9418). Only the plain hypothesis is wrong.

Flags (L4, already noted by the explainer):
- High fasting blood sugar lowers the predicted risk (28% to 14%).
- A fixed thallium defect lowers the risk (n = 15).
- Cholesterol has a negative term inside the flat-slope gate.
