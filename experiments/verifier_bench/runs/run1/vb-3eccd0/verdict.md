# Verdict: vb-3eccd0 (Heart Disease, genome 7c355ee7)

**Overall: weakened.** Every record number, closed form, count and all 78 claims reproduce; five plain-language statements are imprecise about shape or scope. Nothing is refuted.

Started 2026-11-10T23:14:46+13:00, finished 2026-11-10T23:23:27+13:00 (9 minutes).

## Weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_32.hypothesis[0] | L3 | ST depression raises risk "in an uneven way" | A clean hinge: nothing at all up to about 1.23 mm (typical risk 11.9% flat), then a steady rise in the score (13.3% at 1.4 mm to 35.9% at 3.0 mm) | `probe --sweep oldpeak=0:3:16` |
| drafted.readdown.ann_32.evidence[1] (same wording in blocks[1], ann_32 evidence[2-3], ann_35 evidence[1], worth_checking[5]) | L3 | "uneven, mainly once above about 1.2 mm (between 0.2 and 1.2 mm for different patients)" | Numbers right, but "mainly" should be "only" and the rise is steady. Folds are 0.11, 0.17 or 1.23 mm; downsloping patients fold at 2.9 to 5.1 mm | probe, per-patient folds from the closed form |
| drafted.readdown.ann_32.evidence[3] | L3 | raises risk "only if the ST segment does not slope down"; with downsloping, risk stays at 42% | True only up to 2.8 mm. The typical downsloping patient goes 41.6% at 5.0 mm to 59.1% at 6.0 mm; the part is active in 4 of the 17 downsloping patients | `probe --set slope=3 --sweep oldpeak=0:6:13`, `subgroup slope=='3'` |
| drafted.readdown.ann_34.hypothesis[0] | L3 | the part does nothing until about 147 bpm, stated for everyone | 147 bpm holds only with non-anginal chest pain (70 patients). For the other 172 the fold is at about 174 to 175 bpm: the typical asymptomatic patient is off at 175 and on by 180 | `probe --sweep thalach=130:190:13 --sweep cp=3,4` |
| drafted.readdown.ann_34.evidence[3] | L2 | an abnormal thallium scan raises the risk more without non-anginal chest pain | A whole-model contrast, given without "across the whole model". Through this part an abnormal scan slightly *lowers* the risk (-0.007 / -0.055 log-odds), and the gap (1.47 vs 1.34) comes from other terms | C56 group fields, closed form |

## Upheld

- Record: coverage 1.0/1.0, hierarchy complete, preservation 2e-16, closed-form error 0.0269, op counts, splits, unused inputs, summary (19 direct edges, core SD 4.90, gate SDs 0.775/0.558/0.487, AUC 0.9467 full / 0.9299 core / 0.9382, 0.9405, 0.9446 mean-ablated). All upheld.
- ann_35 (whole model): bias, all direct weights, gate weights, accuracy 0.876, AUC 0.947, RMSE 0.293 and the prose all reproduce. SHAP values were not rechecked.
- ann_32: closed form, 47.9% active (83.3% with thal:7, 27.0% without), max 4.72 and the thal:7 amplification (27% to 76% vs 12% to 32%) are upheld. fbs above 120 mg/dL amplifies about as much.
- ann_33: the AND gate is exact. It is active in 19 of 19 women with exercise angina and 0 of the other 223. The truth-table counts and rates, the -0.004 pre-activation for men with exercise angina, +2.07 logit and -2.18/+1.14 all reproduce.
- ann_34: 3818 takes the values {0, 0.702}. The part is active in 80% of cp:3 patients and 10.5% of the rest (74 of 242), max 2.88, threshold shift -0.076.

## Flags (L4, domain)

- Fasting blood sugar above 120 mg/dL, a fixed thallium defect and typical angina all lower the predicted risk. The model really does this, but it is clinically odd. The explanation already flags it.
- A downsloping ST segment suppresses the ischaemia gate (slope:3 weight -1.08). This is clinically counter-intuitive, and the explainer notes it.
