# Verdict vb-e37014 (genome 1f2cf7ea, Heart Disease)

**Overall: refuted.** Coverage, function preservation, closed forms, all 79 claims and all prose numbers reproduce, but the ischaemia pathway's plain hypothesis states the blood-pressure interaction backwards, the root's technical hypothesis makes a false structural statement about thal:7, and one stated mean risk does not reproduce.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_39.hypothesis[0] (C34, C42; C43) | L1 refuted | ST depression raises risk "more so when resting blood pressure is high" | Opposite. Typical patient, 0 to 2.8 mm: part adds +0.03 to +1.09 at 115 mm Hg, only +0.03 to +0.63 at 150 mm Hg. C43 also says stronger at or below 130 mm Hg; cited C42 is about thallium | `probe --hold typical --sweep oldpeak=0:2.8:8 --sweep trestbps=115,150` |
| operations.heart_disease_logit.hypothesis | L1 refuted | "oldpeak and thal:7 influence the output ONLY through these gates" | thal:7 has a direct edge to the output (weight -0.065), listed in the same hypothesis's linear core | structure dump |
| operations.ischaemia_pathway.evidence[1] (and .hypothesis) | L1 refuted | 35 patients open gate 290 from low BP alone, "mean predicted risk 0.13" | Count 35 reproduces; mean predicted risk is 0.181 (median 0.057) | forward pass, 242 training rows |
| drafted.readdown.ann_36.hypothesis[0] (C49, C55); also evidence[1], blocks[2] | L3 weakened | steady from the start "for patients with a normal resting ECG or a reversible defect" | Holds only with both (44 patients). Typical patient (normal ECG) has the 0.9 mm hinge; reversible defect + LVH ECG is off until about 0.7 mm | `probe --set thal=7 --sweep oldpeak=0:1.2:7 --sweep restecg=0,2` |
| drafted.readdown.ann_36.evidence[2], [3] (C53, C52); blocks[2] | L2 weakened | "ST depression counts more with a normal resting ECG (19% to 48% vs 26% to 37%)" | Direction right for this part, but these are whole-model risks. This part carries about a quarter of the difference (0.49 vs 0.27 log-odds; ischaemia pathway 0.86 vs 0.27) | `probe --sweep oldpeak=0,2.8 --sweep restecg=0,2` |
| operations.ischaemia_pathway.hypothesis | L2 weakened | age term adds risk "only" when 290 is quiet | Age node also active for 17 of 135 patients with 290 open (12.6%), as the evidence itself says | forward pass |
| record.summary | L2 weakened | "an age term active only without ischaemia evidence" | Same as above; other summary numbers reproduce | forward pass |

**Upheld parts**
- ann_40 (whole model): weights, 18 direct edges, AUC 0.942 / 0.932, logit SD 3.18 vs 0.78, readout-removal AUC changes, plain hypothesis and evidence all reproduce (except the thal:7 statement above).
- ann_39: gate weights, 1834 threshold 1.78, shares 56% / 26% (78% diseased) / 64% vs 13%, 203 of 242 acted on, age hinge near 53 years, the block paragraph's low-BP direction.
- ann_36: closed form, 48% active, gradients, correlation 0.88, 117 of 242 (77 diseased), hinge near 0.9 mm (0.7 to 1.7 mm).
- ann_37 (copy): identical weights, +1.088 into 290, ablation AUC 0.942 to 0.937.
- ann_38: always-on affine relay; ablation changes risk by at most 0.004 to 0.006, AUC unchanged.

**Flags (L4, domain review)**: high fasting blood sugar lowers risk (C18); typical angina lowest risk (C09, C14, C22); fixed thallium defect lowers risk, n = 15 (C20); ST depression and reversible defect count more with a normal resting ECG (C53, C54); ST depression counts more at lower blood pressure (C43).

Not checked: the whole-model SHAP ranking (no SHAP tool in the verify CLI).
