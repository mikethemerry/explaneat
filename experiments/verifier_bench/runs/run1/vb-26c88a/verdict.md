# Verdict vb-26c88a (Heart, genome 1f2cf7ea)

**Weakened.** Everything reproduces except one side figure in the ischaemia pathway's technical hypothesis.

Time: 23:39 to 23:41 (2.1 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.ischaemia_pathway.hypothesis (and evidence[1]) | L1 weakened | the 35 patients for whom 290 opens from low BP alone have mean predicted risk 0.13 | mean 0.181 (median 0.057; observed rate 0.20); the counts 135 / 100 / 35 are right | node values over training rows |

Upheld:
- ann_40 (whole model): all 18 direct-edge weights and the 4 readouts match; linear core alone AUC 0.932; logit SD 3.18 vs 0.78; pathway removal -0.0056 AUC.
- ann_39 (ischaemia pathway): convex in ST depression, larger with a reversible defect; age term from about 53 years; 1834 and 1987 activity shares match.
- ann_36 (ST depression and thallium signal): off up to about 0.9 mm, then steady; acts on 117 of 242.
- ann_38 (thal:6 relay): affine and near-inert, arithmetic matches. ann_37: an exact copy of ann_36.

Flags (already listed): high fasting blood sugar lowers risk; ST depression counts more with a normal resting ECG.
