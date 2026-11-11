# Verdict: dv-b0aeaa (genome b19b9536, Heart Disease)

Started 2026-11-11 16:18:55 +13:00, finished 16:24:30 +13:00 (about 6 minutes). Scope: the plain-language draft and the claims it cites.

**Overall: weakened.** All 78 claims and every number in the draft reproduce, and the probes confirm the thresholds, counts and directions. Two sentences are worded in a misleading way.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_32.evidence[1] (also blocks, annotations) | L2 | Being a woman lowers risk "even more so" without exercise angina: 20% to below 1%, and 27% to about 1% with it | Holds on the log-odds scale (drop of 4.89 vs 3.65). In the percentages the sentence quotes, the drop is smaller without exercise angina (19.5 vs 26.4 points) | probe sex x exang, typical patient: 19.7 -> 0.2%, 27.4 -> 1.0% |
| drafted.readdown.ann_36.evidence[0] (also blocks, annotations) | L2 | "These inputs add up independently, biggest first" | Typical angina (-4.43) and being a woman (-3.65) are larger than flat ST (3.60), cholesterol and blood pressure but come last, and the lowering inputs are not in size order either | effect sizes of C25 to C33 |

Upheld:
- ann_37 (whole model): typical risk 19.7%; vessels 19.7, 81.9, 98.8% at 0, 1, 2; flat ST 90.0%; resting ECG changes nothing; six parts added.
- ann_36 (linear terms): directions and steady steps confirmed (2.3 log-odds per vessel).
- ann_34 (ST hinge): off at 1.4 mm, on from 1.5 mm; 45.4% at 2.8 mm, 61.6% at 3.5 mm; acts on 70, 50 with disease.
- ann_32 (sex by exercise angina gate): acts on the 160 without exercise angina (47 events); -1.67 for women vs -0.43 for men.
- ann_33 (fixed defect gate): 15 patients, 9 events; typical fixed-defect case 99.5% without the part and 89.9% with it.
- ann_31 (vessels and thallium code 4): constant 0.60 per vessel; code 4 effect stated with n = 2.
- ann_35 (atypical angina flag): max change 1.03%, AUC 0.9437 -> 0.9434.
- Hypothesis revisions: all consistent with the checks above.

Flags (domain, correctly raised in worth-checking): fixed thallium defect lowers risk (observed rate 60% vs 43%); high fasting sugar lowers risk (45% vs 44% observed); thallium code 4 raises risk on 2 rows.
