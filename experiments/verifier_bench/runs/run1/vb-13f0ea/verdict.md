# Verdict vb-13f0ea (Heart, genome 7c355ee7)

**Refuted.** Two plain hypotheses contradict the model: the high heart rate gate lowers risk (it is said to raise it), and the ST depression gate does not act more with a flat ST segment. All numbers reproduce.

Time: 23:51 to 23:52 (1.5 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_34.hypothesis[0] | L3 refuted | high heart rate raises the risk through the part | contribution 0 to 140 bpm, then -0.17 at 150 down to -2.91 at 200 bpm | probe cp=3, thalach sweep |
| drafted.readdown.ann_32.hypothesis[0] | L3 refuted | more so with a flat ST segment [C41] | flat and upsloping give the same contribution (+1.43 at 3 mm); downsloping switches it off; C41 is about a reversible thallium defect | probe oldpeak x slope |

Upheld:
- ann_35 (whole model): AUC 0.947, linear core 0.930, per-gate mean-ablations 0.938 / 0.940 / 0.945, core SD 4.90.
- ann_32 numbers: active 116 of 242 (83% with a reversible defect, 27% without).
- ann_33 (women with exercise angina): exact AND gate, truth table and rates match.
- ann_34 numbers: active 80% with non-anginal pain vs 10.5%; on from about 147 bpm.

Flags (already listed): a downsloping ST segment switches off the ST gate; high fasting blood sugar lowers risk.
