# Verdict vb-efb747 (Heart, genome b19b9536)

**Refuted.** The ST depression hinge is said to act only with a flat ST segment, but it reads only ST depression and acts the same for every slope. The vessel count boost hypothesis only restates its formula. All numbers reproduce.

Time: 23:55 to 23:56 (1.6 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_33.hypothesis[0] | L2 refuted | only for patients with a flat ST segment | reads only oldpeak; +1.411 at 3 mm for both upsloping and flat | vars; probe oldpeak x slope |
| drafted.readdown.ann_30.hypothesis[0] | L4 weakened | weights vessels and thal 4, adds a constant, passes it on | not meaningful (restates the formula): each extra vessel adds the same risk on top of the direct effect (20% -> 45% at 2 vessels) | L4 reading; ablate |

Upheld:
- ann_35 (whole model): core SD 4.96, readouts 1.41; readouts at means AUC 0.9437 -> 0.9295.
- ann_33 numbers: on above about 1.5 mm, 70 active, disease 71% vs 33%, ablation 0.9400.
- ann_31 (no exercise angina, by sex): three values 0 / 0.959 / 3.702 with the stated counts.
- ann_32 (fixed defect with normal sugar): 3.386 for 10 patients, 0.018 for 5; ablation 0.9388.
- ann_34: inert.

Flags (already listed): a fixed thallium defect and high fasting blood sugar lower risk.
