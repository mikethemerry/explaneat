# Verdict vb-eccf26 (Adult, genome 7a8237f3)

**Weakened.** All numbers, kinks, ablations and shapes reproduce; two sentences use a number from a claim they do not cite, and one row count is off by one.

Time: 23:43 to 23:47 (4.5 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_125.evidence[0], drafted.blocks[3] | L1 weakened | "less than about 50 hours" cited to C143 | 50 is right, but it is C142's threshold, not in C143 | recheck prose |
| operations.age_profile.hypothesis | L1 weakened | Thai respondents 27 rows | 26 rows (the same text also says 26) | frame count |

Upheld:
- ann_131 (whole model): additive; acc 0.858, AUC 0.912.
- ann_129 (backbone): indicator-scale effects (-2.87, -0.50, -1.05, -59, +30) match the weights; ablation AUC 0.790.
- ann_130 / ann_122 / ann_123 (capital gains): bends near $4.9k (switched off for Jamaica) and $5.2k seen in the probe; ablations 0.894, 0.903, 0.911.
- ann_121 (age): lowest at 46 to 47 years, higher either side; ablation 0.905.
- ann_125 (short hours): contributions 2.04 / 1.53 / 1.02 / 0.51 / 0.25 / 0 exactly; active 80%.
- ann_124, ann_128, ann_126, ann_127: claims and ablations reproduce.

Flags (already listed): rare one-hots as extreme switches; low grades lowering the chance of a low income.
