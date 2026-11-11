# Verdict: lad-be3c9 (genome 38f7d721)

**Overall: refuted.** Every number reproduces. One reading does not hold: the record says unit 2's "strongest regime" is the 43 blocked b1-c1-d1 positions where the model is wrong. In fact the unit is strongest where x wins every time. There are also two scope overclaims.

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| operations.xBottomLeftRow.hypothesis | L3 | refuted | The strongest regime (x on b1, c1, d1; 43 rows; model 77%, x wins 28%) is fitted and does not match the game | 43 / 77% / 28% reproduce. But those rows have mean unit value 2.73 and hold 0.5% of the unit's contribution, with none of its top 100 values. Its top pattern (x on b1 and c1, open ends, o on f1 and g1; n=150, mean 4.21) predicts 93% and x wins 100% | `verify subgroup --where "c_a1 == 0 and c_b1 == 1 and c_c1 == 1 and c_d1 == 0 and c_e1 == 0 and c_f1 == -1 and c_g1 == -1"`; pattern table (an4.py) |
| record.resisted | L3 | refuted | Same sentence | Same | Same |
| record.summary | L3 | weakened | x on column d raises the win chance and o there lowers it | True for d1-d4, reversed for d6 (x 32% / empty 62% / o 83% on probe; counterfactual 0.466 vs 0.826; observed 43.5% vs 74.0%) and slightly for d5 | `verify probe --sweep c_d6=-1,0,1`; subgroups c_d6 == ±1 |
| operations.boardEvaluation.hypothesis | L3 | weakened | "Lower centre (the centre column d ...)" | As above | As above |
| operations.xHighColumnD.hypothesis | L4 | weakened | A brake on x on d3 "rather than a separate idea" | Only 31% of its total contribution is in x-on-d3 rows; 69% is in the 24318 other active rows | Contribution shares (an3.py) |

**Upheld:** record coverage, function preservation, closed forms and counts (L1).
- ann_84 xCentreAboveOBottom: all numbers hold. The "stacked" wording describes two additive preferences.
- ann_86 xCentreThirdRow: holds. Minor: d1 and e1 outweigh the listed d4.
- ann_87 oBottomCentre, ann_88 xOnC2, ann_89 xLowColumnD and ann_91 xRightOCentre: every probe, subgroup and AUC holds.
- Evidence records: all on the full training split, with matching populations (L2).

**Leaf meaning:** meaningful: units 1, 2, 3, 4, 5 and 6. Unit 2 is more meaningful than the record says. Partial: units 7 and 8.

**Flags:** none.
