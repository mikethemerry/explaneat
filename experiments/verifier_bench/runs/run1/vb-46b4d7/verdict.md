# Verdict vb-46b4d7 (MONK-2, genome 60ee493c)

**Weakened.** All claims, rules, counts and ablations reproduce, including over all 432 configurations; only the stated size of the veto is off.

Time: 23:30 to 23:33 (3.2 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.atleast3_of_five_veto.hypothesis | L1 weakened | veto -70 or less when it fires | smallest firing value -58.2 (the stated formula with the stated minimum 5.6 gives -58.2); still forces the output to 0 | exits over training rows and all 432 configurations |
| record.summary (and root hypothesis) | L1 weakened | down to -70 for 3 or more | -58.2 to -269.3 | as above |

Upheld:
- ann_32 (whole model): over all 432 configurations, negative for at most one and for three or more first values; 113 of 114 and 23 of 38 for exactly two; 96.3% agreement with count5 == 2; 88.0% with the target rule.
- ann_28 (veto): fires on exactly the 112 training rows with three or more first values; removing it drops accuracy to 0.562.
- ann_29 (gate): zero exactly when a5 = 1 and a3 = 2 (40 rows); bonus ranges match; accuracy 0.690 without it.
- ann_30 (part on a5): exact a5 in {2, 4} indicator.
- ann_31: zero on every row. ann_27: linear, claims reproduce.

Flag: the model treats a5 != 1 as a bonus rather than a5 = 1 as one of the "first values" (already flagged).
