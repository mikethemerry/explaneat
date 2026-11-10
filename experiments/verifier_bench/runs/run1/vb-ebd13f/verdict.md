# Verdict vb-ebd13f (MONK-2 genome 60ee493c)

**Overall: refuted.** The plain hypothesis for the part on a5 is contradicted by the model, and two technical hypotheses carry numbers that do not reproduce. Coverage, function preservation, closed forms and all 65 claims reproduce.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_30.hypothesis[0] (refuted) | L2 | "decides the outcome whenever a5 = 2 or 4: it always turns the prediction to 'no'" | 64 of 173 training cases with a5 = 2 or 4 are predicted 'yes'; switching the part off flips 1 of 345 predictions; it is a fixed -0.72 on the score. Typical case with a3 = 2, a5 = 2: 100% | subgroup "a5 in ['2','4'] and risk >= 0.5"; ablate ann_30; probe a5 x a3 |
| operations.atleast3_of_five_veto.hypothesis (refuted) | L1 | "with two it is at least +5.6 ... 274 > 0 exactly when two or more ... (112 rows)" | Node 274 pre-activation is -3.4 to -0.2 with two first values and +5.6 to +8.8 with three; it fires on the 112 rows with three or more (236 rows have two or more) | pre-activation by count; subgroup counts; enumeration of 432 configurations |
| operations.a5_a3_bonus_gate.hypothesis (refuted) | L1 | "1223 = 0 exactly when a5 = 1 and a3 = 2 (52 rows)" | Exactly when, yes, but on 40 training rows (its evidence record says 40) | subgroup "a5 == '1' and a3 == '2'" |
| operations.atleast3_of_five_veto.hypothesis (weakened) | L1 | "a penalty of -70 or more" | -58.2 to -93.3 with three first values (typical case -66.4) | enumeration; probe |
| record.summary, operations.two_of_five_peak.hypothesis (weakened) | L1 | veto "down to -70" / "<= -70" for three or more | same as above; sign conclusion holds | enumeration |

Upheld:
- ann_32 (whole model): additive composition. Score is below 0 for every configuration with at most one or at least three first values. 113/114, 23/38, 96.3% / 95.9% agreement, accuracy 0.887 and AUC 0.925, ablation accuracies 0.562 and 0.690 and rule agreement 88.0% all reproduce.
- ann_27 (attribute sum): linear, with per-value credits as stated. Node 1948 never clips (0.70/0.89).
- ann_28 plain hypothesis and evidence record (three-of-five veto): reproduce.
- ann_29 plain hypothesis and evidence record: reproduce. The 1717 sub-unit moves 1223 by at most 0.07.
- ann_30 technical hypothesis: reproduces (1.198 indicator, -0.72, 0.887 to 0.890).
- ann_31: inert. Exit is 0 everywhere and the maximum pre-activation is -0.55.
- Lead, blocks and worth-checking items: all numbers are in the cited claims and the directions agree with the probes.

Not checked: the sensitivity figures in the ann_29 hypothesis (a5:1 1.65, a3:1 1.25).

Flags: none. MONK-2 is synthetic, so no domain conflict applies.
