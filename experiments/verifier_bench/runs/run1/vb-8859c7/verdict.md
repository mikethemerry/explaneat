# Verdict vb-8859c7 (MONK-2, genome e324239d)

**Refuted.** The ramp's active count is wrong (226 stated, 206 in the model), and its technical rule for a1 = 1 says "at most one" first value where the unit is on for at most two. The rest reproduces.

Time: 23:41 to 23:43 (1.6 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| claims.C27 | L1 refuted | ramp acts on 226 of 345 | 206 (59.7%) | recheck; ablate |
| drafted.readdown.ann_24.evidence[0], drafted.blocks[0] | L1 refuted | 226 of 345 | 206 of 345 | recheck prose |
| operations.few_first_values_ramp.hypothesis | L3 refuted | a1 = 1: on iff at most one of a2, a3, a4, a6 is first | pairs give pre-activation 1.22 to 4.01 (on); triples -7.0 to -4.45 (off): at most two | all 432 configurations |

Upheld:
- ann_28 (whole model): positive rates by a1 and count (100 / 74 / 0 for a1 = 1; 29% at 4 for a1 = 2; a1 = 3 all 107 positive), 85.5% predicted positive, accuracy 0.472, logit AUC 0.734.
- ann_24 behaviour: falls 6.8 to 9.3 per first value; removing it flips 112 decisions, AUC 0.734 -> 0.537.
- ann_27 (direct sum), ann_25 (a5/a3 unit, 3 flips), ann_26 (tiny a6 unit, 0 flips).

Flag: no lower edge, so the model disagrees with the exactly-two rule for 0 or 1 first values (already flagged).
