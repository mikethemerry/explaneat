# Verdict: lad-da6de (genome 63b736bc), Connect-4 dense ReLU MLP

**Overall: weakened.** Every stated number reproduces and the structure (one leaf, one root, full coverage, exact function preservation) is sound. The centre-control reading holds at population level. But some probes behind it are outside the data support, the top rows go against the centre reading, and about a quarter of one override group does not need the thresholds.

## Weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.board_detectors.hypothesis | L3 | "These overrides are the position-specific corrections that the thresholds add" | Group 1 (111 rows): yes, the straightened model puts all of them above 1/2. Group 2 (1450 rows): the straightened model still puts 24% (352) above 1/2 | an2.py, affine fit of each unit |
| operations.board_detectors.hypothesis | L3 | "a1 change carries it: d1 alone gives 68%"; "d1 alone 39%" | The swaps reproduce and keep 4 x and 4 o pieces. The one-piece flips make 5-3 or 3-5 boards, and no training row has those, so these numbers are extrapolation. In-support, a1 is the most negative common square (-0.21 mean change in a swap) | an2.py, an3.py |
| operations.win_vote.hypothesis | L3 | row 34, o's a1 piece turned to x: 70% to below 1%, plus the unit values | Every value reproduces, but the board has 5 x and 3 o pieces, which never occurs in training. It shows the mechanics, but it is not evidence about trained positions | an2.py |
| record.summary (and overall_function) | L3 | x in the centre columns raises it; x on the outer bottom-row squares lowers it; "much like the classic count" | True for rows 1-4 of the centre and for the corners a1 and g1. b1 (-0.07) and f1 (-0.02) are near neutral. x high in the centre lowers the prediction (c6 -0.52, d6 -0.34, c5 -0.22, d5 -0.14), against the table. The score explains 55% of the logit variance; score alone AUC 0.820 vs 0.948 | an3.py, an.py |

## Upheld
- Record: coverage 1.0 / 1.0, hierarchy complete, function preservation 0, closed-form error 0.069. Closed-form coefficients match the weights. The resisted text (0.020, 80% to 99%, the 26-letter split limit with 32-way fan-out) checks out.
- board_detectors (ann_42): all subgroup counts and means (centre, column d, outer bottom row, six square-value bands) and the linearisation numbers (0.948, 0.854, unit 5 at 0.020) reproduce. The overrides' agreement with the outcomes holds on the test split too (29% vs 32%; 75% vs 77%). "No unit is a clean feature" passes an own check: correlation with the score is at most 0.42, and the best single-unit AUC is 0.656. Leaf meaning: **partial**.
- win_vote (ann_43): the positive and negative unit lists match the output weights. The ablations reproduce (0.65 / 0.625; -0.34 / 0.728).

## Flags
None. Note: the recheck report has no `target_hash` for a record target, so it is null in verdict.json.
