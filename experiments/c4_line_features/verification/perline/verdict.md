# Verdict: c4lf-perline-default-s2 (genome 94fb8bdc-784b-4f75-a63c-25b1ad31da15)

**Overall: weakened.** Every stated number, count, sweep, ablation and the gate rule reproduce (0 refuted, 10 weakened, 15 upheld); the weaknesses are readings that restate closed forms, one size comparison that holds per sd but not per line, and a few statements outside the data.

Verified 2026-11-12 21:38 to 22:00 (+13:00), 22 minutes, 80 statements checked. Recheck: 0 mismatch, 0 needs review (recheck.json; no target_hash was returned for a record target).

## Weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.game_terms (V7) | L2 | the o centre-vertical positions "still have above-average x-win rates" | o on open d1-d4: x wins 46.4% vs 70.0% in the rest; o on open e1-e4: 65.0% vs 66.0%. True only for the x lines (c3-f6 75.2% vs 64.4%, b6-e3 70.0% vs 65.4%). The "not a data pattern" conclusion still holds | `subgroup --where '`o_line_d1-d4` >= 1'` |
| record.leaves (V8) | L4 | all six leaves graded "meaningful" | two score 1.0 with the input itself as concept; the record's own text calls xThrEvenUnit "described, not explained"; my grades: 1 meaningful, 2 partial, 3 closed-form-only | reading against the concepts used |
| operations.xDiagD6G3Gate.hypothesis (V11) | L4 | "x on d6-g3 removes a small bonus" | restates relu(0.0138 - 1.47 z) x 0.043; +0.02 logit when x is absent from one line | - |
| operations.xB5E2vsOB3B6Gate.hypothesis (V13) | L3 | 81% at three x pieces on b5-e2; 35% at four o pieces on b3-b6 | both lines range 0 to 2 in training; the probe marks 3 and 4 as outside the data | `probe --sweep 'x_line_b5-e2=0:3:4'` |
| operations.xThrEvenUnit.hypothesis (V15) | L4 | linear rescaling of its input, "described, not explained" | an always-on ReLU (54045 of 54045 rows); no reading beyond the formula | `subgroup --where 'node_2829 > 0'` |
| operations.xLinesDirect.hypothesis (V17) | L3 | three x pieces on a1-d1 give 97% | a1-d1 ranges 0 to 2; 96.6% is extrapolation | `probe --sweep 'x_line_a1-d1=0:3:4'` |
| operations.xLinesDirect.hypothesis (V18) | L4 | rewards x's low and central open lines, penalises row-six diagonals; signed sum scores 0.80 | the reading in board terms alone (rows 1-3 and column d, minus lines touching row six) scores r2 0.356; the 0.80 is the formula's sign pattern over its 35 connected lines | `concept --part ann_161 --concept '<board-region sum>'` |
| operations.oLinesDirect.hypothesis (V20) | L2 | o's open horizontals on rows two to four are threats to x; diagonals "into the left-centre" | row four is mixed: a4-d4 -0.380, d4-g4 -0.178 but b4-e4 +0.148, c4-f4 +0.041; c5-f2 runs to the right side. The reading alone still scores r2 0.767 | closed-form signs; `concept --part ann_162` |
| operations.aggregatesDirect.hypothesis (V22) | L3 | open twos worth about as much as open singles | per sd yes (0.687 vs 0.686); per line one more x single moves the typical chance 49.2 to 53.8%, one more x two 49.2 to 62.2% (a two is worth about three singles); x_open_1 + 3 x_open_2 + 2 centre_x - o_open_1 - 3 o_open_2 scores 0.946 vs 0.817 | `probe --sweep 'x_open_1=5:13:9'`, `--sweep 'x_open_2=0:4:5'` |
| operations.aggregatesDirect.hypothesis (V23) | L2 | an o even-row threat is "nearly decisive" | typical case 49.2 to 8.8%, but on the 665 real rows with one the mean predicted chance is 23.2% (observed 22.9%) | `subgroup --where 'o_thr_even >= 1'` |

## Upheld

- record: coverage 1.0 / 1.0, hierarchy complete, preservation 8.88e-16, closed-form error 0.00858, 6 + 1 annotations, 1 split, 3 identity nodes; summary AUCs 0.912 / 0.841 / 0.871 / 0.869 / 0.910 / 0.9119; gate off on 3115 rows; resisted scores 0.80, 0.83, c3-f6 rate 0.75 vs 0.64; 72 unconnected inputs (66 per-line, open threes, total threats, parities).
- structure: every entry/exit set and every quoted weight matches the recomputed closed forms; readouts 0.88, 0.043, -0.098 recovered from probe contributions.
- winLogit (ann_164): sum of six parts at the output; linearised 0.9119; AUC 0.91197, accuracy 0.83797; open-twos sweeps.
- xDiagD6G3Gate (ann_158): on for all 49405 no-piece rows, off for all 4640 others; 49.2 to 48.7%; mean change 0.0020.
- xB5E2vsOB3B6Gate (ann_159): on iff x count >= o count on every training row and on the full 3x3 grid (ties (1,1) and (2,2) on); 3040 off rows at 56.2% vs 66.4%; mean change 0.0725, AUC 0.910. Reading meaningful.
- xThrEvenUnit (ann_160): always on; 49.2 / 55.1 / 61.0%; 594 rows at 82.8%, 54 with two; AUC 0.9119.
- xLinesDirect (ann_161): ten edges off 0.8488; a1-d1 49.2 / 75.0 / 90.3%; c3-f6 49.2 / 15.7 / 3.4%; 7274 rows at 75.2% observed and predicted; part off 0.8414, mean |change| 0.154; concept 0.8045.
- oLinesDirect (ann_162): ten edges off 0.8797; b2-e2 49.2 / 34.3 / 22.0 / 13.2%; d1-d4 49.2 / 64.9 / 77.9%; 9484 rows at 46.4% observed, 46.3% predicted vs 70.0%; part off 0.8688; concept 0.8291.
- aggregatesDirect (ann_163): x open twos 36.3 to 82.6%, o open twos 61.8 to 17.1%, centre 49.2 to 78.3%, o even threat 49.2 / 8.8 / 1.0%; 665 rows at 22.9% / 23.2%; part off 0.8706; concept 0.817.

## Leaf meaning

| Leaf | Grade | Why |
|---|---|---|
| xDiagD6G3Gate | closed-form-only | the formula in words; input itself is the concept |
| xB5E2vsOB3B6Gate | meaningful | who holds the b5 crossing; exact on all rows and the grid |
| xThrEvenUnit | closed-form-only | always-on rescaling; the author says so |
| xLinesDirect | closed-form-only | the 0.80 concept is the sign pattern; board-region reading alone r2 0.356 |
| oLinesDirect | partial | board-region reading alone r2 0.767; row four mixed; vertical discount is an offset |
| aggregatesDirect | partial | material reading confirmed in direction; equal worth of twos and singles wrong per line |

No flags.
