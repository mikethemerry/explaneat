# Verdict: lad-97ea3 (genome 7ca33eef, Connect-4)

**Refuted.** Every stored number reproduces and the structure is sound. But the record's overall reading puts too much weight on d1: it says o on the bottom centre square d1 is the model's strongest single penalty, and the model contradicts that. Two leaf hypotheses misdescribe their squares.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.overall_function | L3 | refuted | Column d matters "especially on d1", and o on d1 is "the strongest single penalty". | Of the four lower squares in column d, d1 has the smallest effect when its colour is flipped from o to x: d1 +1.94 log-odds, against d2 +4.25, d3 +7.54 and d4 +3.25. In legal positions, swapping colours between d1 and d3 (4457 rows) favours x on d3 by 4.31 log-odds; between d1 and d2 (9260 rows) it favours x on d2 by 1.50. | colour flips and legal swaps over the training rows (analyse.py, analyse2.py) |
| operations.root_win_score.hypothesis | L3 | weakened | "Strong reward/penalty" on d1. | The centre-column reading holds: column d has the widest spread of the seven columns, 98% / 67% / 27%. The d1 emphasis is overstated, as in the row above. | same as above, plus column-control subgroups |
| operations.bottom_row_centre_B.hypothesis | L3 | weakened | Rewards x in the bottom-row centre over x near the left edge (d1 against b1). | The part's bottom-row profile, as the change from flipping o to x: c1 +1.17, d1 +1.66, a1 -0.85, b1 -1.05, e1 -2.06, f1 -1.49. The right of centre is penalised more than the left edge. Its largest effects on common squares are g3 and e1. | per-part colour flips |
| operations.second_row_centre.hypothesis | L1 | weakened | Squares "just above the bottom row in columns b to d (b3, c2, d3)". | b3 and d3 are in row 3. Its largest effect is on c3 (+2.57), which is not named. b4 (+2.17) and a4 (+1.86) also count. | board geometry, per-part colour flips |

Upheld:
- **Record:** coverage 1.0 and 1.0. Function preservation 0. Closed-form error 0.0404 (each leaf's "off by up to" figure matches). Counts: 42 splits, 24 units, 5 leaves and 1 root. Summary numbers (2710 at 98%, 2708 at 27%, AUC 0.938 to 0.823). The "resisted" numbers (all-affine AUC 0.855, 8616 rows).
- **bottom_row_centre_A (ann_294):** d1 is its main square (+3.28). 75% of its input variance comes from the bottom row. All output weights are negative. The subgroup means and the self-probe reproduce.
- **low_board_mixed (ann_296):** the d2 reward and the c1 correction are confirmed. The self-probe reproduces.
- **upper_rows_mixed (ann_298):** d3 (+3.36) and b3 (-3.06) are its two largest effects. 68% of its input variance comes from rows 3 and 4. The self-probe reproduces.
- **Evidence (L2):** each single-square subgroup gap matches the sign and rough size of the controlled effect of that square.

Flags: none. The centre emphasis matches Connect-4 knowledge.

Note: the recheck report has no `target_hash` field for this record target, so `target_hash` is null in verdict.json.
