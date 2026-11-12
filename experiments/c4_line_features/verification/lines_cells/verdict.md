# Verdict: c4lf-lines_cells-heavy-s2 (genome 9e843d20)

**Overall: weakened.** Nothing is refuted: every stated number reproduces (coverage 1.0/1.0, preservation 2.7e-15, closed forms, the 87 provenance-backed values, all 14 probe sweeps, every subgroup count, 13 ablation AUCs, the linearisation). What is weakened is scope: the open-two sweep is a constructed empty board (real positions go 40% to 90%, not 28% to above 99%), the cell triggers named for units 2, 4, 6, 8 and 9 cover a minority of each unit's active rows, and the four closed-form-only parts restate their formulas (as the record says). Findings: 0 refuted, 11 weakened, 27 upheld, 0 flags. 171 factual statements checked; 13 minutes.

## Weakened findings

| id | location | level | stated | what the model does | check |
|---|---|---|---|---|---|
| V07 | `record.summary` (ann_130) | L3 | x's open twos raise the predicted chance (28% at zero to above 99% at six in the sweep); the same sentence in  | the sweep reproduces (27.7% at zero, 99.7% at six) but its held case is an empty board with x_open_1 = 9, not an eight-ply position; on real positions the mean predicted chance is 39.7% at zero open twos (n 11218), 60.2% at one, 74.2% at two, 83.1% at three, 8 | probe sweep x_open_2 0..6 at the typical case; subgroup by open-two count on the training rows |
| V17 | `operations.direct_terms.hypothesis` (ann_127) | L4 | the linear part, described by its 58 weights | not meaningful: restates the closed form (the record grades it closed-form-only itself). What it does: a linear score dominated by single cells (e2 against x, e1 for x) with x line counts positive and o line counts negative. | L4 reading |
| V20 | `operations.unit1.hypothesis` (ann_117) | L4 | stored meaning: the negative of x's open twos plus x on b1, c1, d1 and e1 (partial, 0.67) | the stored concept is a fitted sum, not a game concept, and the hypothesis itself says the unit is "described by its formula rather than explained"; cells carry 81% of its |weight| per sd (top: c1 -0.63, b1 -0.62, x_open_2 -0.62, d2 +0.56, d1 -0.53, e1 -0.52). | L4 reading; closed-form weights |
| V22 | `operations.unit2.hypothesis` (ann_118) | L2 | a gate that lowers x's predicted chance when c3 is x; off with c3 blank or o (in the sweep) | on the data 10361 of its 15368 active rows (67%) have c3 blank: it is on for 24.4% of the 42385 blank-c3 positions and 71.4% of the 4453 positions with o at e2; the rule "x at c3" covers 32% of where it acts (the sweep statement is true only at the typical cas | subgroup c_c3 == 0 and c_e2 == -1 |
| V25 | `operations.unit4.hypothesis` (ann_120) | L2 | its bulk behaviour is positional: an x at g3 switches it on | it acts on 2439 of the 3384 positions with x at g3 (72%), but 4682 of its 7121 active rows (66%) have no x at g3; and positions with x at g3 get a lower whole-model chance (59.4% vs 65.0%; outcome 59.6% vs 66.2%), so the gate raises the score on positions the  | subgroup c_g3 == 1 |
| V28 | `operations.unit6.hypothesis` (ann_122) | L2 | it switches on when c3 is x | 10999 of its 14084 active rows (78%) have no x at c3; it is on for 25.4% of the 42385 blank-c3 positions and 4.0% of the o-at-c3 positions | subgroup c_c3 == 0 |
| V29 | `operations.unit6.hypothesis` (ann_122) | L4 | a positional gate, described not explained | not meaningful: restates the closed form (the record grades it closed-form-only). What it does: a positive gate on for 26% of positions, most often with x at c3 or x on the bottom row, whose active rows are won 80.1% of the time. | L4 reading |
| V31 | `operations.unit7.hypothesis` (ann_123) | L4 | a positional gate whose slope is set by the open-two count and whose level is set by cells | not meaningful: restates the closed form (the record grades it closed-form-only). What it does: lowers the score on 70% of positions, less so the more open twos x has. | L4 reading |
| V33 | `operations.unit8.hypothesis` (ann_124) | L2 | it switches on when c3 is o (off with x there) | 28730 of its 34275 active rows (84%) do not have o at c3: it is on for 46.7% of the 5498 positions with x at c3 and 61.7% of the 42385 blank-c3 positions. "Off with x there" holds only at the typical case. | subgroup c_c3 == 1, c_c3 == 0 |
| V35 | `operations.unit9.hypothesis` (ann_125) | L2 | it needs x at c1 together with o on the bottom row at b1, g1 or a1 | 1773 of its 9743 active rows (18%) have no x at c1 and 2233 (23%) fail the full rule; among the 18798 positions that satisfy the rule it is on for only 40.0%; the rule as a concept scores 0.34 (AUC 0.67). "Needs" overstates a tendency. | subgroup of the rule; verify concept |
| V36 | `operations.unit9.hypothesis` (ann_125) | L4 | a bottom-row positional gate, described not explained | not meaningful: restates the closed form (the record grades it closed-form-only). What it does: a small positive gate on 18% of positions, mostly with x at c1 and o beside it on the bottom row. | L4 reading |

## Upheld parts (one line each)

- **win_logit** (ann_130): worst grade weakened at L3; upheld checks V08, V12
- **raise_gates** (ann_128): worst grade upheld at L3; upheld checks V13
- **lower_gates** (ann_129): worst grade upheld at L3; upheld checks V14
- **direct_terms** (ann_127): worst grade weakened at L4; upheld checks V15, V16
- **unit1** (ann_117): worst grade weakened at L4; upheld checks V18, V19
- **unit2** (ann_118): worst grade weakened at L2; upheld checks V21
- **unit3** (ann_119): worst grade upheld at L3; upheld checks V23
- **unit4** (ann_120): worst grade weakened at L2; upheld checks V24
- **unit5** (ann_121): worst grade upheld at L3; upheld checks V26
- **unit6** (ann_122): worst grade weakened at L2; upheld checks V27
- **unit7** (ann_123): worst grade weakened at L4; upheld checks V30
- **unit8** (ann_124): worst grade weakened at L2; upheld checks V32
- **unit9** (ann_125): worst grade weakened at L2; upheld checks V34
- **unit10** (ann_126): worst grade upheld at L3; upheld checks V37, V38

Record and structure: V01 to V06, V08 to V11 upheld (coverage, preservation, closed-form error, operation counts, readout signs, hierarchy, threat counts, centre-count effect on real positions 59% to 84%).

## Leaf meaning (verifier's own grades)

| leaf | grade | reads | why |
|---|---|---|---|
| unit1 | closed-form-only | cells | The stored concept is a fitted sum of a count and four cells (score 0.67) and the hypothesis itself calls the unit "described by its formula rather than explained"; it is an offset that shrinks as x gains open twos, with cells carrying 81% of its weight. |
| unit2 | partial | cells | It is on for 90% of positions with x at c3 (score 0.78) and lowers the score there as a correction, but two thirds of its active rows have c3 blank, so the cell rule is only part of its behaviour. |
| unit3 | partial | both | A centre-column reading the model confirms on real positions (acts on 81%, 98%, 100%, 100% of positions with 1 to 4 centre pieces, contribution rising +1.6 to +7.2), but 40% of its active rows have no x centre piece. |
| unit4 | partial | cells | The x-at-g3 rule covers 72% of g3 positions but only a third of its active rows; its one clear game reading, firing on every x odd-row threat (+7.2 on 66 rows), is real but rests on too few rows. |
| unit5 | partial | features | A weak-position gate: on when x has no centre piece and few open twos (acts on 84.7% at zero open twos falling to 4.1% at five or more; 68.8% without a centre piece vs 31.4% with); -(centre_x + x_open_2) scores 0.63, a better match than the stored x_open_2 (0.53). |
| unit6 | closed-form-only | cells | The x-at-c3 trigger accounts for 22% of its active rows and no concept reaches 0.5; it is a bottom-row cell pattern with no game reading. |
| unit7 | partial | both | It lowers the score less the more open twos x has (acts on 85% at zero to 30% at five or more, contribution -2.1 to -0.3), a consistent direction on real positions, but cells (e1, c3, e3, a4) set its level and the concept score is 0.33. |
| unit8 | closed-form-only | both | The o-at-c3 rule describes 16% of its active rows (it is on for 47% of x-at-c3 positions and 62% of blank-c3 ones); the growth with open ones is real but the unit has no reading that holds for most of where it acts. |
| unit9 | closed-form-only | cells | The stated rule (x at c1 with o at b1, g1 or a1) scores 0.34: the unit is on for only 40% of positions satisfying it and 23% of its active rows do not. |
| unit10 | partial | both | It rises with x centre pieces and x at d2 on real positions (contribution -3.5 to -14.1 for 1 to 4 pieces, acting on 92% to 100%) and caps the centre column's positive effect; a compensating term with a partial centre reading. |
| direct_terms | closed-form-only | both | A 58-term linear sum; the stated largest weights reproduce, but the part has no reading beyond its coefficients (concept score 0.08). |

## Flags

None. One domain note, not a conflict: unit4's largest effect is firing on every x odd-row threat (+7.2 on the 66 such rows), which is a real Connect-4 concept; the record set it aside for its small n, and that choice is defensible.

## Notes

- The typical held case behind every probe sweep is an empty board with x_open_1 = 9, not a legal eight-ply position; every direction was therefore also confirmed on training subgroups.
- `verify recheck` wrote no `target_hash` for this record target; the field is null in verdict.json.
- Read only: no database writes, no edits to the target directory.
