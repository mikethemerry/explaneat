# Verdict: c4lf-lines-heavy-s0 (genome 6ebf7e0c)

**Overall: weakened.** Every stated number reproduces (coverage 1.0 / 1.0, preservation 8.9e-16, closed-form error 0.020, all counts, AUCs, probe values and concept scores); nothing is refuted, but five statements overreach and one leaf hypothesis restates its closed form. 36 findings: 0 refuted, 7 weakened, 27 upheld, 2 flags. Checked 2026-11-12 18:16 to 18:31 (+13:00), read only.

## Refuted and weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.summary (V6) | L3 | a single o threat switches the o-lines ramp and the cap off | True at the typical case (74.5% -> 36.7%; units 2 and 3 -> 0) and general for the ramp (off on all 765 o-threat rows), but the cap is still on for 149 of the 765 (19.5%) | `probe --sweep o_thr=0:2:3`; `subgroup --where "o_thr > 0"` |
| record.summary (V8), operations.quiet_o_x_centre_gate.hypothesis (V18) | L3 | the quiet-o gate fires when o has few open lines **and x holds the centre** | 13329 / 93.6% reproduce, but 3785 of the 13329 firings (28%) have no x centre piece; at the typical such position the gate is 0.64 and stays on up to 9 o open singles. Few o lines alone switch it; x's centre pieces shift the threshold | `subgroup --where "node_1 > 0 and centre_x == 0"`; `probe --hold "where:node_1 > 0 and centre_x == 0" --sweep o_open_1=1:21:11 --nodes 1` |
| record.summary (V11) | L3 | o's open singles lower x's chance | Holds for usual counts (99.8% at 1 to 7.5% at 21 at the typical case) but reverses once units 1, 2 and 8 floor: held at x_open_2 >= 3, 78.9% at 17 singles then 88.4% at 21 (inside the data), from the direct +0.20 per sd term | `probe --hold "where:x_open_2 >= 3" --sweep o_open_1=1:21:11 --nodes 1,2` |
| operations.win_probability_root.hypothesis (V15) | L3 | x's open singles: chance 13% at 0 to 81% at 10, through the pair and two gates | 12.8% / 81.4% reproduce, but 0 is below the training minimum of 1 (16.8% at 1); at the typical case gate 7 is off for the whole sweep and 10_a only on from 8, so the stated rise runs through the pair. Minor | `probe --sweep x_open_1=0:10:11 --nodes all` |
| operations.x_lines_vs_o_centre_pair.hypothesis (V34) | L3 | both units raise x's chance when x has more open lines | True for open singles; for open twos unit 8 falls (2.35 at 0, 0.83 at 6, 0.08 at 9) and feeds positively, so through unit 8 more x open twos lower the chance. The pair's contribution rises +0.02 -> +0.84 over 0..3 open twos then falls to +0.44 at 6 | `probe --sweep x_open_2=0:9:10 --nodes 3,8,9`; `probe --sweep x_open_2=0:6:7 --wide` |
| operations.x_singles_vs_twos_gate.hypothesis (V29) | L4 | rises with x's open singles, falls with x's open twos, even-row playable columns and odd-row threats | Not meaningful: restates the closed form. The predicate is accurate (concept AUC 0.90) and the part adds up to +0.57 per unit on 45% of positions, but no game reading is given | `concept --part ann_58 --concept "x_open_1 - x_open_2 - n_even_empty_playable"` |

## Upheld parts

- **win_probability_root (ann_62)**: AUC 0.8815, accuracy 0.810, affine replacement 0.8595 (gain 0.022); ramp / cap / direct terms off -> 0.8414 / 0.8328 / 0.8457; x's open twos 44%, 92%, 97% at 0, 2, 6; threats coincide with open threes (0 rows differ); one root, eight leaves, exits all feed node 0. Weakened only by V11 and V15.
- **quiet_o_x_centre_gate (ann_54)**: 13329 (24.7%), 93.6% vs 56.8%, AUC -> 0.8793, +0.009; 3.07 -> 1.02 and >99% -> 97% with two x centre pieces; off from 7 o singles at the typical case; concept 0.95 / 0.90. Weakened by V18.
- **o_lines_ramp (ann_55)**: on 86.3%; 0 of 765 o-threat rows on; AUC -> 0.8414, +0.188; 3.36 -> 1.92, 2.17 -> 0.00, 88% -> 8%, 1.74 -> 0 with one o threat; rises with o's centre pieces and odd-row columns; concept 0.91 / 0.83 and 0.66. Upheld.
- **x_open_twos_cap (ann_56)**: floor on 16079 (29.8%), 82.5% vs 58.8%; 25058 / 9321 / 342; 1.57, 0.78, 0.00; 44% -> 92% -> 97%; AUC -> 0.8328, -0.137; concept 0.94 / 0.88; floors at 4 open twos when o holds the centre (other inputs move the cut-off, as stated). Upheld.
- **o_centre_x_singles_gate (ann_57)**: 9018 (16.7%), 70.9% vs 64.8%, AUC -> 0.8796, +0.011; 0 at three singles, 0.03 at four, 0.90 at ten; o's centre 0, 0, 0.75 ... 2.48; falls with x's centre pieces; 1426 firings with no o centre piece; concept 0.82 / 0.65 (partial, as stated). Upheld.
- **x_singles_vs_twos_gate (ann_58)**: 24466 (45.3%), 70.6% vs 61.9%; 10 and 25727 rule counts; 0 at 7 singles, 0.06 at 8, 0.38 at 10; 1.07 -> off from 3 even-row columns; AUC -> 0.8780, +0.025; concept 0.90 / 0.81. Weakened at L4 only (V29).
- **centre_column_block (ann_59)**: unit 6 on 49.0% (readout +0.29); unit 76 on 9826 (18.2%), 53.6% vs 68.5%, readout -1.35; rule counts 7713 / 2498 / 385; unit 4 on 6984 (12.9%), 2275 with < 2 x centre pieces, readout +1.47, only 64 firings with none; grid 1.02 / 0.74 / 99%, 2.57 / 47%, 43%, 74%; block contribution +1.17 at x3/o0, -0.92 at x0/o3, -2.53 at x1/o3; AUC -> 0.8713; concept 0.97 / 0.95. Upheld (note: whole-model chance at x0/o3 is 87.7%, above an empty centre's 74.5%, because the ramp and gate 7 rise with o's centre pieces).
- **x_lines_vs_o_centre_pair (ann_60)**: 2510 positive on all 54045; unit 8 on 83.5% (readout +0.53), unit 9 on 30176 (55.8%), 55.8% vs 78.5% (readout -1.01); 3842 / 28.0%; 0.48 -> 2.28, 2.91 -> 0.53, 13% -> 81%; 2.10 -> 0 with o's centre; AUC -> 0.8548, 0.087; concept 0.81 / 0.61. Weakened by V34.
- **direct_terms (ann_61)**: weights 0.88, 0.20, -0.17, -0.15, 0.08, 0.06, -0.04 per sd; R-squared 0.86 on x's open twos; exit -0.99 -> 3.11; AUC -> 0.8457, 0.096. Upheld.

## Flags (domain sense, no data conflict)

- **o_lines_ramp**: the unit falls as much with x's own open twos (-0.60 per sd) as with o's open twos (-0.55), so through this part x's open twos lower x's chance (unit 2: 2.21 at 0 twos, 0 at 5). The hypothesis lists the term and calls its reading partial; the game sense is not given.
- **direct_terms**: o's open singles add +0.20 per sd to x's logit directly, an odd sign in game terms; it is hidden at usual counts and surfaces once the gated units floor (V11).

## Leaf meaning

meaningful: quiet_o_x_centre_gate, x_open_twos_cap, centre_column_block. partial: o_lines_ramp, o_centre_x_singles_gate, x_lines_vs_o_centre_pair, direct_terms. closed-form-only: x_singles_vs_twos_gate.

Self-probe sweeps that start one step outside the data (o's or x's open singles at 0, training minimum 1; even-row playable columns at 7, maximum 6) are noted in V15, V17, V20, V25, V28 and V33; the in-range values reproduce the stated shape in every case.
