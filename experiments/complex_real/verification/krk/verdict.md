# Verdict: krk-default-s2 (genome da7d295b-7705-4c44-b8a6-d95cdce44efa)

**Overall: refuted** (1 refuted, 3 weakened, 24 upheld, 0 flags). One stated number does not reproduce; the structure, every closed form, the hinge shapes, the dead unit, the additive composition, the root formula and the ground-truth comparisons all hold. Started 2026-11-13 10:44, finished 11:02 (18 minutes). Training split, 22,444 rows; read only.

## Refuted and weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.rook_right_of_bk.hypothesis (also evidence[2].narrative, record.leaf_assessments[1].why) | L1 | 28% of draws (rook exactly one file right of the king) get a small penalty | 796 of 2,237 draws = 35.6% have the rook one file right (802 = 35.8% have 812 > 0). 636 of 2,237 = 28.4% is the other side (black king one file right of the rook) | `verify subgroup --where "wr_file - bk_file == 1 and outcome == 1"`, `--where "outcome == 1 and node_812 > 0"` |
| operations.rook_right_of_bk.hypothesis | L3 (weakened) | 812 ~ 0.77*max(0, wr_file - bk_file - 0.79): 0.16 at one file right; the penalty is 0.16*1.92 = 0.31 logit; 7455 takes 0.234 / 0.074 / 0 | 812 depends on the rook's absolute file: at one file right it is 0.320 (rook on b) to 0.046 (rook on h), mean 0.162; with the rook on file 4 the steps are 0.229 / 1.017 / 1.806; the kink is at 0.49 + 0.056*wr_file files. Net penalty on those draws -0.22 logit (direct -0.30, +0.08 back through 7455 -> 2804 -> 99). 7455 at 812 = 0.16 is 0.047, not 0.074 | `verify probe --set wr_file=4 --set wr_rank=4 --set bk_rank=4 --sweep bk_file=1:8:8 --nodes 357,812,7455,2711,99,2804`; gt_checks.py |
| operations.bk_right_of_rook.hypothesis (also evidence[2].narrative, record.leaf_assessments[0].why) | L1 (weakened) | zero on every draw; 0 of 8,743 positions with 357 > 0 are draws | 357 > 0 on 10,773 rows, 636 of them draws (357 at most 0.0045 there, within the stated tolerance); 8,743 rows and 0 draws hold for 357 > 0.01 | `verify subgroup --where "node_357 > 0"`; n357.py |
| record.resisted | L3 (weakened) | the fourth side (rook below the king) is only a linear ramp of 0.72 logit per rank | a linear ramp, yes, but 0.797 per rook rank and 0.757 per black-king rank (13.6% -> 25.9% -> 43.7% -> 63.3% for rook ranks 1 to 4 under a king on rank 4); 0.72 is the surrogate's fitted coefficient | `verify probe --set wr_file=4 --set bk_file=4 --set bk_rank=4 --sweep wr_rank=1:8:8`, same with `--sweep bk_rank` |

## Upheld

- **Record**: coverage 1.0 / 1.0, hierarchy complete, Null inputs -1, -2; function preservation 2.2e-16; closed-form error 0.0322; 4 splits, 2 identity nodes, 4 leaves, 2 compositions, 11 hidden nodes; AUC 0.9971, accuracy 0.960; the capture-or-stalemate rule reproduces all 22,444 labels; 8 stalemates (mean 0.183), 131 defended-adjacent (0.514), captures 0.597; adjacency AUC 0.99987, adjacency-alone AUC 0.9955, P(draw | adjacent) 0.945, P(draw | not adjacent) 0.0003; 1,453 predicted draws, all adjacent (61.6%), 907 adjacent predicted wins; within-adjacency AUC 0.673 and regression +0.755 / -0.694; cell means 0.70/0.78/0.78, 0.57/0.62, 0.39/0.44/0.44 against true 0.895 to 0.981; rook 2+ ranks below within one file: 2,552 rows, 0 draws, mean 0.129; surrogate R^2 0.9996, 99.85% agreement; unit activity 48.0 / 38.4 / 54.3 / 68.8 / 95.2%; the "not a DAG" tool failure reproduces. Not reproducible: the leaf-level restore gains and the 0.5616 all-leaves-linear AUC (that tool fails); the node-level values 0.7468 / 0.8480 / 0.9908 and 99 at 0.9970 agree with the stated linearised-alone numbers.
- **ann_17 draw_score**: logit formula reproduces (typical case 1.3417; slopes 0.797 / -0.620; readouts -2.69, -1.25, -1.92); every predicted draw adjacent; white king structurally Null (SHAP magnitudes not recomputed). Worst grade weakened (record.resisted ramp).
- **ann_16 far_from_rook**: 99 = relu(2.15*357 + 3.29*2711 + 0.154*2804 - 0.285); 2804 = 0.331 per king rank + 1.47*7455 + constant, r = 0.978; 99 mean 0.102 adjacent / 6.141 otherwise, active 95.2%; ReLU -> identity keeps AUC 0.9970 with max output change 0.0948. Meaningful as an additive distance penalty.
- **ann_12 bk_right_of_rook**: 357 = 0.715*max(0, bk_file - wr_file - 1) to 0.0045 on every row; 0 / 0 / 0 / 0 / 0.002 / 0.716 / 1.43 / 2.144 across the files; readouts -1.25, +2.15, +1.14 confirmed; affine alone AUC 0.7468. Worst grade weakened (the "357 > 0" count).
- **ann_15 rook_above_bk**: 2711 = 0 up to one rank above (mean 0.010, max 0.036), 0.639 at two (0.60 to 0.68), +0.64 per rank; switched on two ranks below the king when the king is 4 files right (357 = 2.146); relays always on, 3.29x into 99; 1 of 2,237 draws has 2711 >= 0.2; 0 of 10,442 rows with 2711 > 0.5 are draws; affine alone 0.848; no hinge on the rook-below side.
- **ann_14 dead_rank_unit**: 3965 pre-activation at most -0.2418 (rank 1), 0 active rows; zeroing it or the part changes nothing (0 rows, 0 flips); 7180 = 1.972 on every row; identity_812 = -0.2395.
- **ann_13 rook_right_of_bk** (other statements): 812 > 0.5 exactly where the rook is 2+ files right (6,213 rows, 6 draws); 0 with the king on or right of the rook's file; 0.934 at two files; affine alone 0.9908; 7455 formula within 0.0005.

## Leaf meaning

- ann_12 bk_right_of_rook: **meaningful** (a chess condition the model confirms: no draws where it is on).
- ann_13 rook_right_of_bk: **partial** (the 2+-files side holds; the one-file-right penalty on 35.6% of draws and the absolute-file dependence are not domain behaviour).
- ann_14 dead_rank_unit: **closed-form-only** (a verified Null plus a bias).
- ann_15 rook_above_bk: **meaningful** (a chess condition the model confirms).

## Flags

None. The model's blind spots (white king unseen, lopsided box) are stated by the author and confirmed by the data, not domain conflicts.
