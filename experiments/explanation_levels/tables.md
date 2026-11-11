# Three levels per study

| Study | n | Faithful: max abs err (worst) | Leaves | Granularity | Coarse (n) | Understood: meaningful share | Partial | Closed-form only | Domain-concept share |
|---|---|---|---|---|---|---|---|---|---|
| Heart | 10 | 4.4e-16 | 3.5 [3-6] | 1.00 [1.00-1.00] | 0 | 100% [33%-100%] | 0% [0%-67%] | 0% [0%-0%] | n/a |
| MONK-2 | 10 | 2.4e-15 | 3 [2-6] | 1.00 [1.00-1.00] | 0 | 33% [0%-100%] | 50% [0%-67%] | 0% [0%-50%] | n/a |
| Adult | 10 | 7.8e-16 | 8 [6-12] | 1.00 [1.00-1.00] | 0 | 100% [88%-100%] | 0% [0%-0%] | 0% [0%-12%] | n/a |
| Heart-100 | 10 | 4.4e-16 | 4 [4-7] | 1.00 [1.00-1.00] | 0 | 83% [75%-100%] | 17% [0%-25%] | 0% [0%-0%] | n/a |
| churn100 | 10 | 1.0e-15 | 7 [5-12] | 1.00 [0.57-1.00] | 0 | 87% [57%-100%] | 6% [0%-43%] | 0% [0%-25%] | n/a |
| churn200 | 10 | 1.3e-15 | 8.5 [6-10] | 1.00 [0.45-1.00] | 1 | 89% [71%-100%] | 11% [0%-29%] | 0% [0%-10%] | n/a |
| Ladder, synthetic (MLP) | 8 | 3.3e-16 | 5 [3-7] | 1.00 [0.00-1.00] | 2 | 61% [0%-75%] | 39% [25%-100%] | 0% [0%-17%] | n/a |
| Ladder, Connect-4 (MLP) | 6 | 0.0e+00 | 4.5 [1-8] | 1.00 [0.00-1.00] | 2 | 0% [0%-0%] | 31% [0%-80%] | 69% [20%-100%] | 0% [0%-0%] |
| Connect-4 capacity (evolved) | 1 | 1.7e-15 | 11 | 1.00 | 0 | 9% | 55% | 36% | 0% |

# The ladder and the Connect-4 capacity network

| Label | Family | Task | Arch | Hidden | Faithful | Leaves | Granularity | Max width | Max depth | Understood | Partial | CFO | Domain share | Ground truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lad-6eb46 | c4 | c4 | 4 | 4 | 0.0e+00 | 4 | 1.0 | 1 | 1 | 0% | 25% | 75% | 0% |  |
| lad-be3c9 | c4 | c4 | 8 | 8 | 0.0e+00 | 8 | 1.0 | 1 | 1 | 0% | 38% | 62% | 0% |  |
| c4cap-main-h10-e50-s1 | c4 |  | evolved | 10 | 1.7e-15 | 11 | 1.0 | 1 | 1 | 9% | 55% | 36% | 0% |  |
| lad-bcc3b | c4 | c4 | 16 | 16 | 0.0e+00 | 5 | 1.0 | 6 | 1 | 0% | 80% | 20% | 0% |  |
| lad-97ea3 | c4 | c4 | 24 | 24 | 0.0e+00 | 5 | 1.0 | 6 | 1 | 0% | 60% | 40% | 0% |  |
| lad-da6de | c4 | c4 | 32 | 32 | 0.0e+00 | 1 | 0.0 | 32 | 1 | 0% | 0% | 100% | 0% |  |
| lad-61e79 | c4 | c4 | 32x16 | 48 | 0.0e+00 | 1 | 0.0 | 32 | 1 | 0% | 0% | 100% | 0% |  |
| lad-6f6c3 | synthetic | hinges3 | 4 | 4 | 0.0e+00 | 4 | 1.0 | 1 | 1 | 75% | 25% | 0% | n/a | recovered (4/4) |
| lad-b58a2 | synthetic | additive | 4 | 4 | 0.0e+00 | 4 | 1.0 | 1 | 1 | 75% | 25% | 0% | n/a | recovered (5/5) |
| lad-12979 | synthetic | threeway | 8 | 8 | 0.0e+00 | 4 | 1.0 | 3 | 1 | 75% | 25% | 0% | n/a | recovered (5/5) |
| lad-cb415 | synthetic | hinges6 | 8 | 8 | 0.0e+00 | 6 | 1.0 | 2 | 1 | 50% | 50% | 0% | n/a | mostly recovered (5 of 6 hinges as their own leaves) (6/7) |
| lad-f6c4c | synthetic | parity3 | 8 | 8 | 0.0e+00 | 6 | 1.0 | 2 | 1 | 50% | 50% | 0% | n/a | recovered (at the root; the leaves are not the true terms) (3/3) |
| lad-6e63b | synthetic | pairwise | 8x8 | 16 | 0.0e+00 | 3 | 0.0 | 8 | 2 | 0% | 100% | 0% | n/a | mostly recovered (at the root; the leaves entangle the two true interactions) (4/5) |
| lad-b29f2 | synthetic | rule | 16x16 | 32 | 0.0e+00 | 7 | 0.5 | 16 | 1 | 71% | 29% | 0% | n/a | recovered (the rule list at the root; the second layer is not annotated) (5.5/6) |
| lad-ae239 | synthetic | hinges6 | 32x16 | 48 | 3.3e-16 | 6 | 0.3333 | 6 | 1 | 0% | 83% | 17% | n/a | recovered at the root and in second-layer leaves; the 32 first-layer units are described only (6.5/7) |

# Connect-4 and KRK leaves against the concept library

| Label | Arch | Leaf | Exits | Best concept (any) | Score | Grade | Best domain concept | Domain score | Domain grade |
|---|---|---|---|---|---|---|---|---|---|
| lad-6eb46 | 4 | xRaisedInCentre | 1 | o_row1 | 0.46 | closed_form_only | o_row1 | 0.46 | closed_form_only |
| lad-6eb46 | 4 | xOwnsCentreColumn | 1 | centre_diff | 0.53 | partial | centre_diff | 0.53 | partial |
| lad-6eb46 | 4 | leftXRightO | 1 | c_f1 | 0.41 | closed_form_only | col_c_o | 0.24 | closed_form_only |
| lad-6eb46 | 4 | columnCOverBottomLeft | 1 | c_a1 | 0.42 | closed_form_only | x_has_thr | 0.35 | closed_form_only |
| lad-be3c9 | 8 | xCentreAboveOBottom | 1 | o_open_1 | 0.54 | partial | o_open_1 | 0.54 | partial |
| lad-be3c9 | 8 | oBottomCentre | 1 | c_c1 | 0.53 | partial | col_c_x | 0.28 | closed_form_only |
| lad-be3c9 | 8 | xCentreThirdRow | 1 | c_d1 | 0.30 | closed_form_only | centre_diff | 0.30 | closed_form_only |
| lad-be3c9 | 8 | xLowColumnD | 1 | centre_diff | 0.44 | closed_form_only | centre_diff | 0.44 | closed_form_only |
| lad-be3c9 | 8 | xOnC2 | 1 | c_c2 | 0.65 | partial | col_c_x | 0.38 | closed_form_only |
| lad-be3c9 | 8 | xRightOCentre | 1 | x_thr_odd | 0.45 | closed_form_only | x_thr_odd | 0.45 | closed_form_only |
| lad-be3c9 | 8 | xBottomLeftRow | 1 | c_b1 | 0.42 | closed_form_only | o_row1 | 0.35 | closed_form_only |
| lad-be3c9 | 8 | xHighColumnD | 1 | c_c2 | 0.39 | closed_form_only | o_row2 | 0.31 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | direct_square_values | 1 | o_row6 | 0.27 | closed_form_only | o_row6 | 0.27 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_1 | 1 | c_c1 | 0.71 | partial | centre_diff | 0.57 | partial |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_7 | 1 | c_b1 | 0.52 | partial | o_row1 | 0.38 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_4 | 1 | c_d1 | 0.57 | partial | centre_diff | 0.44 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_10 | 1 | c_a1 | 0.52 | partial | col_e_o | 0.34 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_9 | 1 | o_row2 | 0.46 | closed_form_only | o_row2 | 0.46 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_6 | 1 | x_row6 | 0.48 | closed_form_only | x_row6 | 0.48 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_8 | 1 | x_row6 | 0.58 | partial | x_row6 | 0.58 | partial |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_5 | 1 | c_b1 | 0.81 | meaningful | col_b_o | 0.62 | partial |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_3 | 1 | c_c3 | 0.47 | closed_form_only | o_row2 | 0.32 | closed_form_only |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_2 | 1 | x_row3 | 0.59 | partial | x_row3 | 0.59 | partial |
| lad-bcc3b | 16 | bottomRowVsUpper | 1 | x_row6 | 0.58 | partial | x_row6 | 0.58 | partial |
| lad-bcc3b | 16 | sideUnits | 6 | c_b1, c_c2, c_d1 | 0.51 | partial | col_b_x, col_c_x, col_f_o | 0.39 | closed_form_only |
| lad-bcc3b | 16 | oCentreBottom | 3 | c_b1, c_c1, c_d1 | 0.62 | partial | centre_diff, col_b_x | 0.42 | closed_form_only |
| lad-bcc3b | 16 | upperRowUnits | 3 | c_c3, o_row6, x_row5 | 0.40 | closed_form_only | o_row6, x_row5 | 0.37 | closed_form_only |
| lad-bcc3b | 16 | xCentreBottom | 3 | c_d1, centre_diff | 0.64 | partial | centre_diff | 0.62 | partial |
| lad-97ea3 | 24 | bottom_row_centre_A | 5 | c_c1, c_d1 | 0.61 | partial | centre_diff, col_c_x, x_thr_odd | 0.44 | closed_form_only |
| lad-97ea3 | 24 | bottom_row_centre_B | 6 | c_c1, c_d1, o_row1 | 0.56 | partial | centre_diff, col_c_x, o_row1 | 0.51 | partial |
| lad-97ea3 | 24 | second_row_centre | 5 | c_a2, c_b2, c_c2 | 0.51 | partial | col_b_o, col_b_x, o_row6 | 0.32 | closed_form_only |
| lad-97ea3 | 24 | upper_rows_mixed | 3 | c_b3, c_d1, x_mean_row | 0.41 | closed_form_only | x_mean_row, x_row6, x_thr_odd | 0.37 | closed_form_only |
| lad-97ea3 | 24 | low_board_mixed | 5 | c_e1, c_f2, c_g1 | 0.45 | closed_form_only | centre_diff, col_e_o, o_row1 | 0.33 | closed_form_only |
| lad-da6de | 32 | board_detectors | 32 | c_a1, c_a3, c_b1 | 0.49 | closed_form_only | centre_diff, centre_o, col_a_x | 0.39 | closed_form_only |
| lad-61e79 | 32x16 | board_square_features | 32 | c_a1, c_a2, c_b1 | 0.50 | closed_form_only | centre_diff, col_b_x, col_c_x | 0.39 | closed_form_only |
