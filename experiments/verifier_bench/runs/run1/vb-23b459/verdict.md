# Verdict vb-23b459 (Heart, genome 0057df7a)

**Refuted.** The flat-ST-segment part leaves out its cholesterol relay, so coverage is not full and its formula does not reproduce; the age and heart rate gate is said to depend on exercise angina when it depends on the thallium scan. Everything else reproduces.

Time: 23:13 to 23:20 (7.0 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.st_slope_chol_gate.subgraph_nodes | L1 refuted | nodes -9_b, -18_a, -20_b, 1476 | relay 3464 (chol) feeds 1476 but is in no part; -9_b and 3464 uncovered | recheck |
| record.structural_coverage / full_coverage | L1 refuted | 1.0 / true | 0.9474 / false | recheck |
| record.closed_form_max_abs_err | L1 refuted | 0.011 | 1.77 | recheck |
| layer.parts.ann_32.closed_form, claims.C53, drafted.readdown.ann_32.evidence[4] | L1 refuted | formula reproduces exactly | max error 1.77 for the annotated subgraph | recheck |
| claims.C71 | L1 weakened | 2 switching units | 1 | recheck |
| record.resisted | L1 refuted | nothing resisted coverage | two nodes uncovered | recheck |
| drafted.readdown.ann_31.hypothesis[0] | L2 refuted | exercise angina raises risk through the gate, 1% to 4% | gate does not read exang (contribution -1.538 either way); the 1% to 4% is the thallium change (C32) | probe exang=0,1 and thal=3,7 on young high-heart-rate patients |
| operations.age_hr_reserve_gate.hypothesis | L2 refuted | gate fires when no exercise-induced angina | gate blocked by a reversible thallium defect (active 3.3% vs 50.7%) | probe; subgroup thal=='7' |

Upheld:
- ann_34 (whole model): linear core weights, readouts, AUC 0.930 core alone and 0.943 full, logit SDs 3.28 / 0.91, all reproduce.
- ann_31 numbers (80 of 242 on, 11% vs 60% disease, AUC 0.943 -> 0.938) reproduce; only the variable named is wrong.
- ann_33 (ST depression hinges): off up to 0.8 mm, on from 1.0 mm, steeper after about 1.8 mm; kinks and slopes match the weights.
- ann_32 behaviour (flat-slope detector, ST depression up, cholesterol down) matches probes; only the structure and formula fail.

Flags (domain, already listed by the explainer): high fasting blood sugar and a fixed thallium defect lower predicted risk; cholesterol lowers risk inside the flat-ST gate.
