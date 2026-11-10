# Verdict vb-a33fbd (Adult, genome 9c06b29f)

**Refuted.** The youth part leaves out relay node 8375, so coverage is not full and its formula does not reproduce, and the age hinge is placed at about 41 years in the record and the youth hypothesis when it is at about 35.

Time: 23:48 to 23:51 (3.0 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.youth_hinge.subgraph_nodes | L1 refuted | -1, 599, 3328, identity_age | 599 -> 8375 -> 3328 exists; 8375 in no part | recheck; weights |
| record.structural_coverage / full_coverage / closed_form_max_abs_err / resisted | L1 refuted | 1.0 / true / 0.065 / nothing resisted | 0.9844 / false / 1.20 / 599, 8375 uncovered | recheck |
| layer.parts.ann_113.closed_form, claims.C087, drafted.readdown.ann_113.evidence[2] | L1 refuted | formula exact | error 1.19 | recheck |
| claims.C121 | L1 weakened | 2 always-on units | 1 | recheck |
| record.summary | L3 refuted | youth hinge below ~41 | flat from 35 years (risk 61.9% at 35, 38, 41, 44) | probe age sweep |
| operations.youth_hinge.hypothesis | L1 refuted | z = -0.280, ~41 years | 38.6 - 0.28*13.7 = 34.8 years | arithmetic; probe |

Upheld:
- ann_120 (whole model): bias, 40 direct edges and the listed weights; ablations 0.890 (capital gain) and 0.900 (age).
- ann_113 plain text (steady fall to about 35, then flat; 0.161 logit per year).
- ann_114 (capital gains): kink at z 0.554; ablation 0.8896.
- ann_117, ann_118, ann_115, ann_119, ann_116: counts and claims reproduce.

Flags (already listed): low grades and capital losses lower the chance of a low income.
