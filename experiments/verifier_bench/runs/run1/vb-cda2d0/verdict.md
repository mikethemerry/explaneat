# Verdict vb-cda2d0

**Overall: refuted.** The ST-depression hinge formula carries a wrong bias (-0.286 instead of -0.386) in the closed form and technical hypothesis; the plain hypothesis of the sex/exercise-angina part claims an effect for all patients when it acts only on those without exertional chest pain.

Genome `b19b9536-00dd-4e95-9f5a-def6dd8d6889`. Started 2026-11-10T23:36:32+13:00, finished 2026-11-10T23:38:11+13:00 (2 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `layer.parts.ann_33.closed_form` | L1 | refuted | relu(1.0 x_oldpeak - 0.286) (bias -0.286) | the recomputed bias is -0.386, not -0.286 (sympy -0.386047 vs stated -0.286047). In raw units the node is on above 1.261 / 0.845 = 1.49 mm, i.e. z = 0.385, which needs a bias of -0.386; with -0.286 the hinge would open at z 0.285 (about 1.4 mm) | verify recheck closed forms + indicator coefficients |
| `operations.oldpeak_hinge.hypothesis` | L1 | refuted | Node 2985 = ReLU(1.002*oldpeak - 0.286) ... zero for patients with oldpeak below +0.385 SD | the stated bias does not reproduce (it is -0.386) and is inconsistent with the stated +0.385 SD threshold; the same wrong value is in operations.oldpeak_hinge.evidence[0] latex. The rest reproduces: zero for 71.1% of rows, 70 active, disease 0.71 vs 0.33, max  | recomputed closed form + forward pass |
| `drafted.readdown.ann_31.hypothesis[0]` | L2 | weakened | This part lowers the predicted risk for all patients, and lowers it most for women without chest pain on exercise. | not for all patients: the part is exactly 0 for the 82 training patients with chest pain on exercise (19 women, 63 men) and acts only on the 160 without it (C39), lowering the logit by 0.43 for men and 1.67 for women. Right direction and right strongest group, | node 791 values by sex and exercise angina |

## Parts

- `ann_35`: upheld (L1)
- `ann_33`: refuted (L1)
- `ann_31`: weakened (L2)
- `ann_32`: upheld (L4)
- `ann_30`: upheld (L1)
- `ann_34`: upheld (L1)

## Flags

- `drafted.worth_checking[2]`: known risk factor given a protective effect (20% -> 3%)
- `drafted.worth_checking[3]`: clinically odd and fitted to 10 patients; the explainer already calls it likely overfitted

Upheld findings: 12 (see verdict.json).
