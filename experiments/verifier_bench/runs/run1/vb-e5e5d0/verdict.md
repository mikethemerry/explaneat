# Verdict vb-e5e5d0

**Overall: refuted.** The veto annotation leaves out node 2786, so the stated full coverage, closed-form error and the veto formula claim do not reproduce; the veto hypothesis also overstates the minimum penalty (-58, not -70). The other parts hold.

Genome `60ee493c-bce5-4a2b-b7af-d06843835ae6`. Started 2026-11-10T23:31:58+13:00, finished 2026-11-10T23:34:11+13:00 (2 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `structure.atleast3_of_five_veto` | L1 | refuted | subgraph nodes -1_b, -4_b, -7_b, -9_b, -16_b, 274, id_suppress (operations.atleast3_of_five_veto.subgraph_nodes) | the veto subgraph omits hidden node 2786, which feeds id_suppress (id_suppress = -7.14*274 - 6.56*2786, as the technical hypothesis itself says). Recomputed structural coverage 0.947 with nodes 274 and 2786 uncovered and edges 274->2786, 2786->id_suppress outs | verify recheck structure + operations.json |
| `record.structural_coverage` | L1 | refuted | 1.0 (and full_coverage true) | recomputed 0.9474; full coverage false (uncovered 274, 2786) | verify recheck |
| `record.closed_form_max_abs_err` | L1 | refuted | 0.170 (3-s.f. rounding) | recomputed 93.6: the veto closed form built from the stated subgraph misses the 2786 path (part error 93.7 against exit scale 269) | verify recheck |
| `claims.C26` | L1 | refuted | the veto formula reproduces the part (max err 4.3e-14) | recomputed max error 93.74; reproduces = false | verify recheck |
| `claims.C60` | L1 | refuted | the veto has 2 switching units | recomputed 1 switching unit within the stated subgraph (2786 is outside it) | verify recheck |
| `drafted.readdown.ann_28.evidence[3]` | L1 | refuted | Its formula reproduces the part exactly on all 345 training cases. | does not reproduce from the explanation's operations (max error 93.7); recheck lint: cites C26, which is not verified | verify recheck |
| `layer.parts.ann_28.closed_form` | L1 | refuted | relu(... 274 ...) with the 0.605 / 0.609 / 6.56 terms of node 2786 | the stated coefficients 0.605, 0.609, 6.56 are not in the closed form recomputed from the annotation (only -7.14 relu(...)). The stated formula matches the network, but not the annotated part as recorded | verify recheck closed forms |
| `operations.atleast3_of_five_veto.hypothesis` | L1 | refuted | Veto is exactly 0 for at most two first values and <= -70 for three or more ... a penalty of -70 or more (-269 with all five) | with three first values the veto is between -58.2 and -93.3, not <= -70: the weakest triple (a1, a4, a6) gives pre-activation 5.6 and id_suppress = -11.11*5.6 + 4.00 = -58.2; maximum id_suppress when 274 fires is -58.2 over the grid. The rest reproduces: 274 > | indicator pre-activations + all 432 configurations |

## Parts

- `ann_32`: upheld (L2)
- `ann_28`: refuted (L1)
- `ann_27`: upheld (L1)
- `ann_29`: upheld (L1)
- `ann_30`: upheld (L1)
- `ann_31`: upheld (L1)

## Flags

- none

Upheld findings: 11 (see verdict.json).
