# Verdict vb-d9fb05

**Overall: refuted.** Refuted: the heavy-daytime-use part leaves out hidden node 6038, so coverage is 0.978 not 1.0 and its closed form and formula claim do not reproduce; and the account-length part is said to act mainly with an international plan, which it does not read.

Genome `1f180718-4e8b-43bc-be1d-9a4276f234c3`. Started 2026-11-10T23:40:15+13:00, finished 2026-11-10T23:41:31+13:00 (1 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V1 | `record.structural_coverage` | L1 | refuted | 1.0 (and full_coverage true; same in verified_by_orchestrator) | recomputed 0.9783, full coverage false; uncovered nodes -62_ab and 6038 | verify recheck |
| V2 | `record.closed_form_max_abs_err` | L1 | refuted | 0.0280 | recomputed 0.521 (the day/evening hinge closed form no longer matches its part) | verify recheck |
| V4 | `structure.day_eve_minutes_hinge` | L1 | refuted | subgraph -59_b, -62_ab, 3227 (formula uses an inner evening-minutes relu) | hidden node 6038 (relu(0.177 + 0.382 evening minutes)) that feeds 3227 is in no annotation, and -62_ab is left uncovered; the stated subgraph does not contain the computation its own formula describes | verify recheck + operations.json |
| V5 | `layer.parts.ann_82.closed_form` | L1 | refuted | closed form with 0.177, 0.371, 0.382 terms reproduces exactly (6.7e-16) | the recomputed closed form of the stated subgraph has none of those terms; error 0.524 | verify recheck |
| V6 | `claims.C089` | L1 | refuted | formula reproduces the part (max error 6.7e-16) | recomputed max error 0.524; not reproduced | verify recheck |
| V7 | `drafted.readdown.ann_82.evidence[3]` | L1 | refuted | its formula reproduces the part exactly on all 4000 training customers | cites C089, which does not reproduce (error 0.524) | verify recheck |
| V8 | `claims.C134` | L1 | refuted | 2 switching units | recomputed 1 switching unit in the stated subgraph | verify recheck |
| V17 | `drafted.readdown.ann_86.hypothesis[0]` | L3 | refuted | raises the chance steadily with a longer account, mainly for customers with an international plan | the part reads only account length; its contribution is the same with and without an international plan (-0.137 at 48, 0 at 100, +0.137 at 152 in both) | probe account length x international plan |

## Parts upheld

- ann_87: upheld (checked to L3).
- ann_80: upheld (checked to L4).
- ann_81: upheld (checked to L3).
- ann_84: upheld (checked to L1).
- ann_83: upheld (checked to L1).
- ann_85: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[0]`: night minutes raise churn, night charge lowers it -- the two columns are the same information; offsetting weights
