# Verdict vb-ced675 (churn, genome 1f180718)

**Refuted.** The customer-service-calls part is said to barely matter in Oregon, where the calls count most, and one evidence sentence cites the wrong claim. The rest reproduces.

Time: 23:27 to 23:30 (3.0 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_81.hypothesis[0] | L3 refuted | in Oregon the calls barely matter | Oregon: contribution -1.96 -> +1.79 from 0 to 3 calls (0.5% -> 17.6%), against +0.99 -> +1.79 in West Virginia | probe calls x state |
| drafted.readdown.ann_82.evidence[0] | L1 refuted | 747 / 269 / 5% / 23% cited to C086 | those numbers are in C088, not C086 | recheck prose check |

Upheld:
- ann_87 (whole model): additive; mean-ablation AUC losses (0.084, 0.079, 0.038, 0.003) reproduce.
- ann_80 (plans, usage and state): directions and AUC 0.814 without it reproduce.
- ann_82 (heavy daytime use): off in the lower deciles, on for 93% of the top decile (+2.32 logit, churn 0.49).
- ann_81 numbers: flat for 0 to 2 calls, then +1.25 logit per call.
- ann_84, ann_83, ann_86, ann_85: claims and closed forms reproduce.

Flags: the Oregon exemption (84 customers) and the opposite-signed night minutes and night charge.
