# Verdict vb-0d1892

**Overall: weakened.** Numbers, thresholds and directions reproduce; two small weakenings: the record says the network reproduces the 0.88 churn of frequent low-usage callers (it predicts about 65%), and one technical count (304 active rows) is 306.

Genome `6a26c823-5e60-424d-a968-74f44d675efa`. Started 2026-11-10T23:28:16+13:00, finished 2026-11-10T23:31:08+13:00 (3 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V5 | `record.summary` | L2 | weakened | frequent callers with low usage churn (empirical 0.88), which the network reproduces (also in operations.csc_usage_gate.hypothesis) | empirical 88.5% (n=113) for <160 min, but mean predicted risk there is 65.4%; for <180 min 75.0% empirical vs 57.9% predicted. The network ranks this group high but does not reproduce the 0.88 rate | subgroup calls>=4 and day minutes<160 / <180 |
| V13 | `operations.csc_hinge_7208.evidence[1]` | L1 | weakened | active_rows: 304 (CSC>=4, never for state:49) | active on 306 rows (318 with 4+ calls minus the 12 in West Virginia); the plain claim C127 gives 306 | subgroup calls>=4; ablate ann_92 |

## Parts upheld

- ann_95: upheld (checked to L4).
- ann_90: upheld (checked to L1).
- ann_91: upheld (checked to L3).
- ann_93: upheld (checked to L1).
- ann_94: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[1]`: voice-mail plan lowers churn but more messages raise it -- opposite signs for plan and message count are hard to read in domain terms; offsetting weights
- `drafted.worth_checking[2]`: service-call hinge switched off in West Virginia -- a rare-state switch fitted to 12 customers; no domain reason
