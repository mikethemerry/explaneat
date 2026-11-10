# Verdict vb-fd4da7 (churn, genome 6a26c823)

**Refuted.** The many-service-calls part is said to raise the chance steadily from the first call, but it is off until 4 calls, and its stated formula has the wrong bias. The rest reproduces.

Time: 23:20 to 23:27 (6.6 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_92.hypothesis[0] | L3 refuted | raises steadily from the first call | contribution 0 at 0 to 3 calls, +0.59 at 4, +2.43 at 7 (Ohio customer) | probe sweep of service calls |
| layer.parts.ann_92.closed_form | L1 refuted | bias -0.479 | bias -0.679 | recheck; node weights |
| operations.csc_hinge_7208.evidence[0] | L1 refuted | latex bias -0.479 | -0.679 | recheck |
| operations.csc_hinge_7208.hypothesis | L1 refuted | relu(0.54 CSC - 0.37 state:49 - 0.48) | bias -0.68 (rest holds) | weights; subgroup |
| operations.csc_usage_gate.hypothesis | L2 weakened | frequent callers with low use churn 0.88, "which the network reproduces" | empirical 88.5% but predicted 65.4% (4+ calls, under 160 min) | subgroup |

Upheld:
- ann_95 (whole model): additive, AUC 0.901, backbone zeroed 0.54, gate zeroed 0.76.
- ann_90 (fixed-weight inputs): steady day-minute effect and all cited claims reproduce.
- ann_91 (calls with low usage): low-usage hinge whose kink moves up with calls (on to 200 min at 4 calls, to about 170 min at 1 call); linearising costs 0.05 AUC.
- ann_93 (voice-mail discount) and ann_94 (state units): small, AUC unchanged when removed.

Flags: the West Virginia switch-off of the many-calls part (12 customers) and the Nevada offset with the wrong sign have no domain reason.
