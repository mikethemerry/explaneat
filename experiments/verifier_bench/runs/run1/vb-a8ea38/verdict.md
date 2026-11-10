# Verdict vb-a8ea38

**Overall: refuted.** The technical hypothesis of the service-call hinge says it switches on at 3 calls; it switches on at 4. The voice-mail part is described in causal terms the evidence cannot carry. The rest reproduces.

Genome `6a26c823-5e60-424d-a968-74f44d675efa`. Started 2026-11-10T23:28:38+13:00, finished 2026-11-10T23:31:50+13:00 (3 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `drafted.readdown.ann_93.hypothesis[0]` | L2 | weakened | Having a voice-mail plan keeps customers from leaving: this part shows the plan itself makes them stay. | causal overclaim: the part only shows what the model predicts. Through it a voice-mail plan lowers the typical customer's predicted chance from 1.5% to 0.8% (less than a point); nothing here shows the plan causes customers to stay | claims + probe |
| `operations.csc_hinge_7208.hypothesis` | L1 | refuted | Active only when CSC >= 3 calls (7.6% of train rows) | 7208 is never on at 3 calls: 0% active at 0-3 calls, 97% at 4 (all but West Virginia), first active value 4 calls; 306 rows (7.65%). Probe in Wyoming: off at 3 calls, on at 4 (+0.59), 5 (+1.20). The evidence record and the plain text say 4 calls correctly | node 7208 activity by number of service calls; probe outside West Virginia |

## Parts

- `ann_95`: upheld (L2)
- `ann_90`: upheld (L4)
- `ann_91`: upheld (L3)
- `ann_93`: weakened (L2)
- `ann_92`: refuted (L1)
- `ann_94`: upheld (L1)

## Flags

- `drafted.worth_checking[1]`: opposite signs for plan and message count; correlated inputs; for the domain reviewer
- `operations.csc_hinge_7208.hypothesis`: a state-specific exemption from the service-call effect has no plausible business reason; fitted to 12 customers

Upheld findings: 17 (see verdict.json).
