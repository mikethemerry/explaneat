# Verdict vb-acd834

**Overall: refuted.** One number in the ischaemia-pathway hypothesis (and its evidence record) does not reproduce: the 35 patients opened by low blood pressure have mean predicted risk 0.18, not 0.13. Everything else holds.

Genome `1f2cf7ea-5c30-452a-9277-0b4043e888d9`. Started 2026-11-10T23:45:35+13:00, finished 2026-11-10T23:47:18+13:00 (2 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `operations.ischaemia_pathway.hypothesis` | L1 | refuted | 290 active 56% (100/135 of its active cases have 204 on; 35 open from low BP alone, mean predicted risk 0.13) | the 35 patients in whom 290 is open while 204 is off have a mean predicted risk of 0.18 (median 0.06, disease rate 0.20), not 0.13. The rest reproduces: 290 active 135 rows (56%), 100 with 204 on; 1834 active 26%, all with 204 and 290 on, 78% diseased; 1987 on | forward pass on the 35 rows |
| `operations.ischaemia_pathway.evidence[0]` | L1 | refuted | 35 cases open it from low resting BP alone and these patients have low predicted risk (mean 0.13) | mean predicted risk 0.18 for those 35 rows; the qualitative point (low risk) stands | forward pass on the 35 rows |

## Parts

- `ann_40`: upheld (L2)
- `ann_39`: refuted (L1)
- `ann_36`: upheld (L4)
- `ann_38`: upheld (L1)

## Flags

- `drafted.worth_checking[0]`: known risk factor given a protective effect (19% -> 10%)
- `drafted.worth_checking[4]`: clinically one would expect an abnormal resting ECG to add to, not weaken, these signs
- `drafted.worth_checking[5]`: counter-intuitive interaction (high blood pressure raising the bar for the ischaemia signal); for the domain reviewer

Upheld findings: 13 (see verdict.json).
