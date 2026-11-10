# Verdict vb-8f4b05

**Overall: refuted.** Claim C080 and its read-down sentence say the service-call part takes its typical customer to about 8%; the model gives about 3%. A few percentages are rounded up one point and one hidden unit is described too narrowly.

Genome `4b0eb5a6-5859-4ee8-80fe-18ab72f22959`. Started 2026-11-10T23:38:19+13:00, finished 2026-11-10T23:42:45+13:00 (4 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `claims.C080` | L1 | refuted | typical active customer: about 2% without the part, 8.12% with it (delta 0.0586) | recomputed 3.49% with it (delta 0.0123). Probe at the 19 stated reference values: risk 3.4%, part contribution +0.447 log-odds, so 2.2% without and 3.4% with | verify recheck + probe at the claim's own reference values |
| `drafted.readdown.ann_81.evidence[0]` | L1 | refuted | for the typical customer it acts on, the chance is about 2% without it and about 8% with it | the model gives about 3% with the part (3.4-3.5%), not 8%; the block sentence drafted.blocks[1][0] correctly says about 3% | same check |
| `operations.serviceCallsHinges.hypothesis` | L3 | weakened | 5395 ... is on at >=4 calls unless day z > ~0.9, or 3 calls with below-average day usage (810 rows) | 810 rows reproduce, but 284 of them have 0-2 calls (7 at 0, 78 at 1, 199 at 2): the unit also fires for low-call customers with very low day usage, which the description leaves out | node 5395 activity by number of calls |
| `drafted.readdown.ann_79.evidence[2]` | L1 | weakened | without a voice-mail plan ... rises steadily from about 2% to about 5% | C086 gives 4.45%, which rounds to about 4%, not 5% (last-digit rounding); same in drafted.blocks[2][2] | verify recheck prose check |
| `drafted.blocks[2]` | L1 | weakened | from 112 to 249 minutes the chance goes from about 2% to about 5% without one | C086: 2.26% -> 4.45% (about 4%) | verify recheck prose check |
| `drafted.readdown.ann_83.evidence[2]` | L1 | weakened | daytime minutes ... from about 2% at 112 to about 5% at 249 minutes | C012: 2.26% -> 4.45% (about 4%) | verify recheck prose check |

## Parts

- `ann_83`: weakened (L1)
- `ann_81`: refuted (L1)
- `ann_79`: weakened (L1)
- `ann_80`: upheld (L2)
- `ann_82`: upheld (L1)

## Flags

- `drafted.worth_checking[0]`: opposite signs for plan and message count; for the domain reviewer

Upheld findings: 12 (see verdict.json).
