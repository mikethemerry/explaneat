# Verdict vb-e95890

**Overall: refuted.** One plain evidence sentence for the direct sum reverses the direction of five effects; every number and the technical hypotheses reproduce, and the ramp hypothesis only restates its formula.

Genome `e324239d-f591-4c8a-b149-7699ec2d0346`. Started 2026-11-10T23:13:42+13:00, finished 2026-11-10T23:19:00+13:00 (5 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `drafted.readdown.ann_24.hypothesis[0]` | L4 | weakened | a weighted sum of a1, a2, a3, a4 and a6 plus a constant, kept only where it is above zero and passed on with a fixed weight | not meaningful: restates the closed form. What the part does (L3): it gives a large push towards yes while at most two of a2, a3, a4, a6 take their first value (a1=1), at most three for a1=2, and it is off for every a1=3 case | L4 meaning |
| `drafted.readdown.ann_27.evidence[1]` | L1 | refuted | Through it, a4 = 1, a3 = 2 and a6 = 2 raise the probability and a2 = 1 or 2 instead of 3 lowers it | the directions are reversed. The cited claims say a4=1 lowers (-4.90 log-odds), a3=2 lowers (-2.19), a6=2 lowers (-0.24), a2=1 raises (+0.58) and a2=2 raises (+0.12) through the part; the linear weights agree (a4=1 -4.90, a3=2 -2.19, a6=2 -0.24, a2=1 +1.20 vs  | cited claims and indicator coefficients |

## Parts

- `ann_28`: upheld (L3)
- `ann_24`: weakened (L4)
- `ann_27`: refuted (L1)
- `ann_25`: upheld (L1)
- `ann_26`: upheld (L1)

## Flags

- none

Upheld findings: 24 (see verdict.json).
