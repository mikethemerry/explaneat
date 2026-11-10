# Verdict vb-61bbd6

**Overall: weakened.** Nothing is refuted; one technical hypothesis (the Washington / area-code unit) misstates two node values, and the rest reproduces.

Genome `1f180718-4e8b-43bc-be1d-9a4276f234c3`. Started 2026-11-10T23:42:54+13:00, finished 2026-11-10T23:45:27+13:00 (3 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `operations.area415_state47.hypothesis` | L1 | weakened | 595 ... Zero in area 415 (2,016 rows); 0.16 for other areas (logit +0.03); 3.19 for state 47 rows outside area 415 | not zero for all of area 415: the 30 Washington customers in area 415 get 1.03 (logit +0.21); the other 1986 area-415 rows are 0. Rows outside 415 get 0.11 (logit +0.02), not 0.16. The main point holds: 3.19 for the 42 Washington customers outside 415 | node 595 values by area code and state |

## Parts

- `ann_87`: upheld (L2)
- `ann_80`: upheld (L4)
- `ann_82`: upheld (L3)
- `ann_81`: upheld (L2)
- `ann_84`: upheld (L4)
- `ann_83`: upheld (L1)
- `ann_86`: upheld (L1)
- `ann_85`: weakened (L1)

## Flags

- `drafted.worth_checking[0]`: two copies of the same information pull in opposite directions; the net is small; for the domain reviewer
- `operations.state_switches.hypothesis`: state-specific effects of up to 2 logits are hard to justify for a phone company and likely fitted to noise

Upheld findings: 19 (see verdict.json).
