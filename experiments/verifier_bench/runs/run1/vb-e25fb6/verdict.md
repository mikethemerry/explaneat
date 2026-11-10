# Verdict vb-e25fb6

**Overall: upheld.** Every number, claim, hypothesis and sentence I checked reproduces; only domain flags (education one-hots and capital losses pointing the unexpected way, use of the survey weight).

Genome `c32e8fc8-410f-4126-ab7e-974c658ef3fc`. Started 2026-11-10T23:47:27+13:00, finished 2026-11-10T23:50:44+13:00 (3 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|

## Parts

- `ann_124`: upheld (L2)
- `ann_116`: upheld (L3)
- `ann_115`: upheld (L1)
- `ann_120`: upheld (L1)
- `ann_121`: upheld (L1)
- `ann_117`: upheld (L1)
- `ann_118`: upheld (L1)
- `ann_122`: upheld (L1)
- `ann_119`: upheld (L1)
- `ann_123`: upheld (L1)

## Flags

- `drafted.worth_checking[0]`: odd on its face; in census data people with capital losses tend to be better off; for the domain reviewer
- `drafted.worth_checking[3]`: less schooling giving a lower chance of low income conflicts with expectation; probably offset by the collinear education-num term
- `drafted.worth_checking[4]`: conflicts with expectation; again likely compensated by education-num
- `drafted.worth_checking[6]`: a survey weight should carry no information about the person

Upheld findings: 18 (see verdict.json).
