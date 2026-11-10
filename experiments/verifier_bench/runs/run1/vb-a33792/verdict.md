# Verdict vb-a33792

**Overall: weakened.** Nothing is refuted; the capital-gain hypothesis says 95% of people have no capital gains where the training data have 91.6%. Everything else reproduces.

Genome `9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66`. Started 2026-11-10T23:50:53+13:00, finished 2026-11-10T23:53:33+13:00 (3 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `operations.capital_gain_penalty.hypothesis` | L1 | weakened | +0.41 logit at cg=0 (95% of rows) | 91.6% of training rows have zero capital gains, not 95% (same wording in the evidence record). The rest reproduces: contribution +0.406 at 0, -1.46 at the 95th and -8.99 at the 99th percentile, -73.7 at the maximum, kink at z 0.554, slopes -2.79 and -5.71, cor | share of training rows with zero capital gains |

## Parts

- `ann_120`: upheld (L2)
- `ann_114`: weakened (L1)
- `ann_113`: upheld (L3)
- `ann_117`: upheld (L1)
- `ann_118`: upheld (L1)
- `ann_115`: upheld (L1)
- `ann_119`: upheld (L1)
- `ann_116`: upheld (L1)

## Flags

- `drafted.worth_checking[0]`: odd on its face; for the domain reviewer
- `drafted.worth_checking[3]`: conflicts with expectation; likely compensation for the collinear education-num term
- `drafted.worth_checking[4]`: a survey weight should carry no information about the person

Upheld findings: 16 (see verdict.json).
