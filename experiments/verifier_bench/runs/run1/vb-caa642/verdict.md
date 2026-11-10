# Verdict vb-caa642

**Overall: refuted.** The capital-gains bend threshold (claim C126 and the prose citing it) is wrong: the bend starts at about 4883, not 4283 US dollars; and the short-hours hypothesis says the effect continues to the longest weeks when it stops at 50 hours. The rest reproduces.

Genome `7a8237f3-74cb-40bd-8268-b605f64dac2b`. Started 2026-11-10T23:13:43+13:00, finished 2026-11-10T23:22:00+13:00 (8 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V10 | `drafted.blocks[0]` | L1 | refuted | second part adds nothing until about 4283 US dollars | the bend switches on at about 4883 US dollars (recomputed grid point 4882.8); at 4500 its exit is still 0 | recheck of C126 + probe |
| V14 | `claims.C126` | L1 | refuted | threshold/grid point 4282.8 (4283 US dollars) | recomputed 4882.8 (4883 US dollars) in shape.grid[13], pieces and points | verify recheck |
| V15 | `drafted.readdown.ann_123.hypothesis[0]` | L1 | refuted | adds nothing until capital gains reach about 4283 US dollars | part exit 0 at 4000 and 4500, first positive at about 4800 (0.0002), 0.0163 at 5000; onset about 4.8-4.9k, not 4283 | recheck + probe sweep |
| V16 | `drafted.readdown.ann_123.evidence[1]` | L1 | refuted | nothing until about 4283 US dollars, then steadily | same: onset about 4883 US dollars | recheck + probe sweep |
| V23 | `operations.age_profile.hypothesis` | L1 | weakened | for Thai respondents (27 rows) | 26 Thai rows in training (the same record says 26 later) | subgroup native-country==37 |
| V25 | `drafted.readdown.ann_125.hypothesis[0]` | L3 | refuted | falls steadily as hours rise, all the way to the longest working weeks | the part is flat from 50 hours on: contribution +0.507 at 40, +0.099 at 48, 0 at 50, 55, 60, 80 and 99 hours; risk stays 50.8% from 50 to 99 hours. Claim C142 itself has a fold at 49.9 hours | probe hours sweep (typical) |

## Parts upheld

- ann_131: upheld (checked to L1).
- ann_122: upheld (checked to L3).
- ann_129: upheld (checked to L4).
- ann_128: upheld (checked to L1).
- ann_124: upheld (checked to L1).
- ann_126: upheld (checked to L1).
- ann_127: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.readdown.ann_129.evidence[3]`: 7th-8th grade lowers chance of low income to about 35% (vs high-school graduate) -- less schooling giving a lower predicted chance of low income runs against what is known; it arises because education level (1-16) is held fixed while the category changes; data not contradicted
- `drafted.worth_checking[7]`: census sampling weight lowers predicted chance -- a survey weight describes the sampling, not the person; the model uses it
