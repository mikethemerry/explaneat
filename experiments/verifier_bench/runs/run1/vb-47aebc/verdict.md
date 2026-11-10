# Verdict vb-47aebc

**Overall: refuted.** Numbers reproduce, but the not-in-a-family part is said to lower the chance for everyone when it acts only on people not in a family; the capital-gains hypothesis only restates its formula, and the missing-occupation hypothesis is worded causally.

Genome `c32e8fc8-410f-4126-ab7e-974c658ef3fc`. Started 2026-11-10T23:31:35+13:00, finished 2026-11-10T23:34:53+13:00 (3 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V9 | `drafted.readdown.ann_115.hypothesis[0]` | L4 | weakened | multiplies capital gains by a fixed weight, adds a constant, never cut off at zero on the data | not meaningful: restates the closed form. What it does: makes higher capital gains lower the chance of a low income a little more (contribution +0.10 at 0, -0.54 at 6849, -1.77 at 20000; on top of the direct capital-gain edge) | probe capital gain |
| V11 | `drafted.readdown.ann_120.hypothesis[0]` | L3 | refuted | lowers the predicted chance of a low income for everyone, and most for people not living in a family | the part is exactly 0 for everyone who is not "not in a family": active in 0 of 29006 such people (contribution 0.0); it acts only on the 10067 not in a family (-1.03 logit). It does not lower the chance for everyone | probe relationship sweep; subgroup relationship != 1 |
| V13 | `drafted.readdown.ann_121.hypothesis[0]` | L2 | weakened | Not reporting an occupation pushes people into a low income; this part captures that effect | causal overclaim: the evidence shows only that the model predicts a higher chance (62% -> 80%) for a missing occupation (observed 90.7% low income among the 2197); it says nothing about what not reporting does to income | subgroup occupation==0; recheck C101 |

## Parts upheld

- ann_124: upheld (checked to L4).
- ann_116: upheld (checked to L4).
- ann_117: upheld (checked to L1).
- ann_118: upheld (checked to L1).
- ann_122: upheld (checked to L1).
- ann_119: upheld (checked to L3).
- ann_123: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[3]`: 5th-6th grade lowers chance of low income to about 30% -- counter to what is known; caused by holding the 1-16 education level fixed
- `drafted.worth_checking[4]`: master's and bachelor's degrees raise the chance of low income -- counter-intuitive category offsets on top of the education level
- `drafted.readdown.ann_116.hypothesis[0]`: born in Colombia makes low income almost certain -- a rare-category memorisation (64 people) riding on the age unit
