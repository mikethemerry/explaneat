# Verdict vb-b9ec8f

**Overall: refuted.** Nearly everything reproduces; one count in the age hypothesis is wrong (27 Thai respondents where the training data have 26), and the hours sentences cite a claim that does not hold their 50-hour threshold.

Genome `7a8237f3-74cb-40bd-8268-b605f64dac2b`. Started 2026-11-10T23:22:21+13:00, finished 2026-11-10T23:28:29+13:00 (6 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `operations.age_profile.hypothesis` | L1 | refuted | for Thai respondents (27 rows) it adds -4.0 directly and pushes 2990 permanently on | there are 26 people born in Thailand in the training split (node 2134 active on exactly 26 rows; 4 more in the test split), not 27; the same paragraph later says 26. The mechanism itself reproduces (2990 on for every Thai row; age-part change Thai vs US +0.10) | count of Thai rows on the training split |
| `drafted.readdown.ann_125.evidence[0]` | L1 | weakened | It acts on 31364 of 39073 training people, those working less than about 50 hours a week ... [C143] | the 50 hours is right (C142 threshold 49.9 h; the part is on for every row below 50 h and off from 50 h) but it is not in the cited claim C143; recheck flags it. Same in drafted.blocks[3][0] | verify recheck prose check |
| `drafted.blocks[3]` | L1 | weakened | This part acts on the 31364 people who work less than about 50 hours a week ... [C143] | same citation gap as the readdown sentence: 50 hours is supported by C142, not by the cited C143 | verify recheck prose check |

## Parts

- `ann_131`: upheld (L2)
- `ann_130`: upheld (L2)
- `ann_122`: upheld (L1)
- `ann_123`: upheld (L3)
- `ann_129`: upheld (L4)
- `ann_121`: refuted (L1)
- `ann_125`: weakened (L1)
- `ann_128`: upheld (L1)
- `ann_124`: upheld (L1)
- `ann_126`: upheld (L1)
- `ann_127`: upheld (L1)

## Flags

- `drafted.worth_checking[0]`: a capital loss lowering the chance of a low income reads oddly, though in this census data people with capital losses tend to have assets and higher incomes; for the domain reviewer
- `drafted.worth_checking[4]`: less schooling giving a lower chance of low income conflicts with expectation; likely offset by the separate education-num term (the one-hot and the 1-16 scale are collinear)
- `drafted.worth_checking[7]`: a survey weight should carry no information about a person; the model uses it (66% -> 60%)

Upheld findings: 22 (see verdict.json).
