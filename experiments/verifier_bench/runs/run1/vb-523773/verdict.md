# Verdict: vb-523773 (MONK-2, genome e324239d)

**Overall: refuted, narrowly.** Coverage, function preservation, closed forms, all 59 claims and every number in the prose reproduce. One root evidence payload has a wrong count, already disclosed in the record. One root hypothesis phrase has counterexamples.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `operations.upper_edge_count_classifier.evidence[2]` | L1 | a1=3: "108/108 pos" | 107 training cases with a1 = 3, all predicted 'yes' (107/107). The record's `resisted` field and the hypothesis text both give 107. | `verify subgroup --where "a1 == '3'"` |
| `operations.upper_edge_count_classifier.hypothesis` (weakened) | L3 | negative once about four of a2, a3, a4, a6 are at their first value (a1 = 2) | With a1 = 2 and all four at 1: a5 = 2 gives 99.8% (score +6.4), a5 = 3 gives 98.3%. Only a5 = 1 (0%) and a5 = 4 (27%) are negative. On the training data, 2 of the 3 such cases are predicted 'yes'. The per-count table that includes a5 is right. | `verify probe --set a1=2 --set a2=1 --set a3=1 --set a4=1 --set a6=1 --sweep a5=1,2,3,4` |

## Upheld

- **Record.** Coverage 1.0/1.0, preservation 8.33e-16, closed-form error 0.065, the operation counts and the summary numbers all reproduce: 263 starts at about 19 and drops 6.75 to 9.29 per first value; linear weights -15.7 and -14.8, plus +24.9.
- **ann_28 (root).**
  - The logit decomposes exactly as stated: 2.58, 1, 5.14, 1.53, 0.105.
  - The (a1, count) table reproduces (74%, 29%), as do 85.5% predicted positive, accuracy 0.472 and AUC 0.807 / 0.734.
  - There is no lower edge (counts 0 to 1 have the highest scores), and a1 = 3 is always positive.
  - The plain hypothesis and both worth-checking flags hold.
- **ann_24 (ramp).** Indicator weights, the pair and triple bounds, the a1 = 2 and a1 = 3 behaviour, and the range 0 to 24.79 all reproduce. It is zero on 40.3% of rows. Switching it off flips 112 predictions (AUC 0.734 to 0.537).
- **ann_27 (direct sum).** All indicator weights reproduce from probes. a1 and a5 dominate. Switching it off flips 50 predictions.
- **ann_25 (a5/a3 unit).** It takes the values 0, 0.295 and 0.977 as stated. Switching it off flips 3 predictions (AUC 0.7335 to 0.7329). It is active on 259 cases, 82 of them 'yes'.
- **ann_26 (a6 unit).** It equals 0.743 when a6 = 2. Switching it off flips nothing (largest change 0.078 log-odds).

## Not checked and flags

- I did not recompute the quoted sensitivity figures.
- There are no domain flags: MONK-2 is synthetic.
