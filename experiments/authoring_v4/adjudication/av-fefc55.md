# Adjudication: av-fefc55 (genome b19b9536, Heart Disease)

There are 6 errors, built from 8 verifier findings (A: V12, V14, V16; B: V1 to V5). Neither verifier graded anything refuted.

| Error | Location | Verifiers | Genuine | Layer | Mechanism |
|---|---|---|---|---|---|
| E1 | ann_32 "acts on the 15 ..., 9 of whom" (C44) | A V16, B V4 | weakened | draft | g |
| E2 | ann_31 "about 1% both without it and with it" (C39) | A V14 | weakened | draft | f |
| E3 | ann_33 "above 99% both without it and with it" (C34) | A V12 | weakened | draft | g |
| E4 | "each extra vessel adds about the same" (ann_35 hypothesis, blocks[0], blocks[4]) | B V1, V2 | weakened | draft | d |
| E5 | blocks[2] "being a woman lowers the risk more when ..." (C38) | B V3 | weakened | draft | d |
| E6 | record.summary "Four ... corrections (combined logit SD 1.41)" | B V5 | **refuted** | database | d |

## Notes

- **E6 is upgraded to refuted.** I recomputed the readout contributions on the 242 training rows. All five readouts together have SD 1.410. The four one-sided corrections, which are what the sentence names, have SD 1.158. So 1.41 does not reproduce for the quantity the sentence names. It is the whole-readout value attached to a subset of it.
- **E1, E2, E3.** Each statement is literally true. Each one overstates or hides what the part does:
  - E1: 5 of the 15 patients get only -0.017 logit.
  - E2: the part cuts this patient's risk from 0.99% to 0.64%.
  - E3: the reference patient sits at the ceiling, 99.89% to 99.95%.
- **E4 and E5.** Both are true in log-odds but are stated on the risk scale, where the sentence's own numbers do not support them:
  - E4: the risk steps are +62, +17 and +1 points.
  - E5: the drop is -19.5 points without exercise angina against -26.4 points with it.

Counts: genuine refuted 1 database / 0 draft, genuine weakened 5, verifier-wrong 0.
