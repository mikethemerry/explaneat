# Adjudication av-ff605d

**Result.** 11 errors. Genuine refuted: 1 database, 0 draft. Genuine weakened: 9 (7 database, 2 draft). Verifier wrong: 1.

| Error | Location | Verifiers | Genuine | Layer | Mech | Why |
|---|---|---|---|---|---|---|
| E1 | state_offsets.evidence[1] "161_a contributes <0.03 logit" | B refuted | **refuted** | database | d | 161_a's readout weight (-0.0277) is written as a logit contribution. California's offset is -0.21, and zeroing 161_a moves risk by up to 5 points |
| E2 | csc_hinge_7208.hypothesis "lose up to 2.75 logits" | A refuted, B weakened | weakened | database | d | -2.75 is the sign_analysis whole-model state:49 on/off minimum (hinge -2.43 plus other paths about -0.32). It reproduces as the state effect, not as the hinge's loss |
| E3 | csc_hinge_7208.hypothesis "never active for state:49" | A, B weakened | weakened | database | e | True on every observed row. The hinge turns on at 9 calls, an off-support constructed case |
| E4 | draft "switched off in West Virginia" | A, B weakened | weakened | draft | e | Same as E3, in the plain-language layer |
| E5 | draft ann_92 evidence[1] (C125) 16% -> 26% | A weakened | weakened | draft | d | The part's own effect. In the whole model the change is 12.2% -> 15.6% |
| E6 | csc_hinge_7208 payload active_rows 304 | A, B weakened | weakened | database | f | 306 |
| E7 | record.summary and gate evidence[2]: kink moves "from 3 onward" | A, B weakened | weakened | database | e | The kink already moves at 1 and 2 calls; the step at 3 calls is only the largest |
| E8 | gate hypothesis and evidence[2]: "the network reproduces 0.88 / 0.24" | A, B weakened | weakened | database | e | The model predicts 65.4% / 40.8%: right order, levels not reproduced |
| E9 | state_offsets.hypothesis: 6730 "+0.43" | B weakened | weakened | database | f | The value is +0.448, and the same sentence's -0.45 is right |
| E10 | churn_logit.hypothesis "16 states" | B weakened | weakened | database | f | 17 (the total of 22 is right) |
| E11 | churn_logit.evidence[2], [3] "no result" | A weakened | none | database | other | The payloads carry the per-input effects and the SHAP values |

Checks run: probe (state sweep with 161_a and 6730; WV 7-call row, state 49 vs 1; WV calls 6..9 with node 7208; claim C125), ablate (node 161_a), subgroup (CSC>=4 & not WV; CSC>=4 & day<160; WV & CSC==7), recheck null_nodes, and a read of the operations payloads.

Genuine refuted 1 / 0 (database / draft), genuine weakened 9, verifier-wrong 1.
