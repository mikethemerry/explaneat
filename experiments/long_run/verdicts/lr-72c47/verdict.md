# Verdict: lr-72c47 (genome 2e9f788d-56ef-4ddd-84e1-43ec57236fe7)

**Overall: refuted**, on a single count word: the record says "most states" have no edge, but 31 of the 51 state indicators feed the score. Every other number, threshold, shape and subgroup count I checked reproduces. A few statements are weakened.

Findings: 2 refuted, 6 weakened, 22 upheld (165 statements checked). The record has no `target_hash`, because the recheck report for this kind of record carries none.

## Refuted and weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.churn_model.hypothesis | L1, refuted | "Inputs with no edge (most states, ...)" | Only 20 of 51 states have no edge. 31 do (3 + 1 + 5 + 22 across the four state leaves) | recheck null nodes; entries of the state leaves |
| record.overall_function | L1, refuted | "Most state indicators ... have no edge" | Same as above | same |
| record.overall_function | L3, weakened | heavy use moving the calls threshold is "the one interaction in the model" | The voice-mail plan gates the day-use effect. At 300 day minutes the risk is 99.9% without a plan and 0.4% with one. At 150 minutes it is 1.5% both ways | probe day minutes x plan |
| operations.churn_model.hypothesis | L3, weakened | "The joint effect inside the model is heavy use with customer-service calls" | The heavy-use detector is itself a joint effect of day use, evening use and the plan | same probe |
| record.summary | L2, weakened | a "heavy daytime use" detector | 124 of the 360 customers with the detector on (34%) have under 250 day minutes. Evening use also switches it on | subgroup `node_469 > 0 and day minutes < 250` |
| operations.other_state_units.hypothesis | L1, weakened | "Four single-state units" | Unit 15308 reads both Vermont and Delaware, and Delaware enters only through it. All the risks and exits reproduce | incoming edges per unit |
| operations.state11_units.hypothesis | L1, weakened (rounding) | AUC 0.920 -> 0.919 | 0.91954 without the part, which rounds to 0.920. The stated value was rounded twice (from a stored 0.9195) | ablate --part ann_84 |
| operations.eve_night_hinges.hypothesis | L1, weakened (rounding) | AUC 0.920 -> 0.919 | 0.91954, same double rounding | ablate --part ann_86 |

## Upheld parts

- **usage_service_calls (ann_82):** The calls curve reproduces exactly (1.2%, 1.1%, 2.5%, 9.9%, 66.7%), and unit 4823 is off from three calls. The day-minutes curve reproduces too: the detector comes on between 250 and 255 minutes without a plan and stays off with one. So does the joint effect at 270 minutes (52.7%, 50.9%, then 71.2% at five calls). The counts 360, 354 and 6, and 318 vs 3682 customers, reproduce, as does the AUC drop to 0.658.
- **state35_chain (ann_83):** the exits, risks, group sizes and churn rates reproduce.
- **tenure_hinge (ann_87):** the hinge sits between an account length of 84 and 85 (the data give no unit), the unit is on for 2604 customers, and the risk goes from 1.1% to 2.3%.
- **area_code_offsets (ann_88):** reproduces; the model makes almost no use of area code.
- **plan_and_calls_linear (ann_89):** 1.1% vs 15.5%, 19.2% at one international call and 11.5% at eight, the counts 380 and 3620, and the AUC drop to 0.841 all reproduce.
- **state_direct_linear (ann_90):** the order of the largest offsets is right (California, Washington up; Virginia, Rhode Island, Georgia, Colorado, West Virginia down). The part is honestly labelled closed-form only. Idaho has no edge at all.
- **state_effects (ann_91):** the parts add on the score, and the extremes are Hawaii, Vermont, California and Washington.
- **Record:** coverage 1.0 / 1.0, function preservation 4.4e-16, closed-form error 0.032 and the operation counts all reproduce.

## Flags

None. Heavier evening use lowering the risk on its own fits the data: among customers with the detector off, those with an evening charge above $20 churn at 7.1%, against 8.1% below $14.
