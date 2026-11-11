# Adjudication av-a3ee6a

Both verifiers agree that every stated number reproduces. The errors they found are all about framing, and four of the five are in the plain-language draft.

| Id | Layer | Location | Grades (A / B) | Genuine | Mech. |
|---|---|---|---|---|---|
| E1 | draft | ann_100.hypothesis[0] (C001): international plan raises risk "more than anything else" | - / refuted | refuted | e |
| E2 | draft | heavy day use 2% -> 7% at 271 min (C058 and others; several sentences) | weakened / weakened | weakened | d |
| E3 | draft | day charge above $41.59: 2% -> 8% (C116) | weakened / weakened | weakened | d |
| E4 | draft | voice-mail messages 2% -> 6%/10%, plan below 1% (C002, C067 and others) | weakened / weakened | weakened | d |
| E5 | database | dayEveChargeCascade formulas "all off by up to 0.0077" | weakened / - | weakened | f |

Checks:
- E1: going from 0 to 6 service calls takes the typical customer from 2.3% to 85.7%, and 6 calls is within the normal range of the data. Heavy daytime use with minutes and charge moved together reaches 50% at 271 minutes. Both are larger than the plan's 2% -> 24%.
- E2 and E4: no training row has more than 270 day minutes with a day charge under $40. No customer without a voice-mail plan has any messages. The stated numbers come from combinations of inputs that never occur.
- E3: changing the charge across the whole model gives 15.9% -> 11.9%, not 2% -> 8%. The threshold holds.
- E5: node 5604 is off by 0.0177 (relative error 0.0034). The coefficients are right.

Counts: genuine refuted 0 database / 1 draft; genuine weakened 4 (1 database, 3 draft); verifier-wrong 0.
