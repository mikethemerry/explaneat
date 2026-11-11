# Adjudication av-75bfdb

Both verifiers agree that every stated number reproduces (162/162 claims). Every finding is about reference cases or scope in the plain-language draft, plus one scope slip in the record summary.

| Error | Location | Verifiers | Genuine | Layer | Mech |
|---|---|---|---|---|---|
| E1 | very little schooling lowers it, 1st-4th 10% / 7th-8th 19% (readdown ann_115, blocks[0], worth_checking[0]) | A ref, B ref | refuted | draft | d |
| E2 | typical high-school graduate 62% (lead) | A wk, B wk | weakened | draft | d |
| E3 | wife 27% vs husband 62% | A wk, B wk | weakened | draft | d |
| E4 | not married 87% | A wk, B wk | weakened | draft | d |
| E5 | women 80% vs men 62% | A wk, B wk | weakened | draft | d |
| E6 | a doctorate makes no difference for Cambodia-born | B ref | weakened | draft | d |
| E7 | capital gains lower it slowly then faster | B wk | weakened | draft | d |
| E8 | record.summary: ann_116 "16,819 rows it acts on" | B wk | weakened | database | d |

Key checks:
- E1: the 9.9% / 19.3% need education-num held at 10, which no real person has. With matching levels the probe gives 89.8% / 86.5% against 73.8% for a high-school graduate, so the direction is reversed for real people.
- E6 is downgraded from B's refuted. At a fixed education level the doctorate category changes nothing for Cambodia-born people in the whole model (65.6% both ways), so the sentence holds model-wide and not only through the part. It fails only if "doctorate" includes the jump to level 16. No Cambodian with a doctorate exists in training (n = 0).
- E8: ann_116 adds -0.43 to every row aged 35 or over, so it acts on all rows. The 0.98 and the 16,819 are right only for the under-35 group.

Counts: refuted 0 database / 1 draft; weakened 7 (1 database, 6 draft); verifier wrong 0.
