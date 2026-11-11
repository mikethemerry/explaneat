# Adjudication av-fbe877

13 errors, from 11 findings by A and 11 by B.

**Genuine refuted (database 2, draft 0)**
- E1: "Thai respondents (27 rows)": there are 26 (subgroup). A typed count.
- E2: "State-gov gives 0.155" for node 1624: the value is 0.1368 for every state-government row, and no training row exceeds 0.15. Found by A only.

**Genuine weakened (database 3, draft 8)**
- Database:
  - E3: the 2990 formula leaves out the Thailand term.
  - E4: "all nonlinearity is univariate" overclaims, because country and education indicators switch the hinges.
  - E5: "zero for everyone else", while node 1624 is on for 4.1% of rows (the text does name State-gov).
- Draft:
  - E6: the age minimum "about 47" is really 46 (an artefact of the code that computes the claims' turning points).
  - E7: "about 4883" should be about 4795 (an artefact of the same code).
  - E8, E9: the higher starts for Jamaica and Ecuador are whole-model numbers credited to the capital-gain parts.
  - E10, E11: the 7th-8th, 9th and 10th grade figures and the 9th-grade capital-gain start hold education-num at 10, a combination with n = 0.
  - E12: the typical "high-school graduate" sits at education-num 10.
  - E13: "50 hours" is missing from its citation.

**Disagreement resolved:** B graded E10 refuted. The figures reproduce exactly as the education category's own effect with everything else held, which is how the draft reads every one-hot effect, and the item is already in the worth-checking list. So E10 is weakened (an off-support constructed case), not refuted, even though read plainly about real people the direction is reversed.

Verifier-wrong: none.
