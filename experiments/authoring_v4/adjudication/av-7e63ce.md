# Adjudication av-7e63ce (genome 7a8237f3, adult_ohe_v2)

Both verifiers agree that every number in the database layer reproduces. All their findings are about the plain-language draft. I grouped 18 graded findings into 13 errors.

**Refuted (draft, 2)**
- E1, C137: "9th-grade people: 44% -> 6%". The case keeps education level at 10. No 9th-grade row has that (n = 0), because every 9th-grade row has level 5. With real coding the model gives 84.4% -> 29.2%. B refuted this and A weakened it; I side with B, because the sentence names a group of real people.
- E10, C161: "only once age rises above about 46 years". For the 20 Thailand-born training people under 46, the part is on and acts (20 of 20); at 45 years it already raises the score (+0.33). Both verifiers graded it weakened, but under the rule for "only" a real counterexample makes it refuted.

**Weakened (draft, 10)**
- Part-alone values given as the model's chance: E4 Yugoslavia (33% vs 41.7%), E5 state government (63% vs 71.2%), E6 some college (64% vs 60.2%), E7 Trinidad (92% vs 94.1%).
- Shape stated on the wrong scale: E11, "slowly at first, then faster" holds for the log-odds, but the chance falls about linearly.
- Off-support or constructed cases: E2 typical "high-school graduate" at level 10 (a real one gets 71.5%), E12 Ecuador "37 people" (1 of them has gains above 6000), E13 wife 41% / woman 74%.
- Grid-point thresholds: E8 lowest chance at "47" years (it is 46.1), E9 thresholds 5258 / 4883 dollars (about 5160 / 4790).

**Verifier wrong (1)**
- E3, C016, the 7th-8th grade case: it is explicitly scoped "on top of the education-level effect" and is already a worth-checking flag.

Counts: refuted 0 database / 2 draft; weakened 10; verifier-wrong 1.
