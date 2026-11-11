# Adjudication av-eaefd8 (genome 4b0eb5a6, churn_onehot)

Grouping the two verdicts gives 11 errors. The verifiers agree on 8 of them, and A alone found E9 to E11. I found no verifier finding that was wrong.

| Id | Layer | Statement | Genuine | Mech | Found by |
|---|---|---|---|---|---|
| E1 | database | voice-mail plan linear term -2.37 (hypothesis + record.summary) | refuted | d | A, B |
| E2 | database | international plan linear term +2.67 | weakened | d | A, B |
| E3 | database | rareStateUnits "zero measurable logit effect" (record.resisted) | refuted | e | A, B |
| E4 | draft | small state part moves the chance "by about a point at most" | refuted | d | A, B |
| E5 | draft | heavy-daytime part acts on 1033, "all without a voice-mail plan" | refuted | e | A, B |
| E6 | draft | day minutes raise it "only" without a plan / "add nothing" with one | weakened | e | A, B |
| E7 | draft | "about 5%" for 4.455% | weakened | f | A, B |
| E8 | database | 5395 rule (>=4 calls, or 3 with low use) | weakened | e | A, B |
| E9 | database | "<=2 calls: ~0" in the service-call table | weakened | e | A |
| E10 | draft | day charge lowers it "up to about 15.64 USD, then no further" | weakened | b | A |
| E11 | draft | evening charge counts "only" above 30.69 USD day charge | weakened | e | A |

Checks run:
- Claims C001, C002, C013 and C014: the voice-mail plan's direct effect is -1.96, against -2.39 for the whole model.
- A subgroup of voice-mail plan holders: the heavy-daytime part acts on 38 of them.
- Ablating rareStateUnits (largest risk change 8.5 points), plus a Utah / North Dakota probe (+0.34 / -0.24 logit).
- A probe of plan holders across daytime minutes and daytime charge.
- Subgroups at 2 or fewer calls with low day use (+0.36) and at exactly 2 calls.
- A sweep of daytime charge at 3 calls.
- A probe of evening charge against daytime charge.

Genuine refuted: 2 database / 2 draft; genuine weakened: 7 (3 database, 4 draft); verifier wrong: 0.
