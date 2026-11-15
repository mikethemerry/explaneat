# Verdict: lr-10a59 (genome d03ac194)

**Overall: refuted.** Every stated number, closed form, coverage and function-preservation figure reproduces, but the daytime leaf calls evening minutes a minor partner of the heavy-use unit when, at a heavy day charge, evening use moves the predicted risk from 5% to 93%. The service-call reading also misses that 0 to 2 calls change nothing, and "hardly for heavy users" holds only above about 50 dollars.

Verified 2026-11-15, 19:16 to 19:23 (+13:00). Checks were on the training split (n = 4000), read only. 168 statements checked.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.daytime_use.hypothesis (V1, **refuted**) | L3 | The heavy-use unit grows with the day charge "and a little with evening minutes" | At a 48-dollar day charge the risk is 5.1% / 45.7% / 93.0% at 100 / 200 / 300 evening minutes. Data, day charge 40-45: predicted 5.9% (under 150 min) against 53.8% (250+ min). Per sd, evening minutes weigh about half as much as the day charge | probe with consistent minute/charge pairs; subgroups |
| operations.daytime_use.hypothesis (V2) | L3 | The "heavy-use unit" 2470 is pushed down by calls | It is on for 3307 of 4000 customers, including every customer with 0 to 2 calls at any day charge. It works as a "few calls" gate | probe --nodes; node counts; concept (calls AUC 0.98) |
| operations.service_calls.hypothesis (V3) | L3 | Risk "rises steeply with calls" (1.4% at 1 call, 16.1% at 4, ...) | Flat from 0 to 2 calls (1.4% / 1.4% / 1.4%), then steep. The onset moves from 3 calls (up to 35 dollars) to 8 calls (55 dollars) | probe grid of day charge by calls |
| record.summary (V4) | L3 | Calls raise risk "hardly for heavy ones (90.9% and 90.1% at 55 dollars)" | Only 16 customers have a day charge of 55 dollars or more. At 45 dollars (the summary's own heavy threshold) the risk rises from 22.7% at 4 calls to 64.7% at 6 | probe grid; counts |
| operations.usage_and_plans.hypothesis (V5) | L3 | The heavy-use unit cancels the calls edge, so calls matter for light users, not heavy ones | The cancellation (1.75 against 1.73 per sd) applies to everyone while 2470 is on. Heavy users only keep it on to more calls | node values, probe |
| record.overall_function, summary, clinical_terms (V6) | L3 | Driven "mainly by daytime use", "heavy daytime users" | The high end of the U depends on day plus evening use. Evening minutes are never named as a driver | as V1 |
| operations.daytime_use.hypothesis (V7) | L3 | 98.3% / 98.1% at 60 dollars | Reproduces, but 60 dollars is outside the data (max 59.76) | probe range |

**Upheld parts:**
- service_calls: all of its numbers reproduce. It is weakened only on shape (V3).
- night_use: the small rise, steeper below about 125 minutes, reproduces, and the chain is off for 257 customers.
- evening_calls_hinge: on from 116 calls (857 customers), 14.6% against 14.0%.
- intl_charge_hinge: on from 2.70 dollars (2258 customers), 15.6% against 12.2%.
- direct_usage_terms: the edges, the plan probes and the subgroups reproduce.
- state_single_units: each unit fires exactly for its state, and the Georgia unit is dead.
- state7_chain: fires for exactly the 73 DC customers, contribution -1.84.
- state42_night_calls_chain: it cannot switch on (0 rows).
- direct_location_terms: the offsets and the small groups reproduce.
- location: the per-state offsets reproduce.
- churn_risk: the AUCs (0.568 and 0.894) and the international-plan probes reproduce.
- Record: coverage 1.0 and 1.0, preservation 4.4e-16, the closed forms identical, the ignored inputs correct.

**Flag (L4):**
- direct_location_terms (V8): logit offsets of -2.7 for Hawaii rest on 3 churners among 73 customers, and +1.5 for California on 11 among 41. Effects this large at state level are probably overfitting. The record itself says they are fitted to small groups.
