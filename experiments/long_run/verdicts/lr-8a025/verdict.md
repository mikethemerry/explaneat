# Verdict: lr-8a025 (genome a5e92f55, telecom churn)

**Overall: weakened.** Every stated number reproduces (recheck: 0 mismatch, 0 needs review; 45 free-text sentences checked by hand) and no hypothesis is contradicted, but six statements are imprecise. 0 refuted, 6 weakened, 22 upheld, 2 flags.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.summary | L3 | heavy-use hinge opens "when the day charge and evening minutes are both high" | a threshold on combined use: 58 dollars of day charge with 60 evening minutes gives 99.7%; 350 evening minutes with a 35-dollar day charge gives 100%; 66 of the 267 open-gate customers have below-median evening minutes | probe day x evening sweeps; subgroup `node_469 > 0 and eve minutes < 201.4` |
| operations.heavy_use_gate.hypothesis | L3 | with a voice-mail plan the risk "crosses over at 60 (93%)" | reproduces, but the crossover (50% at about 59.5 dollars) is at the edge of the data (max 59.76). No voice-mail customer has a day charge above 58, so this is extrapolation. The data only show that the risk stays low (13 customers above 50 dollars, none churned) | probe `voice mail plan=1`, day charge 57:60; subgroups |
| operations.heavy_use_gate.hypothesis | L2 | "with the gate closed, node 4823 ... is on for 3207 customers" | 3207 counts all rows. With the gate closed the count is 2943 | subgroup `node_4823 > 0 and node_469 == 0` |
| operations.eve_charge_hinge.hypothesis | L3 | switches on "above roughly the typical value" | switches on at about 15.3 dollars, against a median of 17.12. It is on for 65% of customers | probe eve charge 5:30; subgroup `node_263609 > 0` |
| operations.account_length_gate.hypothesis | L2 | "51 days", "201" | the numbers reproduce, but the data give account length without a unit, so "days" is assumed | verify vars; probe |
| structure.night_calls_tangle | L1 | record: meaning "partial"; hypothesis: "described rather than explained" | the meaning grade stored on the annotation says "meaningful" | annotate operation's `meaning` vs record `leaves` |

**Upheld parts** (each number checked by probe, subgroup or ablate):
- churn_score (root): customer-service-call risk curve, the 318 customers with four or more calls, unconnected inputs, direct edges' AUC 0.928 to 0.658.
- direct_usage_terms: weights, international-plan shift 2% to 25%, equal steps, AUC 0.668.
- direct_state_offsets: top-five weights, state probes, Ohio cancellation (-3.91 vs +4.50), AUC 0.918.
- day_charge_hinge: values, 1182 customers, dead unit.
- state_gate_pair: Texas only, on 95 of 95 Texas rows and no others. The evening-charge path is dead and Alabama never opens the gate.
- day_calls: values, 3309 customers.
- intl_plan_hinge: on for 380 of 380 plan holders and none of the others.
- state_gates_lower and state_gates_raise: per-state values and subgroups.
- single_state_boost: the net New Jersey effect is lower risk at every usage and call level tested.
- night_calls_tangle: node values and negligible effect.
- heavy_use_gate: everything not listed in the table, including the 267 customers at 90.3% churn, the area-code and Ohio terms, and AUC 0.649.
- Record coverage (1.0), function preservation (6.7e-16), closed-form error (0.027), operation counts and the resisted text.

**Flags (domain, not refutations)**
- state_gates_lower: Hawaii and Washington DC get -2.1 on the score. The training data agree (6.2% churn, 9 of 146), but on test these customers churn at 17.9% (5 of 28) while the model predicts 8.8%. The offset is likely fitted to noise in small state samples, and the same caution applies to the other state parts.
- day_charge_hinge: below the heavy-use threshold, predicted risk falls as day charge rises (1.6% at 35 dollars, 0.6% at 42.5), then jumps to 28.5% at 45 dollars. That dip conflicts with the usual reading that heavier day use raises churn.

`target_hash`: the recheck report for a record target has no target_hash. It is null in verdict.json.
