# Verifier benchmark score

Items: 48 of 48 run ({'control': 12, 'faulted': 24, 'paraphrase': 12}).

## Detection of injected faults

| | n | refuted (strict) | weakened or refuted (lenient) | at the exact field or claim |
|---|---|---|---|---|
| all | 50 | 84% (42/50) | 96% (48/50) | 86% (43/50) |
| L1 | 16 | 100% (16/16) | 100% (16/16) | 100% (16/16) |
| L2 | 17 | 71% (12/17) | 88% (15/17) | 82% (14/17) |
| L3 | 13 | 100% (13/13) | 100% (13/13) | 77% (10/13) |
| L4 | 4 | 25% (1/4) | 100% (4/4) | 75% (3/4) |
| causal_overclaim | 3 | 33% (1/3) | 100% (3/3) | 100% (3/3) |
| certainty_overclaim | 1 | 100% (1/1) | 100% (1/1) | 100% (1/1) |
| changed_number_claim | 5 | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| changed_number_evidence | 2 | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| closed_form_coefficient | 3 | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| evidence_from_other_part | 1 | 0% (0/1) | 0% (0/1) | 0% (0/1) |
| flipped_direction | 4 | 100% (4/4) | 100% (4/4) | 100% (4/4) |
| population_overclaim | 2 | 50% (1/2) | 100% (2/2) | 100% (2/2) |
| restates_closed_form | 4 | 25% (1/4) | 100% (4/4) | 75% (3/4) |
| rule_threshold_moved | 2 | 100% (2/2) | 100% (2/2) | 50% (1/2) |
| structural_missing_node | 4 | 100% (4/4) | 100% (4/4) | 100% (4/4) |
| subgroup_effect_not_holding | 2 | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| subgroup_effect_reversed | 1 | 100% (1/1) | 100% (1/1) | 100% (1/1) |
| swapped_hypotheses | 1 | 100% (1/1) | 100% (1/1) | 100% (1/1) |
| threshold_moved | 2 | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| wrong_claim_id | 2 | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| wrong_interaction_partner | 5 | 100% (5/5) | 100% (5/5) | 60% (3/5) |
| wrong_shape | 3 | 100% (3/3) | 100% (3/3) | 67% (2/3) |
| wrong_subgroup_n | 3 | 67% (2/3) | 67% (2/3) | 67% (2/3) |

## Level at which faults were caught

| fault level | caught at |
|---|---|
| L1 | L1: 16 |
| L2 | L1: 7, L2: 5, L3: 3, missed: 2 |
| L3 | L1: 7, L2: 3, L3: 3 |
| L4 | L1: 1, L4: 3 |

## False positives (controls and benign paraphrases)

| variant | any refuted | any weakened or refuted | overall grades |
|---|---|---|---|
| control | 33% (4/12) | 67% (8/12) | {'refuted': 4, 'upheld': 4, 'weakened': 4} |
| paraphrase | 17% (2/12) | 75% (9/12) | {'refuted': 2, 'upheld': 3, 'weakened': 7} |
| both | 25% (6/24) | 71% (17/24) | |

Faulted items: overall grades {'refuted': 24}; 19 findings matched no injected fault and no known pre-existing issue.

Time per item: mean 4.9 min, median 3.0 min (n = 48); by variant control 3.6, faulted 4.8, paraphrase 6.3.
