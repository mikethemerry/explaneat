# Consistency reconciliation of the 12 adjudications

Protocol rule: `refuted` only if the stated value or claim is wrong under every reasonable reading of the sentence. Otherwise `weakened`. `none` if it is not an error. Below, one grade rule is given for each recurring kind of error, and that rule is applied to every error of the kind in all 12 items. The reading is taken from the sentence's own wording. A grouped error (one statement in several locations of the same layer) is graded by its most definite location. No errors were added or removed. No exact duplicates within an item and layer were found, so nothing was merged.

Every error in `adjudication_final/<item>.json` has a `kind`. Changed grades have `reconciled_from` and `reconcile_reason`. Where a kept grade might look like it conflicts with a rule, a `reconcile_note` explains why it does not.

## Kinds and rules

| Kind | Description | Rule | Justification |
|---|---|---|---|
| K1 | Plain-language number from a constructed case no real row has: a one-hot category moved while its linked numeric input stays fixed (education category with education-num 10), a male wife or female husband, a day-minutes change with the charge held, voice-mail messages without a plan | `refuted` when the sentence's subject is real people ("people whose highest education is ...", "very little schooling lowers it", "the model gives people whose ...") **and** the direction or comparison it states or implies is wrong for every real member of that group. `weakened` when the wording is a one-input change to the typical case ("a highest education of X lowers it", "being a wife rather than a husband ...", "for a typical customer ..."), or when the direction holds for real people and only the size comes from the constructed case. `none` when the sentence states the construction itself ("on top of the education-level effect") | If the wording is about real people and they go the other way, the claim fails under every reading. If the wording is a one-input change, the counterfactual is a reasonable reading of it, and the number reproduces under that reading |
| K2 | "only", "all", "never", "no", "nothing", "zero", "at most", an explicit bound ("< 0.03"), or a superlative ("more than anything else"), broken by a counterexample | `refuted` when real training rows (or the model's structure, e.g. a direct edge) break it and the sentence itself states no exception. `weakened` when the only counterexample is a constructed case outside the data, when the same sentence or passage states the exception, or when the sentence quantifies the exception group itself ("for the others it stays at about 2%") and that quantity holds | A universal word has no weaker reading. A real counterexample makes it false under every reading, unless the sentence supplies the qualification itself |
| K3 | A part's own effect stated as the whole-model effect, or the reverse (also a hinge's loss vs the input's total effect) | `weakened` when the number reproduces under the other scope (within rounding or inside the per-person range). `refuted` only if it fails under both scopes | The other scope is a reasonable reading of a sentence that does not exclude it |
| K4 | A typed or transcribed number (count, share, value, bound) that does not reproduce for the population or quantity the sentence names | `refuted` when it is outside the rounding of its own stated precision for the named population under every reading (exact counts off by any amount, a two-decimal value off by 0.018, a share for the wrong population). `weakened` when another reasonable reading reproduces it ("incl. 16 states" with 17, absolute vs relative error), or when it is only a rounding issue inside "about" | A wrong value with no alternative reading is the protocol's refuted case |
| K5 | Thresholds or turning points read from a coarse grid, or a median threshold given with "about" | `weakened` when the true value is within a grid step or within the stated spread, and the shape and place hold | "About" covers a displacement within the grid. (A group for which the threshold does not exist at all is K2, not K5) |
| K6 | "or" where the model needs "and", or a reversed condition | `refuted` when one named condition alone fails for the reference patient or for real patients under every reading. `weakened` when the sentence itself names the other condition as a modifier and holds for the reference patient | The conjunction is then partly stated, and the remaining error is one of size |
| K7 | A log-odds shape or comparison ("slowly then faster", "about the same", "more") stated on the chance scale | `weakened` | True on the score scale, which is a reasonable reading |
| K8 | An always-on or constant-offset part described as acting on a subset ("for a few states only", "rows it acts on") | `weakened` | True for the between-group differences, which is a reasonable reading |
| K9 | Implicit exhaustiveness: a list, ranking, rule or formula written as complete or exact when it is a simplification, with no universal word | `weakened` | The non-exhaustive or approximate reading is reasonable |
| K10 | A vague size word or rounding that hides or overstates an effect ("tiny", "small", "a little", "~0", "near-inert", "about 1% both ways", a dropped "slightly") | `weakened` | Scale-dependent words have a reading under which they hold |
| K11 | A typical-case value or a code-computed exception attributed to a whole group, or a group value applied to an individual | `weakened` when the value holds for the typical member or for part of the group | The typical-member reading is reasonable |
| other | One-off errors (comparison base unstated, a citation gap, "nearly all" for all, a domain-sense generalisation, a right count with the wrong denominator, "reproduces" for direction only, a verifier misreading) | Adjudicator's grade kept; none violates the protocol rule | |

## Grades changed (8)

| Item | Error | Kind | From | To | Reason |
|---|---|---|---|---|---|
| av-0491c5 | E1 | K2 | weakened | refuted | "Adds to the risk only once ST depression rises above about 0.9 mm", with no spread or exception given. 18 real training patients (normal ECG, reversible defect, oldpeak < 0.8) have ann_34 acting; probe gives ann_34 +0.05, +0.25, +0.49 and risk 34.6%, 39.2%, 45.2% at 0, 0.4, 0.8 mm. Same rule as av-7e63ce E10 |
| av-0491c5 | E2 | K2 | weakened | refuted | The same unqualified "only once ... above about 0.9 mm" in the ann_37 sentence. ann_37 acts on all 18 of those patients and rises from +0.08 to +0.52 over 0 to 0.8 mm |
| av-7d3005 | E3 | K2 | weakened | refuted | "Active only when the score is quiet" / "active only without ischaemia evidence": 17 of 135 real patients have 1987 on with 290 on. The 13% exception appears only in the ischaemia_pathway passage. The heart_disease_logit hypothesis and record.summary state "only" without it |
| av-eaefd8 | E6 | K2 | weakened | refuted | "With a voice-mail plan they add nothing: the chance stays below 1%". 47 real plan holders have more than 270 day minutes, and ann_79 acts on 38 of them (mean +0.21; typical case 8.9%). Same rows and rule as E5 in this item, which was already refuted |
| av-eaefd8 | E1 | K3 | refuted | weakened | -2.37 is the whole-model voice-mail-plan effect (mean -2.39, per-customer range -3.64 to -1.96), labelled as the linear term. It is the same slip as E2 (+2.67 under the same label), which is graded weakened |
| av-fbe877 | E11 | K1 | weakened | refuted | "For people whose highest education is 9th grade the fall is smaller, from about 44% to about 6%": every real 9th-grader starts near 84%, higher than the others' 63%, and falls by 55 points against 54. Same sentence and rule as av-7e63ce E1 (refuted) |
| av-ff605d | E6 | K4 | weakened | refuted | The payload gives "active_rows: 304". node_7208 > 0, > 0.01 and > 0.05 each give n = 306. Same rule as av-fbe877 E1 (27 for 26) |
| av-ff605d | E9 | K4 | weakened | refuted | "6730 (constant +0.43 logit ...)": the probe gives 6730 = 1.305, which contributes +0.448 (the same sentence's -0.45 when it switches off); the part contributes +0.44. Same rule as av-fbe877 E2 (0.155 for 0.137) |

## Grades checked against a rule and kept

- K1: av-75bfdb E1 stays refuted (it speaks of real people and the direction is reversed). av-7e63ce E1 stays refuted. av-981a97 E1 and av-fbe877 E10 stay weakened (they change one input for the typical person). The wife, woman, not-married and typical-case items (av-75bfdb E2 to E5, av-7e63ce E2 and E13, av-981a97 E2, av-fbe877 E12) and the linked-usage items (av-a3ee6a E2 and E4, av-b528c6 E6) stay weakened, because the direction holds for real people. av-7e63ce E3 stays none.
- K2: these stay refuted: av-7d3005 E2, av-7e63ce E10, av-a3ee6a E1, av-eaefd8 E3, E4 and E5, av-ff605d E1. These stay weakened: av-ff605d E3 and E4 (no real counterexample), av-fbe877 E5 (the exception is in the sentence), av-fbe877 E4 (the sentence names the indicators), av-eaefd8 E11 (the sentence's own "stays at about 2%" for the others holds: 2.3% to 2.6% on real rows).
- K4: these stay refuted: av-7d3005 E1, av-981a97 E4, av-fbe877 E1 and E2, av-fefc55 E6. These stay weakened: av-a3ee6a E5 (the relative-error reading holds), av-eaefd8 E7 (rounding inside "about"), av-ff605d E10 ("incl.").
- K6: av-7d3005 E6 and E7 stay refuted (a normal ECG alone fails for the reference patient). av-7d3005 E4 stays weakened (the ECG is named as a modifier, and the error is "a little").

## Final counts

| Item | Refuted database | Refuted draft | Weakened database | Weakened draft | None |
|---|---|---|---|---|---|
| av-0491c5 | 0 | 2 | 2 | 1 | 0 |
| av-75bfdb | 0 | 1 | 1 | 6 | 0 |
| av-7d3005 | 3 | 2 | 2 | 4 | 0 |
| av-7e63ce | 0 | 2 | 0 | 10 | 1 |
| av-8a4d18 | 0 | 0 | 0 | 1 | 0 |
| av-981a97 | 1 | 0 | 1 | 3 | 0 |
| av-a3ee6a | 0 | 1 | 1 | 3 | 0 |
| av-b528c6 | 0 | 0 | 3 | 4 | 0 |
| av-eaefd8 | 1 | 3 | 4 | 3 | 0 |
| av-fbe877 | 2 | 1 | 3 | 7 | 0 |
| av-fefc55 | 1 | 0 | 0 | 5 | 0 |
| av-ff605d | 3 | 0 | 5 | 2 | 1 |
