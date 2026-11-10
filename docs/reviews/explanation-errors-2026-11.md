# How errors got into the stored explanations, and how to stop them

Review of the 2026-11-11 errata (E1 to E15), the `recheck` v2 sweep and verifier-benchmark run 1. Written 2026-11-11 on branch `review/explanation-errors`. It is read-only: no records or database rows were changed. The reproduction scripts and their outputs are in `docs/reviews/explanation-errors-2026-11/`. "The explaining agent" is the AI agent that built each explanation, and "the drafting agent" is the AI agent that wrote the plain-language prose.

## Executive summary

- **No error came from the MCP tools' default sample.** I recomputed every numeric error on four row sets: train, the 200-row default, test and train+test. None reproduces the wrong value. The default sample does reproduce two *listed* AAAI numbers exactly: heart-paper's 70% and 17.5% are a 200-row, seed-42 sample of train+test.
- **The errors came from hand work on correct outputs.** Seven of the 15 are transcription or drafting slips (f):
  - a count back-derived from a rounded share (27 rather than 26);
  - a table summed by hand (108 rather than 107);
  - three percentages rounded twice;
  - two missing citations.
- **Three are scope or unit confusions (d).** Two of them (E8, E9) are correct whole-model averages that the text labelled as the direct linear term. The third (E6) treats a z-scored one-hot "off" level as 0.
- **The rest:**
  - two are simplifications written as exact (e);
  - one used one regime cell's mean for a two-cell group (b);
  - one is a claim-generator semantic gap that the drafting agent then amplified (g).
- **None of the errors is stale (c).** The operation model is effectively immune to staleness, because evidence lives inside its annotation. One caveat: 63 of 80 drafts had their claims regenerated and were only re-linted, not re-read.
- **One erratum is itself wrong.** E13's "about 20%" was correct: the typical patient's risk is 19.80%. The new rounding lint matched "20%" to an unrelated category share (0.193) in a cited claim, and the errata changed correct text to "about 19%". That should be reverted.
- **One more genuine error is not in the errata.** A benchmark verifier refuted heart-s5's root claim that thal:7 acts "ONLY through these gates": it also has a small direct edge. It is an (e)-type overclaim and should be added.
- **Where the errors entered:**
  - 8 of the 15 were written by the explaining agent into free text (hypotheses, narratives, hand-typed payloads);
  - 5 by the drafting agent;
  - 1 by the claim generator;
  - 1 by the errata process itself.
- **Nothing checked the free text at write time.** `annotate`, `update_narrative` and `add_evidence` store any text and any payload. Evidence cannot be amended, and the outputs carry no split, row count or provenance.
- **`recheck` v2 is not a sufficient safety net.**
  - It reached only 1 of the 8 database-text errors (E3).
  - Of its "mismatch" findings, 1 in 53 was genuine.
  - 465 numeric sentences in the ten corrected genomes were outside its patterns.
  - The 47 false positives are mostly a referent problem (which node, which population, which quantity), which no regex can solve.
- **The time box was not the pressure.** The median explaining agent finished in 17 of its 180 minutes.
- **Top fixes, in order:**
  1. Revert E13 and fix the percentage lint.
  2. Give the explaining agents the verifier's query tools (`subgroup`, `probe`, `enumerate`) and make every stated number a typed reference to a stored query, carrying its split, population, scope and n.
  3. Validate text at write time, with an amend path for evidence.
  4. Have code fill and round the numbers in the drafts, and derive the citations from them.
  5. Fix the generator's exception rule.

  `recheck` at the end of the protocol and "≈" formulas are worth having but catch little on their own.

## Part 1: root cause of each error

**Mechanisms:**
- (a) the MCP default sample;
- (b) a different denominator, split or subgroup;
- (c) stale;
- (d) a unit, scale or scope confusion;
- (e) a simplification written as exact;
- (f) a transcription or drafting slip;
- (g) a claim-generator artefact;
- (h) unknown.

**Sources:**
- "Script" refers to a file in `docs/reviews/explanation-errors-2026-11/`.
- "Transcript" refers to the explaining or drafting agent's session log, cited by its line number.
- `split_sensitivity.py` tests mechanism (a) for E1, E2, E5, E6 and E11, and `e8_e9_churn100_s1.py` tests it for E8 and E9. The other errors are prose, formula, citation or generator errors, for which the row set does not apply.

| # | Explanation | Wrong -> correct | Mechanism | Evidence (reproduction) | Confidence |
|---|---|---|---|---|---|
| E1 | heart-s5 `ischaemia_pathway` hypothesis and narrative | mean predicted risk of the 35 low-BP patients 0.13 -> 0.18 | (b) **mismatched subgroup.** The agent summed two cells for n and copied one cell's mean. | The transcript shows the steps: the `regime_map` output (line 127, train) has 30 rows with "290 on, 204 off, 1987 off" (mean 0.1345) and 5 with "1987 on" (mean 0.4623). The agent wrote "35 ... mean 0.13" 13 s later (line 129). `e1_heart_s5.py` reproduces 0.1345 for 30 rows and 0.1813 for 35. On the other row sets the 35-row mean is 0.21 (train-200), 0.34 (test, the default) and 0.21 (train+test), so it is not (a). | confirmed |
| E2 | adult-s5 `age_profile` hypothesis | Thai respondents "(27 rows)" -> 26 | (f) **count back-derived from a rounded share** | The transcript shows the agent's own table printing "2134 frac_active 0.0007" (line 48); 0.0007 x 39,073 = 27.35. The hypothesis was written at 23:28 (line 68), before the agent counted the Thai rows (26, line 86, 23:31). The later evidence narrative says 26; the hypothesis was never updated. Script: `split_sensitivity.py`. | likely (the sequence is confirmed; the multiplication is inferred) |
| E3 | adult-s5 `age_profile` hypothesis | `2990=relu(1.446 age-0.48)` -> `+0.137*2134` | (e) **simplification written as exact** | The agent's own edge table lists 2990's inputs as age 1.446 and 2134 0.137 (transcript line 30). The term is zero except on the 26 Thai rows, so the agent dropped it but kept "=". `recheck` flags it (max error 4.45 on 26 rows). | confirmed |
| E4 | adult-s5 plain-language draft | "about 50 hours a week" cited C143 only -> also C142 | (f) **citation slip** by the drafting agent | The number is right (C142's threshold is 49.9 h). The sentence cited C143 from the first v5 draft onwards (commit `3d82914`). Script: `prose_history.py`. | confirmed |
| E5 | churn100-s4 `area415_state47` hypothesis | "zero in area 415" -> zero for 1,986 of 2,016; 1.03 for the 30 Washington rows | (e) + (d) **incomplete hand case analysis** of a z-scored formula | The agent evaluated relu(-0.924 + 0.41 state47 - 1.077 area415) for three of the four cells and skipped area 415 with WA (transcript line 71). Its own printout "595 active frac 0.503" (= 2,014 / 4,000, which includes the 30 WA rows) contradicts "zero in area 415". Script: `split_sensitivity.py`. | confirmed |
| E6 | churn100-s4, same hypothesis | "~0.15" / "0.16 for other areas (logit +0.03)" -> 0.11 (logit +0.02) | (d) **z-scaled "off" level taken as 0** | Hand evaluation with state 47 "off" at z = 0 gives relu(-0.924 + 1.077 x 1.008) = 0.1617. The true off level is z = -0.135, which gives 0.1061. The agent's own heatmap payload records the -0.135. Script: `split_sensitivity.py` (cell table and hand evaluation). | confirmed |
| E7 | heart-s2 draft (C30) | "for patients with a reversible defect ... it rises in a jump: from about 12% to 28%" -> this part is off at every age for the typical patient with a reversible defect | (g) **generator semantic gap**, then (f) the drafting agent attached the typical patient's numbers to the exception | The relevant code is `plain_language_shapes.py`, lines 376-397. It names an exception group by **precision only**: at least 80% of the patients whose shape is the exception share a category, at least 30 points more than among the others. It does not check what share of that category shows the exception. It also drops unaffected (flat) patients before characterising the group, although most reversible-defect patients are unaffected. C30 records 25 "step" patients against 152 "hinge" ones. C33, which was not cited, gives this part's effect as 0 in the reversible-defect group. The template text ("moves mostly in a jump") still carries the exception. | confirmed |
| E8 | churn100-s1 root hypothesis and record summary | voice-mail plan "-2.37 for holders" -> -1.96 | (d) **whole model vs direct path.** The number is right for another quantity. | The whole-model mean change in logit (plan switched off -> on) on train is -2.3915 over all customers and **-2.3679 among holders**. The direct edge gives -0.8641 / 0.4415 = -1.957. The 3460 gate adds the rest. No transcript line shows the computation, but the 3-figure match for both E8 and E9 is strong. Script: `e8_e9_churn100_s1.py` (also rules out (a): w/sd on any split is -1.94 to -1.97). | likely (strong) |
| E9 | churn100-s1, same | international plan "+2.67" -> +2.62 | (d) as for E8 | The whole-model mean is +2.6676 (+2.6664 among holders), against +2.6226 for the direct path. | likely (strong) |
| E10 | churn100-s1 draft | "about 5%" -> about 4% | (f) **double rounding** (4.455% -> 4.5% -> 5%) during the v5 re-draft | The cited C086 value was 0.04455 when it was drafted (commit `88db9ae`, drafted in a batch of several labels). The **earlier v3 draft said "about 4%"** for the same claim (commit `a8ba540`), so the re-draft introduced the error. Script: `prose_history.py`. | confirmed |
| E11 | monk2-s0 root evidence payload | "108/108 pos" -> 107/107 | (f) **hand-summed table** in a hand-typed payload, which could not be amended | The agent's own train table gives 8+32+35+22+9+1 = 107 (transcript line 61). The payload was a dict literal typed into an `add_evidence` call (line 87). The agent noticed the error, but `add_evidence` is append-only: it fixed the narrative and disclosed the payload error in `resisted`. Script: `split_sensitivity.py` (107 on train; 61, 37 and 144 on the other row sets). | confirmed |
| E12 | churn100-s7 draft | "about 15%" -> about 14% | (f) double rounding (14.48%) | C055's `risk.to` was 0.144801 at the first draft (`3c28916`). | confirmed |
| E13 | heart100-s0 draft lead | "about 20%" -> "about 19%" | **Not an error. The correction is wrong.** It is a lint artefact. | The lead describes the typical patient. The layer's own `reference_patient.predicted_risk` is 0.197958, recomputed here as 0.197958, so 19.80% -> "about 20%" is correct. The lint added on 2026-11-11 compares only against the cited claims. Its risk-key regex accepts "share", and C55 contains `evidence.who.categories[3].share_out = 0.192982`, a category share and not a risk. The lint matched it and declared "rounded twice". Script: `e13_heart100_s0.py`. | confirmed |
| E14 | heart-s5-c1 draft | "about 28% at 118", "about 27% at 66" -> 27%, 26% | (f) double rounding (27.4963%, 26.45%) | The cited values were the same at the first draft (`a0fe2f1`). Both sit at a rounding boundary. | confirmed |
| E15 | heart100-s7 draft | "about 168 bpm" cited C82 only -> also C83 | (f) citation slip | C83 = 168.176 throughout. The sentence cited C82 from its first draft (`bec3d10`). | confirmed |

**Items listed but not corrected:**

| Item | Mechanism | Evidence | Confidence |
|---|---|---|---|
| heart-paper (AAAI): "active 70%", "active 17.5%" (recheck: 68.2%, 19.0%) | **(a)/(b) default sampling of train+test** | On a 200-row, seed-42 sample of train+test, 155_a is active on exactly 70.0% and 3040 on exactly 17.5%. Train gives 68.2% and 19.0%, test 72.1% and 26.2%, and the full train+test 69.0% and 20.5%. This matches the `load_split_data` and API defaults (`split="both"`) with `max_samples=200`. Script: `listed_heart_paper.py`. | confirmed (the numbers reproduce exactly; the call itself is not in a transcript) |
| heart-paper and adult-paper test-split AUCs | (b) stated on test, with no split in the text | The AUCs are stated on the test split and recheck cannot check them on train. Both are disclosed in the text. | confirmed |
| heart-s2 template text C30 ("moves mostly in a jump") | (g) | The same generator rule as E7. | confirmed |
| Approximate formulas (heart-s2, heart-s2-c1, churn200-s0, heart100-s4, monk2-s4) | (e), disclosed as "nearly" or "simplified" in most cases | `recheck` max errors range from 0.01 to 0.37, each on a few rows. | confirmed |
| heart100-s6 "0.905 -> 0.958" | (h) | The method is not stated. Removing both readouts gives 0.9036. | unknown |
| E8/E9 (the corrections) | n/a | The corrected values (-1.96, +2.62) are the direct effects that the label "linear term" calls for, so they are consistent with it. But the original numbers were correct whole-model averages. A better fix states both. | n/a |

**Tally of the 15, by primary mechanism:**

| Mechanism | n | Errors | Confidence |
|---|---|---|---|
| (a) MCP default sample | 0 | none: tested for E1, E2, E5, E6, E8, E9 and E11 | confirmed negative |
| (b) denominator or subgroup | 1 | E1 | confirmed |
| (c) stale | 0 | none | confirmed negative (git history of every prose error; evidence is stored inside its annotation) |
| (d) unit, scale or scope | 3 | E6, E8, E9 (E5 secondary) | 1 confirmed, 2 likely |
| (e) simplification as exact | 2 | E3, E5 | confirmed |
| (f) transcription or drafting slip | 7 | E2, E11 (counts); E10, E12, E14 (rounding); E4, E15 (citations) | 6 confirmed, 1 likely |
| (g) generator artefact | 1 | E7 (with (f) on top) | confirmed |
| not an error (erratum wrong) | 1 | E13 | confirmed |
| (h) unknown | 0 | none | |

## Part 2: the explanation pipeline and where text can diverge from the model

```
 explaining agent (unattended, one genome)
   |
   |-- MCP evidence tools ---------- defaults: split="test", max_samples=200 (seed 42)
   |   (compute_viz_data, ...)       outputs carry no split, n or sampling info
   |-- local scratch scripts ------- forward passes, tables, hand arithmetic (unlogged)
   |                                 most stated numbers come from here or from memory
   v
 apply_operation(split / identity / annotate + hypothesis text)
   |   checks: structure only (preconditions); hypothesis text is free
 update_narrative (overwrites the hypothesis) . . . no check
 add_evidence (type, payload, narrative) . . . . . no check; append-only, cannot amend
   v
 Explanation.operations (DB JSONB)  -->  final report  -->  verify_record.py
                                              checks coverage, function preservation,
                                              closed-form error; not the text
   v
 plain-language extract (code): claims C01..Cnn, shapes, templates
   |   checks: claim self-checks (consistency, test split); no cross-claim check
   v
 drafting agent (skill) --> lint (refuses a failing draft at store time)
   |   claims re-extracted later -> draft re-linted, not re-read ("revalidated")
   v
 export: per-genome PDF, verification packets, review packs (copies of the text)
   v
 after the fact: recheck v2 sweep . blind verifiers (L1-L4) . errata (hand-applied)
```

At each point in the table, text or numbers can diverge from the model. "Check now" says what checks it today. "Write-time" means at the moment the text is stored, and "after" means at a later review.

| Stage | How text or numbers diverge | Check now | Errors of the 15 that entered here | Notes and benchmark |
|---|---|---|---|---|
| 1. MCP evidence tools | Each surface has a different default: MCP tools `split="test"`, `max_samples=200`; the API `split="both"`, fraction 0.1, max 1000; the React UI `max_samples=100`. The outputs (`compute_viz_data` returns `viz_type`, `data`, names) do not say which rows were used. | None at write time. A protocol rule (pass `"split": "train"` and `"max_samples": 100000`) is followed by hand, often through the agent's own wrapper. | 0 | Two AAAI activity shares come from a default-style sample (Part 1). Of 1,077 stored evidence records in the 60 records, 648 (60%) state a split (all "train"), 399 (37%) state n, and 294 (27%) name a tool or method. None records the tool's arguments or result (`provenance_stats.py`). |
| 2. Agent's local scripts and arithmetic | Hand evaluation of z-scored formulas, hand sums, reading the wrong cell, choosing the quantity (direct or total). | None. Scripts sit in scratch directories; the numbers reach the text by retyping. | 7 (E1, E2, E5, E6, E8, E9, E11) | This is the dominant entry point. The verify tools that would have answered each question in one call (`subgroup`, `probe`, `enumerate`) existed only for verifiers. |
| 3. `annotate` hypothesis text | Free text with numbers, formulas and population claims. | Structural validation only. | E2, E3, E5, E6, E8, E9 (written at annotate or update) | — |
| 4. `update_narrative` | Overwrites the hypothesis. Numbers written before a computation are not revisited (E2). | None. | E1 (hypothesis copy) | — |
| 5. `add_evidence` | Hand-typed payload JSON and narrative. Append-only, with no amend. | None (only that the JSON parses). | E1 (narrative), E11 (payload) | E11 stayed wrong because it could not be amended. |
| 6. Final report and `verify_record.py` | The summary and `resisted` fields restate hypothesis numbers. | Write-time checks of coverage, function preservation and closed-form error; the text is unchecked. | E8, E9 (summary copies) | `agrees_with_agent` covers only `full_coverage`. |
| 7. Claim generator | Shape classes, exceptions, folds and typical-patient risks are code output. Semantic gaps become "verified" claims. | The claim self-checks (consistency share, robustness, test split). No cross-claim consistency check: C30 and C33 contradict each other. | E7 (root cause) | Benchmark: `wrong_shape` faults were caught 3/3 (strict). |
| 8. Drafting agent and lint | The agent retypes numbers from claims (rounding, citations, attaching numbers to the wrong group). | A write-time lint (the store refuses a failing draft). But the lint rules arrived after most drafts: the "round once" rule dates from 2026-11-11 00:31, after every draft. After a claim re-extraction, the draft is only re-linted: 63 of 80 drafts carry `revalidated`. | 5 (E4, E10, E12, E14, E15), plus E7's amplification | E10 was correct ("about 4%") in v3 and became wrong in the v5 re-draft. |
| 9. Export | Rendered copies (PDFs, packets, packs) go stale when the text changes. | Regeneration by hand. | 0 | The errata found stale packets and regenerated them in `4d619be`. |
| 10. After-the-fact review | `recheck` v2, blind verifiers, errata applied by hand. | After the fact. | 1 introduced: E13 | `recheck` found 7 of the 15 (E3, E4, E10, E12, E13, E14, E15), and E13 is a false positive. The blind verifiers found 8 (E1, E2, E4, E5, E7, E8, E10, E11). |

**Tool defaults.** None of the 15 is a default-sample error. The explaining agents mostly computed in their own scripts on the full training split, or wrapped the tools with `split="train"`. For example, heart-s5's wrapper forced `split="train"` and `max_samples=1000`. Still, the defaults are a live hazard:
- The **test** split is the MCP default, and the protocol forbids it.
- A 200-row sample makes counts and shares wrong for any population of interest.
- The outputs do not show which rows were used.
- The AAAI materials show the hazard is real: two shares in them are exactly a 200-row train+test sample.

**No provenance on numbers.** This is the common factor in E1, E2, E5, E6, E8, E9 and E11.
- A stated number carries nothing that says what was computed: no tool, population, split, n or scope (part, direct edge or whole model).
- The reviewer, and `recheck`, have to reverse-engineer each number from the English around it.
- `recheck` then guesses the referent (nearest node id, keyword) and the population (regex on the following words). That guessing is exactly where its false positives come from (see below).
- The two correct-but-mislabelled numbers (E8, E9) show the cost. Without a scope label, even a right number reads as an error.

**The time box.** It was not binding:
- Over the 60 transcripts, the explaining agents took 8 to 72 minutes, median 16.6, against a 3-hour box (`transcript_index.py`).
- No agent stopped on time.
- The pressure was the structure of the task, not the clock:
  - the success criteria in the final report are all structural (coverage, hierarchy, function preservation);
  - nothing in the protocol asks for the source of a number;
  - text is written in one pass, straight after the computation. In heart-s5 the 35/0.13 sentence was written 13 seconds after the regime table appeared, and adult-s5's hypothesis was written before the count it quotes.
- The v5 drafts were written in batches of several genomes per drafting agent, and churn100-s1's draft was stored 3.5 minutes after its drafting agent started. Speed here came from the batching, not the box.

**The 47 false positives of `recheck` v2.**
- They are 47 rows but only 34 distinct statements: the same statement is often checked in both the hypothesis and an evidence narrative.
- They make up 45 of the 53 "mismatch" findings. Only 1 of the 53 was a genuine error (E3), and 7 were listed AAAI or ambiguous items.
- Categorised by the settled explanation in the triage (`fp_categories.py`, `fp_categories.csv`):

| Category | Rows | Example | What would make the checker precise |
|---|---|---|---|
| A. Scope: a share or count within a subgroup compared with the whole training set | 15 | "879 active 46.4% of no-voice-mail rows" checked against 34.1% overall | Evaluate the stated subgroup, or mark it unchecked. The `SUBGROUP_TAIL` regex misses "of no-voice-mail rows" and "for 81 of 82 angina patients". With typed numbers, the population is stored. |
| B. Wrong referent: the nearest-node heuristic picked another node | 9 | relay R3195 compared with the gate; 10233 taken for 9165 | Node references bound in the stored number. The heuristic picks the last node id in the sentence. |
| C. Wrong quantity: a number of another kind caught by a keyword | 7 | "0.9380" (an AUC) read as a readout weight; a fitted slope `2.14*290` read as a readout | Adjacency rules (the number must follow the keyword directly) or typed quantities. |
| D. Another configuration | 4 | an AUC under linearisation or core-only; a counterfactual "without the veto" count | Typed quantities that name the configuration. |
| E. Notation | 2 | the alias "s0"; a formula in raw board units | Declared units and aliases in the formula statement. |
| F. Prose direction word from another clause or an idiom | 5 | "cannot go much lower" (a saturation idiom) | Check only clauses whose cited claim is identified; add the idiom list. |
| G. Prose number in a skipped field or other units | 5 | 97% = C01's `estimate.value`; "0.2 points" = 0.0018 | Include `estimate.value` for purity claims; convert percentage points. |

Categories A to E (37 of the 47) exist only because the checker has to infer the referent, population and quantity from English. They disappear if numbers are stored with their query (Recommendation 2). Better regexes could remove some of A and C, but never reliably.

The checker's **recall** problem is larger than its precision problem:
- On the pre-errata text of the ten corrected genomes, the free-text checker made 302 checked statements and left **465 numeric sentences unchecked** (`recheck_coverage.py`).
- Of the eight database-text errors, it reached only E3. E1, E2, E5, E6, E8, E9 and E11 all fell into its "unchecked" list.
- E2's sentence was merged into the one before it, because the sentence splitter does not split before a sentence that starts with a digit ("2134 is a Thailand indicator...").

**Benchmark context.**
- Blind verifiers caught 100% of the injected L1 faults but 71% (strict) of the L2 faults and 25% of the L4 faults.
- They found 8 of the 15 errors, including every error that came from hand computation except E6 and E9, which were found while confirming a neighbouring error.
- The 19 extra findings on faulted items were triaged as follows:
  - Four refutations of `record.closed_form_max_abs_err` (on vb-23b459, vb-a33fbd, vb-d9fb05 and vb-e5e5d0) are knock-on effects of the injected `structural_missing_node` fault. So are two of `record.resisted`.
  - Two are E1 and E2 seen again on other variants: vb-e8ef4b V14 for E1, vb-caa642 V23 for E2.
  - **One is a genuine error that the errata do not list.** vb-e37014 V2 refutes the heart-s5 root hypothesis's "oldpeak and thal:7 influence the output ONLY through these gates". thal:7 also has a direct edge (`-26_a` -> 0, weight -0.065), which the same hypothesis lists in its linear core. It is a small (e)-type overclaim and should be added to the errata.
  - The remaining weakened findings (scope and paraphrase) have not been triaged.

## Part 3: recommendations, in priority order

"Prevents" counts the errors among the 15 that the change would have stopped at write time. E13 is counted separately as a process error. Effort is S (under a day), M (days) or L (a week or more).

### P0. Repair the errata and the lint (S)

- **Revert E13** in `heart100-s0` ("about 19%" -> "about 20%"), and record it in the errata as a withdrawn correction.
- **Fix `lint_percents`:**
  - add `reference_patient.predicted_risk` (and the outcome base rate) to the candidate set;
  - drop `share`/`share_out` keys that are category shares, not risks;
  - flag a sentence only when *no* candidate rounds once to the stated value, which is already the rule, but over the right candidate set.
- **Annotate E8/E9** in the errata: the original values were correct whole-model averages for holders (-2.37, +2.67). State both quantities in the record.
- **Errata rule:** a correction must reproduce the old value as wrong under every reasonable reading, not only show the new value under one reading. A second reviewer checks it.
- **Prevents:** E13-type regressions. **Risk:** none.

### P1. Give the explaining agents the verifier's query tools, and make every number a typed reference (L)

This is the parallel fixes 1 and 2, done properly.
- **Expose `verify subgroup`, `probe`, `enumerate`, `ablate` and `linearise` as MCP tools** for the explaining agents. One call would have answered each of these:
  - E1 ("mean risk where 290 on and 204 off"), with n;
  - E2 and E11 (counts);
  - E5 and E6 (`enumerate` gives every category cell of node 595);
  - E8 and E9 (`probe` gives the direct and the total effect separately).
- **Store every tool result** as a numbered computation record holding:
  - the tool and its full arguments (split, where-expression, n, seed);
  - the value, the unit (z, raw, logit or probability) and the scope (direct edge, part or whole model);
  - the operation sequence number and a code hash.
- **Write numbers in hypotheses, narratives and payloads as references:** `{q17.mean_risk}`, `{q17.n}`. Code renders them with their rounding.
- **`add_evidence` should accept a computation id rather than a hand-typed payload.** E11's payload was a hand-typed dict.
- **Prevents:** E1, E2, E5, E6, E8, E9 and E11 (7). E5 also needs a lint against number words ("zero", "all", "none", "never", "always") outside a reference, because "zero in area 415" contains no digit.

**What the parallel fixes miss:**
- Placeholders "filled from tool outputs" only move the problem if the output is the wrong cell (E1) or the wrong quantity (E8, E9). The placeholder must name the **query** (population, quantity, scope), not just point at a number in a JSON blob.
- A full-train default stops the AAAI-type errors (2 listed items) but **none of the 15**. Most numbers did not come from the MCP evidence tools at all; they came from scratch scripts. So the typed-reference rule must also cover local computations: make the query tools good enough that scratch scripts are unnecessary, and log any scratch result that is cited.
- Split stamping must reach the **text**, not only the payload.

**Effort and risks:**
- **Effort:** L, for a schema, renderer, MCP tools and a protocol change.
- **Risk 1:** the agents fall back to writing numbers in words. A write-time lint is needed.
- **Risk 2:** the prose gets stilted. Mitigate by rendering the references with their rounding, as the plain-language templates already do.
- **Risk 3:** the old records do not carry references. Accept them as legacy, with the sweep as their check.

### P1. Write-time validation, and the verifier in the loop (M)

- **`annotate`, `update_narrative` and `add_evidence` run the free-text checker** on the text being written and **return the findings to the agent at once.** The agent fixes them while the computation is fresh, rather than leaving them for a sweep a week later.
- Bare digits outside references produce a warning; under the new protocol they are refused.
- **Formulas:** any `node = relu(...)` statement is evaluated on train when it is written. If it is off on any row by more than its tolerance, it must be written with "≈" and the tolerance (parallel fix 4).
- **Add `amend_evidence`,** which keeps the history. E11 shows that append-only evidence turns a noticed error into a permanent one.
- **Re-validate after structural operations and claim re-extraction:**
  - when an annotation's subgraph or the claims change, mark the dependent references stale and re-render them;
  - do not merely re-lint. 63 of 80 drafts were re-linted after their claims changed and were never re-read.
- **Prevents:** E3 (via the formula check), plus everything P1 references cover, earlier. Without references, the write-time free-text check catches what `recheck` catches: E3 and the prose warnings.
- **Risk:** false positives at write time cost the agents time. Gate on category: subgroup, referent and configuration statements are "unchecked" until they are typed.

### P2. Code-filled numbers and citations in the drafts (M)

This is the drafting analogue of P1.
- The drafting agent writes **slots** (`{C086.groups[1].to}`), and code fills in the value, rounded once.
- The **citations are derived from the slots,** so a sentence cannot quote a number without citing its claim.
- A claim re-extraction re-renders the slots and fails if a slot's meaning changed.
- **Prevents:** E4, E10, E12, E14 and E15 (5). E7's drafting half ("for a typical patient ... 12% to 28%" attached to the exception group) is prevented if each slot's population label is rendered with it.
- **Effort:** M, since templates already render numbers this way.
- **Risk:** the drafts become more template-like. The skill already demands claim-faithful text.

### P2. Fix the claim generator's exception rule, and add cross-claim consistency (M)

This is parallel fix 5. In `plain_language_shapes.py`, an exception group is named only if three things hold:
1. **Recall:** at least half of the group shows the exception shape. Count this over *all* sampled patients in the group, including the unaffected (flat) ones.
2. **Precision:** the existing rule.
3. **Size:** a minimum n.

Otherwise the text says "for some patients (25 of 242)" without naming a group. Templates should carry the group's count ("for 29 of the 90 patients with a reversible defect").

Add a **cross-claim check** at extract time. A direction or shape claim's exception group must not be a group in which an interaction claim on the same part reports zero effect. Here C30 and C33 contradict each other.

- **Prevents:** E7 (1).
- **What parallel fix 5 might miss:** the unaffected patients are dropped *before* characterisation. Fixing the threshold alone is not enough; the denominator must include them.
- **Risk:** fewer named exceptions, so more generic "uneven" wording.

### P3. `recheck` v2 at the end of the protocol, as a net and not a gate (S)

This is parallel fix 3.
- **Worth doing,** but it would have caught 6 of the 15 (E3, E4, E10, E12, E14, E15) and flagged E13 falsely. Its "mismatch" precision on the sweep was 1 in 53.
- **Lowering the false-positive rate** by better parsing helps with categories A and C. The real fix is P1's typed numbers, which turn the free-text checker into a lookup.
- **Until then:**
  - run it at the end of the protocol;
  - make the agent **answer every finding**, either by fixing it or by explaining why it is not an error;
  - report the **unchecked-sentence count** as a metric, not only the mismatches.
- **Risk:** agents learn to word statements so that the checker's regexes do not match them. Typed numbers remove the incentive.

### P3. Protocol changes (S)

- **Add accuracy to the success criteria.** The final report should state the numbers in the hypotheses and summary, how many cite a computation, and the end-of-protocol `recheck` result.
- **Add a "number audit" step:** for each number in the hypotheses, the agent re-runs its query (P1 tools) and pastes the command.
- **Require the scope word** for every effect: "direct", "through this part", or "whole model, mean over holders".
- **Keep the time box.** It does no harm, but don't expect it to drive accuracy.
- **Draft one genome per drafting agent,** or have a second agent re-read each batch against the claims.
- **Prevents:** reinforces all the above, and on its own is likely to catch E2 (a number written before it was computed) and E8/E9 (scope).

### P3. Formulas marked "≈" or "=" and checked (S)

This is parallel fix 4.
- **Prevents:** E3 (1), and it makes the five listed approximate formulas explicit.
- **What it misses:** formulas in raw units (the krk false positive) need a declared unit. Without a numeric tolerance next to "≈", "≈" becomes a blanket escape.

### What this would have done to the 15

| Change | Errors prevented |
|---|---|
| P1 typed references and query tools | E1, E2, E5, E6, E8, E9, E11 |
| P1 write-time formula check, or "≈" | E3 |
| P2 slots in the drafts | E4, E10, E12, E14, E15 |
| P2 generator recall rule and cross-claim check | E7 |
| P0 lint fix | E13 (the false erratum) |

Together these cover all 15. Without P1 and P2, the five parallel fixes would have stopped about 8 of the 15:
- E3 (fix 4);
- E7 (fix 5);
- the five prose errors E4, E10, E12, E14 and E15, through fix 1 if it covers the drafts or fix 3 as a net;
- E11 only if placeholders replace hand-typed payloads.

They would not have stopped the database-text errors that come from hand computation and scope (E1, E2, E5, E6, E8, E9).

## Reproduction files

All are in `docs/reviews/explanation-errors-2026-11/`. They read the database only through the verify cache, which is filled in a READ ONLY transaction, or through `load_db_operations` (also READ ONLY), and they take operations from the pre-errata snapshot. Run each with `uv run python <script>`.

| File | Output | What it shows |
|---|---|---|
| `common.py`, `make_pdf.py` | — | Loads the snapshot operations and the cached data; emulates the MCP default sample. |
| `e1_heart_s5.py` | `e1_heart_s5.out.txt` | E1: every candidate population and statistic. |
| `split_sensitivity.py` | `split_sensitivity.json` | E1, E2, E5, E6 and E11 recomputed on train, train-200, test (the default) and train+test, plus the hand evaluation for E6. |
| `e8_e9_churn100_s1.py` | `e8_e9_churn100_s1.out.txt` | E8 and E9: direct versus whole-model effects, w/sd on every split. |
| `e13_heart100_s0.py` | `e13_heart100_s0.out.txt` | E13: the reference risk is 19.80%, and the 0.193 the lint matched is a category share. |
| `listed_heart_paper.py` | `listed_heart_paper.out.txt` | The AAAI 70% and 17.5% reproduce on a 200-row sample of train+test. |
| `prose_history.py` | `prose_history.json` | Each prose error traced through the git history of its layer (not stale). |
| `recheck_coverage.py` | `recheck_coverage.json` | How much free text `recheck` checks, and whether it reached each error. |
| `fp_categories.py` | `fp_categories.csv`, `.out.txt` | The 47 false positives by category. |
| `provenance_stats.py` | `provenance_stats.json` | Split, n and tool fields on the 1,077 evidence records. |
| `transcript_index.py` | `transcript_index.json` | Time use of the 60 explaining agents (it reads the local session logs). |
