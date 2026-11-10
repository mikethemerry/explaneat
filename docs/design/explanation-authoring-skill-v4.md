# Explanation-authoring skill v4: plan

Design document, 2026-11-11, branch `design/human-verification`. Nothing is implemented. This is a plan for a skill, `explanation-authoring` (v4), used by the explaining agent. The agent builds the annotation hierarchy of one genome and writes the hypotheses and evidence. The skill replaces `experiments/10seed/EXPLAIN_PROTOCOL.md` (protocol v3) as the agent's instructions, and it sits on top of the integrity tools being built on `fix/explanation-integrity`. Those tools are:
- query tools for explaining agents, with provenance-stamped results;
- references in the text in place of typed numbers;
- write-time checks;
- `amend_evidence`;
- code-filled drafts.

That branch had no commits on 2026-11-11, so this plan assumes the interfaces that the review recommends (`docs/reviews/explanation-errors-2026-11.md`, P1 to P3). They are listed as assumptions A1 to A7 in `docs/design/human-cross-verification.md`, section 1. The plain-language drafting skill stays separate. Its v5 rules are unchanged, apart from the use of slots (A5).

## 0. Summary

- **Why a skill and not only tools.** The tools stop the mechanical errors:
  - retyping (f);
  - double rounding;
  - citation slips;
  - hand-typed payloads (E11).

  They cannot choose the quantity or the words. Of the 15 errors:
  - three were the wrong quantity or scope (d: E6, E8, E9);
  - one was the wrong subgroup (b: E1);
  - two were simplifications written as exact (e: E3, E5);
  - one more overclaim ("ONLY through these gates") was found later.

  Each of these would survive a perfect reference system if the agent asked the wrong query, or described a right query with the wrong words.
- **What the skill adds** is judgement, in five places:
  - **which quantity** answers the sentence: direct, part or whole model; over whom; in which unit;
  - **scope wording**: the scope words and number words that need proof;
  - **when "≈" is honest**: a stated tolerance and the rows it is off on;
  - **when a part is explained**: a prediction that survives a probe, not a restated formula;
  - **the query-first habit**: ask the tool, then write. Never compute in a scratch script and retype.
- **Success criteria are accuracy criteria**, not only structural ones:
  - zero unresolved write-time findings;
  - every number provenance-backed;
  - every effect with a scope word that matches its query;
  - a number audit that reproduces 100%;
  - a self-probe of the root and the three largest parts.
- **The evaluation re-explains 6 benchmark bases:**
  - three with known errors (heart-s5, churn100-s1 and adult-s5), chosen for the mechanisms they cover;
  - three without (adult-s6, churn100-s2 and heart-s9), from a seeded draw.

  The new and the original explanations go to two blind verifiers each, with the arms hidden. Adjudicated genuine errors per 100 checkable statements are then compared. **The decision rule** is in section 5.6.

## 1. Judgement the skill must carry

### 1.1 Choosing the quantity

The review's E8 and E9 were right numbers for the wrong quantity, and E1 was the n of one population with the mean of another. The skill gives a decision table. The agent picks the row **before** calling a tool, and the row fixes the query and the scope word.

| The sentence wants to say | Quantity | Tool (A1) | Scope word in the text |
|---|---|---|---|
| How much this input moves the score along its own edge | direct term: weight / sd per unit, in logit | `get_node_info` edge, or `probe --nodes` | "directly", "through its own edge" |
| How much this part adds for the patients it acts on | the part's contribution (exit x readout) and its mean over **the rows where it acts** | `subgroup` (acts-on n, mean contribution) | "through this part", "for the N it acts on" |
| What happens to a patient's prediction when the input changes everywhere | whole-model change: mean over rows, with the population stated | `probe --sweep` (or `--claim`) | "in the whole model, on average over X" |
| What the part is worth to the model | ablation (AUC, accuracy, share flipped) with `--to baseline` | `ablate --part` | "switching this part off changes ..." |
| Whether a rule holds on a discrete space | the agreement count and every counterexample | `enumerate --rule` | "for every combination", or "except N cases" |
| A count, a share or an outcome rate in a subgroup | n, share and rate from **one** query | `subgroup --where` | "N of M training patients with ..." |

Three rules go with the table:
- **One sentence, one query per statistic group.** An n and a mean in the same clause must come from the same query (E1).
- **Both quantities when they differ.** If an input feeds a part as well as its direct edge, and the direct and whole-model values differ by more than the rounding, say both: "a direct term of -1.96; in the whole model, -2.37 on average for holders, because the 3460 gate adds the rest". This is the errata's own recommended fix for E8 and E9.
- **Units are part of the quantity.** Inputs are z-scored, so a one-hot "off" level is not 0 (E6). Ask `probe` or `enumerate` in real units and never evaluate a z-scored formula by hand.

### 1.2 Scope and number words

These words make claims about **every** row of a population: "only", "all", "every", "none", "no", "zero", "never", "always", "exactly", "solely" and "the same". The checkers cannot see such claims without a digit (E5: "zero in area 415"). The skill's rule:

| Word | Allowed only with |
|---|---|
| "zero", "none", "never", "no X" | an `enumerate` or `subgroup` reference showing the count is 0, **over every cell of the stated population**. E5's agent skipped one of four cells. |
| "only", "solely" | an `ablate` reference for every other path. For "only through these gates", also `ablate --direct` or the edge list, to show that there is no direct edge (the thal:7 overclaim). |
| "all", "every", "always" | a `subgroup` reference with share = 1.0 |
| "exactly", "=" | the write-time exact check (A3), with max error at most 1e-9 |
| anything else | rewrite with the number: "for 1,986 of the 2,016 customers in area 415" |

The write-time lint (A3) refuses these words outside a reference under protocol v4. The skill explains **why**, so that the agent does not reword to escape the lint. "Practically none" with no number is worse than "28 of 2,016".

### 1.3 When "≈" is honest

- **"="** is allowed only when the formula passed the exact check: every training row within 1e-9 of the node value.
- **"≈"** must carry its tolerance and its rows: "2990 ≈ relu(1.446 age - 0.48) (off by up to 4.45 on the 26 Thai rows; exact elsewhere)". Without the tolerance, "≈" is a blanket escape (review P3).
- **When a dropped term matters,** keep the full formula and describe the simplification in words. A term matters when it is non-zero on a group that the hypothesis talks about, or when its error exceeds 5% of the node's range. In E3 the Thailand term was the subject of the same hypothesis.
- **Raw-unit formulas** must say so ("in mm Hg, not z"), so that the checker and the reader use the same units (the false-positive category E).

### 1.4 When a part is truly explained

The structural criteria (coverage, hierarchy, function preservation) say that a part exists. They do not say that its hypothesis is true. A part counts as **explained** when all four of these hold:
1. **Prediction.** The hypothesis states at least one behaviour that a probe can test: a threshold, a direction, a partner, a subgroup or a rule.
2. **It survives a self-probe.** The agent constructs that case (`probe --sweep` across the threshold, or the partner at two levels) and the model does it. The probe is stored as evidence (a computation id).
3. **It is not a restatement.** The hypothesis says what the part does to the prediction and for whom, in domain terms. A sentence that is the closed form in words ("a weighted sum of X and Y, kept where positive") does not count. This is the verifier's L4 rule.
4. **Small regimes are named.** Any regime the part acts in on fewer than 30 rows, or fewer than 1% of the training rows, is stated with its n. If it looks fitted, say so: "fitted to 14 rows".

A part that fails 2 is **refuted by the agent itself**: rewrite the hypothesis, or record the part in `resisted`. A part that fails 3 is "described, not explained", which is a valid, honest outcome (protocol rule 5).

### 1.5 Query tools instead of scratch scripts

The review found that **most stated numbers came from scratch scripts or memory**, not from the evidence tools. Seven of the 15 errors entered there. The skill makes the query tools the first resort. Scratch work is allowed only for exploration, never as the source of a stated number. The rules:
- **Before writing any number,** find the tool that answers it (section 1.1). If none does, call `log_computation` (A1; a stored scratch result with its code). Never retype.
- **Write after you compute,** never before. E2's hypothesis quoted a count 3 minutes before the agent computed it.
- **One pass is not enough.** After the hierarchy is complete, the number audit (section 2) reruns every reference.

## 2. Success criteria

The final report keeps every v3 field and adds an `accuracy` block. The orchestrator's `verify_record.py` re-checks it, as it does coverage today.

| Criterion | Gate (must) or target | Field |
|---|---|---|
| Full coverage, hierarchy complete, function preserved < 1e-9 | gate, as in v3 (partial is still a valid, reported result) | as in v3 |
| Open write-time findings | gate: 0 | `accuracy.open_findings` |
| Numbers in hypotheses, narratives, payloads, summary and resisted that are references | gate: 100% (bare digits only for node ids, years and codes, as the lint allows) | `accuracy.numbers`, `accuracy.provenance_backed` |
| Effects with a scope word that matches the query's scope | gate: 100% | `accuracy.scope_mismatch` (0) |
| "=" formulas passing the exact check; "≈" with a tolerance | gate: 100% | `accuracy.formulas` |
| Number audit: every reference rerun and matching | gate: 100% | `accuracy.audit` |
| End-of-protocol `recheck`: mismatches; unchecked sentences | gate: 0 mismatches, each finding answered; target: unchecked at most 5 | `accuracy.recheck` |
| Self-probe of the root and the three largest parts (by ablation AUC loss) | gate: done, each with a stored probe; refutations fixed or listed in `resisted` | `accuracy.self_probe` |
| Hypotheses graded "restates the closed form" by the agent's own L4 pass | target: 0 | `accuracy.restatements` |
| Time | as in v3: a 3-hour box, kept but not expected to drive accuracy | as in v3 |

`summary` and `resisted` follow the same rules: they are text, and they hold references.

## 3. Draft skill outline

Front matter, as a draft:
```
name: explanation-authoring
description: Use when explaining one ExplaNEAT genome end to end (refactoring, leaf and
  composition annotations, hypotheses, evidence, final report) with the provenance-stamped
  query tools, so every stated number is a reference to a stored query with the right
  quantity and scope, every scope word is proved, formulas are "=" only when exact, each
  part's hypothesis survives a self-probe, and the report meets the accuracy gates.
```

| Section | Content | Source |
|---|---|---|
| 0. Purpose | What the tools guarantee and what they do not (the referent gap). The model's view, not the world's. Never name the assistant. | review, Part 2 |
| 1. Ground rules | v3's hard rules 1 to 8 (one genome; function-preserving operations only; train split; time box; honesty; full sample; read-only repo). New: no stated number from a scratch script; `amend_evidence` instead of append-and-disclose. | EXPLAIN_PROTOCOL v3 |
| 2. Workflow | v3's six steps, plus: (2b) plan the queries for each candidate part; (4b) the self-probe for each leaf; (6b) number audit; (6c) end-of-protocol `recheck`; (6d) answer every finding. | v3, review P3 |
| 3. Numbers are references | The query-first rule; the reference syntax; `log_computation` for the rare scratch result; rounding is the renderer's job. | review P1, P2 |
| 4. Choosing the quantity | The decision table of section 1.1; worked E8/E9 and E1 examples. | E1, E8, E9 |
| 5. Scope and number words | The table of section 1.2; worked E5 and thal:7 examples. | E5, thal:7 |
| 6. Formulas | The rules of section 1.3; the E3 example. | E3, P3 |
| 7. Categorical inputs | Real-unit levels; z-scored "off" is not 0; ask `enumerate` for every cell. | E5, E6 |
| 8. When a part is explained | The criteria of section 1.4; how to record "described, not explained". | verifier L3, L4 |
| 9. Evidence | Computation ids, not payload literals; `amend_evidence` with a reason; what each evidence type must contain. | E11 |
| 10. Write-time findings | Fix while the computation is fresh; never reword to dodge a checker; how to answer a false positive (state the referent, add the reference). | review P1, P3 |
| 11. Final report | The v3 JSON plus the `accuracy` block of section 2. | section 2 |
| 12. Worked examples | Section 4 below. | errata |
| 13. Self-check | The checklist below. | |

**Self-check checklist** (skill section 13):
- [ ] Every number in every hypothesis, narrative, payload, summary and resisted is a reference, and the audit reran all of them.
- [ ] Every effect states its scope, and the scope matches its query's `scope` field. Where direct and whole-model differ, both are given.
- [ ] Every n and statistic in one clause comes from one query.
- [ ] Every "zero", "only", "all", "never", "exactly" or "=" has its proof reference (section 1.2).
- [ ] Every formula is "=" (exact check passed) or "≈" with a tolerance and the rows it is off on.
- [ ] Categorical levels were queried in real units, every cell.
- [ ] Every leaf has a stored self-probe of its stated behaviour; the root and the three largest parts were probed.
- [ ] No hypothesis is a restatement of its closed form.
- [ ] Regimes with fewer than 30 rows are named with their n.
- [ ] Zero open write-time findings; `recheck` mismatches are 0 and every warning is answered.
- [ ] The final report's `accuracy` block is filled in from the tools, not by hand.

## 4. Worked examples from the real errors

Each example shows what was written, what the skill makes the agent write instead, and the rule involved. The references are shown in the assumed syntax (A2).

| Error | Written | v4 version | Rule |
|---|---|---|---|
| E1 | "35 patients ... mean predicted risk 0.13" (n from two cells, mean from one) | "{q4.n} patients with 290 on and 204 off; their mean predicted risk is {q4.mean_risk}", where q4 = `subgroup --where "n290>0 & n204a==0 & n204b==0"` | one query per statistic group |
| E2 | "Thai respondents (27 rows)", typed before counting, back-derived from 0.0007 x 39,073 | "{q7.n} respondents born in Thailand", where q7 = `subgroup --where "native_country == 'Thailand'"` | write after you compute |
| E3 | `2990 = relu(1.446 age - 0.48)` | the full formula with `+ 0.137*2134`, or "≈ ... (off by up to 4.45 on the 26 Thai rows)" | "=" only when exact |
| E5 | "zero in area 415" | "zero for {q9.cells[...].n} of the {q9.n} area-415 customers; {q9...value} for the 30 in Washington", where q9 = `enumerate --group-by "area_code, state_47"` | every cell; number words need proof |
| E6 | "~0.15 for other areas" (state 47 "off" taken as z = 0) | the value from the same `enumerate` table, in real units | z-scored "off" is not 0 |
| E8, E9 | "voice-mail plan -2.37 logit for holders" under "linear term" | "a direct term of {q12.direct} ({q12.scope}); in the whole model {q13.mean_change} on average for holders" | choose the quantity; give both |
| E11 | the payload literal `{"a1=3": "108/108 pos"}` | `add_evidence` with computation id `q15` (`subgroup "a1 == '3'"`); if it is wrong, `amend_evidence` | no hand payloads |
| thal:7 | "influence the output ONLY through these gates" | "mainly through these gates; thal:7 also has a small direct edge ({q18.weight})", with q18 = `ablate --direct` | "only" needs every other path ablated |
| E7, the drafting half | the typical patient's numbers attached to the exception group | (drafting skill) each slot renders its population label; the authoring skill's part: name exception groups with their counts ("29 of 90") | population labels travel with numbers |

## 5. Evaluation plan

### 5.1 Question

Do the v4 tools and skill together reduce the genuine errors in explanations, at the same structural quality? And is any of the error mechanisms they target still present?

### 5.2 Genomes

There are six of the twelve benchmark bases (`experiments/verifier_bench/items`, four items each):

| Group | Genomes | How chosen | Known adjudicated errors in the original |
|---|---|---|---|
| With errors | heart-s5, churn100-s1, adult-s5 | Chosen for coverage of the mechanisms: (b) E1; (d) E8, E9; (e) E3 and the thal:7 overclaim; (f) E2, E4, E10 | heart-s5: 2 (E1, thal:7); churn100-s1: 3 (E8, E9, E10); adult-s5: 3 (E2, E3, E4) |
| Without known errors | adult-s6, churn100-s2, heart-s9 | `numpy.random.default_rng(20261111).permutation([...])[:3]` over the six bases with no adjudicated error (adult-s4, adult-s6, churn100-s2, heart-s7, heart-s9, monk2-s4) | 0 refuted; weakened findings untriaged |

Two notes on the choice:
- **Monk2 is not in the sample.** Add monk2-s0 (E11) as a seventh genome if time allows.
- **The selection bias is disclosed.** The error bases were chosen because errors were found, so their original counts sit above the average. The clean group, and the fresh matched-effort verification of both arms (section 5.4), are there to correct for that.

### 5.3 Arms

- **Original:** the stored explanation as it was before the errata (the snapshot tagged `explanations-pre-errata-2026-11-11`), with its v5 plain-language draft.
- **v4:** a fresh explanation of the same genome by an explaining agent with the v4 tools and skill, then a drafting agent with code-filled slots.
  - **Where it goes:** a separate explanation that never overwrites the original. Either it is a new label (`<label>-v4`) with operations kept in a file and replayed in memory, as benchmark items already are, or it is a second `Explanation` row if the schema allows one. This is an assumption to check before running.
  - **Same conditions:** the same time box and dataset split, and the same structural gates.

### 5.4 Verification (blind, matched effort)

1. **Render both arms as benchmark items** with the existing item format (`explanation.md`, `record.json`, `operations.json`, `plain_language.json`). The v4 references are rendered to text with their rounding, and the provenance appendix is removed, so that the verifier cannot tell the arms apart.
2. **Run two independent blind verifiers per item** with the existing `explanation-verifier` skill. That makes 6 genomes x 2 arms x 2 = 24 runs, at about 5 minutes each (the run 1 median was 3 minutes).
3. **Inject one fault per item** (nice-to-have) with the benchmark builder, to check that detection is the same in both arms. The format differs a little, and the detection rates must match for the comparison to hold. Score these with `score.py` as in run 1.
4. **Adjudicate every refuted or weakened finding** blind to the arm. Mike does this, or a second agent with Mike spot-checking. Use run 1's ADJUDICATION procedure and the errata rule: a finding counts only if the stated value is wrong under every reasonable reading.
5. **Use the known errata as a check of the fresh run.** In the original arm, the share of the 8 known errors that the fresh pair of verifiers finds estimates the detection rate in this run.

### 5.5 Metrics

| Metric | Definition | Role |
|---|---|---|
| Genuine errors | Adjudicated `verifier right` findings at refuted level, per explanation, split into database text (hypotheses, narratives, payloads, record) and drafts | primary (count) |
| Error density | Genuine errors per 100 checkable statements, where the denominator is the numeric atoms plus number words (human-cross-verification, section 3.1) | primary (rate; corrects for length) |
| Targeted-mechanism errors | Genuine errors of mechanism (b), (d), (e) or (f) | key secondary |
| Genuine weakened | Adjudicated weakened findings: scope overclaims, citation gaps, rounding | secondary |
| Provenance | Share of numeric atoms that are references; U-badge count | process |
| Structural quality | Full coverage, function preservation, and the verifier's L4 "restates the closed form" count | non-inferiority |
| Cost | Explaining-agent minutes; verifier minutes per item; optionally, Mike's review minutes with the queue | process |
| Detection parity | Injected-fault detection, v4 arm against the original arm | validity check |

### 5.6 Decision rule

This rule should be fixed before the run (commit this document).

**Adopt v4 as the explaining protocol for all new explanations if all of these hold:**
1. **Fewer errors.** Total genuine errors in the v4 arm are at most half of those in the original arm under matched effort, **and** at most 2 across the six genomes.
2. **Mechanisms eliminated.** There are zero genuine errors of mechanisms (b), (d), (e) or (f) in v4's **database text**. These are the mechanisms the tools and the skill target. One such error is a defect in the tool or skill to be fixed, not noise.
3. **No structural loss.** Full coverage on at least 5 of 6, and function preservation below 1e-9 on all. The L4 restatement count is not higher than the original arm's.
4. **Detection parity.** If faults are injected, detection in the v4 arm is within 1 fault of the original arm's.
5. **Acceptable cost.** The median explaining-agent time is at most 60 minutes. The originals took 17, and v4 adds the audit and the self-probes.

**Outcomes:**
- **If 1 and 3 hold but 2 fails:** fix the tool or skill gap that let the error through, re-explain only the affected genome, and re-verify it. This is allowed once.
- **If 1 fails:** do not adopt. Report the results as they are, and read the new errors' mechanisms. A new mechanism that is not in the review is a finding in itself.
- **If 4 fails:** the comparison is not valid. Re-render the items so that the two arms look the same, and re-run the verification.

**Statistics.** With 6 genomes, the counts are small, so the rule is a pre-specified engineering decision, not a significance test. Report, as description only:
- the counts and densities, with exact Poisson intervals;
- the conditional binomial test of v4 errors given the total, with p = 0.5 under no difference.

The originals' known count across the six is 8 genuine errors under full effort. The original arm's count under matched effort will be lower, and that is the number the rule compares against.

### 5.7 Cost and order

| Step | Effort |
|---|---|
| The integrity tools merged (a prerequisite, not part of this plan) | n/a |
| Write the skill from section 3 (about 300 lines, modelled on the verifier skill) | half a day |
| Dry run on one non-sample genome (e.g. heart-s7); fix the skill | half a day, plus about 1 h of agent time |
| Six v4 explanations and drafts | about 6 to 8 h of agent time, run 3 at a time |
| Build 24 items; run 24 blind verifiers | half a day of setup; about 2 h of agent time |
| Adjudication | about 2 to 3 h of Mike's time (run 1 had about 30 findings to triage) |
| Write-up in `experiments/verifier_bench/runs/v4eval/RESULTS.md` | half a day |

In total, about 2 to 3 working days, of which about 3 hours are Mike's.
