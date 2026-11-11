# Explanation-authoring skill v4: evaluation results

2026-11-11, branch `skill/authoring-v4`. The design and the decision rule are in
`docs/design/explanation-authoring-skill-v4.md`, sections 5.1 to 5.6. The rule was fixed before
this run.

## What was run

- **Genomes:** the design's six. heart-s5, churn100-s1 and adult-s5 have known errata;
  adult-s6, churn100-s2 and heart-s9 have none.
- **Original arm:** the explanations as they stood before the errata. Records and layers come
  from the tag `explanations-pre-errata-2026-11-11`. The operations are the verifier-benchmark
  dump of 2026-11-10, which is identical to the errata snapshot.
- **v4 arm:** a fresh explanation of each genome, written by an explaining agent following
  `.claude/skills/explanation-authoring/SKILL.md`. A drafting agent then wrote the v5
  plain-language draft with code-filled slots.
  - **Where the data lives:** a local SQLite copy (`~/.explaneat/authoring_v4/eval.db`, built by
    `make_db.py`). It holds the evidence log and strict mode, and only the six genomes, their
    data and their configs. Postgres was read only; nothing stored was changed.
  - **Dry run:** one dry run on heart-s7, outside the sample, before the six. The skill was
    revised from it, and was then frozen for the six.
- **Blinding:** the 12 items were built in the verifier-benchmark format with random ids. In the
  v4 items, templates, provenance ids, query metadata and the accuracy and recheck blocks were
  removed. The arm key was kept outside the repository until the end; it is now `arms.json`.
- **Verification:** two blind verifiers per item, each following the explanation-verifier
  skill, for 24 runs at matched effort (about 15 minutes each). No faults were injected.
- **Adjudication:**
  - **First pass:** one blind adjudicator per item. It grouped every refuted or weakened finding
    into errors and re-checked each with the tools. It then graded each error `refuted` (wrong
    under every reasonable reading), `weakened` or `none`, and gave its layer and mechanism.
  - **Consistency pass:** one blind consistency reviewer then applied one rule per kind of
    error across all 12 items. It changed 8 grades (`adjudication_final/RULES.md`).
  - **Mike's review:** Mike has not spot-checked these grades yet; see "What Mike needs to do"
    below.

## Results

**Genuine errors are adjudicated `refuted` findings.** The checkable statements are numeric
atoms plus number words in the rendered text (`count_statements.py`). Intervals are exact
Poisson 95% intervals.

| | Original | v4 |
|---|---|---|
| Genuine errors, total | **17** | **6** |
| in database text (hypotheses, evidence, record) | 11 | **0** |
| in plain-language drafts | 6 | 6 |
| Checkable statements (database / draft) | 2925 (1946 / 979) | 2427 (1208 / 1219) |
| Errors per 100 statements | 0.58 (0.34 to 0.93) | 0.25 (0.09 to 0.54) |
| Database errors per 100 database statements | 0.57 | 0.00 |
| Targeted mechanisms (b, d, e, f) in database text | 11 | **0** |
| Genuine weakened (database / draft) | 39 (15 / 24) | 32 (7 / 25) |
| Weakened per 100 statements | 1.33 | 1.32 |
| Verifier findings judged not an error | 1 | 1 |
| Verifier agreement: genuine errors found by both verifiers | 61% | 47% |
| Verifier agreement: refuted errors found by both | 12 of 17 | 4 of 6 |
| Full coverage; function preservation below 1e-9 | 6 of 6; 6 of 6 | 6 of 6; 6 of 6 |
| Hypotheses graded "restates the closed form" (L4) | 0 | 0 |
| Explaining time, median (records) | 17 min | 21.6 min |
| Explaining-agent tokens, median | not recorded | 266k |
| Drafting time and tokens, median | not recorded | 7.5 min, 221k |
| Verifier time and tokens per run, median (both arms) | 7.9 min, 153k | (pooled) |

**Per genome** (refuted, as database / draft):

| Genome | Original | v4 |
|---|---|---|
| heart-s5 | 3 / 2 | 0 / 2 |
| churn100-s1 | 1 / 3 | 0 / 1 |
| adult-s5 | 2 / 1 | 0 / 2 |
| adult-s6 | 1 / 0 | 0 / 1 |
| churn100-s2 | 3 / 0 | 0 / 0 |
| heart-s9 | 1 / 0 | 0 / 0 |

**Descriptive statistics:**
- **All errors:** a conditional binomial test of the v4 count given the total, with p = 0.5,
  gives 6 of 23, one-sided p = 0.017.
- **Database text only:** 0 of 11, p ≈ 0.0005.

**Detection check:** the fresh verifiers found all 8 known errata of the original arm.
- **Refuted:** E1, the thal:7 "only" and E2.
- **Weakened:**
  - E3, E4, E9 and E10;
  - E8, which the consistency pass moved from refuted to weakened, on the same rule as E9. The
    -2.37 is a correct whole-model value under the label "linear term".

They also found further genuine errors in the original arm that are not in the errata:
- **churn100-s1:** "all without a voice-mail plan" (38 holders), "zero measurable logit effect"
  and "about a point at most";
- **churn100-s2:** 161_a "< 0.03 logit", 304 for 306, and +0.43 for +0.448;
- **adult-s5:** node 1624 0.155 for 0.137;
- **adult-s6:** "95% of rows" for 91.6%;
- **heart-s9:** an SD of 1.41 attached to four of the five readouts.

**Provenance (review kit, real store):**
- **Badges:** across the six v4 explanations' database text there are 244 C sentences, 88 U,
  13 F and 417 with no number (`badges.json`).
- **Reruns:** of the 1225 references, 1178 were rerun and matched exactly. 47 come from tools
  that cannot be rerun, so they are `unavailable`.
- **No mismatch:** no reference failed to reproduce.
- **The 13 F sentences** are the kit's scope check ("direct" wording on whole-model queries).
  They are flagged for review; none of them was an adjudicated error.

## Decision rule (section 5.6)

| Criterion | Result | Holds? |
|---|---|---|
| 1. Total genuine errors at most half the original's, **and** at most 2 across the six | 6 is at most half of 17, but 6 is more than 2 | **no** |
| 2. Zero errors of mechanisms (b), (d), (e) or (f) in v4 database text | 0 | yes |
| 3. Full coverage on at least 5 of 6; function preservation below 1e-9 on all; restatements not above the original's | 6 of 6; 6 of 6; 0 against 0 | yes |
| 4. Detection parity with injected faults | no faults injected (the design calls this nice to have); the 8 known errata were all found | not applicable |
| 5. Median explaining time at most 60 minutes | 21.6 min | yes |

**Outcome: criterion 1 fails, so the rule says do not adopt v4 as it stands.**
- **The one cause:** all six remaining v4 errors are in the plain-language drafts.
- **What changed:**
  - In the database text written under the v4 skill, genuine errors went from 11 to 0, with
    fewer weakened findings too (15 to 7).
  - In the drafts, errors stayed at 6 against 6.
- **Why the drafts are untouched by v4:** the drafts come from the separate drafting skill and
  the claim generator, and v4 changes neither except for the use of slots.
- **What the rule asks next:** read the new mechanisms.

## The errors that remain (v4 drafts)

| Item (genome) | Error | Kind |
|---|---|---|
| adult-s5 | "9th-grade people go from about 44% to about 6%". The education category was changed while education-num stayed at 10, and no 9th-grade row has that; the real values are 84% to 29%. | constructed case off the data (new) |
| adult-s5 | "the age part acts only above about 46 years": it acts at every age for the 26 Thai-born | "only" with a counterexample |
| adult-s6 | "very little schooling lowers it: 1st-4th grade about 10%". Education-num is held at 10 again; real 1st-4th graders are at 90 to 95%. | constructed case off the data (new) |
| churn100-s1 | the international plan raises risk "more than anything else": service calls take it from 2% to 86% | overclaim |
| heart-s5 | "adds risk only once ST depression rises above about 0.9 mm": 18 real patients have the part acting at 0 mm | "only" with a counterexample |
| heart-s5 | the same 0.9 mm threshold paired with a reference patient whose own threshold is about 1.8 mm | scope |

**A new mechanism, not in the review: constructed cases off the data.**
- **What happens:** the claim generator changes one input of the typical person and holds
  everything else fixed. For one-hot categories that are tied to a numeric input, the result is
  a case no real row has. Examples are an education category with education-num, and a "wife"
  who stays male.
- **How it goes wrong:** the drafting agent then states that number about real people.
- **Where it appears:** the same mechanism is in the original arm's drafts too (adult-s5,
  adult-s6). In weakened form it is the most common draft finding in both arms.
- **Why the tools miss it:** the number reproduces exactly, so neither the slots nor `recheck`
  can catch it.

**What would fix the draft layer:**
- the claim generator moves linked inputs together, or marks claims whose case is off support;
- the drafting skill names the population ("for the typical person, changing only the
  category"), not "people with ...";
- the drafting skill gets the same "only" rule as the authoring skill.

**The rule's re-run allowance does not apply.** The design allows one re-run when criteria 1
and 3 hold but 2 fails. Here criterion 1 fails, so no re-run was done.

## Caveats

- **Small numbers:** six genomes, so this is an engineering decision, not a significance test.
- **Selection:** three of the bases were chosen because errors had been found in them.
- **Adjudication:** it was done by agents, not by Mike. The consistency pass changed 8 grades
  (7 up, 1 down) on one rule per kind of error, blind to the arm. Pairs of items share a genome,
  so blinding across the pair is not perfect.
- **Hints in the skill:** the skill teaches from the review's errors. Node ids, values and
  countries were removed from its examples, but the general lessons ("only through" may hide a
  direct edge; subgroup n against split n) apply to the same genomes.
- **Length:** the v4 database text is shorter, with 1208 checkable statements against 1946.
  That is why the per-statement rate is reported.
- **Tool fix during the run:** two v4 explaining agents hit an `amend_evidence(query_id=...)`
  strict-mode bug. It was fixed after the six explanations were written (commit "amend_evidence
  ... merge the stored query's values after the lint"), so all six ran on the same code.
- **Agreement metric:** verifier agreement is the share of genuine errors found by both
  verifiers. It is lower in v4, where the remaining findings are fewer and subtler.

## Files

- **Scripts:**
  - `make_db.py`: builds the SQLite copy;
  - `check_db.py`: checks it gives the same model and data as Postgres;
  - `export_v4.py`;
  - `build_items.py`: builds the blinded items;
  - `count_statements.py`;
  - `badges.py`;
  - `tabulate.py`.
- **Results:**
  - `table.json`: every number above;
  - `statements.json`;
  - `badges.json`;
  - `arms.json`: the unblinding key;
  - `usage.csv`: time and tokens per agent.
- **Adjudication:**
  - `adjudication_final/`: per item, plus `RULES.md`;
  - `adjudication/`: the first pass.
- **Not committed:**
  - **Large and regenerable:** `items/`, `runs/` (verdicts), `v4/`.
  - **Local only:** the SQLite database is at `~/.explaneat/authoring_v4/`.

## Draft-layer re-run (drafts v2, 2026-11-11)

Branch `fix/draft-layer`. The results above are unchanged. This section adds a re-run of the
draft arm only, after fixing the two mechanisms behind the six remaining draft errors. Files are
in `drafts_v2/`.

**The fixes:**
- **Generator (layer v5g, `explaneat/analysis/plain_language_links.py`).**
  - **Detecting links:** linked inputs are found from the training data. A link needs
    structural zeros that rule out at least 2% of the other input's distribution, with a
    codebook override.
    - Found: education and education-num; relationship, sex and marital status; occupation and
      work class (both "?"); each minutes and charge pair; voice-mail plan and messages.
    - Not found: rare associations such as country of birth and race. Heart has no links.
  - **Moving them:** every counterfactual (typical-case risks, per-row effects, shapes, folds)
    moves the linked inputs to their typical value among the real rows with the new value.
    - The typical person is aligned too: a high-school graduate now has education-num 9, not
      the overall median 10.
    - Claims record what moved (`linked.moved`).
    - If the case cannot be put on the data, the claim gets `off_data` with the reason.
  - **Test checks** stay one-input.
- **Proof rule.**
  - Threshold, "only if", subgroup and rule claims carry a `support_check` over the whole
    population. For example, heart-s5's ST-depression threshold has 88 of 242 exceptions, and
    adult-s5's age threshold has 26.
  - Templates say "only" only when there is no exception, and leave `off_data` claims out.
  - The lint refuses:
    - real-people wording for an `off_data` claim (`lint_off_data`);
    - "only / all / never / nothing" and superlatives without a cited check that shows no
      exception (`lint_universal`).
- **Drafting skill:** a v5g section covers naming what moved, `off_data`, and the proof rule.

**How the re-run was done:**
- **Database text:** the same six genomes and the same v4 database text (SQLite `eval.db`,
  unchanged).
- **Claims:** re-extracted with the fixed generator.
- **Drafts:** a fresh drafting agent per genome, following the revised skill. Median about 4
  minutes; all six passed the lint and were stored.
- **Items:** blinded as before, in `drafts_v2/items`, with the key in `key.json`.
- **Verification:** two blind verifiers per item (12 runs). The item README limited the scope
  to the plain-language layer and the claims it cites.
- **Adjudication:** one blind adjudicator per item, with the same `adjudication_final/RULES.md`.
  No separate consistency reviewer was run. Instead, the two grade changes from the verifiers
  were checked against the rules (below).

| | Original | v4 | v4 database text + drafts v2 |
|---|---|---|---|
| Genuine errors in database text | 11 | 0 | 0 (the v4 text, not re-verified) |
| Genuine errors in drafts | 6 | 6 | **0** |
| **Total** | **17** | **6** | **0** |
| Draft checkable statements | 979 | 1219 | 857 |
| Draft errors per 100 draft statements | 0.61 | 0.49 | 0.00 (95% upper bound 0.43) |
| Genuine weakened in drafts | 24 | 25 | 41 |

**Per genome** (refuted / weakened in the drafts):

| Genome | Original | v4 | drafts v2 |
|---|---|---|---|
| heart-s5 | 2 / 4 | 2 / 1 | 0 / 7 |
| churn100-s1 | 3 / 3 | 1 / 3 | 0 / 8 |
| adult-s5 | 1 / 7 | 2 / 10 | 0 / 10 |
| adult-s6 | 0 / 3 | 1 / 6 | 0 / 5 |
| churn100-s2 | 0 / 2 | 0 / 4 | 0 / 10 |
| heart-s9 | 0 / 5 | 0 / 1 | 0 / 1 |

**Decision rule (section 5.6), applied to v4 database text plus drafts v2:**

| Criterion | Result | Holds? |
|---|---|---|
| 1. At most half the original's 17, and at most 2 | 0 | yes |
| 2. No errors of mechanisms (b), (d), (e) or (f) in v4 database text | 0 (unchanged) | yes |
| 3. Structure | unchanged from v4: 6 of 6; 6 of 6; 0 against 0 | yes |
| 4. Detection parity | not applicable, as before | n/a |
| 5. Median explaining time | 21.6 min (unchanged) | yes |

**Outcome: with the draft-layer fix, the rule's criteria hold, so the rule says adopt v4
together with drafts v2. That reading has caveats, set out next.**

**Caveats:**
- **Two downgrades decide the outcome.** Two errors that verifiers graded refuted were graded
  weakened by the adjudicators under K3: a number that reproduces at the part's scope, written
  as the model's.
  - **adult-s5:** "state government ... about 72% to about 71%". This is the part's effect; the
    whole model goes to 78%.
  - **churn100-s1:** an interaction stated for a part that does not carry it. The whole-model
    numbers reproduce.
  - **Under a stricter reading:** if both were refuted, the total would be 2. That is still
    within the rule, but only just.
- **The weakened count rose (41 against 25), and the drafts are shorter.** The verifiers
  looked at the drafts only, so this is not a like-for-like comparison.
  - **By kind:** most are scope slips (K3, 7), omissions (K9, 8), roundings that hide a change
    ("about 1% to about 1%", K10) and one-offs ("other", 10).
  - **K1:** two K1 errors remain at weakened. Both are typical-case numbers where the moved
    inputs carry most of the effect (for example, sex together with marital status).
- **The universal-word lint is a regex.** The drafters reported two problems:
  - false positives such as "never married" (now exempt) and "nothing changes until";
  - slots cannot name `linked.moved` values whose input names contain spaces, so one drafter
    typed "29 messages".
- **The re-run is not fully independent.**
  - The same mechanisms were fixed on the same six genomes.
  - The skill's examples avoid these genomes' numbers, but the lessons are general.
  - Adjudication was by agents; Mike has not spot-checked it.
- **The fix changes the numbers in stored layers.** The typical person moves for the adult
  data, and claims that involve linked inputs move. Re-extracting a stored 10-seed layer
  therefore changes its claims, so the stored drafts would need redrafting. They were not
  touched.

**Files** (in `drafts_v2/`):
- `build_items.py`, `tabulate.py`;
- `table.json`, `key.json`;
- `adjudication/` (one file per item);
- `<label>/draft.json`;
- `runs/{A,B}/<item>/verdict.*`.

Not committed: `items/` and the `<label>/pl/` layers. They are large and can be regenerated.
