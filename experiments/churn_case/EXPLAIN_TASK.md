# Explaining-agent task (churn case, Experiment 1.2)

The prompt given to the explaining agent, with `<...>` filled for the selected genome. It is the
line-features prompt (`experiments/c4_line_features/EXPLAIN_TASK.md`) with the network
description, the data description and the independence list changed. Written before the run's
best genome was inspected.

---

Explain ONE network end to end, unattended, following the frozen methodology:
- the skill `.claude/skills/explanation-authoring/SKILL.md` (authoring v4; read it fully first),
- protocol v4, `experiments/10seed/EXPLAIN_PROTOCOL.md` (its hard rules apply),
- strict mode with provenance (on by default for a new explanation; do not turn it off),
- function-preserving operations only,
- a **3-hour box** from your first tool call (run `date` first and check it regularly; stop when
  the 3 hours are up, wherever you are, and report what you have).

Do NOT write plain-language drafts (skip that layer entirely).

**Network**
- label: `<LABEL>`
- genome_id: `<GENOME_ID>`
- dataset_id: `802129c2-b11b-43a5-b6af-8bcf80e4c252` (`churn_onehot`), dataset split id:
  `45962ea6-3c78-49f9-a01e-3665f6809b1e` (training split only for evidence)
- `DATABASE_URL=postgresql://localhost/explaneat_dev` (write only to this genome's explanation)
- What it is: a PropNEAT network (evolved topology, ReLU hidden units, sigmoid output, direct
  input-to-output edges) trained by backprop within evolution, 200 generations.
- Data: PMLB telecom churn, 5,000 customers; the label is 1 when the customer left. 71 inputs:
  `state` one-hot (51 levels, PMLB's sorted label encoding of the state abbreviations), `area code`
  one-hot (408, 415, 510), `international plan` and `voice mail plan` (0/1), and 15 numeric columns
  (account length, voice-mail messages, day / evening / night / international minutes, calls and
  charges, customer-service calls). Each `total * charge` is an exact multiple of its
  `total * minutes` column (after z-scoring they are the same column), so the model may use either.
  Every input is named by its feature after `rename_inputs.py`. Use `probe_concept` on every leaf
  and `set_meaning`; prefer the inputs' real names and customer terms (heavy daytime use, an
  international plan, many customer-service calls) in hypotheses.

**Where to work**
- Working directory (the repository): `/Users/mike/dev/explaneat/.claude/worktrees/churn-case`.
  Run every command from there, e.g.
  `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python experiments/10seed/xn.py get_model_state '{"genome_id":"<GENOME_ID>"}' 2>/dev/null`.
  Batch many calls with `--calls <scratch>/calls.json`. Names with hyphens go in backticks in
  `where` expressions.
- Scratch directory (yours alone): `<SCRATCH>/`.
- First step: `uv run python experiments/10seed/rename_inputs.py <GENOME_ID> 802129c2-b11b-43a5-b6af-8bcf80e4c252`.

**Independence**
- Do not open anything under `experiments/churn/`, `experiments/10seed/explanations*/`,
  `experiments/10seed/plain_language*/`, `experiments/explanation_levels/` or
  `experiments/churn_case/` other than this file, and do not look at other genomes'
  explanations. Work from this genome, its data and the tools.
- Do not use the test split.

**Big networks.** Plan the hierarchy before annotating (a node sits in only one annotation's
subgraph). Group units by what they do on the data (`unit_activity`, `suggest_candidates`,
`query_linearise`, `classify_nodes`, `detect_splits`). A leaf may hold several units. Long linear
sums go in `analytical_formula` evidence (section 6 of the skill). Partial coverage, "described,
not explained" and `resisted` are valid results: say honestly what could not be explained and why.

**Output**
1. Before your final message, write your final report JSON (the skill's section 9 format) to
   `<SCRATCH>/report.json`, and the record you pass to `--recheck` to `<SCRATCH>/record.json`.
2. Add these fields to the report:
   - `"leaves"`: one entry per leaf annotation:
     `{"name": "...", "units": [...], "meaning": "meaningful" | "partial" | "closed-form-only", "probe_score": <number or null>, "best_concept": "...", "why": "one sentence"}`
     (`meaningful` = a domain-level reading that its self-probe confirms; `partial` = a reading that
     holds for part of its behaviour; `closed-form-only` = described by its formula, no reading).
   - `"overall_function"`: 2-5 sentences, in words, on what the whole network computes from which
     inputs: the main effects, which inputs interact with which and how (gates, thresholds, rules),
     and which inputs it ignores. No numbers needed here; if you give any, use placeholders as
     elsewhere.
   - `"customer_terms"`: 2-5 sentences on what the network does in customer terms (which
     behaviours and plans raise or lower the predicted chance of leaving, and what the gates
     detect), or "none" if the leaves have no such reading.
   - `"stuck"`: what slowed you down or stopped you (operations, coverage, leaf meaning, tools,
     time), or empty.
3. Your last message is the report JSON. Never name the assistant in anything you write.
