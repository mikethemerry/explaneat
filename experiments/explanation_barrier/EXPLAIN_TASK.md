# Explaining-agent task (explanation complexity ladder)

The prompt given to each explaining agent, with `<...>` filled per network. Frozen before the first
network was explained.

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
- dataset_id: `<DATASET_ID>`, dataset split id: `<SPLIT_ID>` (training split only for evidence)
- `DATABASE_URL=postgresql://localhost/explaneat_dev` (write only to this genome's explanation)
- What it is: a plain dense ReLU network (hidden layers `<ARCH>`) with a sigmoid output, trained
  by backprop and imported as a genome. Every hidden unit is a ReLU node; there are no direct
  input-to-output edges unless the structure shows them.
- Data: <DATA_DESCRIPTION>

**Where to work**
- Working directory (the repository): `/Users/mike/dev/explaneat/.claude/worktrees/agent-a9d0e271a15a01f2b`.
  Run every command from there, e.g.
  `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python experiments/10seed/xn.py get_model_state '{"genome_id":"<GENOME_ID>"}' 2>/dev/null`.
  Batch many calls with `--calls <scratch>/calls.json`.
- Scratch directory (yours alone):
  `/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad/barrier/<LABEL>/`.
- First step: `uv run python experiments/10seed/rename_inputs.py <GENOME_ID> <DATASET_ID>`.

**Independence**
- Do not open anything under `experiments/explanation_barrier/` (it holds how the data were made),
  `experiments/complex_real/explanations/` or `experiments/complex_real/plain_language/`, and do not
  look at other genomes' explanations. Work from this genome, its data and the tools.
- Do not use the test split.

**Big networks.** If the network is large, plan the hierarchy before annotating (a node sits in
only one annotation's subgraph). Group units by what they do on the data (`unit_activity`,
`suggest_candidates`, `query_linearise`, `classify_nodes`, `detect_splits`). A leaf may hold
several units. Long linear sums go in `analytical_formula` evidence (section 6 of the skill). Partial coverage, "described, not explained" and `resisted` are valid results:
say honestly what could not be explained and why.

**Output**
1. Before your final message, write your final report JSON (the skill's section 9 format) to
   `<scratch>/report.json`, and the record you pass to `--recheck` to `<scratch>/record.json`.
2. Add these fields to the report:
   - `"leaves"`: one entry per leaf annotation:
     `{"name": "...", "units": [...], "meaning": "meaningful" | "partial" | "closed-form-only", "why": "one sentence"}`
     (`meaningful` = a domain-level reading that its self-probe confirms; `partial` = a reading that
     holds for part of its behaviour; `closed-form-only` = described by its formula, no reading).
   - `"overall_function"`: 2-5 sentences, in words, on what the whole network computes from which
     inputs: the main effects, which inputs interact with which and how (gates, thresholds, rules),
     and which inputs it ignores. No numbers needed here; if you give any, use placeholders as
     elsewhere.
   - `"stuck"`: what slowed you down or stopped you (operations, coverage, leaf meaning, tools,
     time), or empty.
3. Your last message is the report JSON. Never name the assistant in anything you write.
