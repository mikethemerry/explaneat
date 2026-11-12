# Explaining-agent task (Connect-4 line-features base test)

The prompt given to each explaining agent, with `<...>` filled per network. Frozen before the first
network was explained. It is the ladder's prompt (`experiments/explanation_barrier/EXPLAIN_TASK.md`)
with the network description and the independence list changed.

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
- What it is: a PropNEAT network (evolved topology, ReLU hidden units, sigmoid output, direct
  input-to-output edges) trained by backprop within evolution.
- Data: Connect-4 eight-ply positions (4 x and 4 o pieces, x to move, no forced move); the label
  is 1 when x wins with perfect play, 0 for a loss or a draw. The inputs are <INPUTS>. Every input
  is named by its feature (after `rename_inputs.py`); the dataset's `additional_metadata.codebook`
  (`get_dataset`) gives one line per input saying what it counts. The concept library of this
  dataset offers every line feature as a domain concept, their x-minus-o differences<CELL_LIB>.
  Use `probe_concept` on every leaf and `set_meaning`; prefer the leaf's own inputs' names and
  game terms (open twos, open threes, threats, centre column, playable-cell parity) in hypotheses.

**Where to work**
- Working directory (the repository): `/Users/mike/dev/explaneat/.claude/worktrees/c4-lines`.
  Run every command from there, e.g.
  `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python experiments/10seed/xn.py get_model_state '{"genome_id":"<GENOME_ID>"}' 2>/dev/null`.
  Batch many calls with `--calls <scratch>/calls.json`. Names with hyphens go in backticks in
  `where` expressions.
- Scratch directory (yours alone): `<SCRATCH>/`.
- First step: `uv run python experiments/10seed/rename_inputs.py <GENOME_ID> <DATASET_ID>`.

**Independence**
- Do not open anything under `experiments/c4_line_features/` other than this file,
  `experiments/c4_capacity/`, `experiments/complex_real/explanations/`,
  `experiments/explanation_barrier/` or `experiments/explanation_levels/`, and do not look at
  other genomes' explanations. Work from this genome, its data and the tools.
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
   - `"game_terms"`: 2-5 sentences on what the network does in Connect-4 terms (which features
     raise or lower x's predicted chance, and what the gates detect), or "none" if the leaves
     have no game reading.
   - `"stuck"`: what slowed you down or stopped you (operations, coverage, leaf meaning, tools,
     time), or empty.
3. Your last message is the report JSON. Never name the assistant in anything you write.
