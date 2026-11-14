# Explaining-agent task (synthetic U-shape and saddle study)

The prompt given to each explaining agent, with `<...>` filled for the selected genome. It is the
churn case prompt (`experiments/churn_case/EXPLAIN_TASK.md`) with the network description, the data
description and the independence list changed. Written before any run's best genome was inspected.
The agent is told the inputs' clinical names and the dataset's size, not how the labels were made.

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
- dataset_id: `<DATASET_ID>` (`<DATASET_NAME>`), dataset split id: `<SPLIT_ID>` (training split
  only for evidence)
- `DATABASE_URL=postgresql://localhost/explaneat_dev` (write only to this genome's explanation)
- What it is: a PropNEAT network (evolved topology, ReLU hidden units, sigmoid output, direct
  input-to-output edges) trained by backprop within evolution, 100 generations.
- Data: a synthetic clinical-style tabular dataset, 1,500 training rows, binary outcome (1 = the
  outcome occurred; about 40% of rows). 10 inputs in raw units: `age` (years), `bmi` (kg/m²),
  `sbp` (systolic blood pressure, mmHg), `heart_rate` (beats/min), `sodium` (serum sodium, mmol/L),
  `creatinine` (serum creatinine, µmol/L), `smoker`, `diabetes`, `male` (0/1) and `noise` (a
  standard normal column). The data were generated from a known function that is withheld from
  you; some inputs may carry no signal, and an input's effect need not be monotone or additive.
  Describe what the **model** does with each input, including the shape of its response (where it
  rises, where it falls, where it turns) and any interaction between inputs, in clinical terms. Use
  `probe_concept` on every leaf and `set_meaning`; for a leaf whose response is not monotone or
  that joins two inputs, try ad-hoc expressions (for example `(sodium - 137) ** 2`,
  `heart_rate * sbp`) as well as the library, and state in the hypothesis the shape you found and
  the input values where it turns.

**Where to work**
- Working directory (the repository): `/Users/mike/dev/explaneat/.claude/worktrees/ushape`.
  Run every command from there, e.g.
  `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python experiments/10seed/xn.py get_model_state '{"genome_id":"<GENOME_ID>"}' 2>/dev/null`.
  Batch many calls with `--calls <scratch>/calls.json`. Names with hyphens go in backticks in
  `where` expressions.
- Scratch directory (yours alone): `<SCRATCH>/`.
- First step: `uv run python experiments/10seed/rename_inputs.py <GENOME_ID> <DATASET_ID>`.

**Independence**
- Do not open anything under `experiments/ushape_saddle/` (it holds how the data were made) other
  than this file, nor `experiments/explanation_barrier/`, `experiments/10seed/explanations*/`,
  `experiments/10seed/plain_language*/` or `experiments/explanation_levels/`, and do not look at
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
     inputs: the main effects, the shape of each (monotone, hinged, U-shaped, ...), which inputs
     interact with which and how (gates, thresholds, opposing directions), and which inputs it
     ignores. No numbers needed here; if you give any, use placeholders as elsewhere.
   - `"clinical_terms"`: 2-5 sentences on what the network does in clinical terms (which inputs
     raise or lower the predicted risk, over which ranges, and what any joint effect detects), or
     "none" if the leaves have no such reading.
   - `"stuck"`: what slowed you down or stopped you (operations, coverage, leaf meaning, tools,
     time), or empty.
3. Your last message is the report JSON. Never name the assistant in anything you write.
