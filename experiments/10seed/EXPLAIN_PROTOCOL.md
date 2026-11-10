# Explanation protocol v4 (Claude-assisted, 10-seed experiment)

v4 (2026-11-11, after the errata and `docs/reviews/explanation-errors-2026-11.md`): **every number
in a hypothesis or evidence comes from a stored computation**, written as a placeholder; the
verifier's query tools replace scratch scripts; formulas are tagged exact or approximate;
`recheck` runs at the end and its summary goes into the record.

Each explanation agent explains exactly ONE genome, following the paper workflow
(reframed.tex §Method). It works unattended and does not ask the user anything.

## Tools

Call ExplaNEAT MCP tool functions through the shell wrapper (the same code as the MCP server):

```bash
cd /Users/mike/dev/explaneat
uv run python experiments/10seed/xn.py --list                       # signatures
uv run python experiments/10seed/xn.py begin_explanation '{"genome_id":"<G>"}'
uv run python experiments/10seed/xn.py get_model_state '{"genome_id":"<G>"}'
uv run python experiments/10seed/xn.py apply_operation '{"genome_id":"<G>","operation_type":"split_node","params":"{\"node_id\":\"-9\"}"}'
```

`params` / `node_ids` / `payload` arguments are JSON *strings* inside the JSON kwargs.
Source: `mcp_server/tools/*.py`. Read a tool's docstring when unsure of its params.
Append `2>/dev/null` to the calls (compute_shap writes a tqdm progress bar to stderr).

Query tools (read only, training split, every result stored with a provenance id):
- `query_subgroup '{"genome_id":"<G>","where":"node_290 > 0 and node_204_a == 0"}'`: n, outcome
  rate, mean predicted risk and where each part acts, for any subgroup (`where` is a pandas query:
  quote category codes `thal == '7'`, backticks for names with spaces, `node_<id>` for node values).
- `query_probe` (constructed cases, sweeps; whole-model effects), `query_ablate` (parts, nodes,
  edges or the direct edges switched off; `direct: true` for the direct terms), `query_enumerate`
  (every cell of a small discrete input space), `query_linearise` (restore gains).
- `get_query` / `list_queries`: a stored record and every value's path.

Helpers:
- `uv run python experiments/10seed/rename_inputs.py <genome_id> <dataset_id>`: names
  every input node after its feature. Run it once, first, before any split. Don't rename
  inputs by hand.
- `uv run python experiments/10seed/check_explanation.py <genome_id>`: the automatic
  checks (function preservation, closed-form error, and a hierarchy check: one root that
  contains the output, with every annotation under it).

Scratch files go in your own directory:
`/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad/<label>/`.

## Hard rules

1. **Only your genome.** Never call `apply_operation`, `undo_operation`, `add_evidence`,
   `update_narrative` or `save_snapshot` on any other genome. Never touch the paper
   genomes (`9022a9eb`, `4bc8fa07`, `23dde0fa`) or other seeds.
2. **Function-preserving refactoring only:** `split_node`, `consolidate_node`,
   `add_identity_node`, `rename_node`, `rename_annotation`. **Never** use `add_node`,
   `remove_node`, `disable_connection` or `enable_connection`, even though the MCP
   server instructions suggest them. The explained network must compute the same
   function as the evolved one.
3. **Training split only for evidence.** The evidence tools (`compute_viz_data`,
   `get_viz_summary`, `compute_shap`, `get_input_distribution`, `compute_performance`) and
   the query tools default to the full training split; leave `split` at "train". Do not look
   at test-split numbers. The dataset split id is in your task. Every result says which rows it
   used (`"split": {"split", "n", "sampled", "label"}`).
4. **Time box: 3 hours wall clock** from your first tool call. Record `date` at the start.
   Check `date` regularly. If 3 hours pass, stop immediately, wherever you are.
5. **Honesty:** if something can't be explained cleanly, say so in the notes. Don't
   force an annotation whose hypothesis the evidence doesn't support. Partial coverage
   is a valid result.
6. **Full sample** is the default (no `max_samples`). Sample only for `compute_shap` on large
   data (e.g. `"max_samples": 500`), and quote the n its `split` block reports.
7. **Ablations** may be computed locally with read-only repo code (forward passes on
   in-memory copies) on the training split, in your scratch dir. Never through
   `disable_connection`.
8. Don't edit repository code. If a tool is broken, work around it with other tools and
   record the bug in your notes. If that is impossible, stop and report.
9. **Numbers are references, not typed.** Every evidence and query tool returns
   `provenance.id` (`q_...`). Write each number in a hypothesis or evidence narrative as
   `{{q:<id>.<path>|<fmt>}}`: `<path>` is the value's path in the result (`get_query` lists
   them), `<fmt>` a Python format (`.2f`, `.1%`), `pct` (a risk as a rounded percentage) or `d`.
   The text is stored rendered, with the template. New explanations are **strict**: a bare typed
   number is refused (`allow_numbers` for a literal such as a count in a name). Number words
   ("zero", "all", "only", "never") also need a query behind them: "zero in area 415" was wrong
   for 30 rows (E5). Prefer `add_evidence(..., query_id="q_...")` to a hand-typed payload; fix
   a wrong record with `amend_evidence` (its history is kept).
10. **No scratch-script numbers.** Use the query tools for counts, shares, subgroup means and
    effects. If a scratch computation is unavoidable, state it as approximate and name the script.
11. **Say the scope of every effect:** "direct" (the edge's weight), "through this part", or
    "whole model, mean over <group>" (E8/E9 were whole-model averages labelled as direct terms).
12. **Formulas:** `node = relu(...)` is **exact**: it must reproduce the node on the training
    rows at the precision written (checked when you write it; refused in strict mode if not).
    Write `≈` (and the tolerance, e.g. "≈ ... (max error 0.34 on 9 rows)") for a simplification
    (E3 dropped a term but kept "=").

## Known tool behaviour (protocol v3, after the tool fixes merged in 88c1b00)

Heart was explained with v2 tooling. The following were fixed afterwards and no longer
need workarounds: the output node is now sigmoid in every engine; composition formulas
and composition evidence include the children; compositions may use descendants at any
depth; `child_annotation_ids` accepts names or `ann_N`; `detect_splits`/`classify_nodes`
flag input fan-out and report `already_covered`; SHAP no longer zeroes small effects;
`line_plot` is accepted; `get_coverage` returns a real `compositional_coverage` and a
`hierarchy` block.

Still true:
- **A node can sit in only one annotation's subgraph.** Plan the whole hierarchy before
  annotating. Split any node (hidden or input) that feeds two different branches.
- `undo_operation(seq)` removes that operation **and every later one**.
- `update_narrative` **overwrites** the hypothesis, so write the full hypothesis plus the
  evidence summary.
- **Unconnected inputs** (inputs with no enabled phenotype edges) count towards
  structural coverage. List them as zero-effect entry nodes **of the root annotation**,
  not as a separate annotation. A separate root-level annotation breaks the single-root
  requirement.
- compute_shap at full sample is slow (minutes). Run it in the background or with a
  smaller `max_samples` on large datasets, and record the sample size used.

## Workflow

1. **Analyse the phenotype:** `begin_explanation`, `get_model_state`,
   `get_input_distribution` (train), and `get_node_info` as needed. Map input node ids
   to feature names.
2. **Propose candidate subgraphs from the topology:** hidden-node gates, serial chains,
   and the group of direct input→output edges. Choose the candidates yourself (no user
   to ask).
3. **Refactor:** `classify_nodes` / `detect_splits` on each candidate. Fix violations with
   `split_node` (an entry or intermediate node with external outputs) or
   `add_identity_node` (an exit with external inputs). Use `validate_operation` for dry
   runs.
4. **Leaf annotations:** `apply_operation` `annotate` with `name`, `entry_nodes`,
   `exit_nodes`, `subgraph_nodes`, `subgraph_connections` and `hypothesis`. Then
   `get_formula` (use `force: true` if needed) for the closed form, plus at least one
   empirical piece of evidence (`get_viz_summary`, `compute_viz_data` or a query tool).
   Attach the evidence with `add_evidence` (`query_id` for a query's result; `evidence_type`
   e.g. `analytical_formula`, `empirical_viz`; the payload includes the LaTeX). Refine the
   hypothesis with `update_narrative`. **Read the `text_check` every write returns** and fix
   its findings while the computation is fresh.
5. **Composition annotations:** annotate with `child_annotation_ids`, explaining how the
   children combine (additive, gated, serial), up to a root annotation that covers the
   output.
6. **Coverage:** `get_coverage` (structural, compositional and hierarchy) and
   `check_explanation.py` (the split is looked up from the genome). Iterate until structural and
   compositional coverage are 1.0 **and** `hierarchy.complete` is true, or the time box
   ends. Also confirm that `function_preservation_max_abs_err` is below 1e-9.
7. **Number audit:** for each number in your hypotheses and summary, check it is a placeholder
   (or allow-listed literal) and that its query is the right population, quantity and scope.
8. **Recheck (last step):**
   ```bash
   uv run python -m explaneat verify recheck <genome_id> --no-claims \
       --summary /private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad/<label>/recheck_summary.json --strict
   ```
   It re-checks the free text against the model and the provenance of every referenced number.
   **An explanation cannot be marked complete with an unresolved `mismatch`**: fix each one
   (`update_narrative`, `amend_evidence`) and rerun. Answer every `needs_review` item, by fixing
   it or saying in `recheck_notes` why it is not an error. Put the summary JSON in the record as
   `recheck_summary`.

## Final report (your last message)

Return exactly this JSON block (plus any short prose you like after it):

```json
{
  "label": "heart-s5",
  "genome_id": "...",
  "start": "<date at first call, ISO>",
  "end": "<date at last call, ISO>",
  "structural_coverage": 0.0,
  "compositional_coverage": 0.0,
  "full_coverage": false,
  "within_budget": true,
  "n_leaf_annotations": 0,
  "n_composition_annotations": 0,
  "n_splits": 0,
  "n_identity_nodes": 0,
  "hidden_nodes": 0,
  "hidden_nodes_single_leaf_no_split": 0,
  "hidden_nodes_in_any_annotation": 0,
  "hierarchy_complete": false,
  "function_preservation_max_abs_err": 0.0,
  "closed_form_max_abs_err": 0.0,
  "resisted": "what resisted explanation and why, or empty",
  "recheck_summary": {"mismatch": 0, "needs_review": 0, "needs_review_items": [],
                      "unchecked_freetext_sentences": 0, "complete_ok": true},
  "recheck_notes": "for each needs_review item: fixed, or why it is not an error",
  "numbers": {"in_text": 0, "referenced": 0, "allow_listed": 0},
  "tool_bugs": "bugs or workarounds, or empty",
  "summary": "2-4 sentences on what the network computes"
}
```

`hidden_nodes_single_leaf_no_split` = the number of the phenotype's original hidden nodes
that end up inside exactly one leaf annotation without being split.
`hidden_nodes_in_any_annotation` = the number of original hidden nodes (or all their split
copies) inside some leaf or composition subgraph.
`full_coverage` = structural 1.0 AND compositional 1.0 AND `hierarchy_complete`.
`recheck_summary` = the JSON written by step 8 (`explaneat verify recheck --summary`); the record
is complete only with `recheck_summary.mismatch == 0` (`verify_record.py` reruns it and sets
`verified_by_orchestrator.complete_ok`). `numbers` = how many numbers the hypotheses and summary
state, how many are query references, and how many are allow-listed literals.
