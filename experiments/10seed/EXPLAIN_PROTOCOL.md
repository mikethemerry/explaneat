# Explanation protocol (Claude-assisted, 10-seed experiment)

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

## Hard rules

1. **Only your genome.** Never call `apply_operation`, `undo_operation`, `add_evidence`,
   `update_narrative` or `save_snapshot` on any other genome. Never touch the paper
   genomes (`9022a9eb`, `4bc8fa07`, `23dde0fa`) or other seeds.
2. **Function-preserving refactoring only:** `split_node`, `consolidate_node`,
   `add_identity_node`, `rename_node`, `rename_annotation`. **Never** use `add_node`,
   `remove_node`, `disable_connection` or `enable_connection`, even though the MCP
   server instructions suggest them. The explained network must compute the same
   function as the evolved one.
3. **Training split only for evidence:** pass `"split": "train"` to every evidence tool
   (`compute_viz_data`, `get_viz_summary`, `compute_shap`, `get_input_distribution`,
   `compute_performance`). Do not look at test-split numbers. The dataset split id is in
   your task.
4. **Time box: 3 hours wall clock** from your first tool call. Record `date` at the start.
   Check `date` regularly. If 3 hours pass, stop immediately, wherever you are.
5. **Honesty:** if something can't be explained cleanly, say so in the notes. Don't
   force an annotation whose hypothesis the evidence doesn't support. Partial coverage
   is a valid result.
6. Don't edit repository code. If a tool is broken, work around it with other tools and
   record the bug in your notes. If that is impossible, stop and report.

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
   empirical piece of evidence (`get_viz_summary` or `compute_viz_data` on train).
   Attach the evidence with `add_evidence` (`evidence_type` e.g. `analytical_formula`,
   `empirical_viz`; the payload includes the LaTeX). Refine the hypothesis with
   `update_narrative`.
5. **Composition annotations:** annotate with `child_annotation_ids`, explaining how the
   children combine (additive, gated, serial), up to a root annotation that covers the
   output.
6. **Coverage:** `get_coverage`. Iterate on uncovered nodes or edges and missing
   compositions until both structural and compositional coverage reach 1.0, or the time
   box ends.

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
  "resisted": "what resisted explanation and why, or empty",
  "tool_bugs": "bugs or workarounds, or empty",
  "summary": "2-4 sentences on what the network computes"
}
```

`hidden_nodes_single_leaf_no_split` = the number of the phenotype's original hidden nodes
that end up inside exactly one leaf annotation without being split.
