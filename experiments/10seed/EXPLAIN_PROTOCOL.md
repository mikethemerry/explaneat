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
Append `2>/dev/null` to the calls (compute_shap writes a tqdm progress bar to stderr).

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
3. **Training split only for evidence:** pass `"split": "train"` to every evidence tool
   (`compute_viz_data`, `get_viz_summary`, `compute_shap`, `get_input_distribution`,
   `compute_performance`). Do not look at test-split numbers. The dataset split id is in
   your task.
4. **Time box: 3 hours wall clock** from your first tool call. Record `date` at the start.
   Check `date` regularly. If 3 hours pass, stop immediately, wherever you are.
5. **Honesty:** if something can't be explained cleanly, say so in the notes. Don't
   force an annotation whose hypothesis the evidence doesn't support. Partial coverage
   is a valid result.
6. **Full sample:** pass `"max_samples": 100000` (and `"sample_fraction": 1.0` where the
   tool accepts it) to evidence tools. The defaults subsample the training split.
7. **Ablations** may be computed locally with read-only repo code (forward passes on
   in-memory copies) on the training split, in your scratch dir. Never through
   `disable_connection`.
8. Don't edit repository code. If a tool is broken, work around it with other tools and
   record the bug in your notes. If that is impossible, stop and report.

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
   empirical piece of evidence (`get_viz_summary` or `compute_viz_data` on train).
   Attach the evidence with `add_evidence` (`evidence_type` e.g. `analytical_formula`,
   `empirical_viz`; the payload includes the LaTeX). Refine the hypothesis with
   `update_narrative`.
5. **Composition annotations:** annotate with `child_annotation_ids`, explaining how the
   children combine (additive, gated, serial), up to a root annotation that covers the
   output.
6. **Coverage:** `get_coverage` (structural, compositional and hierarchy) and
   `check_explanation.py` (the split is looked up from the genome). Iterate until structural and
   compositional coverage are 1.0 **and** `hierarchy.complete` is true, or the time box
   ends. Also confirm that `function_preservation_max_abs_err` is below 1e-9.

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
  "tool_bugs": "bugs or workarounds, or empty",
  "summary": "2-4 sentences on what the network computes"
}
```

`hidden_nodes_single_leaf_no_split` = the number of the phenotype's original hidden nodes
that end up inside exactly one leaf annotation without being split.
`hidden_nodes_in_any_annotation` = the number of original hidden nodes (or all their split
copies) inside some leaf or composition subgraph.
`full_coverage` = structural 1.0 AND compositional 1.0 AND `hierarchy_complete`.
