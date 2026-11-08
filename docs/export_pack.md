# Export pack

The export pack writes the full explanation of one explained genome as files: the annotation
hierarchy, every annotation's hypothesis, closed form and evidence, the refactoring operations,
the function-preservation checks, coverage, and authorship. It is meant as an appendix
"certificate" that lets a reader check a network against the Ch 5 coverage definitions without
the database.

**The JSON is the canonical record.** Every other format (LaTeX, TikZ, PDF, DOCX) is rendered
from the JSON alone; nothing reads the database after the JSON is written.

## 1. The data model (what is stored, and where)

This section records what the export reads, found by reading the code and the database
(8 Nov 2026), before the schema was designed.

### 1.1 Genomes and their experiments (DB)

- `Experiment` (`experiments`): `id`, `name`, `dataset_id` → `Dataset`, `split_id` →
  `DatasetSplit`, `random_seed` (the 10-seed runs set it; the July paper runs do not),
  `git_commit_sha` (the commit the *training* ran on), `neat_config_text`, `config_json`.
- `Population` (`populations`): one row per generation; `experiment_id`, `generation`.
- `Genome` (`genomes`): `id` (UUID), `genome_id` (NEAT key), `fitness`, `genome_data` (the
  serialised NEAT genome), `population_id`. `Genome.to_neat_genome(config)` rebuilds it.
- `Dataset` / `DatasetSplit`: `feature_names`, `num_classes`, `x_data`/`y_data`; the split
  holds `train_indices`, `test_indices` and the scaler (`scaler_type`, `scaler_params`).
  `mcp_server.helpers.load_split_data` applies the scaler, as training did.

### 1.2 Phenotype and model state (computed, not stored)

- **Phenotype** (`get_phenotype`): `ExplaNEAT(neat_genome, config).get_phenotype_network()`,
  a `NetworkStructure` (nodes with `type`, `bias`, `activation`, `response`, `aggregation`;
  connections with `weight`, `enabled`, `innovation`; `input_node_ids`, `output_node_ids`).
  Output nodes carry `activation="sigmoid"`, the activation the trained model applies
  (`NeuralNeat` parity, `tests/test_core/test_engine_parity.py`).
  `StructureNetwork(structure).model_output_activation` gives the same value
  (`structure_output_activation`; "identity" only for survival / Cox models).
- **Model state**: `ModelStateEngine(phenotype)` replays `Explanation.operations`
  (`mcp_server.helpers.build_engine`). `engine.current_state` is the explained model;
  `engine.original_phenotype` the unmodified phenotype; `engine.annotations` the
  `AnnotationData` list (with `parent_annotation_id` set from the children lists and
  `display_name` from `rename_annotation`).

### 1.3 The explanation: one JSONB event stream (DB)

- `Explanation` (`explanations`): `id`, `genome_id`, `name`, `description`, `created_at`,
  `updated_at`, cached `structural_coverage` / `compositional_coverage` (NULL for every
  explanation checked; not used), and **`operations`**, a JSONB array. Every explained genome
  has exactly one `Explanation` (86 rows, 86 distinct genomes); `build_engine` takes
  `.first()`.
- Each operation is `{seq, type, params, result, created_at, notes?}`. Types in the DB:
  `rename_node` (cosmetic display names), `split_node`, `add_identity_node`,
  `consolidate_node` (function-preserving), `annotate`, and, outside the studies exported here,
  `prune_node` and `retrain` (not function-preserving).
- **Annotations are `annotate` operations.** `params` holds `name`, `hypothesis`,
  `entry_nodes`, `exit_nodes`, `subgraph_nodes`, `subgraph_connections`,
  `child_annotation_ids` (names or `ann_<seq>` ids; non-empty means a composition) and
  `evidence`. The tools' annotation id is `result.annotation_id` or `ann_<seq>`. The
  operation's `created_at` is the time the annotation was (last) written. There is no author
  field.
- The `annotations` *table* (`Annotation` model: `hypothesis`, `evidence`, entry/exit/subgraph
  nodes, `parent_annotation_id`, `explanation_id`) holds one legacy row (a 2025 backache
  genome). It is not used by the tools and none of the exported genomes has rows there.
- **Evidence** lives inside the annotate op, `params.evidence`:
  - `{"records": [{type, payload, narrative, timestamp}, …]}`, written by `add_evidence` /
    `save_snapshot`. Record types seen: `analytical_formula` (471), `empirical_viz` (439),
    `ablation_study` (97), `shap_importance`, `shap`, `input_output_sampling`,
    `performance_metrics`, `performance`, `local_table`, `empirical_table`,
    `exhaustive_enumeration`, `sign_analysis`, `activation_profile`, `ablation`,
    `local_ablation`, `rare_category_offsets`.
  - The payload is free-form JSON. The data split is in `payload.split` (or `split_choice`)
    when the author recorded it; nothing enforces it.
  - Legacy category lists (`{"visualizations": [{narrative, viz_config, svg_data, …}]}`)
    exist only on two 2025 backache genomes.
  - `list_evidence` flattens both forms; `get_annotations` reports counts and types.

### 1.4 Formulas (computed)

`get_formula` builds `AnnotationFunction.from_structure(expanded_annotation, model_state)`.
For a composition the annotation is first expanded over its descendants
(`explaneat.analysis.composition.expand_annotation`), and `latex_collapsed` is the
"children as named functions" form (`child_annotations=…`, `to_latex(expand=False)`).
`to_sympy` rounds weights and biases to 3 significant figures (`sympy.Float(w, 3)`); the
closed-form error of `check_explanation.py` is measured on that displayed form.

### 1.5 Coverage (computed)

`explaneat.analysis.coverage_report.coverage_report(engine.current_state, engine.annotations)`
backs both the MCP `get_coverage` tool and the API coverage route:
- candidate nodes are all nodes except outputs;
- a node is covered when some annotation (leaf, or a composition expanded over its
  descendants) contains it and all its enabled outgoing edges;
- exact Null nodes (no path to an output over enabled, non-zero-weight edges) and Null edges
  (zero-weight, or touching a Null node) count as covered and are listed under `null`;
- compositional coverage is `compute_compositional_coverage_from_hierarchy` (single global root
  containing every output; parentless, childless annotations made only of Null nodes are
  exempt, listed under `hierarchy.exempt_disconnected`).

### 1.6 Function-preservation and closed-form checks (computed)

`experiments/10seed/check_explanation.py` (training split, every row):
- function preservation: max |output(model state) − output(original phenotype)|;
- per leaf: numeric error of `AnnotationFunction` and error of the displayed (3 s.f.) closed
  form against the forward-pass exit activations;
- the hierarchy check (one root, containing the outputs, every annotation under it).

### 1.7 Explanation records (files)

`experiments/10seed/explanations/<label>.json`, one per 10-seed / supplementary genome (60):
the explaining agent's report (`label`, `genome_id`, `start`, `end`, claimed coverage and
checks, `summary`, `resisted`, `tool_bugs`, `analysis`, …), `protocol_version` (`v3` for 50,
a pilot string for `heart-s5`, absent for 9 Heart records written before the field existed)
and `verified_by_orchestrator` (the independent re-run of `get_coverage` and
`check_explanation.py` by `verify_record.py`; automated checks, not human review).
`mike_verification` (`verified`, `minutes`, `changes`, `notes`, `recorded_at`), written by
`explaneat.analysis.explanation_records.write_verification` or the UI, is **absent from every
record** as of 8 Nov 2026. `explaneat.analysis.explanation_records.find_record_for_genome`
finds a genome's record.

`experiments/10seed/verification/collect_live.py` and `make_packets.py` build Mike's
verification packets from the same sources (DB read-only, then files only); the export reuses
their reading of the operations and their closed-form computation (`force=True`, with a
timeout).

### 1.8 Authorship: what is known

- 10-seed, Heart-100 and churn genomes: drafted by a Claude agent, one per genome, under
  `EXPLAIN_PROTOCOL.md` (`experiments/10seed/README.md` §6; churn: `experiments/churn/PLAN.md`).
  The protocol version is the record's `protocol_version` where present. The agent's model is
  not recorded.
- The three AAAI paper genomes (explanations stored April–July 2026): no author is recorded in
  the DB (`Explanation` has no author field; the operations have no author). The drafter is
  **unknown**.
- Verification by Mike: none recorded (no `mike_verification` in any record; the paper genomes
  have no record file).
- Times: each annotation's `created_at` (from its operation) is the drafting time; the
  record's `start`/`end` bracket the agent session.

## 2. Schema (v1)

JSON Schema: `explaneat/export/schema/explanation-v1.schema.json` (draft 2020-12).
`schema_version` is `1.0.0`. Top level:

| Key | Contents |
|---|---|
| `label` | e.g. `heart-paper`, `adult-s3`; LaTeX labels use `app:<label>:` |
| `provenance` | experiment (id, name, seed, training commit), dataset, split, genome (id, NEAT key, generation, fitness), explanation (id, timestamps, n operations), study (`10seed` set/position or `aaai-paper`), record file, tool (version, git commit, dirty), export timestamp, content hash |
| `conventions` | float format, coverage conventions (incl. output nodes excluded), evidence-kind mapping, closed-form error flag |
| `model` | task, output activation (`StructureNetwork.model_output_activation`), train/test AUC and accuracy at 0.5 |
| `phenotype`, `model_state` | nodes (type, activation, bias, display name), edges (weight, enabled, innovation), counts |
| `operations` | counts by type, and every refactoring operation (params, result, notes, time, function-preserving flag) |
| `function_preservation` | max abs output change over every training row (check_explanation.py), tolerance 1e-9, non-preserving op types |
| `annotations[]` | id, name, kind (leaf/composition), parent, children, depth; subgraph (entry, exit, internal nodes, edges with weights, composition's union with descendants, function entries); covered nodes; hypothesis; closed form (LaTeX, collapsed LaTeX, SymPy strings and exact `srepr`, closed-form and numeric error on train); evidence items (type, kind analytical/empirical, method, result = narrative + scrubbed payload, data split + how it was found, thresholds, time); authorship (drafted by/at, verified by/at) |
| `hierarchy` | roots, global root, max depth, tree |
| `coverage` | `report` = `coverage_report()` verbatim (what `get_coverage` returns); `summary`; `null` (nodes with category `disconnected_input`/`dead_end` and reason, edges `zero_weight`/`touches_null_node`, exempt annotations); uncovered nodes, uncovered edges, missing compositions; `matches_get_coverage` |
| `authorship` | drafter (`ai_agent` / `human` / `unknown`), protocol version, session, human verification (`mike_verification`), automated verification (`verified_by_orchestrator`) |
| `record` | the explanation record file verbatim, or null |
| `issues[]` | audit against Ch 5 (code, severity, annotation, message) |

## 3. Conventions

- **Canonical JSON:** sorted keys, indent 1, UTF-8, trailing newline. Floats use Python's
  shortest round-trip `repr`. Values read from the DB are stored unchanged; values the export
  computes (errors, performance) are rounded to 6 significant figures first. NaN/inf become
  null.
- **Content hash:** SHA-256 of the canonical JSON without `provenance.export_timestamp`,
  `provenance.content_hash` and `provenance.tool`. Exporting twice gives byte-identical JSON
  apart from the timestamp, and the same hash (tested).
- **Coverage:** the current `coverage_report` (output nodes excluded from the candidate set;
  exact Null parts covered and listed). The collector checks its numbers against `get_coverage`
  for every export (`coverage.matches_get_coverage`).
- **Evidence kind:** `analytical_formula`, `sign_analysis`, `exhaustive_enumeration` are
  analytical; the other known record types are empirical. The data split is taken from
  `payload.split` (`explicit`), else inferred from payload key names (`train_auc`, ...) or the
  narrative ("on the test split"), and the inference is recorded.
- **No data rows:** evidence payload lists longer than 50 items (or tables longer than 20
  rows) are replaced by their length and SHA-256; raw-data keys are dropped.
- **Authorship in print:** the JSON keeps the true drafter (`"claude"`). Printed outputs
  (LaTeX, PDF, DOCX) never print that name: the drafter is "AI-assisted draft", the verifier
  "M. Merry" or "unverified" (tested).

## 4. Formats

All non-JSON formats are rendered from the JSON alone (`explaneat export render --from`);
text formats are byte-identical on re-render (tested).

- **LaTeX** (`<label>.tex`): a `\section` appendix for the thesis (`uathesis`), using only
  packages its preamble loads (amsmath, amssymb, booktabs, tabularx, graphicx, xcolor with
  dvipsnames, hyperref via the class, cleveref, TikZ with arrows.meta, fit, backgrounds,
  positioning, calc). Summary table, coverage, hierarchy, figures, operations, one block per
  annotation, audit. `--review-marks` (default) wraps it in the thesis's `drafted`
  environment. Closed forms are printed from the exact `srepr` (`explaneat/export/mathfmt.py`):
  child annotations as named functions, sigmoid outputs as `\sigma(z)` with the logit `z`
  broken across `align*` lines.
- **TikZ** (`<label>-fig-full.tex`, `-fig-level-<k>.tex`): plain TikZ; deterministic layered
  layout; full view and one collapsed view per hierarchy level; scaled to at most `\linewidth`
  and `0.75\textheight`. Pre-registered threshold: more than 60 non-Null nodes, the appendix
  shows the collapsed views (the full view file is still written).
- **PDF:** `latexmk -pdf` of standalone wrappers (`<label>-standalone.tex`, one per figure);
  build files under `build/`.
- **DOCX:** python-docx (optional extra `export`), figures as PNG from the TikZ PDFs
  (`pdftoppm`). Formats a machine cannot produce are reported as `unavailable: install ...`
  per format; JSON and TeX/TikZ always work. `explaneat doctor` reports export capability.

## 5. Commands

```bash
uv run python -m explaneat export --genome <id> --formats json,tex,tikz,pdf,docx --out <dir>
uv run python -m explaneat export --experiment <id> --genome <id> --out <dir>
uv run python -m explaneat export --explanation <id> --out <dir>
uv run python -m explaneat export --experiment <id> --out <dir>        # all explained genomes
uv run python -m explaneat export --study 10seed|paper|thesis --out <dir>
uv run python -m explaneat export render --from <dir>/<label>/<label>.json   # no database
uv run python -m explaneat export verification-pack --first 10 --out <dir>   # PRE order + review.pdf
uv run python -m explaneat export record-verdict <label> --verified --minutes N --changes "..."
```

Batches write `index.json` and `index.tex` (`\input`s every appendix; it sets
`\explaneatcertdir` so the TikZ modules resolve from the subdirectories).

API: `POST /api/genomes/{id}/export`, `POST /api/experiments/{id}/export` (background jobs),
`GET /api/exports/jobs/{job}`, `GET /api/exports/capabilities`,
`GET /api/exports/{export_id}/{file}`. Output under `$EXPLANEAT_HOME/exports/<genome>/<hash>/`.
The UI has an Export tab in the Tools drawer, an Export button on Review rows and on
experiments. The API and CLI use the MCP helpers, so a restricted dataset is refused unless
`EXPLANEAT_ALLOW_ASSISTED_RESTRICTED=1`.
