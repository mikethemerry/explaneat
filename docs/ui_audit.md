# UI / API / MCP audit (November 2026)

Scope: the React Explorer (`web/react-explorer/`), the FastAPI backend (`explaneat/api/`)
and the MCP server (`mcp_server/`), checked against the work done on
`exp/10seed-consistency` since the UI was last developed: the 10-seed explanation
protocol and its 60 Claude-produced explanation records, the churn extension studies
(restore gain, linearisation, one-hot grouping, state offsets, graft/knockout/refit),
the XOR realisation study, and the engine fixes (output-activation parity, compositional
coverage, SHAP `l1_reg`, the phantom-weight mask default).

Method:
- read the code paths of every endpoint the UI calls;
- ran the API against the real database (`postgresql://localhost/explaneat_dev`,
  read-only use) on port 8011 and exercised the endpoints with HTTP requests (results below);
- `tsc --noEmit` and `vite build` for the frontend;
- the Python tests: `uv run pytest -q tests/test_api tests/test_mcp tests/test_core`
  gave **369 passed, 0 failed** before any change on this branch.

No browser automation tool was used: there was none in this session that could reach
`localhost`. UI behaviour was checked through the build, the type-checker and the API.

## 1. What works today

| Area | Status | Checked |
|---|---|---|
| Experiment list, genome explorer, network graph (React Flow), collapse of annotations into function nodes | works | `GET /api/experiments` 200 (but 6.2 s, see W5); `GET /genomes/{id}/model` 200 in 0.03 s |
| Operations: split, consolidate, identity, rename, annotate, undo, validate | works | `GET /operations` 200; `POST /analyze/split-detection` 200 |
| Annotation strategy wizard (identity → splits → annotate) | works | code read; matches `annotation_coverage_model.md` |
| Evidence: line, heatmap, partial dependence, PCA, sensitivity, ICE, SHAP, activation profile, regime map, edge influence | works | `POST /evidence/viz-data` (line, sensitivity) 200 in < 0.1 s |
| Formulas (KaTeX) for leaves | works | `GET /evidence/formula?annotation_id=ann_32` 200 |
| Whole-model performance | works, **parity correct** | heart-s1 train AUC from `POST /evidence/performance` = 0.93922, equal to the stored fitness 0.93922 |
| Snapshots + gallery | works | code read |
| Retrain panel (non-identity ops), experiment launch/resume, config templates | works | code read |

The output-activation parity fix (`ExplaNEAT.get_phenotype_network()` sets
`activation="sigmoid"` on output nodes) flows through every API path, because they
all build the phenotype the same way. The UI has **no hardcoded sigmoid or ReLU** in its
computation; the only activation strings are help texts ("red line marks the ReLU
boundary"), which are still correct for hidden nodes.

## 2. What is wrong now

| # | Finding | Effect | Fix |
|---|---|---|---|
| W1 | `GET /api/genomes/{id}/analyze/coverage` uses the **old coverage semantics**: structural = `|subgraph_nodes| / |all nodes|` (no edge condition, outputs counted, compositions not expanded) and compositional = the cached `Explanation.compositional_coverage` column. | heart-s1, a full-coverage explanation by the paper definitions (MCP `get_coverage`: 1.0 / 1.0, hierarchy complete), is reported as **compositional 0.0**. Any UI built on this endpoint would be wrong. (The UI does not call it today: `getCoverageAnalysis` is unused.) | **Fixed**: the route now calls the same `CoverageComputer` + `compute_compositional_coverage_from_hierarchy` code as MCP `get_coverage` (shared helper), and returns the hierarchy block. |
| W2 | `python -m explaneat api` (documented in `CLAUDE.md`) **did not exist**. The CLI only had `db`. | The documented start command failed (`No such command 'api'`). | **Fixed**: `explaneat api [--host] [--port] [--reload]`. |
| W3 | API startup marks **every** experiment with status `running` as `interrupted`. | Starting the UI while a script (e.g. `run_seed.py`, the other agent's runs) is training corrupts that run's status. | **Fixed**: `explaneat api` skips this unless `--mark-orphans` (env `EXPLANEAT_MARK_ORPHANS=0`). |
| W4 | The API does **not serve the built UI**. `explaneat/static/react_explorer/` is only used by the legacy pyvis/`visualization.py` path; the React app runs only via the Vite dev server (port 3001 in `vite.config.ts`, while `CLAUDE.md` says 5173). The static folder has accumulated 9 stale bundles. | No "one command" way to use the UI; docs inconsistent. | **Fixed**: the API mounts the built bundle at `/` with SPA fallback; the stale bundles were removed when the new build was deployed. (A cross-platform deploy script belongs to the install work on `exp/install`.) |
| W5 | `GET /api/experiments` runs 3 queries per experiment (N+1) and has no name filter. | 6.2 s for 953 experiments; finding `10seed-heart-s1` means paging through hundreds of derived churn experiments. | **Fixed**: aggregate queries and `?q=` name filter; the list has a search box. |
| W6 | `GET /genomes/{id}/model` (and other read routes via `_get_phenotype_and_engine`) **create an `Explanation` row** when the genome has none. | A read is a DB write. Harmless for the 60 explained genomes, but opening an ancestor in the explorer creates an empty explanation row for it. | **Not fixed in the old routes** (behaviour other code relies on). The new tools/lineage/diff routes are read-only, and the lineage diff view draws ancestors without calling `/model`. "Open ancestor in the explorer" does go through `/model`. |
| W7 | Root composition formulas: with > 8 entries and no `force`, `GET /evidence/formula` returns `latex: null, tractable: false` for a composition. | The UI first shows "intractable" for the root of every 10-seed explanation (31 entries on heart-s1). | Already mitigated: `FormulaDisplay` has a "Show anyway" button (`force=true`). No change. |
| W8 | Docs: `react_explorer_requirements.md` refers to `useCollapsedView.ts`, which no longer exists (collapse is server-side via `?collapsed=`); the dev port is 3001, not 5173. | Stale docs. | Updated. |

No incorrect *model computation* was found in the UI after the engine fixes: the
remaining problems are in coverage reporting (W1), tooling (W2–W5) and a read-write
side effect (W6).

## 3. Missing: tools the agents built or needed that the UI/API/MCP lack

From `EXPLAIN_PROTOCOL.md`, `check_explanation.py`, `verify_record.py`,
`experiments/churn/explained_eval.py`, `genome_surgery.py`, the 60 records' `resisted` and
`tool_bugs` notes, and `experiments/xor/RESULTS.md`. Records = how many of the 60 records'
`resisted`/`tool_bugs` notes mention it (keyword count, approximate).

| # | What agents did by hand | Records | In UI? | In API? | In MCP? |
|---|---|---|---|---|---|
| M1 | **Automatic checks**: function preservation, per-leaf closed-form error (numeric and as displayed), hierarchy check (`check_explanation.py`) | 60 (protocol step 6) | no | no | partly (`get_coverage` has hierarchy) |
| M2 | **Coverage + single-root checklist**: which nodes are uncovered, unconnected inputs must be root entries, one root containing the output | 19 | no | wrong (W1) | yes |
| M3 | **Input fan-out splits found by hand** ("not flagged by tools") | 5 | partial (strategy wizard) | `detect_splits` | yes (fixed in v3) |
| M4 | **Dead / always-on / affine hidden units**: fraction of train rows on which each ReLU is active; "never clips (affine)", "never fires (dead)" | 36 | per node only (activation profile, one at a time) | per node | per node (`activation_profile` needs `node_id`) |
| M5 | **Linearisation / restore gain**: AUC with all ReLUs replaced by affine fits; per-unit restore gain (churn study A; heart-s1 "all-ReLU→identity 0.939→0.916") | 3 | no | no | no |
| M6 | **Rare one-hot categories / state offsets**: per-category logit effect vs empirical log-odds offset, n rows; "memorised rule", "fitted to 14 rows" (churn `rare_category_switches`, study A: only 39 % hold on test) | 17 | no | no | no |
| M7 | **One-hot grouping**: collapse `state:*`, `native-country:*` into a base feature in tables/graph | all churn/adult | partial (`_group_onehot_features` used for viz "source view") | partial | no |
| M8 | **Near-cancelling pairs** (direct edge + hidden route cancel) | 7 | no | no | no |
| M9 | **Hypothesis / evidence write-up per leaf** (closed form, hypothesis, evidence summary) | 60 | no editor; hypothesis shown only in a tooltip | `PUT /evidence/narrative` | `update_narrative` |
| M10 | **Review a Claude explanation and record Mike's verification** (`mike_verification` in `explanations/<label>.json`) | 60 pending | no | no | no |
| M11 | **Candidate leaf subgraphs** from topology (hidden-node input closures, hinges with their entries, nodes feeding the output, serial chains) — protocol step 2 | 60 | no | no | no |
| M12 | `get_viz_summary` reports a hidden exit's activation before the output weight; agents computed the logit contribution by hand | 1 | — | — | open issue |
| M13 | Lineage of a top performer, jumps, ancestor diffs | (Mike's request) | no | no | no (lineage core in progress) |

## 4. Survival (Cox) extension: what the UI would need

The Cox extension (`origin/exp/cox-survival`) is now merged into this branch (merge
commit `2657f48`, approved by Mike). Items 1–5 below are implemented in the portal
(`docs/survival.md` §7); item 6 is a plain ridge Cox on the network's inputs, not the
published equation. The original list of what the UI needed:

1. **Task-type flag on the experiment/dataset** (`classification` vs `survival`), set from
   the dataset's time/event columns, used to switch every metric panel.
2. **Performance panel**: Harrell's C-index (with bootstrap CI), time-dependent AUC at
   chosen horizons, calibration at a horizon (predicted vs Kaplan–Meier observed risk by
   decile), integrated Brier score; hide AUC/accuracy/log-loss for survival.
3. **Output semantics**: the output is the log-risk (identity output); all "logit"
   wording becomes "log hazard ratio". Annotation readings: an annotation's exit
   contribution `w·y` is a **log hazard ratio**; show `exp(w·Δy)` as a hazard ratio for
   a one-unit / one-SD / category change of each entry (from the linear terms of the
   closed form, or empirically from the partial-dependence curve).
4. **Linearisation / restore gain** in C-index instead of AUC (same code with the metric
   swapped).
5. **Dataset loading**: time and event column selection, event-stratified splits.
6. **Comparator**: a linear Cox model on the same split (lifelines/scikit-survival or
   the branch's own `cox.py`).

The new evidence code in this branch takes a `metric` callable so (4) is a one-line
change once the survival metrics are available.

## 5. Prioritised gap list (impact × effort)

Impact: H = blocks or misleads the researcher, M = saves substantial time, L = nice to have.
Effort: S < 2 h, M = half a day, L = days.

| Priority | Gap | Impact | Effort | Status on this branch |
|---|---|---|---|---|
| 1 | W1 coverage route wrong | H | S | done |
| 2 | W2–W4 start command, orphan marking, API serves the UI | H | S | done |
| 3 | M1 automatic checks (function preservation, closed-form error, hierarchy) in API + UI | H | S | done |
| 4 | M10 review + verify the 60 Claude explanations (record viewer, `mike_verification`) | H | M | done |
| 5 | M2 coverage/hierarchy checklist, clickable uncovered nodes | H | S | done |
| 6 | M4 unit activity table (dead / always-on / affine) | M | S | done |
| 7 | M5 linearise / restore gain per annotation | M | M | done |
| 8 | M9 hypothesis/evidence template editor | M | S | done |
| 9 | M11 candidate leaf subgraphs | M | M | done |
| 10 | M6/M7 one-hot grouping + rare-category offsets table | M | M | table done (Categories tab, grouped by base feature); collapsing one-hot inputs into one node *in the graph* not done |
| 11 | M13 lineage panel | M | M | done (Lineage tab); real runs need the parent backfill first |
| 12 | W5 experiment list speed + search | M | S | done |
| 13 | Survival display (section 4) | H for VAREANZ | M | blocked on the merge |
| 14 | W7 composition formula force | L | S | already in the UI |
| 15 | M8 near-cancelling pairs detector | L | M | not done |
| 16 | M12 `get_viz_summary` logit contribution | L | S | not done |
| 17 | W6 read routes create `Explanation` rows | L | S | new routes only |
| 18 | Survival task-type display and portal workflow | H for VAREANZ | M | done (`/portal`, survival evidence panel, hazard ratios) |
| 19 | Whole-model evidence panel crashed on survival datasets (`mse` is null) | H | S | fixed: survival bundle rendered instead |
| 20 | `DatabaseBackpropPopulation.resume_from_db` re-saved generation 0 (unique-constraint crash on resume) | H | S | fixed: `run()` continues at the saved generation |

## 6. Where the features are in the UI

| Feature | Where | API | MCP |
|---|---|---|---|
| Review queue for the 60 records | top nav **Review explanations** (`/review`) | `GET /api/records` | — |
| Record vs live checks, Mike's verification, export/import | explorer **Tools → Review & verify** | `GET /tools/record`, `PUT /api/records/{label}/verification`, `GET /tools/operations/export`, `POST /tools/operations/import-preview`, `POST /tools/operations/import` | — |
| Checks + coverage/hierarchy checklist (uncovered nodes highlighted, clickable) | **Tools → Checks & coverage** | `GET /tools/check`, `GET /analyze/coverage` | `check_explanation`, `get_coverage` |
| Dead / always-on / rare / hinge units | **Tools → Unit activity** | `GET /tools/unit-activity` | `unit_activity` |
| Candidate leaf subgraphs | **Tools → Candidates** (click selects; the annotation wizard takes over) | `GET /tools/candidates` | `suggest_candidates` |
| Linearise / restore gain | **Tools → Restore gain** | `POST /tools/linearise` | `linearise_annotations` |
| One-hot groups, rare-category offsets | **Tools → Categories** | `GET /tools/category-offsets` | `category_offsets` |
| Hypothesis / evidence template | evidence panel → **Hypothesis & evidence write-up** | `GET/PUT /tools/annotations/{id}/hypothesis` | `update_narrative` |
| Lineage chart, jumps, ancestor diff | **Tools → Lineage** | `/lineage`, `/lineage/jumps`, `/ancestor`, `/diff/{b}`, `/diff/{b}/graph` | `get_lineage`, `detect_jumps`, `get_ancestor`, `diff_genomes` |

`/tools/...` is short for `/api/genomes/{id}/tools/...`.

The API was exercised on the real database (read-only use, port 8011) for heart-s1
(`d3b1f42b`), churn100-s0 (`3b4b0fe5`) and adult-s1 (`f2932f2e`). Every tool route answers
in under 1 s; `/tools/check` reproduces heart-s1's recorded closed-form error, 0.012276.
No verification or import was written to the real database or to the records.
