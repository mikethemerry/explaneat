# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExplaNEAT is a NEAT (NeuroEvolution of Augmenting Topologies) implementation with backpropagation capabilities for explainable AI research. It combines evolutionary neural networks with gradient-based training and provides tools for analyzing, visualizing, and annotating evolved network structures.

### Theoretical Framework: Functional Decomposition and Recomposition

ExplaNEAT treats neural network explanation as a two-phase process operating on the network's computational graph:

**Phase 1 — Functional Decomposition**: A NEAT network is a composition of primitive functions (sigmoid, relu, etc.) connected in a DAG. Every node computes `f(weighted_sum(inputs))`. The network as a whole computes some function `N(x₁,...,xₙ)` that is the composition of all these primitives. This composition is already implicit in the graph structure — ExplaNEAT makes it explicit by extracting the closed-form mathematical expression via `AnnotationFunction`. Any subgraph can be read as a function from its entry nodes to its exit nodes.

**Phase 2 — Recomposition through Annotation**: The researcher rewrites the fully-decomposed graph into a human-readable form using two operations:

1. **Identity operations** (node splitting, identity node insertion): Restructure the graph without changing its function. Analogous to algebraic identities — they make the structure amenable to annotation without altering the computed function. For example, splitting a dual-purpose node into two copies separates its roles.

2. **Function composition** (annotation/collapse): Replace a subgraph with a named function node. An annotation over entries {a,b,c} with exits {x,y} declares that the subgraph computes `(x,y) = F(a,b,c)`. Collapsing replaces the individual nodes with a single multi-input, multi-output function node carrying the closed-form expression.

The explanation process thus moves between levels of abstraction:

- **Fully decomposed**: all primitive NEAT nodes — the raw evolved network
- **Partially recomposed**: some subgraphs named as functions, e.g., `F(a,b,c) = G(a,b) + H(b,c)`
- **Fully recomposed**: the network expressed as a composition of named, interpretable functions

This is **term rewriting on a DAG** — substituting subexpressions with named functions — which is provably cycle-free. Hierarchical annotations compose naturally: if G and H are child annotations of F, then F's formula expresses how G and H combine. This supports both parallel composition (`F(a,b,c) = Combine(G(a,b), H(b,c))`) and serial composition (`F(a,b,c) = H(G(a,b), c)`).

The key insight is that the network *already is* a functional composition — the researcher's job is to find meaningful named groupings within it, not to impose structure that isn't there.

See `docs/theoretical_framework.md` for the full mathematical treatment.

## Commands

### Environment
**Always use `uv run` for Python commands** - this automatically uses the virtual environment.

```bash
# Install (see README.md; Windows: docs/install_windows.md; VARIANZ: docs/varianz_runbook.md)
# Python 3.12 via uv (.python-version); no Node: the built UI is committed in explaneat/static/react_explorer
uv sync --extra survival --extra dev
uv run python -m explaneat setup    # SQLite DB in ~/.explaneat (EXPLANEAT_HOME); uses the bundled UI
uv run python -m explaneat doctor   # environment check with fixes
uv run python -m explaneat demo     # hello world on synthetic data -> demo_output/report.md (PASS/FAIL per stage)
uv run pytest -m smoke              # ~3 s install smoke test

# Run tests
uv run pytest                           # All tests
uv run pytest tests/test_db/            # Database tests only
uv run pytest -m "not slow"             # Skip slow tests
uv run pytest tests/test_analysis/test_annotation_manager.py::TestAnnotationManager::test_create_annotation -v

# Run the genome explorer CLI
uv run python genome_explorer_cli.py --interactive
```

### Database (PostgreSQL)
```bash
uv run python -m explaneat db create-db   # Create database
uv run python -m explaneat db init        # Create all tables
uv run python -m explaneat db upgrade     # Apply migrations
uv run python -m explaneat db revision "message"  # Create migration
```

### API Server (also serves the built UI at http://127.0.0.1:8000/)
```bash
uv run python -m explaneat api              # 127.0.0.1:8000
uv run python -m explaneat api --port 8080  # Custom port
uv run python -m explaneat api --mark-orphans  # also mark 'running' experiments interrupted (off by default: scripts' runs are 'running' too)
```

### React Explorer Frontend
```bash
cd web/react-explorer && npm install && npm run dev   # Dev server (port 3001, proxies /api to :8000)
cd web/react-explorer && npm ci && npx tsc --noEmit -p . && npm run build   # Type check + production build
uv run python -m explaneat setup --skip-db   # replaces explaneat/static/react_explorer with dist/ (only if built from current source)
# Then COMMIT explaneat/static/react_explorer: the clinical install has no Node. build-info.json records the
# source hash; doctor warns and tests/test_install/test_ui_bundle.py fails if the bundle is stale
```

### MCP Server
```bash
uv run python -m mcp_server   # Run MCP server (stdio transport)
```

To configure in Claude Code settings:
```json
{
  "mcpServers": {
    "explaneat": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server"],
      "cwd": "/path/to/explaneat",
      "env": {
        "DATABASE_URL": "postgresql://localhost/explaneat_dev"
      }
    }
  }
}
```

The MCP server exposes about 40 tools for model analysis (also the analysis tools check_explanation, linearise_annotations, unit_activity, suggest_candidates, category_offsets):
- **Discovery**: list_experiments, get_experiment, get_best_genome, list_genomes, get_genome
- **Structure**: get_phenotype, get_model_state, get_node_info
- **Operations**: list_operations, apply_operation, validate_operation, undo_operation, get_annotations
- **Evidence**: get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution (on a `task_type: "survival"` dataset, `compute_performance` returns Harrell/Uno C, td-AUC, Brier/IBS and calibration at a horizon instead of AUC/accuracy)
- **Coverage**: classify_nodes, detect_splits, get_coverage (its `decomposition` block warns when one part holds a whole wide layer or hidden units sit only in the root or only as entries: coverage can be 1.0 then)
- **Datasets**: list_datasets, get_dataset, get_dataset_splits
- **Snapshots**: save_snapshot, update_narrative, add_evidence, amend_evidence, list_evidence
- **Queries** (verifier tools for the explaining agent, provenance-stamped): query_subgroup, query_probe, query_enumerate, query_ablate, query_linearise, get_query, list_queries, set_strict_mode
- **Lineage**: get_lineage, get_ancestor, diff_genomes, detect_jumps (see `ANCESTRY_TRACKING.md`)

Evidence and query tools default to the **full training split** (no sampling unless asked) and every result carries `split` (rows used) and `provenance.id`: an entry in the `evidence_log` table (tool, arguments, query: population, quantity, scope, unit; split, n, rows hash). Hypotheses and narratives reference numbers as `{{q:<id>.<path>|<fmt>}}`; writes are checked against the model at once (`explaneat/analysis/narrative_check.py`) and, in strict mode (`explanation_integrity`, on for explanations created from now on, off for existing ones), bare typed numbers and exact formulas that do not reproduce are refused. Tables added by migration `l5m6n7o8p9q0` (suggest `db upgrade`; never run it on `explaneat_dev` yourself). `explaneat verify recheck` (v3) is precise (`mismatch` only when confident, else `needs_review`) and checks the provenance; its `--summary` block is the record's `recheck_summary`. Drafts write numbers as claim slots `{{C086.risk.to|pct}}` (`plain_language_slots.py`): filled by code, citations derived.

## Architecture

### Core Layer (`explaneat/core/`)

- **`explaneat.py`**: Main `ExplaNEAT` class - wraps a NEAT genome and provides network analysis (depth, density, skippiness), and methods to get genotype/phenotype `NetworkStructure` representations
- **`neuralneat.py`**: `NeuralNeat` class - converts NEAT genome to PyTorch `nn.Module` for forward passes and training. Uses `NodeMapping` to organize nodes into layers. Works with raw NEAT genomes (integer node IDs only)
- **`structure_network.py`**: `StructureNetwork` class - builds a layered feedforward network from a `NetworkStructure` and runs forward passes with per-node activation storage. Unlike `NeuralNeat`, works with string node IDs (identity nodes, split nodes, etc.). Handles: identity activation, split input nodes (maps to base node's tensor column), partial input connectivity (pruned phenotypes), skip connections
- **`model_state.py`**: `ModelStateEngine` - replays operations (split_node, add_identity_node, annotate, etc.) on the original phenotype to produce the current annotated model state. Stores operations as an ordered list; supports undo via `remove_operation(seq)`
- **`operations.py`**: Operation handlers (`apply_split_node`, `apply_add_identity_node`, `apply_add_node`, `apply_remove_node`, `apply_consolidate_node`) that modify `NetworkStructure` in place. Also contains `validate_operation()` which enforces the three preconditions for annotations
- **`backproppop.py`**: `BackpropPopulation` - extends NEAT's population with backpropagation training loop. Full-batch; minimises `self.criterion` (default `nn.BCELoss()`; any `criterion(pred, target)`, e.g. `CoxCriterion` on an `[N, 2]` survival target)
- **`genome_network.py`**: Data classes for network representation (`NetworkStructure`, `NetworkNode`, `NetworkConnection`, `NodeType`)

### Analysis Layer (`explaneat/analysis/`)

- **`genome_explorer.py`**: `GenomeExplorer` - main entry point for loading genomes from DB and exploring ancestry, performance, and visualization
- **`annotation_function.py`**: `AnnotationFunction` - extracts the mathematical function of an annotation subgraph from a `NetworkStructure`. Supports evaluation (`__call__`), dimensionality queries, and LaTeX formula generation via sympy. Use `from_structure(annotation, structure)` for annotated models
- **`activation_extractor.py`**: `ActivationExtractor` - runs forward passes through a network and extracts activations at annotation entry/exit nodes. Two modes: legacy (NEAT genome + config) and structure mode (`from_structure(structure)` for annotated models)
- **`activation_cache.py`**: LRU cache for activation data to avoid recomputing forward passes
- **`viz_data.py`**: Compute visualization-ready data for Observable Plot rendering: `compute_line_plot`, `compute_heatmap`, `compute_partial_dependence`, `compute_pca_scatter`, `compute_sensitivity`
- **`annotation_manager.py`**: `AnnotationManager` - CRUD operations for annotations (subgraph hypotheses with entry/exit nodes)
- **`explanation_manager.py`**: `ExplanationManager` - manages explanations (groups of annotations with coverage metrics)
- **`node_splitting.py`**: `NodeSplitManager` - handles node splits for dual-function nodes (e.g., node 5 splits into "5_a", "5_b")
- **`lineage.py`**: ancestor chains (`get_lineage`, fitter-parent or parent1 mode), `detect_jumps`, `diff_genomes`, `ancestor_at_generation`, `map_genes_to_annotations`. Reads DB `parent1_id`/`parent2_id`, else the replay sidecar `experiments/lineage/parents/<experiment>.csv.gz`
- **`coverage.py`**: `CoverageComputer` - computes structural and compositional coverage of explanations
- **`collapse_validator.py`**: `CollapseValidator` - validates collapse preconditions and performs collapse/expand operations (pure graph logic, no DB)
- **`subgraph_validator.py`**: `SubgraphValidator` - validates connectivity of annotation subgraphs
- **`visualization.py`**: `GenomeVisualizer`, `InteractiveNetworkViewer` - Pyvis and React-based network visualization

### Survival Layer (`explaneat/survival/`)

Cox proportional-hazards models. The network outputs a log-risk `g(x)` through an identity output, is trained full-batch on the Efron Cox partial likelihood, and gets a Breslow baseline: `S(t|x) = S0(t)^exp(g(x))`. User guide: `docs/survival.md`. Validation results: `experiments/survival/RESULTS.md`. The optional extra `uv sync --extra survival` installs scikit-survival and lifelines.

- **`cox.py`**: `CoxRiskSets` (build once per fold, from that fold's training rows), `cox_ph_loss` (Breslow/Efron, vectorised), `null_cox_loss`
- **`criterion.py`**: `CoxCriterion` for `BackpropPopulation(criterion=...)`; target `[N, 2]` = (time, event)
- **`fitness.py`**: `make_cox_fitness` (fitness = −loss − λ·k/D, never 1/loss), `cox_concordance_fitness` (ablation)
- **`breslow.py`**: `BreslowBaseline`, `fit_survival_predictor`
- **`metrics.py`**: Harrell/Uno C, time-dependent AUC, IPCW Brier, IBS, `calibration_at_horizon` (decile predicted vs KM observed risk), `survival_metrics` bundle
- **`data.py`**: targets, sksurv builtin loaders, `load_survival_csv` (cohort hook), event-stratified splits, DB storage (`task_type: "survival"`, `y_data` `[N, 2]`, no schema change)
- **`evidence.py`**: `survival_performance`, called by `compute_performance` (MCP + API) on survival datasets
- **`explain.py`**: hazard-ratio readings of output terms (direct edges, ReLU hinges)
- **`simulate.py`**: known-truth linear / non-linear simulations (run first)
- Runner and benchmark: `experiments/survival/run_seed.py`, `study.py`. A local SQLite DB for restricted data uses `explaneat/db/sqlite_compat.py`

### API Layer (`explaneat/api/`)

FastAPI REST API serving the React Explorer frontend:

- **`app.py`**: FastAPI app factory with CORS, static file serving, route registration
- **`schemas.py`**: Pydantic models for request/response validation
- **`routes/experiments.py`**: List experiments, link datasets, get experiment splits
- **`routes/genomes.py`**: Get genome details, model state (with operations applied), best genome per experiment
- **`routes/operations.py`**: Add/remove operations on the model state (split, identity, annotate, etc.)
- **`routes/evidence.py`**: Compute visualization data, get formulas, save/list evidence snapshots. Uses `StructureNetwork` via `_build_model_state()` to forward-pass through the fully annotated model
- **`routes/datasets.py`**: List datasets, download from PMLB, create splits
- **`routes/analysis.py`**: Annotation strategy analysis, coverage computation

Run the API: `uv run python -m explaneat api` (default port 8000)

### Database Layer (`explaneat/db/`)

SQLAlchemy models with PostgreSQL/SQLite support:

- **`Experiment`** -> **`Population`** (per generation) -> **`Genome`** (with parent tracking)
- **`Genome`** -> **`Explanation`** (with `operations` JSONB column storing the operation event stream)
- **`Dataset`** (with `x_data`/`y_data` stored as numpy arrays) -> **`DatasetSplit`** (train/test indices)
- **`GeneOrigin`** - tracks when each gene (innovation number) first appeared
- **`pmlb_loader.py`**: Downloads datasets from PMLB and stores them in the database

Key patterns:
- Use `db.session_scope()` context manager for transactions
- `Genome.from_neat_genome()` / `to_neat_genome()` for serialization
- All node IDs are strings to support split nodes (e.g., "5", "5_a"). Split copies are `<base>_<token>`: tokens `a`..`z`, then `Aa`..`Zz` past 26 copies; a consolidated node joins its tokens (`5_a` + `5_Ab` -> `5_aAb`). Parse ids with `explaneat/core/split_ids.py` (never a single-letter regex)
- Operations are stored as a JSONB array on `Explanation`, replayed by `ModelStateEngine`

### React Explorer Frontend (`web/react-explorer/`)

Single-page app built with React + TypeScript + Vite, served as static files:

- **`App.tsx`**: Top-level routing between experiment list and genome explorer
- **`ExperimentList.tsx`**: Lists experiments with dataset setup buttons
- **`GenomeExplorer.tsx`**: Main explorer view with network graph, operations panel, and evidence panel
- **`NetworkViewer.tsx`**: React Flow-based network graph with annotation coloring and collapse support
- **`OperationsPanel.tsx`**: Node selection, operation execution, annotation strategy wizard
- **`EvidencePanel.tsx`**: Evidence visualization for annotations (formula display, dataset selection, viz controls, snapshot management)
- **`VizCanvas.tsx`**: Observable Plot renderer for line plots, heatmaps, PCA scatter, partial dependence, sensitivity
- **`FormulaDisplay.tsx`**: KaTeX rendering of annotation formulas
- **`DatasetSelector.tsx`**: Dataset/split selection with auto-detection from experiment
- **`DatasetSetupModal.tsx`**: Modal for downloading PMLB datasets and creating splits
- **`EvidenceGallery.tsx`**: Display saved evidence snapshots
- **`ToolsDrawer.tsx`**: Tabbed drawer under the graph. Tabs: Review & verify (`RecordPanel.tsx`: Claude record vs live checks, `mike_verification`, export/import), Checks & coverage, Unit activity, Candidates, Restore gain, Categories (one-hot offsets), Lineage (`LineagePanel.tsx`)
- **`HypothesisEditor.tsx`**: Hypothesis write-up with the protocol's leaf template (evidence panel)
- **`RecordsPage.tsx`**: `/review`, the queue of explanation records and their verification status
- **`PortalPage.tsx`**: `/portal`, the "Run experiments" workflow: load a local CSV (survival/classification), launch a seed batch (worker processes, concurrency limit, cancel/resume), and a results table with a ridge Cox / LR comparator. Library: `explaneat/batch/` (`datasets`, `worker`, `jobs`, `results`, `common`; job state in `$EXPLANEAT_HOME/jobs`, default `~/.explaneat`); routes `explaneat/api/routes/portal.py`. `SurvivalEvidence.tsx`: survival metrics + hazard ratios in the evidence panel. User guide and governance: `docs/survival.md` §7–8
- **`api/tools.ts`**: Client for `/api/genomes/{id}/tools/*`, `/api/records`, lineage routes. Library side: `explaneat/analysis/explanation_tools.py`, `explanation_records.py`, `coverage_report.py`; routes `explain_tools.py`, `records.py`, `lineage.py`; governance `explaneat/governance.py`. See `docs/ui_audit.md`

### CLI (`genome_explorer_cli.py`)

Interactive shell for exploring genomes:
- `experiments` / `exp` - list experiments
- `genomes` / `g` - list genomes in current experiment
- `load <genome_id>` - load a genome
- `network-interactive` / `ni` - React visualization
- `annotate` - create annotations
- `splits` - manage node splits
- `coverage` - compute explanation coverage

## Key Patterns

### Network Representations
- **Genotype**: Full genome structure (all nodes/connections)
- **Phenotype**: Pruned to active subgraph (reachable from inputs to outputs)
- **Model State**: Phenotype with all operations (splits, identity nodes, annotations) applied via `ModelStateEngine`. This is the "real" model that the evidence system works on

### Two Forward-Pass Engines
- **`NeuralNeat`**: PyTorch `nn.Module` from raw NEAT genome. Integer node IDs only. Used for training
- **`StructureNetwork`**: NumPy/Torch forward-pass from `NetworkStructure`. String node IDs, handles identity nodes (`identity_5`), split nodes (`5_a`, `5_b`), and split input nodes (`-20_a`). Used for evidence visualization on annotated models
- **Parity**: `NeuralNeat` applies its configured output activation at the output layer (ReLU on hidden layers), whatever the output gene says: sigmoid by default, `identity` for Cox survival models. The setting lives on the NEAT config as `config.output_activation` (`neuralneat.set_output_activation` / `get_output_activation`; absent = sigmoid), is recorded on experiments as `config_json["training"]["output_activation"]` and restored by `load_neat_config`. `ExplaNEAT.get_genotype_network()`/`get_phenotype_network()` set that activation on output nodes (and `metadata["output_activation"]` when non-default), so StructureNetwork, ActivationExtractor, AnnotationFunction and formulas compute the trained model's function. Evidence code must call `sn.override_output_activation(sn.model_output_activation)`, never a literal `"sigmoid"` (`tests/test_core/test_engine_parity.py`)

### Operations-Based Model State
The model state is computed by replaying an ordered list of operations on the base phenotype. Operations are stored as JSONB on the `Explanation` model. Operation types: `split_node`, `consolidate_node`, `add_identity_node`, `add_node`, `remove_node`, `annotate`. The `ModelStateEngine` handles replay, undo, and validation.

### Annotation Hierarchy
Annotations can have parent-child relationships for compositional explanations. Coverage is computed based on how much of the phenotype is explained.

### Collapse vs Coverage
These are distinct concepts. **Coverage** (paper Def 10) measures whether a node's behavior is fully explained. **Collapsibility** measures whether an annotation can be validly replaced by a single node (requires three preconditions: entry-only ingress, exit-only egress, pure exits). See `docs/annotation_collapsing_model.md` for the formalization and `explaneat/analysis/collapse_validator.py` for the validation logic.

### Evidence Pipeline
The evidence system computes visualizations for annotations:
1. `_build_model_state()` replays operations on the phenotype to get the current `NetworkStructure`
2. `ActivationExtractor.from_structure()` runs a forward pass through the full network and extracts activations at annotation entry/exit nodes
3. `AnnotationFunction.from_structure()` builds the annotation's mathematical function from the subgraph
4. `viz_data` module computes visualization data (line plots, heatmaps, PCA, sensitivity, partial dependence)
5. Frontend `VizCanvas` renders using Observable Plot

### Test Markers
```bash
uv run pytest -m db           # Database tests
uv run pytest -m cli          # CLI tests
uv run pytest -m "not slow"   # Skip slow tests
```

## Documentation

**Keep documentation up to date** when making changes to annotation logic, coverage calculations, or UI behavior:

- **`docs/theoretical_framework.md`**: Core theoretical framework — functional decomposition and recomposition of neural networks, term rewriting, cycle freedom proofs, hierarchical composition
- **`docs/annotation_coverage_model.md`**: Mathematical definitions for coverage, annotation strategy (identity nodes, splits, blocking issues), node classification (entry/exit/intermediate)
- **`docs/annotation_collapsing_model.md`**: Mathematical formalization of the collapse operation, three preconditions, composition property, fix mechanisms
- **`docs/annotation_hierarchy_and_splitting.md`**: Node splitting mechanics, hierarchy structure
- **`docs/react_explorer_requirements.md`**: React Explorer UI requirements

When changing annotation strategy logic (in `OperationsPanel.tsx`), update the corresponding section in `annotation_coverage_model.md`. When changing collapse/annotation logic, update the corresponding math docs.

### Annotation Strategy Summary

The UI computes annotation strategies with these rules:

1. **Identity Nodes**: When single exit node has external inputs (partial coverage), add identity node to intercept connections and replace exit
2. **Split Nodes**: When entry/intermediate nodes have external outputs, split them
3. **Blocking Issues**: When intermediate/exit nodes have external inputs, user must expand selection

Execution order: identity nodes → splits → create annotation

## Important Notes

- Do NOT run database migration commands directly - suggest them to the user
- Database resolution: `DATABASE_URL` → `~/.explaneat/config.json` (written by `explaneat setup`, SQLite) → `postgresql://localhost/explaneat_dev`. Mike's research runs live in Postgres `explaneat_dev`; never run `setup` or migrations against it
- Lineage: runs record parents by default (`track_ancestry`); the pre-Nov-2026 runs were backfilled by exact replay (`experiments/lineage/`). API: `explaneat/analysis/lineage.py`
- Survival (Cox): `explaneat/survival/`, `docs/survival.md`. Restricted data (DAP/HDEC): `EXPLANEAT_RESTRICTED=1`; keep data outside the repo; the Claude/MCP path refuses restricted datasets unless explicitly allowed
- The API binds to 127.0.0.1; Swagger docs are off unless `EXPLANEAT_API_DOCS=1`
- Tests use in-memory SQLite for speed
