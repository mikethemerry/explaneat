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
# Install dependencies
uv venv && uv pip install -e .

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

### API Server
```bash
uv run python -m explaneat api              # Start on port 8000
uv run python -m explaneat api --port 8080  # Custom port
```

### React Explorer Frontend
```bash
cd web/react-explorer && npm install && npm run dev   # Dev server (port 5173)
cd web/react-explorer && npm run build                # Production build
cp -R web/react-explorer/dist/* explaneat/static/react_explorer/  # Deploy to static
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

The MCP server exposes 29 tools for model analysis:
- **Discovery**: list_experiments, get_experiment, get_best_genome, list_genomes, get_genome
- **Structure**: get_phenotype, get_model_state, get_node_info
- **Operations**: list_operations, apply_operation, validate_operation, undo_operation, get_annotations
- **Evidence**: get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution
- **Coverage**: classify_nodes, detect_splits, get_coverage
- **Datasets**: list_datasets, get_dataset, get_dataset_splits
- **Snapshots**: save_snapshot, update_narrative, list_evidence

## Architecture

### Core Layer (`explaneat/core/`)

- **`explaneat.py`**: Main `ExplaNEAT` class - wraps a NEAT genome and provides network analysis (depth, density, skippiness), and methods to get genotype/phenotype `NetworkStructure` representations
- **`neuralneat.py`**: `NeuralNeat` class - converts NEAT genome to PyTorch `nn.Module` for forward passes and training. Uses `NodeMapping` to organize nodes into layers. Works with raw NEAT genomes (integer node IDs only)
- **`structure_network.py`**: `StructureNetwork` class - builds a layered feedforward network from a `NetworkStructure` and runs forward passes with per-node activation storage. Unlike `NeuralNeat`, works with string node IDs (identity nodes, split nodes, etc.). Handles: identity activation, split input nodes (maps to base node's tensor column), partial input connectivity (pruned phenotypes), skip connections
- **`model_state.py`**: `ModelStateEngine` - replays operations (split_node, add_identity_node, annotate, etc.) on the original phenotype to produce the current annotated model state. Stores operations as an ordered list; supports undo via `remove_operation(seq)`
- **`operations.py`**: Operation handlers (`apply_split_node`, `apply_add_identity_node`, `apply_add_node`, `apply_remove_node`, `apply_consolidate_node`) that modify `NetworkStructure` in place. Also contains `validate_operation()` which enforces the three preconditions for annotations
- **`backproppop.py`**: `BackpropPopulation` - extends NEAT's population with backpropagation training loop
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
- **`coverage.py`**: `CoverageComputer` - computes structural and compositional coverage of explanations
- **`collapse_validator.py`**: `CollapseValidator` - validates collapse preconditions and performs collapse/expand operations (pure graph logic, no DB)
- **`subgraph_validator.py`**: `SubgraphValidator` - validates connectivity of annotation subgraphs
- **`visualization.py`**: `GenomeVisualizer`, `InteractiveNetworkViewer` - Pyvis and React-based network visualization

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
- All node IDs are strings to support split nodes (e.g., "5", "5_a")
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
- Default database: `postgresql://localhost/explaneat_dev` (override with `DATABASE_URL`)
- Tests use in-memory SQLite for speed
