# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExplaNEAT is a NEAT (NeuroEvolution of Augmenting Topologies) implementation with backpropagation capabilities for explainable AI research. It combines evolutionary neural networks with gradient-based training and provides tools for analyzing, visualizing, and annotating evolved network structures.

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

### React Explorer Frontend
```bash
cd web/react-explorer && npm install && npm run build
cp -R web/react-explorer/dist/* explaneat/static/react_explorer/
```

## Architecture

### Core Layer (`explaneat/core/`)

- **`explaneat.py`**: Main `ExplaNEAT` class - wraps a NEAT genome and provides network analysis (depth, density, skippiness), and methods to get genotype/phenotype `NetworkStructure` representations
- **`neuralneat.py`**: `NeuralNeat` class - converts NEAT genome to PyTorch `nn.Module` for forward passes and training. Uses `NodeMapping` to organize nodes into layers
- **`backproppop.py`**: `BackpropPopulation` - extends NEAT's population with backpropagation training loop
- **`genome_network.py`**: Data classes for network representation (`NetworkStructure`, `NetworkNode`, `NetworkConnection`, `NodeType`). `get_phenotype_with_splits()` applies node splits to create modified phenotype networks

### Analysis Layer (`explaneat/analysis/`)

- **`genome_explorer.py`**: `GenomeExplorer` - main entry point for loading genomes from DB and exploring ancestry, performance, and visualization
- **`annotation_manager.py`**: `AnnotationManager` - CRUD operations for annotations (subgraph hypotheses with entry/exit nodes)
- **`explanation_manager.py`**: `ExplanationManager` - manages explanations (groups of annotations with coverage metrics)
- **`node_splitting.py`**: `NodeSplitManager` - handles node splits for dual-function nodes (e.g., node 5 splits into "5_a", "5_b")
- **`coverage.py`**: `CoverageComputer` - computes structural and compositional coverage of explanations
- **`subgraph_validator.py`**: `SubgraphValidator` - validates connectivity of annotation subgraphs
- **`visualization.py`**: `GenomeVisualizer`, `InteractiveNetworkViewer` - Pyvis and React-based network visualization

### Database Layer (`explaneat/db/`)

SQLAlchemy models with PostgreSQL/SQLite support:

- **`Experiment`** -> **`Population`** (per generation) -> **`Genome`** (with parent tracking)
- **`Genome`** -> **`Explanation`** -> **`Annotation`** (subgraph hypotheses)
- **`Explanation`** -> **`NodeSplit`** (one row per split original node, with `split_mappings` JSONB)
- **`GeneOrigin`** - tracks when each gene (innovation number) first appeared

Key patterns:
- Use `db.session_scope()` context manager for transactions
- `Genome.from_neat_genome()` / `to_neat_genome()` for serialization
- All node IDs are strings to support split nodes (e.g., "5", "5_a")

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
- **Phenotype with splits**: Node splits applied to enable annotation of dual-function nodes

### Annotation Hierarchy
Annotations can have parent-child relationships for compositional explanations. Coverage is computed based on how much of the phenotype is explained.

### Test Markers
```bash
uv run pytest -m db           # Database tests
uv run pytest -m cli          # CLI tests
uv run pytest -m "not slow"   # Skip slow tests
```

## Important Notes

- Do NOT run database migration commands directly - suggest them to the user
- Default database: `postgresql://localhost/explaneat_dev` (override with `DATABASE_URL`)
- Tests use in-memory SQLite for speed
