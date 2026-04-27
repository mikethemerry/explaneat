# ExplaNEAT MCP Server Design

## Goal

Create an MCP (Model Context Protocol) server that gives Claude full access to ExplaNEAT's model analysis capabilities. Claude can explore experiments, inspect models, apply operations (splits, annotations), gather evidence (formulas, visualizations, SHAP), and produce comprehensive reports explaining how evolved NEAT models work.

The workflow is collaborative: Claude has full read/write access and can discuss findings with the user as it builds up an explanation.

## Architecture

### Approach: Fine-grained tools, direct Python imports

- **One tool per capability** (~29 tools) — gives Claude maximum flexibility for interactive, discussion-driven analysis
- **Direct imports** of `explaneat.analysis`, `explaneat.db`, `explaneat.core` — single process, no dependency on a running FastAPI server
- **MCP Python SDK** (`mcp[cli]`) with stdio transport — Claude Code launches it as a subprocess
- **Database**: reads `DATABASE_URL` env var, defaults to `postgresql://localhost/explaneat_dev`

### Directory Structure

```
mcp_server/
├── __init__.py
├── server.py          # MCP server setup, tool registration, DB init
├── tools/
│   ├── __init__.py
│   ├── experiments.py  # Experiment & genome discovery (tools 1-5)
│   ├── models.py       # Model structure (tools 6-8)
│   ├── operations.py   # Operations read/write (tools 9-13)
│   ├── evidence.py     # Evidence & analysis (tools 14-20)
│   ├── coverage.py     # Coverage & classification (tools 21-23)
│   ├── datasets.py     # Dataset access (tools 24-26)
│   └── snapshots.py    # Snapshots & narrative (tools 27-29)
└── rendering.py        # Plot rendering to PNG via matplotlib
```

## Tool Inventory

### Experiment & Genome Discovery
1. **`list_experiments`** — List experiments with pagination
2. **`get_experiment`** — Get experiment details (config, dataset info)
3. **`get_best_genome`** — Get highest-fitness genome from an experiment
4. **`list_genomes`** — List genomes with filtering (by experiment, fitness, etc.)
5. **`get_genome`** — Get genome metadata (fitness, node/connection counts)

### Model Structure
6. **`get_phenotype`** — Get pruned active network (nodes, connections)
7. **`get_model_state`** — Get current model state with all operations applied (optionally collapsed)
8. **`get_node_info`** — Get node properties (bias, activation, response, aggregation)

### Operations (read/write)
9. **`list_operations`** — List operation event stream in sequence order
10. **`apply_operation`** — Apply an operation (split_node, consolidate_node, add_identity_node, add_node, remove_node, annotate, rename_node, rename_annotation, disable_connection, enable_connection)
11. **`validate_operation`** — Validate an operation without applying it
12. **`undo_operation`** — Remove an operation and all subsequent ones
13. **`get_annotations`** — List annotations with hierarchy info

### Evidence & Analysis
14. **`get_formula`** — Get symbolic formula for annotation (LaTeX + SymPy)
15. **`compute_viz_data`** — Compute raw visualization data (line, heatmap, PCA, sensitivity, partial dependence, ICE, scatter, distribution, activation profile, edge influence, regime map)
16. **`render_visualization`** — Render visualization to PNG image
17. **`get_viz_summary`** — Compute summary statistics for visualization data
18. **`compute_shap`** — SHAP variable importance for an annotation
19. **`compute_performance`** — Evaluate annotation performance (MSE, accuracy, correlation)
20. **`get_input_distribution`** — Analyze input feature distributions

### Coverage & Classification
21. **`classify_nodes`** — Classify nodes as entry/intermediate/exit
22. **`detect_splits`** — Detect nodes needing splits for annotation
23. **`get_coverage`** — Structural and compositional coverage metrics

### Datasets
24. **`list_datasets`** — List available datasets
25. **`get_dataset`** — Get dataset metadata and info
26. **`get_dataset_splits`** — Get train/test splits

### Snapshots & Narrative
27. **`save_snapshot`** — Save evidence snapshot with visualization and metadata
28. **`update_narrative`** — Update annotation narrative/description
29. **`list_evidence`** — List saved evidence entries

## DB Session Management

- **Database initialized once** at server startup, stored on server instance
- **One session per tool call** via `db.session_scope()` context manager
- **Never return ORM objects** outside session — serialize to dicts within the session
- **Eager load relationships** when needed (`joinedload`) to avoid lazy loading outside session

```python
db = Database()  # once at startup

async def tool_handler(params):
    with db.session_scope() as session:
        # All DB access and serialization within session
        result = {...}
    return result
```

## Tool Schema Patterns

### Read tools
- Input: identifiers (genome_id, experiment_id, etc.) + optional filters
- Output: JSON dict with all relevant data

### Write tools (operations)
- Input: genome_id + operation_type enum + params dict (varies by operation type)
- Output: updated model state + operation sequence number

### Evidence tools
- Input: genome_id + annotation_id + viz_type + dataset_id + split_id + options
- Output varies by mode:
  - `compute_viz_data`: raw numerical arrays (grid points, scatter points, domains)
  - `render_visualization`: PNG image bytes
  - `get_viz_summary`: summary statistics (trends, ranges, correlations)

## Visualization Rendering

For `render_visualization`, matplotlib renders the viz_data output to a PNG temp file, returned as image content. This avoids needing a browser or Observable runtime server-side.

Supported viz types: line, heatmap, partial_dependence, pca_scatter, sensitivity, ice, feature_output_scatter, output_distribution, activation_profile, edge_influence, regime_map.

## Configuration

The MCP server is configured in Claude Code's MCP settings:

```json
{
  "mcpServers": {
    "explaneat": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server"],
      "cwd": "/Users/mike/dev/explaneat",
      "env": {
        "DATABASE_URL": "postgresql://localhost/explaneat_dev"
      }
    }
  }
}
```
