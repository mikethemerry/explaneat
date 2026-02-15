# React Interactive Explorer Requirements

This document captures the functional and technical requirements for the React-based interactive network explorer.

## Architecture

The explorer is an API-backed single-page app:
- **Backend**: FastAPI server (`explaneat/api/`) serving REST endpoints
- **Frontend**: React + TypeScript + Vite (`web/react-explorer/`), built and deployed to `explaneat/static/react_explorer/`
- **Rendering**: React Flow for network graph, Observable Plot for visualizations, KaTeX for formulas

## Functional Requirements

### 1. Experiment Management

- **Experiment List** (`ExperimentList.tsx`): Table of experiments with columns for name, dataset, generations, best fitness
- **Dataset Setup** (`DatasetSetupModal.tsx`): Modal to download PMLB datasets, link to experiments, and create train/test splits
- **Genome Selection**: Click "Explore" on an experiment to load its best genome into the explorer

### 2. Graph Rendering

- **Network Viewer** (`NetworkViewer.tsx`): React Flow-based directed graph
  - Input, hidden, output, and identity nodes with distinct styling
  - Annotation-colored nodes and connections
  - Positive (blue) and negative (red) weighted edges
  - Custom layer-based layout with bezier curves
  - Click to select nodes, Shift+click for multi-select
- **Collapsed View** (`useCollapsedView.ts`): Annotations can be collapsed into single nodes, hiding internal computation
- **Node Types**: Input (green), Hidden (blue), Output (red), Identity (yellow), Annotation (purple)

### 3. Operations Panel

- **Operations Panel** (`OperationsPanel.tsx`): Node selection state, operation execution, annotation strategy
- **Operations**: Split node, consolidate, add identity node, add/remove node, create annotation
- **Annotation Strategy Wizard**: Analyzes selected nodes and computes required operations (identity nodes, splits) to create a valid annotation. Three-precondition validation (entry-only ingress, exit-only egress, pure exits)
- **Operation History**: Scrollable list of all applied operations with undo support

### 4. Evidence Panel

- **Evidence Panel** (`EvidencePanel.tsx`): Visualization and evidence management per annotation
- **Formula Display** (`FormulaDisplay.tsx`): Shows the closed-form mathematical formula of the annotation function, rendered with KaTeX. Fetched via `GET /api/genomes/{id}/evidence/formula?annotation_id=...`
- **Dataset Selector** (`DatasetSelector.tsx`): Select dataset split for visualization. Auto-detects linked split from experiment. Options for train/test/both data and sample fraction
- **Visualization Types** (`VizCanvas.tsx`):
  - **Line Plot**: Function curve with optional scatter overlay of actual data points
  - **Heatmap**: 2D function surface with scatter overlay
  - **Partial Dependence**: 1D or 2D partial dependence (varies selected inputs, fixes others at median)
  - **PCA Scatter**: 2D PCA projection of entry activations colored by exit value
  - **Sensitivity**: Bar chart of per-input sensitivity scores
- **Snapshot Management**: Save visualizations as evidence on annotations with narrative text
- **Evidence Gallery** (`EvidenceGallery.tsx`): View saved evidence snapshots per annotation

### 5. Annotation Management

- **Annotation List**: Collapsible list of all annotations with entry/exit counts, leaf/composition badges
- **Annotation Toggling**: Click annotations to expand/collapse their nodes in the graph
- **Composition Annotations**: Parent-child relationships for hierarchical explanations

## API Endpoints

### Experiments
- `GET /api/experiments` - List all experiments
- `PUT /api/experiments/{id}/dataset` - Link dataset and create split
- `GET /api/experiments/{id}/split` - Get linked split info

### Genomes
- `GET /api/genomes/{id}` - Get genome details
- `GET /api/genomes/{id}/model-state` - Get model state with operations applied
- `GET /api/experiments/{id}/best-genome` - Get best genome for an experiment

### Operations
- `POST /api/genomes/{id}/operations` - Add operation (split, identity, annotate, etc.)
- `DELETE /api/genomes/{id}/operations/{seq}` - Undo operation and all subsequent

### Evidence
- `POST /api/genomes/{id}/evidence/viz-data` - Compute visualization data
- `GET /api/genomes/{id}/evidence/formula` - Get LaTeX formula for annotation
- `POST /api/genomes/{id}/evidence/snapshot` - Save visualization snapshot
- `GET /api/genomes/{id}/evidence?annotation_id=...` - List evidence for annotation

### Datasets
- `GET /api/datasets` - List all datasets
- `POST /api/datasets/pmlb` - Download dataset from PMLB
- `POST /api/datasets/{id}/splits` - Create train/test split

## Evidence Computation Pipeline

The evidence system works on the **fully annotated model** (with identity nodes, split nodes, etc.), not the raw NEAT genome:

1. `_build_model_state()` loads the phenotype and replays all operations via `ModelStateEngine`
2. `StructureNetwork` builds a layered feedforward network from the resulting `NetworkStructure`
3. `ActivationExtractor.from_structure()` runs a forward pass and extracts activations at annotation entry/exit nodes
4. `AnnotationFunction.from_structure()` builds the annotation's mathematical function from its subgraph weights/biases/activations
5. `viz_data` computes visualization-specific data (grid curves, heatmaps, PCA projections, sensitivity scores)
6. Frontend `VizCanvas` renders using Observable Plot

### Split Input Nodes

When `apply_split_node` splits an input node (e.g., `-20` → `-20_a`, `-20_b`), the split nodes inherit `NodeType.INPUT` but `input_node_ids` still lists the original. `StructureNetwork` detects these and maps them to the correct input tensor column via base node ID extraction.

## Non-Functional Requirements

- **Performance:** Handle NEAT genomes with ~70 nodes and ~70 connections smoothly
- **Build:** `npm run build` produces static bundle; deploy with `cp -R dist/* explaneat/static/react_explorer/`
- **Dev:** `npm run dev` on port 5173 with CORS to API on port 8000

## Layout

Custom layer-based layout algorithm in `NetworkViewer.tsx` (not dagre). Nodes positioned left-to-right by depth, with bezier curve edges.

### Depth Computation (`computeNodeDepths`)

Computes the maximum distance from any input node for each node:
- Input nodes get depth 0
- Each node's depth = max(predecessor depths) + 1
- Uses iterative relaxation until convergence

**Cycle detection:** Before computing depths, runs DFS from input nodes to identify back-edges (edges that form cycles). Back-edges are excluded from depth computation, producing a DAG. This handles cycles that can arise from collapsed annotation rerouting (e.g., when an entry node both feeds into and receives from the annotation proxy node). Cycles are logged as warnings.

### Within-Layer Ordering (`reorderByBarycenter`)

Nodes within each layer are ordered using the barycenter heuristic (Sugiyama-style) to minimize edge crossings:
1. **Forward pass** (layers 1→max): sort each layer by the average position of predecessors in the previous layer
2. **Backward pass** (layers max-1→0): refine by the average position of successors in the next layer
3. **Tiebreak:** node type (input < hidden < identity < annotation < output), then numeric ID

### Collapsed View Metadata

When annotations are collapsed (`useCollapsedView.ts`), the model metadata's `input_nodes` and `output_nodes` are filtered to only include nodes visible in the collapsed view. This ensures the depth computation receives correct input/output sets for layer assignment.
