# React Interactive Explorer Requirements

This document captures the functional and technical requirements for the React-based interactive network explorer.

## Architecture

The explorer is an API-backed single-page app:
- **Backend**: FastAPI server (`explaneat/api/`) serving REST endpoints, and the built UI at `/` (SPA fallback)
- **Frontend**: React + TypeScript + Vite (`web/react-explorer/`), built and deployed to `explaneat/static/react_explorer/`
- **Start**: `uv run python -m explaneat api` (binds 127.0.0.1:8000), then open http://127.0.0.1:8000/
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
- **Collapsed View**: annotations are collapsed into function nodes server-side (`GET /model?collapsed=...`)
- **Highlights**: `nodeHighlight`/`edgeHighlight` props mark nodes and edges as added / removed (ghosted) / changed (lineage diffs), uncovered (coverage checklist) or candidate
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

### 4b. Hypothesis editor

- **HypothesisEditor** (`HypothesisEditor.tsx`, in the evidence panel of a real annotation): edits the annotation's hypothesis. "Insert template" adds the explanation protocol's leaf fields (closed form, prefilled from the formula; hypothesis; evidence; caveats). Saving overwrites the hypothesis (`PUT /tools/annotations/{id}/hypothesis`, as MCP `update_narrative`).

### 5. Annotation Management

- **Annotation List**: Collapsible list of all annotations with entry/exit counts, leaf/composition badges
- **Annotation Toggling**: Click annotations to expand/collapse their nodes in the graph
- **Composition Annotations**: Parent-child relationships for hierarchical explanations

### 6. Explanation tools drawer (`ToolsDrawer.tsx`)

A tabbed drawer under the network graph. Everything runs on the training split of the genome's experiment split, read-only unless stated.

- **Review & verify** (`RecordPanel.tsx`): the Claude explanation record for this genome (`experiments/10seed/explanations/<label>.json`, or `$EXPLANEAT_RECORDS_DIR`): summary, what resisted, tool notes, and recorded vs live checks. The **Mike's verification** form (verified, minutes with a timer, changes, notes) writes `mike_verification` into the record file; rebuild the CSVs with `build_explanations_csv.py`. Export the explanation as JSON. Import an exported file or a list of operations: it is replayed in memory and checked first, and it only replaces a non-empty explanation after confirmation.
- **Checks & coverage**: function preservation, structural/compositional coverage (paper Def 10–11), single root, root contains the output, every annotation under the root, full coverage; per-leaf numeric and displayed closed-form error. Uncovered nodes and edges are highlighted in the graph (orange dashed) and clickable.
- **Unit activity**: each hidden ReLU's status on train: dead, rare (≤ 5 rows), always on (affine on the data) or hinge, with mean/SD and direct logit SD.
- **Candidates**: candidate leaf subgraphs from the topology (hidden gate with its inputs; the hidden node's input closure), with the splits each needs. Also lists the direct input→output edges and the unconnected inputs, which must be root entries. Clicking a candidate selects its nodes, and the annotation strategy wizard takes over.
- **Restore gain**: linearise annotation units (least-squares affine fit of the exits on the entries, fitted on train). Shows the all-linear AUC, each unit linearised alone, and each unit's restore gain.
- **Categories**: one-hot inputs grouped by base feature (`state:*`). For each level: rows, rate, the model's logit effect, the empirical and warranted offsets, and a flag when a large effect is not supported by the level's Wilson interval.
- **Lineage** (`LineagePanel.tsx`): fitness along the ancestor chain (fitter-parent or parent-1 line) against the population best and mean per generation. Steps are coloured as structural, weights-only or elite copy, and jumps are shown as red bands. Click the chart, or type a generation, to show that ancestor's diff against the descendant, against its parent, or against that generation's best. The diff appears in the graph: added nodes and edges green, removed ones ghosted, weight/bias changes amber. The jump table lists each jump's added genes and the annotations of the final explanation that contain them. When the run has no recorded parents, the tab says so and points to the replay/backfill.

### 7. Review queue (`RecordsPage.tsx`, `/review`)

All explanation records with full coverage, agreement, closed-form error and verification status, filterable. **Review** opens the genome.

### 7b. Run experiments (`PortalPage.tsx`, `/portal`)

This is the portal workflow for survival and binary classification; the user guide is `docs/survival.md` §7.

1. **Load a local CSV.** The file is read by path and never uploaded. A column preview shows no raw rows. Pick a role for each column (numeric or one-hot feature, time/event or target, ignore), plus imputation, the time unit, the restricted flag, and stratified train/val/test fractions.
2. **Launch a batch.**
   - Templates: `survival-cox-v3` (protocol v3) and `classification-propneat`.
   - Settings: seeds, generations, population, epochs, parsimony λ, and the maximum number of concurrent processes.
   - Each run shows progress, cancel and resume.
3. **Results table.** For each seed, test metrics with bootstrap CIs, the comparator fitted on the same split, the paired Δ, and structure. The table ends with the across-seed mean and t-interval, and exports to CSV. **Explain** opens the selected network.

**Survival evidence** (`SurvivalEvidence.tsx`) is shown in the output node's evidence panel when the dataset is survival:
- Harrell's and Uno's C, td-AUC per horizon, IBS, Cox loss vs null, and the calibration plot at the horizon (E/O);
- hazard-ratio readings of the output terms and of each annotation exit (`GET /api/portal/genomes/{id}/hazard-ratios`).

### 8. Data governance

- The API binds to 127.0.0.1 by default. The UI bundle loads no external scripts or fonts (KaTeX fonts are bundled), and the UI calls only `/api`.
- `EXPLANEAT_RESTRICTED=1` disables the routes that make outbound calls (PMLB/UCI search and download) and the whole MCP path.
- A dataset with `additional_metadata.restricted = true` is refused on the MCP (Claude-assisted) path unless `EXPLANEAT_ALLOW_ASSISTED_RESTRICTED=1`. Explain it in the local UI.
- `GET /api/governance` reports the settings.

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

### Explanation tools (`/api/genomes/{id}/tools`)
- `GET /check`, `POST /linearise`, `GET /unit-activity`, `GET /candidates`, `GET /category-offsets`
- `GET /record`, `GET /operations/export`, `POST /operations/import-preview`, `POST /operations/import`
- `GET|PUT /annotations/{annotation_id}/hypothesis`
- `GET /api/genomes/{id}/analyze/coverage`: paper coverage (same as MCP `get_coverage`)

### Portal (`/api/portal`)
- `POST /csv/preview`, `POST /datasets/csv`, `GET /datasets`, `GET /templates`
- `POST /batches`, `GET /batches`, `GET /batches/{id}`, `POST /batches/{id}/cancel[?run_id=]`, `POST /batches/{id}/resume[?run_id=]`
- `GET /batches/{id}/results`, `GET /batches/{id}/results.csv`, `GET /results?experiment_ids=`
- `GET /genomes/{id}/hazard-ratios`

### Records
- `GET /api/records`, `GET /api/records/{label}`, `PUT /api/records/{label}/verification`

### Lineage
- `GET /api/genomes/{id}/lineage?mode=`, `GET /api/genomes/{id}/lineage/jumps`, `GET /api/genomes/{id}/ancestor?generation=`
- `GET /api/genomes/{a}/diff/{b}`, `GET /api/genomes/{a}/diff/{b}/graph`

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
- **Build:** `npm run build` produces the static bundle. Deploy it by replacing `explaneat/static/react_explorer/` with `dist/`, removing old `assets/` first.
- **Type check:** `npx tsc --noEmit -p .` (clean)
- **Dev:** `npm run dev` on port 3001 (`vite.config.ts`), which proxies `/api` to the API on port 8000

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

When annotations are collapsed (server-side, `?collapsed=`), the model metadata's `input_nodes` and `output_nodes` are filtered to only include nodes visible in the collapsed view. This ensures the depth computation receives correct input/output sets for layer assignment.
