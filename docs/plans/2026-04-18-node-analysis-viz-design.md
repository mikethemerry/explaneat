# Node Analysis Visualizations — Design

## Problem

Understanding a NEAT node requires knowing its **operating regime** — not just its weights, but where in input space it's active, how strongly, and how that interacts with other nodes. Current tools show weights and formulas but don't answer: "Is this node mostly dead?", "Which edges actually matter?", "How many distinct operating modes does this subgraph have?"

## Scope

Four deliverables:

1. **Node Activation Profile** — histogram + activation rate
2. **Regime Map** — ReLU on/off patterns across a subgraph
3. **Effective Influence** — per-edge contribution variance
4. **Resizable EvidencePanel** — draggable width

## 1. Node Activation Profile

**What:** For a selected node, show the distribution of its activation values across the dataset.

**Backend (`viz_data.py`):**
- New function `compute_activation_profile(activations: np.ndarray, activation_fn: str) -> dict`
- Returns: `{bin_edges, counts, x_label, stats: {mean, std, min, max, median, count, activation_rate, zero_fraction}}`
- `activation_rate` = fraction of samples where output > 0 (meaningful for ReLU)
- `zero_fraction` = fraction exactly at 0 (indicates ReLU clamping)

**Backend route (`evidence.py`):**
- Add `"activation_profile"` to `POST /viz-data` dispatch
- When `node_id` is provided: run forward pass, call `_net.get_node_activation(node_id)`, pass to `compute_activation_profile`
- Also needs the node's activation function name from the `NetworkStructure`

**Frontend (`VizCanvas.tsx`):**
- Reuse existing `renderHistogram` renderer
- Add activation rate / zero fraction display as text overlay on the chart

**Data shape:** Same as existing histogram format — no new renderer needed.

## 2. Regime Map

**What:** For an annotation's subgraph (or a node's ancestor subgraph), identify the distinct ReLU activation patterns across the dataset. Each unique pattern = a "regime." Within each regime, the subgraph reduces to a linear function.

**Backend (`viz_data.py`):**
- New function `compute_regime_map(node_activations: Dict[str, np.ndarray], relu_node_ids: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict`
- For each sample, compute binary vector: `[act > 0 for each ReLU node]`
- Group by unique patterns
- For each regime: count, accuracy, mean prediction, class distribution

**Per-regime linear equations:**
- Within a regime, all ReLUs are either on (linear passthrough) or off (zero). The subgraph becomes affine.
- Use `AnnotationFunction._steps` to build the simplified expression per regime by substituting ReLU(x) → x (when on) or ReLU(x) → 0 (when off)
- Return as LaTeX strings per regime

**Backend route (`evidence.py`):**
- Add `"regime_map"` to `POST /viz-data` dispatch
- Scope: annotation subgraph (all ReLU nodes within) or node ancestor subgraph via `_compute_node_subgraph`
- Run forward pass, collect activations for all nodes in subgraph, filter to ReLU nodes

**Return format:**
```json
{
  "regimes": [
    {
      "pattern": {"219": true, "444": false, "345": true},
      "count": 3500,
      "fraction": 0.72,
      "mean_prediction": 0.73,
      "accuracy": 0.85,
      "class_distribution": {"0": 1200, "1": 2300},
      "latex": "y = 0.37x_{hours} - 0.84x_{marital} + 3.6"
    }
  ],
  "relu_nodes": ["219", "444", "345"],
  "total_samples": 4880
}
```

**Frontend (`VizCanvas.tsx`):**
- New `renderRegimeMap` renderer
- Table layout: one row per regime, columns for pattern (color-coded on/off dots), count, fraction, accuracy, formula
- Sort by count descending (most common regime first)
- Optionally highlight the "dominant" regime

## 3. Effective Influence

**What:** For each edge in a subgraph, compute how much it actually contributes in practice: `var(weight * source_activation)` across the dataset. High variance = this edge differentiates predictions. Low variance = vestigial.

**Backend (`viz_data.py`):**
- New function `compute_edge_influence(connections: List, node_activations: Dict[str, np.ndarray]) -> dict`
- For each connection: `influence = var(weight * activations[from_node])`
- Also compute `mean_contribution = mean(weight * activations[from_node])` for the direction/sign
- Return sorted by influence descending

**Return format:**
```json
{
  "edges": [
    {
      "from": "219", "to": "output",
      "weight": 2.86,
      "influence": 1.45,
      "mean_contribution": 2.1,
      "normalized_influence": 1.0
    }
  ]
}
```

`normalized_influence` scales to [0, 1] relative to the max influence edge.

**Backend route (`evidence.py`):**
- Add `"edge_influence"` to `POST /viz-data` dispatch
- Scope: annotation subgraph or node ancestor subgraph

**Frontend — EvidencePanel bar chart (`VizCanvas.tsx`):**
- New `renderEdgeInfluence` renderer
- Horizontal bar chart, one bar per edge
- Bar length = normalized influence
- Bar color = sign of mean_contribution (green for positive, red for negative)
- Label = "from → to (weight)"

**Frontend — NetworkViewer overlay:**
- New prop `edgeInfluence?: Record<string, {influence: number, mean_contribution: number}>`
- When provided, override edge rendering:
  - Thickness: proportional to normalized influence (min 1px, max 6px)
  - Color: red for negative mean_contribution, blue/green for positive
- Toggle button in EvidencePanel: "Show influence on graph"
- When toggled, compute influence data and pass to NetworkViewer

## 4. Resizable EvidencePanel

- Add a vertical drag handle on the left edge of the EvidencePanel
- On mousedown, track mousemove to resize panel width
- Store width in state (default ~400px, min 300px, max 800px)
- CSS: `cursor: col-resize` on the handle, `user-select: none` during drag

## Implementation Order

1. Resizable EvidencePanel (quick, improves UX for everything)
2. Node Activation Profile (builds on existing histogram infra, validates the per-node activation extraction pattern)
3. Effective Influence (medium complexity, high value — validates forward-pass + per-edge computation)
4. Regime Map (most complex, builds on patterns established in 2 and 3)

## Files to Modify

| File | Changes |
|------|---------|
| `explaneat/analysis/viz_data.py` | Add `compute_activation_profile`, `compute_regime_map`, `compute_edge_influence` |
| `explaneat/api/routes/evidence.py` | Add dispatch cases for 3 new viz types, add helper to extract all node activations |
| `web/react-explorer/src/components/VizCanvas.tsx` | Add `renderRegimeMap`, `renderEdgeInfluence` renderers |
| `web/react-explorer/src/components/NetworkViewer.tsx` | Add `edgeInfluence` prop, influence coloring mode |
| `web/react-explorer/src/components/EvidencePanel.tsx` | Add new viz type options, influence toggle, resizable width |
| `web/react-explorer/src/components/GenomeExplorer.tsx` | Wire influence data between EvidencePanel and NetworkViewer, handle panel width state |
| `web/react-explorer/src/api/client.ts` | No new endpoints needed — existing `POST /viz-data` handles all via `viz_type` field |
