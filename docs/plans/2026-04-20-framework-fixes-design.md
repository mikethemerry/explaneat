# ExplaNEAT Framework Fixes Design

## Goal

Close the gaps between the 2025 paper's formal definitions and the tool's implementation, based on feedback from the first real-world MCP analysis of a Heart Disease model.

## Changes

### 1. Composition Annotations (junction-only)

The `annotate` operation already accepts `child_annotation_ids` and tracks `parent_annotation_id`. But validation in `operations.py:230` rejects any subgraph node already covered by another annotation, blocking compositions.

**Fix**: When `child_annotation_ids` is non-empty, the annotation is a composition. Its `subgraph_nodes` contains only junction nodes — nodes between children not covered by any child. Validation allows children's exit/entry nodes as the composition's boundary nodes but requires `subgraph_nodes` to contain only uncovered junction nodes.

- Compositions' `entry_nodes` can include children's exit nodes (serial) or external nodes
- Compositions' `exit_nodes` can include children's entry nodes or new junction exits
- `AnnotationData` gains `is_composition` property (true when `child_annotation_ids` non-empty)

### 2. Coverage Metrics (use CoverageComputer)

The MCP `get_coverage` tool does a naive hidden-node count. The actual `CoverageComputer` class in `explaneat/analysis/coverage.py` correctly implements paper Defs 10-11 (all candidate nodes + edges, outgoing edge containment, splits).

**Fix**: Replace naive implementation with `CoverageComputer`. Return:

```json
{
  "structural_coverage": 0.75,
  "compositional_coverage": 0.5,
  "node_coverage": {
    "covered": ["5", "3"],
    "uncovered": ["7"],
    "total_candidate": 8,
    "by_annotation": {"A1": ["5", "3"], "A2": ["-1"]}
  },
  "edge_coverage": {
    "covered": [["5", "0"]],
    "uncovered": [["7", "0"]],
    "total": 5
  },
  "annotations_count": 2,
  "leaf_count": 2,
  "composition_count": 0
}
```

Counts all candidate nodes (excludes only output nodes per Def 10), includes edge coverage, compositional coverage (Def 11), per-annotation breakdown, leaf vs composition counts.

### 3. Structured Evidence Records

Currently `evidence` is unstructured JSONB. Define a lightweight schema enforced at API/MCP layer:

```json
{
  "records": [
    {
      "type": "analytical_formula",
      "payload": {"latex": "\\sigma(0.5x_1)", "tractable": true},
      "timestamp": "2026-04-20T12:00:00Z",
      "narrative": "Closed-form shows weighted sum with sigmoid"
    }
  ]
}
```

Evidence types (extensible enum): `analytical_formula`, `shap_importance`, `ablation_study`, `input_output_sampling`, `sign_analysis`, `performance_metrics`, `visualization`.

Each record: `type` (required), `payload` (required, type-specific), `timestamp` (required, auto-set), `narrative` (optional).

Add `EvidenceRecord` Pydantic model in schemas.py. Existing `save_snapshot` format becomes the `visualization` evidence type. Backward compatible with old unstructured evidence.

### 4. MCP Tool Fixes

- **`get_formula` smarter defaults**: Auto-set `force=True` when annotation has ≤8 inputs and ≤5 layers.
- **`get_annotations` enriched**: Add `evidence_count`, `evidence_types`, `is_composition` per annotation.
- **`get_model_state` enriched**: Include evidence summary per annotation.
- **New `add_evidence` tool**: Write typed evidence record. Params: genome_id, annotation_id, evidence_type (enum), payload (JSON), narrative (optional). Auto-sets timestamp.

### Not in scope

- Sign analysis as first-class concept (needs more framework design)
- Tensor-annotation bijection (PropNEAT-specific)
- "Trivial leaf" category (convention, document in paper)
- Node-level renaming (already works via rename_node operation)

## Files affected

| File | Change |
|------|--------|
| `explaneat/core/operations.py` | Composition validation rules |
| `explaneat/api/schemas.py` | EvidenceRecord Pydantic model |
| `mcp_server/tools/coverage.py` | Use CoverageComputer, rich response |
| `mcp_server/tools/evidence.py` | get_formula smart defaults |
| `mcp_server/tools/operations.py` | get_annotations enriched |
| `mcp_server/tools/models.py` | get_model_state enriched |
| `mcp_server/tools/snapshots.py` | add_evidence tool, structured records |
