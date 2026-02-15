# Annotation Hierarchy and Node Splitting

## Overview

This document describes the annotation hierarchy structure and node splitting mechanism, aligned with the Beyond Intuition paper specification.

## Annotation Hierarchy

### Structure

An annotation hierarchy is a **rooted tree** structure where:
- **Leaf annotations** are at the leaves (no children)
- **Composition annotations** are internal nodes (have children)
- Each annotation can have at most one parent
- The hierarchy forms a tree (no cycles)

### Construction-Based Approach

Annotations are built **bottom-up** using a construction-based approach:

1. **Leaf Annotations**: Created first to explain primitive subgraphs in isolation
2. **Composition Annotations**: Created by combining child annotations, explaining how they work together

A composition annotation:
- References its children via `child_annotation_ids` (annotation names) in the annotate operation
- The `ModelStateEngine` sets `parent_annotation_id` on children during replay
- Describes how the children combine (via hypothesis)
- Has its own subgraph (nodes and connections) that may include junction nodes/edges connecting the children
- Can itself be a child of another composition annotation

**Important:** The API computes `parent_annotation_id` by inverting `child_annotation_ids` via a name-based lookup. When multiple annotations share the same name, this back-pointer can be incorrect. The client-side collapsed view (`useCollapsedView.ts`) uses `children_ids` — the forward reference declared on the parent annotation — as the authoritative source for hierarchy traversal. See `docs/annotation_collapsing_model.md` Implementation section for details.

### Example Hierarchy

```
Root Annotation (covers entire model)
├── Composition Annotation A
│   ├── Leaf Annotation A1
│   └── Leaf Annotation A2
└── Composition Annotation B
    ├── Leaf Annotation B1
    └── Leaf Annotation B2
```

### Well-Formed Hierarchy

A well-formed annotation hierarchy must:
1. Have at least one root annotation (no parent)
2. All leaf annotations are valid
3. Full structural coverage: all model nodes covered by leaf annotations
4. Full compositional coverage: all required composition steps explained
5. Root annotation covers the global model

## Operations-Based Model

### ModelStateEngine

All model modifications (splits, identity nodes, annotations) are managed through an **operations event stream** stored as JSONB on the `Explanation` model. The `ModelStateEngine` (`explaneat/core/model_state.py`) replays operations on the base phenotype to produce the current model state.

**Operation types:**
- `split_node`: Split a node into multiple nodes (one per outgoing connection)
- `consolidate_node`: Merge previously split nodes back together
- `add_identity_node`: Intercept connections to a target node through an identity node
- `add_node`: Insert a node into an existing connection
- `remove_node`: Remove a pass-through node, combining its connections
- `annotate`: Mark a subgraph as explained with entry/exit nodes and hypothesis

**Key properties:**
- Operations are ordered (sequence numbers)
- Undo removes an operation and all subsequent operations, then replays
- The engine tracks covered nodes/connections (immutable after annotation)
- Validation enforces the three preconditions before creating annotations

### Operation Storage

Operations are stored in `Explanation.operations` as a JSONB array:

```json
[
  {"seq": 0, "type": "split_node", "params": {"node_id": "-17"}, "result": {"created_nodes": ["-17_a", "-17_b"], "removed_nodes": ["-17"]}},
  {"seq": 1, "type": "add_identity_node", "params": {"target_node": "0", "connections": [["-3_a", "0"]], "new_node_id": "identity_1"}, "result": {"created_nodes": ["identity_1"]}},
  {"seq": 2, "type": "annotate", "params": {"name": "A101", "hypothesis": "...", "entry_nodes": [...], "exit_nodes": [...], "subgraph_nodes": [...], "subgraph_connections": [...]}, "result": {"annotation_index": 0}}
]
```

## Node Splitting

### Purpose

Node splitting handles **dual-function nodes** - nodes that send outputs both within an annotation's subgraph and outside it. Without splitting, such nodes cannot be fully covered by an annotation because they have outgoing connections outside the annotation.

### Full-Split Model

When a node needs to be split via `apply_split_node` (`explaneat/core/operations.py`), it is **fully split** - a dedicated split node is created for **each** outgoing connection:

- **Original Node**: Removed from the model
- **Split Nodes**: One per outgoing connection, named `{base_id}_{letter}` (e.g., `5_a`, `5_b`)
- **Outgoing Connections**: Each split node carries **exactly one** outgoing connection
- **Incoming Connections**: All split nodes receive copies of all original incoming connections
- **Properties**: Split nodes inherit the original's type, bias, activation, response, aggregation

### Naming Convention

Split nodes use alphabetic suffixes: `{base}_{letter}`

- First split: `5_a`, `5_b`, `5_c`, ...
- Re-split of consolidated `5_ac`: restores `5_a`, `5_c`
- Existing suffixes are avoided to prevent collisions

The `_get_base_node_id()` function extracts the base ID: `"5_a"` → `"5"`, `"-20_b"` → `"-20"`

### Split Input Nodes

When an input node is split (e.g., `-20` → `-20_a`, `-20_b`):
- Split nodes inherit `NodeType.INPUT`
- But `input_node_ids` in the `NetworkStructure` still lists the original `-20`
- `StructureNetwork` detects these orphaned input nodes and maps them to the same tensor column as their base node
- Both `-20_a` and `-20_b` read from the same column in the input data (the column for feature `-20`)

### Consolidation

Previously split nodes can be consolidated back via `apply_consolidate_node`:
- Merges split nodes from the same base: `5_a` + `5_c` → `5_ac`
- The consolidated node gets all outgoing connections from both splits
- Incoming connections are deduplicated
- Can be re-split later (restoring the original split letters)

### Constraints

1. **Completeness**: The union of all split nodes' outgoing connections equals the original node's outgoing connections
2. **Non-overlap**: Each outgoing connection belongs to exactly one split node
3. **Uniqueness**: Split node IDs are unique (alphabetic suffix collision avoidance)
4. **Shared Inputs**: All split nodes share the original node's incoming connections
5. **One Connection Per Split**: Each split node has exactly one outgoing connection

## Identity Nodes

### Purpose

Identity nodes intercept a subset of connections to a target node, enabling partial coverage annotations. They have `bias=0` and `activation="identity"`, so they pass through the weighted sum of their inputs unchanged.

### Creation via `apply_add_identity_node`

When an exit node has external inputs (Precondition 3 violation):
1. Create identity node (e.g., `identity_5`)
2. Redirect specified connections from their original target to the identity node (preserving weights)
3. Add a new connection from identity node to original target with weight 1.0
4. The identity node becomes the annotation's exit; the original target is removed from the annotation

### Example

Before: `-3 --0.5--> 0`, `-4 --0.8--> 0`, `-5 --1.0--> 0` (node 0 is output)

Annotation wants to cover `-3` and `-4` only. Node 0 has external input from `-5`.

After `add_identity_node(target=0, connections=[(-3, 0), (-4, 0)], new_node_id="identity_1")`:
- `-3 --0.5--> identity_1`, `-4 --0.8--> identity_1`, `identity_1 --1.0--> 0`, `-5 --1.0--> 0`
- Annotation: entry={-3, -4}, exit={identity_1}, subgraph={-3, -4, identity_1}

## Three Preconditions for Annotations

The `validate_operation()` function in `operations.py` enforces these before creating an annotation:

1. **Entry-Only Ingress (P1)**: All edges entering the subgraph from outside must target entry nodes
2. **Exit-Only Egress (P2)**: All edges leaving the subgraph must originate from exit nodes
3. **Pure Exits (P3)**: Exit nodes receive inputs only from within the annotation

### UI Strategy Wizard

The `OperationsPanel.tsx` annotation strategy wizard analyzes selected nodes and computes fix operations:

1. **P3 violations** → Add identity nodes (partial exit coverage)
2. **P2 violations** → Split nodes (external outputs from entry/intermediate)
3. **P1 violations** → User must expand selection (blocking issue)

Execution order: identity nodes → splits → create annotation

## Evidence on Annotations

Annotations carry evidence (hypothesis + verifiable data). Evidence is stored in the annotation's `params.evidence` field within the operations array:

```json
{
  "type": "annotate",
  "params": {
    "name": "A505",
    "hypothesis": "This subgraph computes...",
    "evidence": {
      "snapshots": [
        {"viz_config": {...}, "svg_data": "...", "narrative": "Shows linear relationship", "timestamp": "..."}
      ]
    }
  }
}
```

Evidence snapshots are managed via the `/api/genomes/{id}/evidence/` endpoints.

## References

- Beyond_Intuition.pdf - Core specification document
- `docs/annotation_coverage_model.md` - Coverage definitions
- `docs/annotation_collapsing_model.md` - Collapse operation formalization
- `explaneat/core/operations.py` - Operation handlers and validation
- `explaneat/core/model_state.py` - ModelStateEngine for operation replay
