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
- References its children via `parent_annotation_id` relationships
- Describes how the children combine (via hypothesis)
- Has its own subgraph (nodes and connections) that may include junction nodes/edges connecting the children
- Can itself be a child of another composition annotation

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

### Database Representation

- Each annotation has an optional `parent_annotation_id` field
- Leaf annotations: `parent_annotation_id IS NULL` and no children
- Composition annotations: have children (inferred from `parent_annotation_id` references)

### Well-Formed Hierarchy

A well-formed annotation hierarchy must:
1. Have at least one root annotation (no parent)
2. All leaf annotations are valid
3. Full structural coverage: all model nodes covered by leaf annotations
4. Full compositional coverage: all required composition steps explained
5. Root annotation covers the global model

## Node Splitting

### Purpose

Node splitting handles **dual-function nodes** - nodes that send outputs both within an annotation's subgraph and outside it. Without splitting, such nodes cannot be fully covered by an annotation because they have outgoing connections outside the annotation.

### Multi-Split Model

A node can be split into **multiple** split nodes (not just binary):

- **Original Node**: The node being split
- **Split Nodes**: Multiple nodes, each with a unique `split_node_id`
- **Outgoing Connections**: Each split node carries a **subset** of the original node's outgoing connections
- **Incoming Connections**: All split nodes share the same incoming connections as the original node

### Constraints

1. **Completeness**: The union of all split nodes' outgoing connections must equal the original node's outgoing connections
2. **Non-overlap**: Each outgoing connection belongs to exactly one split node
3. **Uniqueness**: Each `split_node_id` must be unique per `original_node_id` within an explanation
4. **Shared Inputs**: All split nodes share the original node's incoming connections (computed from graph, not stored)

### Database Representation

Node splits are stored in a separate `node_splits` table with:
- `original_node_id`: The node being split
- `split_node_id`: Unique ID for this split node
- `outgoing_connections`: JSONB array of `[from_node, to_node]` tuples - subset of original node's outgoing connections
- `explanation_id`: Which explanation this split belongs to
- `annotation_id`: Optional - which annotation uses this split (for tracking)

### Coverage with Splits

When computing coverage for a split node:
- Use `split_node_id` with its specific `outgoing_connections` subset
- Use `original_node_id`'s incoming connections (shared by all splits)
- Apply the standard coverage definition: `covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A)`

### Example

Original node `5` has outgoing connections: `[(5, 6), (5, 7), (5, 8)]`

Split 1:
- `split_node_id = 501`
- `outgoing_connections = [(5, 6), (5, 7)]`

Split 2:
- `split_node_id = 502`
- `outgoing_connections = [(5, 8)]`

Both splits share node `5`'s incoming connections (e.g., `[(3, 5), (4, 5)]`).

## Explanation Data Model

### Purpose

An **explanation** groups annotations and node splits into a coherent explanation of a model. Multiple explanations can exist for the same model, allowing different explanatory approaches.

### Structure

An explanation contains:
- **Genome ID**: The model being explained
- **Annotations**: Set of annotations forming a hierarchy
- **Node Splits**: Set of node splits used by the annotations
- **Coverage Metrics**: Cached structural and compositional coverage values
- **Well-Formed Flag**: Whether the explanation meets well-formed criteria

### Usage

When computing coverage or visibility:
1. Select a specific explanation
2. Use only annotations and splits belonging to that explanation
3. Compute coverage metrics within that explanation's context

This allows multiple explanations to coexist for the same model without interference.

## Implementation Details

### Creating Composition Annotations

```python
from explaneat.analysis.annotation_manager import AnnotationManager

# Create composition annotation
composition = AnnotationManager.create_composition_annotation(
    genome_id=genome_id,
    child_annotation_ids=[child1_id, child2_id],
    hypothesis="How children combine...",
    entry_nodes=[...],
    exit_nodes=[...],
    nodes=[...],
    connections=[...],
    explanation_id=explanation_id
)
```

The system automatically sets `parent_annotation_id` on all children.

### Creating Node Splits

```python
from explaneat.analysis.node_splitting import NodeSplitManager

# Create a split
split = NodeSplitManager.create_split(
    genome_id=genome_id,
    original_node_id=5,
    split_node_id=501,
    outgoing_connections=[(5, 6), (5, 7)],
    explanation_id=explanation_id
)
```

### Validating Splits

```python
# Validate that splits for a node are complete
result = NodeSplitManager.validate_splits_complete(
    genome_id=genome_id,
    original_node_id=5,
    explanation_id=explanation_id
)
```

### Phenotype with Splits

To visualize the graph with splits applied:

```python
from explaneat.core.genome_network import get_phenotype_with_splits

phenotype = get_phenotype_with_splits(explanation_id)
```

This returns a `NetworkStructure` where:
- Original nodes are replaced by split nodes where splits exist
- Split nodes use their specific `outgoing_connections`
- All split nodes share the original node's incoming connections

## Direct Connections Annotation

Direct input-output connections can be automatically annotated using the CLI command `create-direct-connections`. This creates a special annotation that groups all inputs with only direct connections to outputs.

**When to use:**
- When you want to treat trivial direct connections as a single annotation
- When inputs connect directly to outputs without intermediate processing
- As a convenience feature to avoid manually creating annotations for simple cases

**How it works:**
- The CLI command scans the phenotype network
- Identifies inputs that have ONLY direct connections to outputs (no other outgoing connections)
- Creates a single annotation covering all such inputs and their direct connection edges
- The annotation can be assigned to an explanation (if one is selected)

**Note:** If an input has both direct connections and other connections, it is NOT included in the direct connections annotation and requires a proper annotation to explain its behavior.

## References

- Beyond_Intuition.pdf - Core specification document
- Compositional Explanation.md - Additional framework details
- docs/annotation_coverage_model.md - Coverage definitions

