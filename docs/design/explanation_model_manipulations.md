# Explanation Model: Event Stream Architecture

## Overview

This document specifies the architecture for building explanations where the **explained model** can differ from the **original model**. Explanations are built by applying a deterministic, ordered sequence of **operations** to the original model. Operations include both structural manipulations and annotations.

```
Original Model + Operations (event stream) = Final Model with Annotations
```

## Core Concepts

### Model Relationship

- **Original Model**: The NEAT genome as evolved/trained - immutable reference
- **Operations**: An ordered event stream of atomic, deterministic operations
- **Final Model**: The result of applying all operations; annotations define immutable regions within it

### Event Stream Architecture

All changes to the explanation are captured as an ordered event stream. Each operation:
- Is atomic (single logical change)
- Is deterministic (same input always produces same output)
- Validates against the current model state (after prior operations)
- Can be undone (popped from stream) or redone (replayed)

```
┌──────────────┐     ┌───────┐     ┌───────┐     ┌───────┐     ┌─────────────────────┐
│ Original     │ ──▶ │ Op 1  │ ──▶ │ Op 2  │ ──▶ │ Op N  │ ──▶ │ Final Model         │
│ Model        │     │ split │     │ annot │     │ ...   │     │ (with annotations)  │
└──────────────┘     └───────┘     └───────┘     └───────┘     └─────────────────────┘
```

### Immutability and Coverage

When an `annotate` operation is applied, the nodes and connections within its coverage become **immutable**:
- No structural operations can modify nodes within annotation coverage
- Other operations outside the coverage remain valid
- This allows interleaving of structural operations and annotations

## Operation Types

### 1. split_node

**Purpose**: Separate a multi-output node into distinct nodes, one per output connection. Enables independent annotation of different functional paths through a node.

**Parameters**: `{ node_id: string }`

**Operation**:
- Input: A node ID with multiple output connections
- Output: Multiple new nodes, each with all original inputs but a single output

**Naming Convention**: Split nodes are named `{original_id}_{letter}` where letters are assigned in alphabetical order based on ascending target node ID of the output connections.

**Example**:
```
Before:
  Node 13: inputs [-2, -3, 4], outputs to [7, 9]
  Connections: [-2→13], [-3→13], [4→13], [13→7], [13→9]

After split_node(13):
  Node 13_a: inputs [-2, -3, 4], output to [7]   (7 < 9, so 'a')
  Node 13_b: inputs [-2, -3, 4], output to [9]   (9 > 7, so 'b')

  Connections:
    [-2→13_a], [-3→13_a], [4→13_a], [13_a→7]
    [-2→13_b], [-3→13_b], [4→13_b], [13_b→9]
```

**Properties preserved**:
- Node bias, activation, response, aggregation: copied to all split nodes
- Input connection weights: copied to all split nodes
- Output connection weights: preserved on respective split nodes

**Constraints**:
- Node must have ≥2 output connections
- Node cannot be an input node
- Node cannot be within annotation coverage

### 2. consolidate_node

**Purpose**: Recombine previously split nodes back together. This is the inverse of `split_node`.

**Parameters**: `{ node_ids: string[] }` (e.g., `["13_a", "13_c"]`)

**Operation**:
- Input: Two or more node IDs that were created from the same original split
- Output: A single node combining their outputs

**Naming Convention**: Consolidated nodes combine their suffix letters alphabetically. `13_a` + `13_c` → `13_ac`. This preserves history even on full reconsolidation (`13_a` + `13_b` + `13_c` → `13_abc`, not back to `13`).

**Example**:
```
Before (after a prior split_node(13)):
  Node 13_a: outputs to [7]
  Node 13_b: outputs to [9]
  Node 13_c: outputs to [12]

After consolidate_node([13_a, 13_c]):
  Node 13_ac: outputs to [7, 12]  (combined outputs, sorted)
  Node 13_b: unchanged

  Input connections to 13_ac are deduplicated (same as original 13 had)
```

**Constraints**:
- All node IDs must be from the same original split (same base ID)
- Cannot consolidate nodes from different bases (e.g., can't combine `13_a` with `14_b`)
- Cannot consolidate nodes that weren't created by `split_node`
- None of the nodes can be within annotation coverage

**Re-splitting consolidated nodes**: A consolidated node like `13_ac` can be re-split, but only back into its constituent parts (`13_a` and `13_c`), not into new subdivisions (`13_ac_a`, `13_ac_b`). The split restores the state before the consolidation.

### 3. remove_node

**Purpose**: Simplify the model by removing pass-through nodes, combining their connections.

**Parameters**: `{ node_id: string }`

**Operation**:
- Input: A node ID with exactly one input connection and one output connection
- Output: The node is removed, replaced by a single connection

**Weight Calculation**: The new connection weight is the product of the two original weights.

**Example**:
```
Before:
  Node 15: input from [-2], output to [4]
  Connections: [-2→15] weight=0.5, [15→4] weight=2.0

After remove_node(15):
  Node 15: removed
  Connection: [-2→4] weight=1.0  (0.5 × 2.0)
```

**Constraints**:
- Node must have exactly 1 input connection
- Node must have exactly 1 output connection
- Node cannot be an input or output node
- Node cannot be within annotation coverage

### 4. add_node

**Purpose**: Insert a node into an existing connection. Useful for adding structure for annotation purposes.

**Parameters**: `{ connection: [string, string], new_node_id: string, bias?: number, activation?: string }`

**Operation**:
- Input: A connection [from, to] and a new node ID
- Output: The connection is split with a new node in between

**Weight Assignment** (matches NEAT mutation behavior):
- New connection [from → new_node]: weight = 1.0
- New connection [new_node → to]: weight = original_weight

**Example**:
```
Before:
  Connection: [-2→4] weight=1.5

After add_node([-2, 4], new_node_id=16):
  Node 16: bias=0.0, activation=identity (or as specified)
  Connections: [-2→16] weight=1.0, [16→4] weight=1.5
```

**Constraints**:
- Connection must exist
- New node ID must not already exist
- Connection cannot be within annotation coverage

### 5. annotate

**Purpose**: Create a hypothesis about a subgraph's function. Establishes an immutable boundary.

**Parameters**:
```
{
  name: string,
  hypothesis: string,
  entry_nodes: string[],
  exit_nodes: string[],
  subgraph_nodes: string[],
  subgraph_connections: [string, string][],
  evidence?: object
}
```

**Operation**:
- Validates the subgraph is connected
- Records the annotation
- Marks all nodes and connections in `subgraph_nodes` and `subgraph_connections` as immutable

**Constraints**:
- Subgraph must be connected (validated)
- All referenced nodes must exist in current model state
- Subgraph nodes/connections cannot already be covered by another annotation (no overlap)

**Effect on other operations**: Once applied, no structural operations (`split_node`, `consolidate_node`, `remove_node`, `add_node`) can modify nodes within this annotation's coverage.

## Data Model

### Operation Record

```python
@dataclass
class Operation:
    id: UUID
    explanation_id: UUID
    sequence_number: int      # Order in the event stream (0-indexed)
    operation_type: str       # 'split_node', 'consolidate_node', 'remove_node', 'add_node', 'annotate'
    parameters: dict          # Type-specific parameters (see above)
    created_at: datetime

    # For annotate operations, this links to the Annotation record
    annotation_id: Optional[UUID]
```

### Explanation

```python
class Explanation:
    id: UUID
    genome_id: UUID                    # Reference to original model
    operations: List[Operation]        # Ordered by sequence_number

    def get_final_model(self) -> NetworkStructure:
        """Apply all operations to original model, return final state."""
        model = self.get_original_model()
        for op in sorted(self.operations, key=lambda o: o.sequence_number):
            model = apply_operation(model, op)
        return model

    def get_covered_nodes(self) -> Set[str]:
        """Return all node IDs covered by annotations (immutable)."""
        covered = set()
        for op in self.operations:
            if op.operation_type == 'annotate':
                covered.update(op.parameters['subgraph_nodes'])
        return covered

    def can_modify_node(self, node_id: str) -> bool:
        """Check if a node can be modified by structural operations."""
        return node_id not in self.get_covered_nodes()
```

### Event Stream Operations

```python
class ExplanationEventStream:
    def append(self, operation: Operation) -> Result:
        """Validate and append operation to stream."""
        # 1. Validate preconditions against current model state
        # 2. Check no immutable nodes are affected
        # 3. Apply operation
        # 4. Increment sequence_number

    def undo(self) -> Operation:
        """Remove and return the last operation."""
        # Pop last operation, recompute model state

    def redo(self, operation: Operation) -> Result:
        """Re-append a previously undone operation."""
        # Validate and append (same as append)

    def replay(self) -> NetworkStructure:
        """Replay all operations from original model."""
        # Used for validation and state reconstruction
```

## Validation Rules

### Before any structural operation:
1. Target node(s) must exist in current model state
2. Target node(s) must not be in annotation coverage
3. Operation-specific preconditions (e.g., node has required connections)

### Before annotate:
1. All referenced nodes must exist
2. Subgraph must be connected
3. No overlap with existing annotation coverage

### State consistency:
- The event stream can always be replayed from the original model
- Undo removes the last operation; the model state is recomputed
- No "dangling" references (all node IDs valid at time of operation)

## Determinism Guarantees

All operations are deterministic:
- `split_node`: Letter assignment based on sorted target node IDs
- `consolidate_node`: Combined suffix is alphabetically sorted
- `remove_node`: Weight product is deterministic
- `add_node`: Fixed weight assignment (1.0, original)
- `annotate`: No randomness

Same operation sequence always produces identical final model.

## Examples

### Example 1: Split, Annotate, Consolidate

```
Original: Node 13 with outputs to [7, 9, 12]

Op 0: split_node(13)
  → 13_a (→7), 13_b (→9), 13_c (→12)

Op 1: annotate({nodes: [13_a, 7], ...})
  → 13_a is now immutable

Op 2: consolidate_node([13_b, 13_c])
  → 13_bc (→9, →12)
  → Valid: 13_b and 13_c are not in annotation coverage

Op 3: split_node(13_a)
  → ERROR: 13_a is within annotation coverage
```

### Example 2: Full History Preservation

```
Op 0: split_node(13)      → 13_a, 13_b, 13_c
Op 1: consolidate_node([13_a, 13_b, 13_c])  → 13_abc (not "13")

Final model has node "13_abc" - the history is preserved in the name.
Original node "13" is not restored; that would require examining the
original model, not just the event stream.
```

### Example 3: Re-split After Consolidate

```
Op 0: split_node(13)              → 13_a, 13_b, 13_c
Op 1: consolidate_node([13_a, 13_c])  → 13_ac, 13_b
Op 2: split_node(13_ac)           → 13_a, 13_c (restored)

Re-splitting 13_ac returns it to 13_a and 13_c, not 13_ac_a, 13_ac_b.
The split "remembers" the consolidation it's reversing.
```

## Migration from Current System

The current `NodeSplit` model will be migrated to this event stream architecture:
- Existing splits become `split_node` operations
- Existing annotations become `annotate` operations
- Sequence numbers assigned based on created_at timestamps
- `split_mappings` JSON replaced by operation parameters
