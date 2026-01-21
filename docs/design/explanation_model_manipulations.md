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

### Graph Assumptions

**DAG Requirement**: The model is assumed to be a Directed Acyclic Graph (DAG). Cycles are not currently supported. This is consistent with standard NEAT feed-forward networks.

**Phenotype Pruning**: Before any explanation operations, the model is the **phenotype** (not genotype), which has been pruned to remove:
- Dead ends in the forward direction (nodes that don't reach any output)
- Dead ends in the backward direction (nodes not reachable from any input)
- Disabled connections

This guarantees that every node in the model is on at least one valid input→output path. There are no disconnected or computationally dead nodes to consider.

## Annotation Collapse Semantics

A key feature of annotations is the ability to **collapse** an annotated subgraph into a single node for simplified visualization. This enables hierarchical explanation: users see the high-level collapsed view and can drill down into details.

### Node Classification

Within an annotation's coverage, nodes are classified as:

| Type | External Inputs | External Outputs | Internal I/O |
|------|-----------------|------------------|--------------|
| **Entry** | ✓ At least one | ✗ None allowed | Outputs internal |
| **Intermediate** | ✗ None | ✗ None | All I/O internal |
| **Exit** | ✗ None | ✓ At least one | Inputs internal, may have internal outputs too |

- **Entry nodes**: Where signals enter the subgraph from outside. Must not have any outputs to nodes outside coverage (no "side effects").
- **Intermediate nodes**: Fully contained within the subgraph. All inputs and outputs connect only to other nodes in coverage.
- **Exit nodes**: Where signals leave the subgraph. May also have outputs to other internal nodes (those paths are absorbed).

### The No Side Effects Rule

**Entry and intermediate nodes cannot have outputs outside the annotation coverage.**

If a node would be both entry (has external input) AND have external output, it must be `split_node` first to separate the paths. This ensures clean collapse boundaries.

```
INVALID (entry node 13 has external output):
  A(outside) → 13(entry) → B(inside)
                    ↓
               C(outside)   ← side effect!

VALID (after split):
  A(outside) → 13_a → B(inside)    ← 13_a is entry
  A(outside) → 13_b → C(outside)   ← 13_b is outside coverage
```

### Collapse Operation

When an annotation is collapsed for visualization:

1. **All internal structure is absorbed** - entry, intermediate, and exit nodes become a single "annotation node"
2. **Inputs preserved** - all connections from outside TO entry nodes become inputs to the annotation node
3. **Outputs preserved** - all connections FROM exit nodes TO outside become outputs from the annotation node

```
Before collapse:
  A ──→ [B(entry) → C → D(exit)] ──→ E
                       ↓
                  F → G(exit) ──→ H

After collapse:
  A ──→ [annotation] ──→ E
                    ↘──→ H

Annotation node: 1 input (from A), 2 outputs (to E, H)
```

### Multiple Exits Are Valid

An annotation can have multiple exit nodes. Each exit's external outputs become outputs of the collapsed annotation node:

```
Coverage: [B, C, D, F, G]
Entry: B (input from A)
Exits: D (output to E), G (output to H)

Collapsed: annotation node with input from A, outputs to E and H
```

### Internal Paths from Exit Nodes

Exit nodes may have internal outputs in addition to external ones. These internal paths are simply absorbed:

```
A → B(entry) → C → D(exit) → E(outside)
                   ↓
                   F → G(exit) → H(outside)

D is an exit (→E) but also outputs to F internally.
This is valid - the D→F→G path is absorbed, G is another exit.
```

### Nested Annotations

Collapsed annotation nodes behave like regular nodes, enabling hierarchical composition:

```
Level 0: Raw model nodes
Level 1: [Annotation A] covers nodes [1,2,3] → collapses to node "A"
Level 2: [Annotation X] covers [A, 4, 5] → collapses to node "X"
```

Users can drill down: X → shows A,4,5 → expand A → shows 1,2,3

### Validation for Collapse

An annotation is **valid for collapse** if:
1. ✓ All entry nodes have no external outputs (no side effects)
2. ✓ All intermediate nodes have no external inputs or outputs
3. ✓ The subgraph is connected (guaranteed by phenotype pruning + annotation validation)
4. ✓ No overlap with other annotations

If validation fails due to side effects, the user must first apply `split_node` operations to separate the conflicting paths.

## Pre-Annotation Split Detection

Before creating an annotation, we must identify nodes that violate the entry/intermediate/exit constraints and require splitting.

### The Violation Condition

A node **requires splitting** if and only if it has:
- **At least one external input** (from outside proposed coverage), AND
- **At least one external output** (to outside proposed coverage)

Such a node cannot be cleanly classified:
- It has external input → would be an "entry"
- But entries cannot have external output → violation

Intermediate nodes (no external I/O) and exit nodes (external output only, no external input) are always valid.

### Detection Algorithm

```python
def detect_required_splits(proposed_coverage: Set[str], model: NetworkStructure) -> List[str]:
    """
    Identify nodes that must be split before annotation can be created.

    Args:
        proposed_coverage: Set of node IDs the user wants to annotate
        model: Current model state

    Returns:
        List of node IDs that require split_node before annotation
    """
    must_split = []

    for node_id in proposed_coverage:
        inputs = model.get_inputs_to_node(node_id)
        outputs = model.get_outputs_from_node(node_id)

        has_external_input = any(conn.from_node not in proposed_coverage for conn in inputs)
        has_external_output = any(conn.to_node not in proposed_coverage for conn in outputs)

        if has_external_input and has_external_output:
            must_split.append(node_id)

    return must_split
```

### Coverage Adjustment After Split

After splitting a violating node, the coverage must be adjusted:

1. **Remove** the original node from coverage
2. **Include** only the split nodes whose outputs are ALL internal to coverage
3. **Exclude** split nodes that have any external output

```python
def adjust_coverage_after_split(
    original_node: str,
    split_nodes: List[str],  # e.g., ["13_a", "13_b", "13_c"]
    proposed_coverage: Set[str],
    model: NetworkStructure
) -> Set[str]:
    """
    Adjust coverage after splitting a node.

    Only split nodes with purely internal outputs are included.
    Split nodes with external outputs are excluded (they become
    part of the "outside" that feeds into entries).
    """
    new_coverage = proposed_coverage - {original_node}

    for split_node in split_nodes:
        outputs = model.get_outputs_from_node(split_node)
        # Check if ALL outputs go to nodes in the (adjusted) coverage
        # Note: other split nodes from same original are considered "internal" for this check
        internal_targets = proposed_coverage | set(split_nodes) - {original_node}

        all_outputs_internal = all(
            conn.to_node in internal_targets
            for conn in outputs
        )

        if all_outputs_internal:
            new_coverage.add(split_node)
        # else: split_node stays outside coverage (becomes external source)

    return new_coverage
```

### Example: Split and Adjust

```
Proposed coverage: {13, 14, 15}
Node 13: inputs from [A(outside), B(outside)], outputs to [14(inside), C(outside)]

Step 1: Detect violation
  - 13 has external input (A, B) AND external output (C)
  - 13 must be split

Step 2: Split node 13 (outputs sorted: 14 < C, assuming C is e.g., node 20)
  - 13_a: outputs to [14]  (14 is inside)
  - 13_b: outputs to [C]   (C is outside)

Step 3: Adjust coverage
  - 13_a: all outputs internal → INCLUDE
  - 13_b: has external output → EXCLUDE
  - New coverage: {13_a, 14, 15}

Result:
  - 13_a is an entry node (external inputs from A, B; internal output to 14)
  - 13_b is outside coverage (external path A,B → 13_b → C)
  - No violations remain
```

### Iterative Resolution

In complex cases, splitting one node may reveal new violations (if the split changes what's "internal" vs "external"). The resolution process is iterative:

```python
def resolve_all_splits(proposed_coverage: Set[str], model: NetworkStructure) -> Tuple[Set[str], List[Operation]]:
    """
    Iteratively split nodes until no violations remain.

    Returns:
        - Final adjusted coverage
        - List of split_node operations to apply
    """
    coverage = set(proposed_coverage)
    operations = []

    while True:
        violations = detect_required_splits(coverage, model)
        if not violations:
            break

        # Split the first violation (could also split all, but sequential is clearer)
        node_to_split = violations[0]
        split_op = Operation(type='split_node', params={'node_id': node_to_split})
        operations.append(split_op)

        # Apply split to model (conceptually)
        model = apply_operation(model, split_op)
        split_nodes = get_split_results(node_to_split, model)  # e.g., ["13_a", "13_b"]

        # Adjust coverage
        coverage = adjust_coverage_after_split(node_to_split, split_nodes, coverage, model)

    return coverage, operations
```

### User Workflow

When a user attempts to create an annotation:

1. **Validate** proposed coverage against current model
2. **Detect** any nodes requiring splits
3. If violations found:
   - **Option A (automatic)**: System suggests/applies required splits, adjusts coverage
   - **Option B (manual)**: System reports violations, user manually applies `split_node` operations
4. Once clean, **create** the `annotate` operation

The automatic option provides a smoother UX, while the manual option gives users full control over the event stream.

## Edge Cases and Special Nodes

### Input Nodes (Negative IDs)

Input nodes represent external inputs to the network and have special constraints:

- **Cannot be split**: Input nodes have no internal computation to split
- **Can be entry nodes**: They naturally have "external input" (from the environment)
- **Cannot be intermediate or exit**: They have no inputs from other nodes

When annotating a subgraph that starts at an input node, the input node is always an entry node.

### Output Nodes

Output nodes represent the network's outputs and have special constraints:

- **Cannot be split**: The output must remain a single node for the network to function
- **Cannot typically be in coverage**: Output nodes usually need to remain as the "outside" that exits connect to
- **Problem case**: Direct input→output paths cannot have the output as an exit

**Solution**: Use `add_identity_node` to intercept connections going to the output node:

```
Before: [-1→0], [-2→0]  (inputs directly to output)

Problem: Want to annotate the -1 path, but output 0 can't be split or included.

Solution: add_identity_node(target_node=0, connections=[[-1, 0]], new_node_id=100)

After: [-1→100], [100→0], [-2→0]

Now annotate: coverage={-1, 100}, entry=[-1], exit=[100]
```

### Nodes with Single Output

A node with only one output connection cannot be split (nothing to split into). This is fine - such nodes are naturally either:
- **Entry**: If they have external input (the single output goes inside)
- **Intermediate**: If all I/O is internal
- **Exit**: If they have internal input and external output

No splitting is ever required for single-output nodes.

### Nodes That Are Both Entry AND Exit

This is the primary violation case. A node with both external input AND external output must be split:

```
External → [Node] → External   ← VIOLATION: entry with side effect
               ↓
           Internal

Solution: split_node to separate the external output path
```

### Already-Split Nodes

Split nodes (e.g., `13_a`, `13_b`) follow the same rules as regular nodes:
- They can be further split if they have multiple outputs
- They can be consolidated back with siblings
- They can be annotated following the same entry/exit rules

The split naming (`13_a`) doesn't change their behavior, only their identity.

## Operation Types

There are six operation types:

| # | Operation | Purpose |
|---|-----------|---------|
| 1 | `split_node` | Separate multi-output node into one node per output |
| 2 | `consolidate_node` | Recombine previously split nodes |
| 3 | `remove_node` | Remove pass-through node, combining connections |
| 4 | `add_node` | Insert node into a single connection |
| 5 | `add_identity_node` | Intercept multiple connections to same target |
| 6 | `annotate` | Create hypothesis about subgraph function |

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

### 5. add_identity_node

**Purpose**: Intercept a subset of connections sharing a common target node, funneling them through an identity node. This creates a clean exit point for annotating paths that converge on a node that cannot itself be an exit (e.g., an output node or a node with other external connections).

**Parameters**: `{ target_node: string, connections: [string, string][], new_node_id: string }`

**Operation**:
- Input: A target node ID, a list of connections that all end at that target, and a new node ID
- Output: The specified connections are redirected through a new identity node

**Weight Assignment**:
- Redirected connections [from → new_node]: preserve original weights
- New connection [new_node → target]: weight = 1.0, bias = 0.0, activation = identity

**Aggregation**: The identity node uses the same aggregation function as the target node (typically `sum`), so the mathematical result is preserved.

**Example**:
```
Before:
  Connections: [a→n] weight=0.5, [b→n] weight=0.8, [c→n] weight=1.2

After add_identity_node(target_node=n, connections=[[a,n], [b,n]], new_node_id=i):
  Node i: bias=0.0, activation=identity, aggregation=sum
  Connections:
    [a→i] weight=0.5  (redirected, weight preserved)
    [b→i] weight=0.8  (redirected, weight preserved)
    [i→n] weight=1.0  (new)
    [c→n] weight=1.2  (unchanged)
```

**Use Case - Annotating Paths to Output Nodes**:

When you want to annotate a subgraph that leads directly to an output node, the output node cannot be split (it must remain the single output). Use `add_identity_node` to create a proper exit:

```
Before:
  Input -1 → node 5 → Output 0
  Input -2 → node 5 → Output 0

Problem: Want to annotate path from -1 through 5 to output.
         But Output 0 can't be in coverage (it's the global output)
         and node 5 can't be an exit if -2→5 is external input.

Solution:
  Op 1: add_identity_node(target_node=5, connections=[[-1, 5]], new_node_id=100)

  Result:
    [-1→100] → [100→5] → [5→0]
    [-2→5] → [5→0]

  Op 2: annotate(coverage={-1, 100}, entry=[-1], exit=[100])

  Now -1 is entry, 100 is exit (clean boundary).
  The annotation captures "the path from input -1 through identity to node 5"
```

**Constraints**:
- Target node must exist
- All connections in the list must exist and end at target_node
- At least one connection must be specified
- Cannot specify ALL connections to target (must leave at least one direct, or use a different approach)
- New node ID must not already exist
- None of the connections can be within annotation coverage

**Inverse Operation**: The identity node can later be removed with `remove_node` if it has exactly one input and one output remaining (after potential annotation changes).

### 6. annotate

**Purpose**: Create a hypothesis about a subgraph's function. Establishes an immutable boundary that can be collapsed for hierarchical visualization.

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
- Validates entry/exit/intermediate node classification (see Collapse Semantics)
- Records the annotation
- Marks all nodes and connections in `subgraph_nodes` and `subgraph_connections` as immutable

**Constraints**:
- Subgraph must be connected (validated)
- All referenced nodes must exist in current model state
- Subgraph nodes/connections cannot already be covered by another annotation (no overlap)
- Entry nodes must have no external outputs (no side effects)
- Intermediate nodes must have no external inputs or outputs
- If constraints are violated, user must `split_node` first to separate paths

**Effect on other operations**: Once applied, no structural operations (`split_node`, `consolidate_node`, `remove_node`, `add_node`) can modify nodes within this annotation's coverage.

## Data Model

### Operation Record

```python
@dataclass
class Operation:
    id: UUID
    explanation_id: UUID
    sequence_number: int      # Order in the event stream (0-indexed)
    operation_type: str       # 'split_node', 'consolidate_node', 'remove_node', 'add_node', 'add_identity_node', 'annotate'
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
4. Entry nodes have no external outputs (no side effects)
5. Intermediate nodes have no external I/O
6. Entry/exit classification is consistent with declared lists

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
- `add_identity_node`: Weight preservation and fixed new connection weight (1.0)
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

### Example 4: Identity Node for Output Path Annotation

```
Original model:
  Input -1 ──→ Node 5 ──→ Output 0
  Input -2 ──→ Node 5 ──┘

  Connections: [-1→5], [-2→5], [5→0]

Goal: Annotate the path from input -1 through node 5.

Problem:
  - Output 0 cannot be in coverage (it's the global output)
  - Node 5 has external input from -2, so it can't be a clean exit
  - Node 5 would need to be split, but it has only one output

Solution using add_identity_node:

Op 0: add_identity_node(target_node=5, connections=[[-1, 5]], new_node_id=100)

  Result:
    Input -1 ──→ Node 100 ──→ Node 5 ──→ Output 0
    Input -2 ────────────────→ Node 5 ──┘

    Connections: [-1→100], [100→5], [-2→5], [5→0]

Op 1: annotate({
  name: "Input -1 processing",
  entry_nodes: [-1],
  exit_nodes: [100],
  subgraph_nodes: [-1, 100],
  subgraph_connections: [[-1, 100]]
})

  Result:
    - Node -1 is entry (external input from environment)
    - Node 100 is exit (output goes to node 5, which is outside coverage)
    - Clean annotation boundary established

Collapsed view:
  [Input -1 processing] ──→ Node 5 ──→ Output 0
  Input -2 ─────────────────→ Node 5 ──┘
```

### Example 5: Combining Split and Identity Node

```
Original model:
  A ──→ Node 10 ──→ B
  C ──→ Node 10 ──→ D
  E ──→ Node 10 ──┘

  Node 10 has inputs from [A, C, E] and outputs to [B, D]

Goal: Annotate paths A→10→B and C→10, leaving E→10→D outside.

Step 1: Split node 10 to separate outputs
  Op 0: split_node(10)
    → 10_a (→B), 10_b (→D)   [B < D alphabetically/numerically]

    A ──→ 10_a ──→ B
    C ──→ 10_a
    E ──→ 10_a
    A ──→ 10_b ──→ D
    C ──→ 10_b
    E ──→ 10_b

Step 2: Use identity node to separate E's path to 10_a
  Op 1: add_identity_node(target_node=10_a, connections=[[A, 10_a], [C, 10_a]], new_node_id=100)

    A ──→ 100 ──→ 10_a ──→ B
    C ──→ 100 ──┘
    E ──→ 10_a
    (10_b paths unchanged)

Step 3: Annotate
  Op 2: annotate({
    entry_nodes: [A, C],
    exit_nodes: [10_a],
    subgraph_nodes: [A, C, 100, 10_a],
    ...
  })

  Wait - 10_a still has external input from E! Need another identity node.

Step 2 (revised): Also isolate E's input
  Op 1: add_identity_node(target_node=10_a, connections=[[A, 10_a], [C, 10_a]], new_node_id=100)

    A ──→ 100 ──→ 10_a ──→ B
    C ──→ 100 ──┘
    E ──────────→ 10_a

  Now annotate {A, C, 100} with 100 as exit:

  Op 2: annotate({
    entry_nodes: [A, C],
    exit_nodes: [100],
    subgraph_nodes: [A, C, 100],
    subgraph_connections: [[A, 100], [C, 100]]
  })

This cleanly captures "the A and C input processing" as a unit.
```

## Migration from Current System

The current `NodeSplit` model will be migrated to this event stream architecture:
- Existing splits become `split_node` operations
- Existing annotations become `annotate` operations
- Sequence numbers assigned based on created_at timestamps
- `split_mappings` JSON replaced by operation parameters
