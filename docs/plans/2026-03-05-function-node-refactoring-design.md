# Design: Function Node Refactoring

**Date**: 2026-03-05
**Status**: Approved

## Problem

The current collapse implementation operates as graph surgery — removing internal nodes and rewiring edges — which can introduce cycles in the DAG. Additionally, collapse logic is duplicated between the Python server (`CollapseValidator`) and the React frontend (`useCollapsedView` hook), leading to inconsistencies.

More fundamentally, the current approach treats collapse as a rendering trick rather than as what it actually is: replacing a subgraph with a named function.

## Core Insight

Explaining a NEAT network is a two-phase process of **functional decomposition and recomposition**:

1. **Decomposition**: The network already is a composition of primitive functions. Identity operations (splits, identity nodes) restructure the graph to make this composition explicit without changing the computed function.

2. **Recomposition**: Annotations replace subgraphs with named multi-input, multi-output function nodes. This is term rewriting on a DAG — provably cycle-free.

An annotation carries three layers of meaning:
- **Structural**: this subgraph has entries {a,b,c} and exits {x,y}
- **Functional**: it computes F(a,b,c) = sigmoid(w1*a + w2*b) — extracted by the system
- **Interpretive**: "this represents the additive effect of drugs A and B on blood pressure" — provided by the researcher

See `docs/theoretical_framework.md` for the full mathematical treatment.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Canonical state | Always fully-expanded | Single source of truth; collapsed views are derived |
| Collapse location | Server-side | Eliminates duplicated logic; simpler frontend |
| Implementation pattern | Pure transform function | Testable, stateless, no side effects |
| Multi-output handling | Indexed outputs on connections | One function node, N outputs; connections carry output_index |
| Activation functions | Shared registry | Fix current hard-coded identity/sigmoid/relu limitation |

## Data Model Changes

### NodeType — add FUNCTION

```python
class NodeType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    HIDDEN = "hidden"
    FUNCTION = "function"
```

### FunctionNodeMetadata — new dataclass

```python
@dataclass
class FunctionNodeMetadata:
    annotation_name: str                        # human-readable label
    annotation_id: str                          # links back to AnnotationData
    hypothesis: str                             # domain-level interpretation
    n_inputs: int                               # number of entry nodes
    n_outputs: int                              # number of exit nodes
    input_names: List[str]                      # original entry node IDs
    output_names: List[str]                     # original exit node IDs
    formula_latex: Optional[str]                # LaTeX representation
    subgraph_nodes: List[str]                   # preserved for expand
    subgraph_connections: List[Tuple[str, str]] # preserved for expand
```

### NetworkNode — add function_metadata field

```python
@dataclass
class NetworkNode:
    id: str
    type: NodeType
    bias: Optional[float] = None
    activation: Optional[str] = None
    response: Optional[float] = None
    aggregation: Optional[str] = None
    function_metadata: Optional[FunctionNodeMetadata] = None  # NEW
```

### NetworkConnection — add output_index field

```python
@dataclass
class NetworkConnection:
    from_node: str
    to_node: str
    weight: float
    enabled: bool
    innovation: Optional[int] = None
    output_index: Optional[int] = None  # NEW: which output of a FUNCTION node
```

For non-function nodes, `output_index` is `None`. For connections from a function node, it specifies which of the N outputs this connection carries.

## New Modules

### `explaneat/core/activations.py` — Activation Registry

Shared registry mapping activation name strings to numpy and sympy implementations:

```python
ACTIVATIONS = {
    "sigmoid": ActivationPair(numpy_fn, sympy_fn),
    "relu": ActivationPair(numpy_fn, sympy_fn),
    "identity": ActivationPair(numpy_fn, sympy_fn),
    "tanh": ActivationPair(numpy_fn, sympy_fn),
    "abs": ActivationPair(numpy_fn, sympy_fn),
    # ... full set from neat-python
}
```

Used by both `StructureNetwork` and `AnnotationFunction`, replacing their hard-coded if/else chains.

### `explaneat/core/collapse_transform.py` — Pure Collapse Transform

```python
def collapse_structure(
    structure: NetworkStructure,
    annotations: List[AnnotationData],
    collapsed_ids: Set[str],
) -> NetworkStructure:
```

Algorithm:
1. For each annotation in `collapsed_ids` (children before parents):
   - Validate preconditions via existing `CollapseValidator.validate_collapsible`
   - Compute formula via `AnnotationFunction.from_structure`
   - Identify internal nodes (subgraph_nodes - entry_nodes) to remove
   - Create a FUNCTION node with FunctionNodeMetadata (including hypothesis)
   - Rewire connections:
     - entry -> internal becomes entry -> function_node (deduplicated)
     - exit -> external becomes function_node -> external with output_index
     - connections between non-internal nodes preserved
     - connections involving internal nodes removed
   - Remove internal nodes from node list
2. Return new NetworkStructure

Key property: pure function, no mutation. Call with different `collapsed_ids` for different views.

## Changes to Existing Modules

### `StructureNetwork` — handle FUNCTION nodes

- Use activation registry instead of hard-coded if/else
- When encountering a FUNCTION node during forward pass:
  - Gather weighted inputs (same as any node)
  - Call the annotation's computation (stored `AnnotationFunction`)
  - Node produces N outputs
  - Downstream connections use `output_index` to select which output they read
- Depth computation works naturally — function node depth determined by entry connections

### `AnnotationFunction` — use activation registry

- Replace hard-coded activation assignment (lines 128-134) with registry lookup from `NetworkNode.activation`
- Replace hard-coded sympy expressions in `to_sympy()` with registry's sympy functions
- Core logic (topological ordering, weight extraction, evaluation) unchanged

### API — model-state endpoint

`GET /api/genomes/{genome_id}/model-state?explanation_id=xxx&collapsed=ann1,ann2`

- If `collapsed` parameter provided, apply `collapse_structure` to produce derived view
- Serialize FUNCTION nodes with their metadata (formula, hypothesis, etc.)
- Serialize `output_index` on connections
- Include which annotations are currently collapsed in response

### Frontend — simplification

- **Remove** `useCollapsedView.ts` hook entirely
- Collapse/expand toggles trigger re-fetch with updated `collapsed` parameter
- `NetworkViewer` renders FUNCTION nodes with distinct visual treatment (different shape, show name + formula on hover)
- `OperationsPanel` collapse UI simplified — just toggle state and re-fetch

## What Stays the Same

- `ModelStateEngine` — still owns expanded state, operation replay, annotations
- `CollapseValidator.validate_collapsible` and `suggest_fixes` — still used for precondition checking
- Evidence routes — always compute from expanded structure
- `AnnotationData` structure — unchanged
- Operation event stream — unchanged (no new persistent operation type)

## What Gets Removed

- `CollapseValidator.collapse()` and `expand()` methods — superseded by `collapse_transform`
- `useCollapsedView.ts` React hook — replaced by server-computed views
- Client-side collapse logic in `NetworkViewer`/`OperationsPanel`

## Testing Strategy

- Unit tests for `collapse_structure` with various topologies (linear, diamond, fan-out)
- Unit tests verifying cycle freedom for all collapse configurations
- Unit tests for activation registry (numpy and sympy agree for all activations)
- Unit tests for `StructureNetwork` forward pass through function nodes
- Integration tests: collapse + forward pass produces same outputs as expanded forward pass
- Round-trip tests: collapse then use subgraph_nodes/connections to verify expand consistency
