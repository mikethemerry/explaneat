# Composed Annotation Formulas Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix composed annotation collapse connectivity and add multi-level formula display (collapsed referencing child names, expanded showing all primitives) with a toggle in the evidence panel.

**Architecture:** Four layers of changes: (1) fix `collapse_transform.py` to auto-derive correct entry/exit nodes for compositional annotations, (2) teach `AnnotationFunction` to handle `NodeType.FUNCTION` nodes in subgraphs and produce both collapsed and expanded sympy expressions, (3) update the formula API endpoint to return both forms, (4) add expand/collapse toggle to `FormulaDisplay` component.

**Tech Stack:** Python (sympy, numpy, dataclasses), FastAPI/Pydantic, React/TypeScript, KaTeX

---

## File Structure

### Files to Modify

| File | Responsibility | Changes |
|------|---------------|---------|
| `explaneat/core/collapse_transform.py` | Pure function: structure + annotations -> collapsed structure | Add `_compute_effective_entries_exits()` to auto-derive correct entry/exit for compositional annotations |
| `explaneat/analysis/annotation_function.py` | Extract callable/symbolic functions from annotation subgraphs | Handle `NodeType.FUNCTION` nodes in `_build_from_structure()` and add `expand` param to `to_sympy()`/`to_latex()` |
| `explaneat/api/routes/evidence.py` | Formula + viz data endpoints | Update formula endpoint to return both collapsed and expanded LaTeX |
| `explaneat/api/schemas.py` | Pydantic request/response models | Update `FormulaResponse` to include `latex_collapsed`, `latex_expanded`, `children` |
| `web/react-explorer/src/api/client.ts` | API client types + fetch functions | Update `FormulaResponse` type |
| `web/react-explorer/src/components/FormulaDisplay.tsx` | KaTeX formula rendering | Add expand/collapse toggle for composed annotations |

### Files to Create

| File | Responsibility |
|------|---------------|
| `tests/test_core/test_collapse_transform_composed.py` | Tests for composed annotation entry/exit derivation and connectivity |
| `tests/test_analysis/test_annotation_function_composed.py` | Tests for AnnotationFunction handling FUNCTION nodes and expand/collapse |

### Existing Test Files (for reference, not modified)

| File | Role |
|------|------|
| `tests/test_core/test_collapse_transform.py` | Existing collapse tests (line 778+: `TestCompositionalAnnotationCollapse`) |
| `tests/test_core/test_collapse_integration.py` | Forward-pass equivalence tests |
| `tests/test_analysis/test_annotation_function.py` | Existing AnnotationFunction tests |

---

## Chunk 1: Fix Composed Annotation Collapse Connectivity

### Task 1: Add tests for composed annotation entry/exit derivation

**Files:**
- Create: `tests/test_core/test_collapse_transform_composed.py`

The core bug: compositional annotations (empty `subgraph_nodes`, children only) use manually-declared entry/exit nodes which may be wrong. Entry/exit should be auto-derived from the effective subgraph's boundary connections.

- [ ] **Step 1: Write tests for `_compute_effective_entries_exits()`**

```python
"""Tests for composed annotation entry/exit auto-derivation."""

import pytest
from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.collapse_transform import (
    _compute_effective_entries_exits,
    _compute_effective_subgraph_nodes,
    collapse_structure,
)


def _make_node(id, type=NodeType.HIDDEN, **kwargs):
    return NetworkNode(id=id, type=type, **kwargs)


def _make_conn(from_node, to_node, weight=1.0, enabled=True, **kwargs):
    return NetworkConnection(
        from_node=from_node, to_node=to_node, weight=weight, enabled=enabled, **kwargs
    )


def _make_annotation(name, entry_nodes, exit_nodes, subgraph_nodes,
                     subgraph_connections=None, parent_annotation_id=None):
    return AnnotationData(
        name=name,
        hypothesis="test",
        entry_nodes=entry_nodes,
        exit_nodes=exit_nodes,
        subgraph_nodes=subgraph_nodes,
        subgraph_connections=subgraph_connections or [],
        parent_annotation_id=parent_annotation_id,
    )


def _megaann_structure():
    """
    Models the real MegaAnn1 topology (simplified):

        -24 --> A1678 --> 1676 ---> identity_7 --> A20608(ann_26) --> 608_a --> out
                                |-> identity_6 --> A20608(ann_24) --> 608_b --> out
        -20_a ----------------------------------------^(ann_26)
        -20_b ----------------------------------------^(ann_24)

    Child annotations:
      A1678: entries=[-24, -4_c], exits=[1676], subgraph=[-24, 1676, 1559_b, -4_c]
      A20608_b: entries=[-20_b, identity_6], exits=[608_b], subgraph=[608_b, -20_b, identity_6]
      A20608_a: entries=[-20_a, identity_7], exits=[608_a], subgraph=[608_a, -20_a, identity_7]

    Parent MegaAnn1 (compositional): children=[A1678, A20608_b, A20608_a]
    """
    nodes = [
        _make_node("-24", NodeType.INPUT),
        _make_node("-20_a", NodeType.INPUT),
        _make_node("-20_b", NodeType.INPUT),
        _make_node("1559_b", bias=0.5),
        _make_node("-4_c", NodeType.INPUT),
        _make_node("1676", bias=1.2),
        _make_node("identity_6", bias=0.0, activation="identity"),
        _make_node("identity_7", bias=0.0, activation="identity"),
        _make_node("608_a", bias=-1.0),
        _make_node("608_b", bias=-1.0),
        _make_node("0", NodeType.OUTPUT, bias=-0.5),
    ]
    connections = [
        # A1678 internals
        _make_conn("-24", "1676", 1.0),
        _make_conn("-4_c", "1559_b", 1.0),
        _make_conn("1559_b", "1676", -1.7),
        # 1676 -> identity nodes (internal wiring between children)
        _make_conn("1676", "identity_6", 0.35),
        _make_conn("1676", "identity_7", 0.35),
        # A20608_a internals
        _make_conn("-20_a", "608_a", -0.58),
        _make_conn("identity_7", "608_a", 1.0),
        # A20608_b internals
        _make_conn("-20_b", "608_b", -0.58),
        _make_conn("identity_6", "608_b", 1.0),
        # Outputs from composed region to rest of network
        _make_conn("608_a", "0", 0.63),
        _make_conn("608_b", "0", 2.19),
    ]
    structure = NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-24", "-20_a", "-20_b", "-4_c"],
        output_node_ids=["0"],
    )

    child1 = _make_annotation(
        "A1678",
        entry_nodes=["-24", "-4_c"],  # -4_c is INPUT, must be entry not internal
        exit_nodes=["1676"],
        subgraph_nodes=["-24", "1676", "1559_b", "-4_c"],
        parent_annotation_id="MegaAnn1",
    )
    child2 = _make_annotation(
        "A20608_b",
        entry_nodes=["-20_b", "identity_6"],
        exit_nodes=["608_b"],
        subgraph_nodes=["608_b", "-20_b", "identity_6"],
        parent_annotation_id="MegaAnn1",
    )
    child3 = _make_annotation(
        "A20608_a",
        entry_nodes=["-20_a", "identity_7"],
        exit_nodes=["608_a"],
        subgraph_nodes=["608_a", "-20_a", "identity_7"],
        parent_annotation_id="MegaAnn1",
    )
    # Parent declares wrong entry/exit -- auto-derivation in collapse_structure
    # will override these with the correct boundary nodes.
    parent = _make_annotation(
        "MegaAnn1",
        entry_nodes=["-24", "-20_a", "-20_b", "identity_6"],  # declared (wrong)
        exit_nodes=["-20_a", "identity_7"],                     # declared (wrong)
        subgraph_nodes=[],                                       # compositional
    )

    all_annotations = [child1, child2, child3, parent]
    return structure, all_annotations


class TestComputeEffectiveEntriesExits:
    """Test auto-derivation of entry/exit nodes for compositional annotations.

    NOTE: These tests call _compute_effective_entries_exits on the ORIGINAL
    (uncollapsed) structure to validate the boundary detection algorithm.
    In collapse_structure(), it is called on the CURRENT structure (after
    children are collapsed), so the exits will be fn_ nodes instead.
    """

    def test_megaann_effective_entries(self):
        structure, annotations = _megaann_structure()
        ann_by_name = {a.name: a for a in annotations}
        children_map = {}
        for a in annotations:
            if a.parent_annotation_id and a.parent_annotation_id in ann_by_name:
                children_map.setdefault(a.parent_annotation_id, []).append(a)

        effective_subgraph = _compute_effective_subgraph_nodes(
            ann_by_name["MegaAnn1"], ann_by_name, children_map
        )
        entries, exits = _compute_effective_entries_exits(
            effective_subgraph, structure
        )
        # True entries: nodes receiving connections from OUTSIDE the effective subgraph
        assert "-24" in entries
        assert "-20_a" in entries
        assert "-20_b" in entries
        # -4_c is also an input feeding into the subgraph from outside
        assert "-4_c" in entries
        # identity_6 and identity_7 are NOT true entries (fed by 1676, which is internal)
        assert "identity_6" not in entries
        assert "identity_7" not in entries

    def test_megaann_effective_exits(self):
        structure, annotations = _megaann_structure()
        ann_by_name = {a.name: a for a in annotations}
        children_map = {}
        for a in annotations:
            if a.parent_annotation_id and a.parent_annotation_id in ann_by_name:
                children_map.setdefault(a.parent_annotation_id, []).append(a)

        effective_subgraph = _compute_effective_subgraph_nodes(
            ann_by_name["MegaAnn1"], ann_by_name, children_map
        )
        entries, exits = _compute_effective_entries_exits(
            effective_subgraph, structure
        )
        # True exits (on original structure): nodes sending connections outside
        assert "608_a" in exits
        assert "608_b" in exits
        # -20_a is NOT a true exit (it only feeds into 608_a, which is internal)
        assert "-20_a" not in exits
        # identity_7 is NOT a true exit (feeds 608_a, internal)
        assert "identity_7" not in exits


class TestComposedAnnotationCollapseConnectivity:
    """Test that collapsing composed annotations produces connected function nodes."""

    def test_fn_megaann_has_outgoing_connections(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_conns_out = [
            c for c in collapsed.connections if c.from_node == "fn_MegaAnn1"
        ]
        assert len(fn_conns_out) > 0, "fn_MegaAnn1 must have outgoing connections"

    def test_fn_megaann_has_incoming_connections(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_conns_in = [
            c for c in collapsed.connections if c.to_node == "fn_MegaAnn1"
        ]
        assert len(fn_conns_in) > 0, "fn_MegaAnn1 must have incoming connections"

    def test_fn_megaann_connects_to_output(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_to_out = [
            c for c in collapsed.connections
            if c.from_node == "fn_MegaAnn1" and c.to_node == "0"
        ]
        # 608_a->0 and 608_b->0 should become fn_MegaAnn1->0 (two connections with different output_index)
        assert len(fn_to_out) == 2
        indices = {c.output_index for c in fn_to_out}
        assert indices == {0, 1}

    def test_fn_megaann_metadata_has_correct_entries_exits(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_node = next(n for n in collapsed.nodes if n.id == "fn_MegaAnn1")
        meta = fn_node.function_metadata
        # Effective entries: input nodes feeding into the composed region
        assert set(meta.input_names) == {"-24", "-20_a", "-20_b", "-4_c"}
        # Effective exits: after children are collapsed, the fn_child nodes
        # are what connect outside — these become the parent's exits
        assert set(meta.output_names) == {"fn_A20608_a", "fn_A20608_b"}

    def test_collapsed_structure_validates(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        result = collapsed.validate()
        assert result["is_valid"], f"Validation errors: {result['errors']}"

    def test_no_cycle_after_composed_collapse(self):
        """Verify cycle freedom (using DFS)."""
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        # Simple DFS cycle check
        adj = {}
        for c in collapsed.connections:
            adj.setdefault(c.from_node, []).append(c.to_node)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n.id: WHITE for n in collapsed.nodes}
        def dfs(u):
            color[u] = GRAY
            for v in adj.get(u, []):
                if color.get(v) == GRAY:
                    return True
                if color.get(v) == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False
        has_cycle = any(dfs(n.id) for n in collapsed.nodes if color[n.id] == WHITE)
        assert not has_cycle
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_collapse_transform_composed.py -v`
Expected: ImportError for `_compute_effective_entries_exits` (does not exist yet), and test failures for connectivity.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_core/test_collapse_transform_composed.py
git commit -m "test: add tests for composed annotation entry/exit derivation and collapse connectivity"
```

---

### Task 2: Implement `_compute_effective_entries_exits()` in collapse_transform

**Files:**
- Modify: `explaneat/core/collapse_transform.py`

- [ ] **Step 1: Add the `_compute_effective_entries_exits` function**

Add after `_compute_effective_subgraph_nodes()` (after line 56):

```python
def _compute_effective_entries_exits(
    effective_subgraph: Set[str],
    structure: NetworkStructure,
) -> Tuple[Set[str], Set[str]]:
    """Derive true entry and exit nodes from the effective subgraph boundary.

    Entry nodes: subgraph nodes that receive at least one connection from
    outside the subgraph (or are input nodes with no predecessors within
    the subgraph).

    Exit nodes: subgraph nodes that send at least one connection to a node
    outside the subgraph.

    This replaces manually-declared entry/exit for compositional annotations,
    which can be wrong when internal wiring between children is mislabeled.
    """
    entries: Set[str] = set()
    exits: Set[str] = set()

    # Build quick lookup of enabled connections
    for conn in structure.connections:
        if not conn.enabled:
            continue
        from_in = conn.from_node in effective_subgraph
        to_in = conn.to_node in effective_subgraph

        if not from_in and to_in:
            # External -> subgraph: to_node is an entry
            entries.add(conn.to_node)
        elif from_in and not to_in:
            # Subgraph -> external: from_node is an exit
            exits.add(conn.from_node)

    # Input nodes in the subgraph with no internal predecessors are also entries
    input_ids = set(structure.input_node_ids)
    for node_id in effective_subgraph:
        if node_id in input_ids:
            # Check if this input has any predecessors within the subgraph
            has_internal_pred = any(
                c.from_node in effective_subgraph
                for c in structure.connections
                if c.enabled and c.to_node == node_id
            )
            if not has_internal_pred:
                entries.add(node_id)

    return entries, exits
```

- [ ] **Step 2: Use effective entries/exits when collapsing compositional annotations**

In `collapse_structure()`, modify the compositional annotation handling block (around line 122-143). Replace the block starting at `if descendant_names:` with:

```python
    for annotation in to_collapse:
        descendant_names = _find_descendant_names(annotation.name, children_map)

        if descendant_names:
            # Compositional annotation: compute effective subgraph and derive
            # true entry/exit nodes from boundary connections.
            original_effective = _compute_effective_subgraph_nodes(
                annotation, annotation_by_name, children_map
            )
            current_node_ids = {n.id for n in result.nodes}
            current_subgraph: Set[str] = set()
            for nid in original_effective:
                if nid in current_node_ids:
                    current_subgraph.add(nid)
            for desc_name in descendant_names:
                fn_id = f"fn_{desc_name}"
                if fn_id in current_node_ids:
                    current_subgraph.add(fn_id)

            # Auto-derive correct entries/exits from the CURRENT structure
            # (after children have been collapsed). Must use current_subgraph,
            # not original_effective, because original node IDs may no longer
            # exist in the structure (replaced by fn_ nodes).
            effective_entries, effective_exits = _compute_effective_entries_exits(
                current_subgraph, result
            )

            # Create a corrected annotation with derived entry/exit.
            # Sort for deterministic output_index assignment.
            corrected = AnnotationData(
                name=annotation.name,
                hypothesis=annotation.hypothesis,
                entry_nodes=sorted(effective_entries),
                exit_nodes=sorted(effective_exits),
                subgraph_nodes=list(current_subgraph),
                subgraph_connections=annotation.subgraph_connections,
                evidence=annotation.evidence,
                parent_annotation_id=annotation.parent_annotation_id,
            )

            result = _collapse_one(
                result, corrected, effective_nodes_override=current_subgraph
            )
        else:
            result = _collapse_one(result, annotation)
```

- [ ] **Step 3: Run the composed tests**

Run: `uv run pytest tests/test_core/test_collapse_transform_composed.py -v`
Expected: All tests pass.

- [ ] **Step 4: Run existing collapse tests for regression**

Run: `uv run pytest tests/test_core/test_collapse_transform.py tests/test_core/test_collapse_integration.py -v`
Expected: All existing tests still pass. The compositional tests in `TestCompositionalAnnotationCollapse` use a simpler topology where the declared entries/exits happen to be correct, so auto-derivation should produce the same results.

- [ ] **Step 5: Commit**

```bash
git add explaneat/core/collapse_transform.py
git commit -m "fix: auto-derive entry/exit nodes for compositional annotation collapse"
```

---

### Task 3: Forward-pass equivalence test for composed collapse

**Files:**
- Modify: `tests/test_core/test_collapse_transform_composed.py`

- [ ] **Step 1: Add forward-pass equivalence test**

Append to `test_collapse_transform_composed.py`:

```python
import numpy as np
import torch
from explaneat.core.structure_network import StructureNetwork


class TestComposedCollapseForwardPass:
    """Verify that collapsing a composed annotation preserves forward-pass output."""

    def test_megaann_forward_pass_equivalence(self):
        structure, annotations = _megaann_structure()

        # Forward pass on expanded structure
        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[0.5, 0.3, -0.2, 0.8]], dtype=torch.float32)
        expanded_out = expanded_net.forward(x).detach().numpy()

        # Collapse all annotations including parent
        collapsed = collapse_structure(
            structure, annotations,
            {"A1678", "A20608_b", "A20608_a", "MegaAnn1"},
        )
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_megaann_batch_forward_pass(self):
        structure, annotations = _megaann_structure()

        x = torch.tensor([
            [0.5, 0.3, -0.2, 0.8],
            [1.0, -1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float32)

        expanded_net = StructureNetwork(structure)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(
            structure, annotations,
            {"A1678", "A20608_b", "A20608_a", "MegaAnn1"},
        )
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)
```

- [ ] **Step 2: Run forward-pass tests**

Run: `uv run pytest tests/test_core/test_collapse_transform_composed.py::TestComposedCollapseForwardPass -v`
Expected: PASS (both tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_core/test_collapse_transform_composed.py
git commit -m "test: add forward-pass equivalence tests for composed annotation collapse"
```

---

## Chunk 2: Teach AnnotationFunction About FUNCTION Nodes

### Task 4: Add tests for AnnotationFunction with FUNCTION nodes

**Files:**
- Create: `tests/test_analysis/test_annotation_function_composed.py`

- [ ] **Step 1: Write tests for AnnotationFunction handling FUNCTION nodes**

```python
"""Tests for AnnotationFunction handling composed annotations with FUNCTION nodes."""

import numpy as np
import pytest
from explaneat.core.genome_network import (
    FunctionNodeMetadata,
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.collapse_transform import collapse_structure
from explaneat.analysis.annotation_function import AnnotationFunction


def _make_node(id, type=NodeType.HIDDEN, **kwargs):
    return NetworkNode(id=id, type=type, **kwargs)


def _make_conn(from_node, to_node, weight=1.0, enabled=True, **kwargs):
    return NetworkConnection(
        from_node=from_node, to_node=to_node, weight=weight, enabled=enabled, **kwargs
    )


def _composed_network_and_annotations():
    """
    Simple composed topology:
        in1 -> A -> B -> out1
               ^child1   ^child2
               ^------parent------^

    child1: entries=[in1], exits=[A], subgraph=[in1, A]
    child2: entries=[A], exits=[B], subgraph=[A, B]  (A is shared entry)
    parent: compositional over child1 + child2
    """
    nodes = [
        _make_node("in1", NodeType.INPUT),
        _make_node("A", bias=0.1, activation="relu"),
        _make_node("B", bias=-0.2, activation="relu"),
        _make_node("out1", NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        _make_conn("in1", "A", weight=0.5),
        _make_conn("A", "B", weight=0.8),
        _make_conn("B", "out1", weight=1.0),
    ]
    structure = NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["in1"],
        output_node_ids=["out1"],
    )

    child1 = AnnotationData(
        name="child1", hypothesis="test",
        entry_nodes=["in1"], exit_nodes=["A"],
        subgraph_nodes=["in1", "A"],
        subgraph_connections=[("in1", "A")],
        parent_annotation_id="parent",
    )
    child2 = AnnotationData(
        name="child2", hypothesis="test",
        entry_nodes=["A"], exit_nodes=["B"],
        subgraph_nodes=["A", "B"],
        subgraph_connections=[("A", "B")],
        parent_annotation_id="parent",
    )
    parent = AnnotationData(
        name="parent", hypothesis="test",
        entry_nodes=["in1"], exit_nodes=["B"],
        subgraph_nodes=[],
        subgraph_connections=[],
    )
    return structure, [child1, child2, parent]


class TestAnnotationFunctionWithFunctionNodes:
    """Test AnnotationFunction on a structure containing collapsed fn_ nodes."""

    def test_expanded_formula_from_uncollapsed_structure(self):
        """Fully-expanded formula from the original (uncollapsed) structure."""
        structure, annotations = _composed_network_and_annotations()
        parent = annotations[2]
        # Use the parent annotation but with correct subgraph from effective computation
        parent_with_subgraph = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1"], exit_nodes=["B"],
            subgraph_nodes=["in1", "A", "B"],
            subgraph_connections=[("in1", "A"), ("A", "B")],
        )
        af = AnnotationFunction.from_structure(parent_with_subgraph, structure)
        result = af(np.array([1.0]))
        # Manual: A = relu(0.5*1.0 + 0.1) = 0.6, B = relu(0.8*0.6 - 0.2) = 0.28
        expected = 0.28
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_collapsed_formula_references_child_names(self):
        """Collapsed formula should use child function names, not primitives."""
        structure, annotations = _composed_network_and_annotations()
        # Collapse only children, then build AnnotationFunction from partially-collapsed structure
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        # The parent's effective subgraph now contains fn_child1 and fn_child2
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1"], exit_nodes=["B"],
            subgraph_nodes=["in1", "fn_child1", "fn_child2"],
            subgraph_connections=[],  # will be derived from structure
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        latex = af.to_latex(expand=False)
        assert latex is not None
        # Should reference child function names
        assert "child1" in latex or "child2" in latex

    def test_expanded_formula_substitutes_child_expressions(self):
        """Expanded formula should inline child expressions to primitives."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1"], exit_nodes=["B"],
            subgraph_nodes=["in1", "fn_child1", "fn_child2"],
            subgraph_connections=[],
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        latex_expanded = af.to_latex(expand=True)
        assert latex_expanded is not None
        # Expanded form should NOT contain child function names
        assert "child1" not in latex_expanded
        assert "child2" not in latex_expanded

    def test_evaluation_with_function_nodes_matches_expanded(self):
        """Numerical evaluation through fn_ nodes matches the expanded version."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1"], exit_nodes=["B"],
            subgraph_nodes=["in1", "fn_child1", "fn_child2"],
            subgraph_connections=[],
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        result = af(np.array([1.0]))
        # Same expected value as the expanded version
        expected = 0.28
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_to_latex_default_is_expanded(self):
        """Default to_latex() (no expand param) returns expanded form."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1"], exit_nodes=["B"],
            subgraph_nodes=["in1", "fn_child1", "fn_child2"],
            subgraph_connections=[],
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        # Default should be the expanded form (backwards compatible)
        default_latex = af.to_latex()
        expanded_latex = af.to_latex(expand=True)
        assert default_latex == expanded_latex
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_analysis/test_annotation_function_composed.py -v`
Expected: Failures - `to_latex()` doesn't accept `expand` param, `_build_from_structure` doesn't handle FUNCTION nodes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_analysis/test_annotation_function_composed.py
git commit -m "test: add tests for AnnotationFunction with FUNCTION nodes and expand/collapse"
```

---

### Task 5: Implement FUNCTION node handling in AnnotationFunction

**Files:**
- Modify: `explaneat/analysis/annotation_function.py`

- [ ] **Step 1: Update `_build_from_structure` to handle FUNCTION nodes**

In `_build_from_structure()` (line 83), after building the `predecessors` dict and computing depths, add handling for FUNCTION nodes. The key changes:

1. Import `NodeType` at the top of the file (line 8 area):

```python
from ..core.genome_network import NetworkStructure, NodeType
```

2. Replace `_build_from_structure()` with a version that recognizes FUNCTION nodes:

```python
    def _build_from_structure(self):
        """Build computation graph directly from NetworkStructure."""
        nodes_by_id = {n.id: n for n in self._structure.nodes}
        enabled_conns = {
            (c.from_node, c.to_node): c
            for c in self._structure.connections
            if c.enabled
        }

        output_ids = set(self._structure.output_node_ids)
        subgraph_set = set(self.subgraph_nodes)

        # If subgraph_connections is empty, derive from structure
        if not self.subgraph_connections:
            self.subgraph_connections = [
                (c.from_node, c.to_node)
                for c in self._structure.connections
                if c.enabled and c.from_node in subgraph_set and c.to_node in subgraph_set
            ]

        # Build predecessors within the subgraph
        predecessors = {n: [] for n in self.subgraph_nodes}
        for from_n, to_n in self.subgraph_connections:
            if from_n in subgraph_set and to_n in subgraph_set:
                predecessors.setdefault(to_n, []).append(from_n)

        # Compute depth within subgraph (entry nodes = depth 0)
        depth = {n: 0 for n in self.entry_nodes}
        changed = True
        while changed:
            changed = False
            for n in self.subgraph_nodes:
                if n in self.entry_nodes:
                    continue
                parent_depths = [depth[p] for p in predecessors.get(n, []) if p in depth]
                if parent_depths:
                    new_d = max(parent_depths) + 1
                    if n not in depth or new_d > depth[n]:
                        depth[n] = new_d
                        changed = True

        self._node_locations = {n: (depth.get(n, 0), 0) for n in self.subgraph_nodes}
        self._exit_is_output = {n: n in output_ids for n in self.exit_nodes}

        # Order internal nodes by depth
        internal_nodes = [n for n in self.subgraph_nodes if n not in self.entry_nodes]
        internal_nodes.sort(key=lambda n: depth.get(n, 0))

        # Track FUNCTION nodes for sympy
        self._function_node_data = {}  # node_id -> {ann_func, meta}

        self._steps = []
        for node_str in internal_nodes:
            node_obj = nodes_by_id.get(node_str)
            if node_obj is None:
                continue

            # Check if this is a FUNCTION node
            if node_obj.type == NodeType.FUNCTION and node_obj.function_metadata:
                meta = node_obj.function_metadata
                # Build a child AnnotationFunction from the metadata
                child_af = self._build_child_annotation_function(meta, nodes_by_id)
                self._function_node_data[node_str] = {
                    "ann_func": child_af,
                    "meta": meta,
                }

                # Get input connections to this function node
                input_node_strs = []
                for conn_key in self.subgraph_connections:
                    if conn_key[1] == node_str:
                        input_node_strs.append(conn_key[0])

                # IMPORTANT: Reorder inputs to match meta.input_names order.
                # The child AnnotationFunction creates symbols x_0, x_1, ...
                # based on meta.input_names order. Connection iteration order
                # is arbitrary and may not match.
                if meta.input_names:
                    name_order = {name: i for i, name in enumerate(meta.input_names)}
                    input_node_strs.sort(key=lambda n: name_order.get(n, len(name_order)))

                self._steps.append({
                    "node": node_str,
                    "input_nodes": input_node_strs,
                    "weights": None,  # Not used for FUNCTION nodes
                    "bias": 0.0,
                    "activation": "function",
                    "is_function": True,
                    "function_meta": meta,
                    "function_af": child_af,
                    "n_outputs": meta.n_outputs,
                    "output_names": meta.output_names,
                })
                continue

            # Regular node (unchanged logic)
            bias_val = node_obj.bias if node_obj.bias is not None else 0.0
            is_output_node = self._exit_is_output.get(node_str, False) and node_str in output_ids
            if node_obj.activation:
                activation = node_obj.activation
            elif is_output_node:
                activation = "sigmoid"
            else:
                activation = "relu"

            input_node_strs = []
            weights = []
            input_output_indices = []  # output_index for each input (None for regular nodes)
            for conn in self.subgraph_connections:
                if conn[1] == node_str:
                    input_node_strs.append(conn[0])
                    conn_obj = enabled_conns.get(conn)
                    w = conn_obj.weight if conn_obj else 0.0
                    weights.append(w)
                    # Capture output_index for multi-output FUNCTION node inputs
                    out_idx = getattr(conn_obj, 'output_index', None) if conn_obj else None
                    input_output_indices.append(out_idx)

            self._steps.append({
                "node": node_str,
                "input_nodes": input_node_strs,
                "weights": np.array(weights, dtype=np.float64),
                "bias": float(bias_val),
                "activation": activation,
                "is_function": False,
                "input_output_indices": input_output_indices,
            })

    def _build_child_annotation_function(self, meta, nodes_by_id):
        """Reconstruct an AnnotationFunction from FunctionNodeMetadata."""
        from ..core.genome_network import NetworkConnection, NetworkNode, NetworkStructure

        # Rebuild mini-structure from metadata (same approach as StructureNetwork)
        mini_nodes = []
        for nid in meta.subgraph_nodes:
            existing = nodes_by_id.get(nid)
            if existing:
                mini_nodes.append(existing)
            else:
                props = meta.node_properties.get(nid, {})
                mini_nodes.append(NetworkNode(
                    id=nid,
                    type=NodeType.HIDDEN,
                    bias=props.get("bias", 0.0),
                    activation=props.get("activation", "relu"),
                ))
        mini_conns = [
            NetworkConnection(
                from_node=f, to_node=t,
                weight=meta.connection_weights.get((f, t), 0.0),
                enabled=True,
            )
            for f, t in meta.subgraph_connections
        ]
        mini_structure = NetworkStructure(
            nodes=mini_nodes, connections=mini_conns,
            input_node_ids=list(meta.input_names),
            output_node_ids=list(meta.output_names),
        )
        child_ann = AnnotationData(
            name=meta.annotation_name, hypothesis=meta.hypothesis,
            entry_nodes=list(meta.input_names),
            exit_nodes=list(meta.output_names),
            subgraph_nodes=list(meta.subgraph_nodes),
            subgraph_connections=list(meta.subgraph_connections),
        )
        return AnnotationFunction.from_structure(child_ann, mini_structure)
```

Note: this requires importing `AnnotationData` at the top. Add to the imports:

```python
from ..core.model_state import AnnotationData
```

- [ ] **Step 2: Update `__call__` to handle FUNCTION steps**

Replace the evaluation loop in `__call__` (line 283 area) with:

```python
        # Evaluate each step in topological order
        for step in self._steps:
            if step.get("is_function"):
                # Evaluate through the child AnnotationFunction
                child_af = step["function_af"]
                # Gather inputs to the function node
                fn_inputs = np.column_stack(
                    [activations.get(n, np.zeros(len(x))) for n in step["input_nodes"]]
                ) if step["input_nodes"] else np.zeros((len(x), 0))
                fn_outputs = child_af(fn_inputs)
                if fn_outputs.ndim == 1:
                    fn_outputs = fn_outputs.reshape(-1, 1)
                # Store first output under the node ID (for single-output case)
                activations[step["node"]] = fn_outputs[:, 0]
                # Store each output under positional key for multi-output lookup
                for oi, oname in enumerate(step.get("output_names", [])):
                    if oi < fn_outputs.shape[1]:
                        activations[oname] = fn_outputs[:, oi]
                        activations[f"{step['node']}__out_{oi}"] = fn_outputs[:, oi]
            else:
                # Regular node evaluation.
                # For multi-output FUNCTION node inputs, use output_index to
                # select the correct output column.
                output_indices = step.get("input_output_indices", [])
                input_vals = []
                for j, n in enumerate(step["input_nodes"]):
                    out_idx = output_indices[j] if j < len(output_indices) else None
                    if out_idx is not None:
                        val = activations.get(f"{n}__out_{out_idx}", activations.get(n, np.zeros(len(x))))
                    else:
                        val = activations.get(n, np.zeros(len(x)))
                    input_vals.append(val)
                inputs = np.column_stack(input_vals) if input_vals else np.zeros((len(x), 0))

                if len(step["weights"]) > 0 and inputs.shape[1] > 0:
                    z = inputs @ step["weights"] + step["bias"]
                else:
                    z = np.full(len(x), step["bias"])

                act_fn = get_numpy_activation(step["activation"])
                activations[step["node"]] = act_fn(z)
```

- [ ] **Step 3: Update `to_sympy()` to accept `expand` parameter**

Replace `to_sympy()` (line 308 area) with:

```python
    def to_sympy(self, expand: bool = True) -> Optional[Dict[str, "sympy.Expr"]]:
        """Extract symbolic expressions for each output.

        Args:
            expand: If True (default), inline child function expressions to
                produce a fully-expanded primitive formula. If False, represent
                child FUNCTION nodes as named sympy Functions (e.g. child1(x_0)).

        Returns None if the subgraph is too complex (>5 inputs or >3 internal layers).
        """
        if self.n_inputs > 5:
            return None

        depths = set()
        for step in self._steps:
            loc = self._node_locations.get(step["node"])
            if loc:
                depths.add(loc[0])
        if len(depths) > 3:
            return None

        try:
            import sympy
        except ImportError:
            return None

        # Create input symbols
        input_syms = {}
        for i, node_str in enumerate(self.entry_nodes):
            input_syms[node_str] = sympy.Symbol(f"x_{i}")

        node_exprs = dict(input_syms)

        for step in self._steps:
            if step.get("is_function"):
                meta = step["function_meta"]
                child_af = step["function_af"]

                # Gather input expressions for the function node
                child_input_exprs = [
                    node_exprs.get(n, sympy.Symbol(n))
                    for n in step["input_nodes"]
                ]

                if expand:
                    # Inline child's sympy expressions
                    child_sympy = child_af.to_sympy(expand=True)
                    if child_sympy is None:
                        return None  # Can't expand this child

                    # Substitute child's input symbols with our input expressions
                    child_input_syms = [
                        sympy.Symbol(f"x_{i}") for i in range(len(step["input_nodes"]))
                    ]

                    # Map each child output to a substituted expression.
                    # Store under both the original output name and a positional
                    # key "{fn_node}__out_{i}" so downstream regular nodes can
                    # look up by output_index for multi-output FUNCTION nodes.
                    for oi, oname in enumerate(step.get("output_names", [])):
                        out_key = f"y_{oi}"
                        if out_key in child_sympy:
                            sub_expr = child_sympy[out_key]
                            for ci, csym in enumerate(child_input_syms):
                                if ci < len(child_input_exprs):
                                    sub_expr = sub_expr.subs(csym, child_input_exprs[ci])
                            simplified = sympy.simplify(sub_expr)
                            node_exprs[oname] = simplified
                            node_exprs[f"{step['node']}__out_{oi}"] = simplified

                    # Default: first output stored under the node ID itself
                    if f"{step['node']}__out_0" in node_exprs:
                        node_exprs[step["node"]] = node_exprs[f"{step['node']}__out_0"]
                else:
                    # Collapsed form: represent as named function application
                    fn_sym = sympy.Function(meta.annotation_name)
                    fn_expr = fn_sym(*child_input_exprs)
                    node_exprs[step["node"]] = fn_expr
                    for oi, oname in enumerate(step.get("output_names", [])):
                        node_exprs[oname] = fn_expr
                        node_exprs[f"{step['node']}__out_{oi}"] = fn_expr
            else:
                # Regular node: sum(w_i * input_i) + bias
                # Use output_index to select the correct output from
                # multi-output FUNCTION nodes (stored as "{node}__out_{idx}").
                expr = sympy.Float(step["bias"])
                output_indices = step.get("input_output_indices", [])
                for j, in_node in enumerate(step["input_nodes"]):
                    w = step["weights"][j]
                    if abs(w) < 1e-12:
                        continue
                    # Check for multi-output function node output
                    out_idx = output_indices[j] if j < len(output_indices) else None
                    if out_idx is not None:
                        lookup = f"{in_node}__out_{out_idx}"
                    else:
                        lookup = in_node
                    in_expr = node_exprs.get(lookup, node_exprs.get(in_node, sympy.Symbol(in_node)))
                    expr += sympy.Float(w) * in_expr

                act_sym_fn = get_sympy_activation(step["activation"])
                expr = act_sym_fn(expr)
                node_exprs[step["node"]] = sympy.simplify(expr)

        # Collect output expressions
        result = {}
        for i, node_str in enumerate(self.exit_nodes):
            if node_str in node_exprs:
                result[f"y_{i}"] = node_exprs[node_str]

        return result if result else None
```

- [ ] **Step 4: Update `to_latex()` to pass through `expand` parameter**

Replace `to_latex()`:

```python
    def to_latex(self, expand: bool = True) -> Optional[str]:
        """Get LaTeX representation of the function.

        Args:
            expand: If True (default), fully expand to primitives.
                If False, use child function names.

        Returns None if sympy extraction fails or is intractable.
        """
        exprs = self.to_sympy(expand=expand)
        if exprs is None:
            return None

        try:
            import sympy
            parts = []
            for name, expr in exprs.items():
                parts.append(f"{name} = {sympy.latex(expr)}")
            return " \\\\ ".join(parts)
        except Exception:
            return None
```

- [ ] **Step 5: Run the composed annotation function tests**

Run: `uv run pytest tests/test_analysis/test_annotation_function_composed.py -v`
Expected: All tests pass.

- [ ] **Step 6: Run existing annotation function tests for regression**

Run: `uv run pytest tests/test_analysis/test_annotation_function.py -v`
Expected: All existing tests still pass (default `expand=True` is backwards-compatible).

- [ ] **Step 7: Commit**

```bash
git add explaneat/analysis/annotation_function.py
git commit -m "feat: teach AnnotationFunction to handle FUNCTION nodes with expand/collapse formulas"
```

---

## Chunk 3: Update API and Frontend

### Task 6: Update FormulaResponse schema and evidence endpoint

**Files:**
- Modify: `explaneat/api/schemas.py`
- Modify: `explaneat/api/routes/evidence.py`

- [ ] **Step 1: Update `FormulaResponse` in schemas.py**

Find and replace the `FormulaResponse` class in `explaneat/api/schemas.py`:

```python
class ChildFormulaInfo(BaseModel):
    name: str
    latex: Optional[str] = None
    dimensionality: List[int]

class FormulaResponse(BaseModel):
    latex: Optional[str] = None                    # backwards-compat: expanded form
    latex_collapsed: Optional[str] = None          # collapsed form (references child names)
    latex_expanded: Optional[str] = None           # fully expanded form
    tractable: bool = False
    dimensionality: List[int]                      # [n_inputs, n_outputs]
    is_composed: bool = False                      # True if annotation has children
    children: List[ChildFormulaInfo] = []           # child annotation formulas
```

- [ ] **Step 2: Add `_build_engine` helper and refactor `_build_model_state` in evidence.py**

Factor out engine construction from `_build_model_state` (lines 50-77). Replace `_build_model_state` with two functions — `_build_engine` and a slimmed `_build_model_state` that delegates:

```python
def _build_engine(session, genome_id: str) -> ModelStateEngine:
    """Build a ModelStateEngine for a genome with all operations replayed.

    Returns the engine (with .current_state and .annotations available).
    """
    from ...core.explaneat import ExplaNEAT

    genome_db, neat_genome, config = _load_genome_and_config(session, genome_id)
    explainer = ExplaNEAT(neat_genome, config)
    phenotype = explainer.get_phenotype_network()

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == uuid.UUID(genome_id))
        .first()
    )

    engine = ModelStateEngine(phenotype)
    if explanation and explanation.operations:
        engine.load_operations({"operations": explanation.operations})
    return engine


def _build_model_state(session, genome_id: str) -> NetworkStructure:
    """Build the current annotated model by replaying operations on the phenotype."""
    return _build_engine(session, genome_id).current_state
```

All existing callers of `_build_model_state` are unchanged.

- [ ] **Step 3: Add `child_annotation_ids` to `_find_annotation_in_operations` return dict**

In `_find_annotation_in_operations()` (line 106-115 in current evidence.py), add the missing `child_annotation_ids` and `parent_annotation_id` fields:

```python
        if ann_id == annotation_id:
            return {
                "id": ann_id,
                "entry_nodes": params.get("entry_nodes", []),
                "exit_nodes": params.get("exit_nodes", []),
                "subgraph_nodes": params.get("subgraph_nodes", []),
                "subgraph_connections": params.get("subgraph_connections", []),
                "name": params.get("name"),
                "hypothesis": params.get("hypothesis"),
                "evidence": params.get("evidence") or {},
                "child_annotation_ids": params.get("child_annotation_ids", []),
                "parent_annotation_id": params.get("parent_annotation_id"),
            }
```

- [ ] **Step 4: Update the formula endpoint in evidence.py**

Replace the `get_formula` function with the composed-aware version. **Key details:**
- `child_annotation_ids` stores annotation **names** (e.g. "A1678"), not synthetic IDs (e.g. "ann_37")
- Match child operations by `params["name"]`, not by synthetic `ann_id`
- Use `_build_engine` (not the deleted `_get_engine` pseudocode) to access `engine.annotations`

```python
@router.get("/formula", response_model=FormulaResponse)
async def get_formula(
    annotation_id: str,
    genome_id: str = Path(...),
):
    """Get closed-form formula for an annotation subgraph.

    For composed annotations (with children), returns both collapsed and
    expanded LaTeX representations.
    """
    with db.session_scope() as session:
        # Build engine ONCE — used for both model_state and annotations
        engine = _build_engine(session, genome_id)
        model_state = engine.current_state
        annotation = _find_annotation_in_operations(session, genome_id, annotation_id)

        ann_fn = AnnotationFunction.from_structure(annotation, model_state)
        n_in, n_out = ann_fn.dimensionality

        # child_annotation_ids stores child annotation NAMES (e.g. "A1678"),
        # not synthetic IDs (e.g. "ann_37").
        child_ann_names = annotation.get("child_annotation_ids", [])
        is_composed = len(child_ann_names) > 0

        if is_composed:
            # For composed annotations, we need the partially-collapsed structure
            # where children are collapsed but parent is not.
            children_info = []

            # Look up child formulas by matching annotation NAME (not synthetic ID)
            explanation = (
                session.query(Explanation)
                .filter(Explanation.genome_id == uuid.UUID(genome_id))
                .first()
            )
            for op in (explanation.operations or []):
                if op.get("type") != "annotate":
                    continue
                params = op.get("params", {})
                op_name = params.get("name")
                if op_name not in child_ann_names:
                    continue
                result_data = op.get("result", {})
                child_ann_id = result_data.get("annotation_id") or f"ann_{op.get('seq', 0)}"
                child_ann = _find_annotation_in_operations(session, genome_id, child_ann_id)
                child_af = AnnotationFunction.from_structure(child_ann, model_state)
                child_latex = child_af.to_latex()
                cn_in, cn_out = child_af.dimensionality
                children_info.append(ChildFormulaInfo(
                    name=op_name,
                    latex=child_latex,
                    dimensionality=[cn_in, cn_out],
                ))

            # Build partially-collapsed structure (children collapsed, parent not)
            from ...core.collapse_transform import collapse_structure, _compute_effective_entries_exits
            child_names_set = set(child_ann_names)
            partially_collapsed = collapse_structure(
                model_state, engine.annotations, child_names_set
            )

            # Build parent annotation dict for the partially-collapsed structure.
            # Include fn_ nodes from collapsed children in the subgraph.
            fn_node_ids = [f"fn_{name}" for name in child_ann_names]
            parent_ann = dict(annotation)  # copy to avoid mutating
            parent_subgraph = (
                set(parent_ann.get("subgraph_nodes", []))
                | set(parent_ann.get("entry_nodes", []))
                | set(parent_ann.get("exit_nodes", []))
                | set(fn_node_ids)
            )
            # Keep only nodes that actually exist in the partially-collapsed structure
            existing_ids = {n.id for n in partially_collapsed.nodes}
            parent_subgraph = parent_subgraph & existing_ids
            parent_ann["subgraph_nodes"] = list(parent_subgraph)

            # CRITICAL: Derive effective entries/exits from the partially-collapsed
            # structure. The original annotation's exit_nodes reference node IDs
            # that may no longer exist (replaced by fn_ nodes during child collapse).
            effective_entries, effective_exits = _compute_effective_entries_exits(
                parent_subgraph, partially_collapsed
            )
            parent_ann["entry_nodes"] = sorted(effective_entries)
            parent_ann["exit_nodes"] = sorted(effective_exits)

            composed_af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
            latex_collapsed = composed_af.to_latex(expand=False)
            latex_expanded = composed_af.to_latex(expand=True)

            # Fall back to direct computation if composed path fails
            if latex_expanded is None:
                latex_expanded = ann_fn.to_latex()

            return FormulaResponse(
                latex=latex_expanded,
                latex_collapsed=latex_collapsed,
                latex_expanded=latex_expanded,
                tractable=latex_expanded is not None or latex_collapsed is not None,
                dimensionality=[n_in, n_out],
                is_composed=True,
                children=children_info,
            )
        else:
            # Leaf annotation: single formula
            latex = ann_fn.to_latex()
            return FormulaResponse(
                latex=latex,
                latex_collapsed=None,
                latex_expanded=latex,
                tractable=latex is not None,
                dimensionality=[n_in, n_out],
                is_composed=False,
                children=[],
            )
```

Also update imports at the top of evidence.py. Add `ChildFormulaInfo` to the existing schema import block:
```python
from ..schemas import (
    VizDataRequest,
    VizDataResponse,
    FormulaResponse,
    ChildFormulaInfo,  # <-- add this
    SnapshotRequest,
    NarrativeUpdateRequest,
    EvidenceEntry,
    EvidenceListResponse,
)
```

Note: `collapse_structure` and `_compute_effective_entries_exits` are imported inline within the `is_composed` branch to avoid circular imports.

- [ ] **Step 5: Run the API server and manually verify the endpoint works**

Run: `uv run python -m explaneat api`
Then: `curl "http://localhost:8000/api/genomes/<GENOME_ID>/evidence/formula?annotation_id=ann_38"` (MegaAnn1's ID)
Expected: JSON response with `is_composed: true`, `latex_collapsed` and `latex_expanded` populated, `children` array with child formulas.

- [ ] **Step 6: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/evidence.py
git commit -m "feat: formula endpoint returns collapsed and expanded forms for composed annotations"
```

---

### Task 7: Update frontend FormulaDisplay with expand/collapse toggle

**Files:**
- Modify: `web/react-explorer/src/api/client.ts`
- Modify: `web/react-explorer/src/components/FormulaDisplay.tsx`

- [ ] **Step 1: Update FormulaResponse type in client.ts**

Find the `FormulaResponse` type in `client.ts` (around line 378) and replace:

```typescript
export type ChildFormulaInfo = {
  name: string;
  latex: string | null;
  dimensionality: [number, number];
};

export type FormulaResponse = {
  latex: string | null;
  latex_collapsed: string | null;
  latex_expanded: string | null;
  tractable: boolean;
  dimensionality: [number, number];
  is_composed: boolean;
  children: ChildFormulaInfo[];
};
```

- [ ] **Step 2: Update FormulaDisplay component with toggle**

Replace the entire `FormulaDisplay.tsx` with:

```tsx
import { useEffect, useState, useRef, useCallback } from "react";
import { getFormula, type FormulaResponse } from "../api/client";
import katex from "katex";
import "katex/dist/katex.min.css";

type FormulaDisplayProps = {
  genomeId: string;
  annotationId: string;
};

export function FormulaDisplay({ genomeId, annotationId }: FormulaDisplayProps) {
  const [formula, setFormula] = useState<FormulaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const mathRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getFormula(genomeId, annotationId)
      .then((data) => {
        if (!cancelled) {
          setFormula(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [genomeId, annotationId]);

  // Determine which LaTeX to render
  const currentLatex = formula
    ? expanded
      ? formula.latex_expanded
      : formula.latex_collapsed ?? formula.latex_expanded  // fall back to expanded for leaf annotations
    : null;

  useEffect(() => {
    if (currentLatex && mathRef.current) {
      try {
        katex.render(currentLatex, mathRef.current, {
          displayMode: true,
          throwOnError: false,
        });
      } catch {
        if (mathRef.current) {
          mathRef.current.textContent = currentLatex;
        }
      }
    }
  }, [currentLatex]);

  const handleToggle = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  if (loading) {
    return <div className="formula-display loading">Loading formula...</div>;
  }

  if (error) {
    return <div className="formula-display error-message">Formula error: {error}</div>;
  }

  if (!formula || !formula.tractable) {
    const [nIn, nOut] = formula?.dimensionality || [0, 0];
    return (
      <div className="formula-display">
        <div className="formula-header">
          <span className="formula-label">Formula</span>
          <span className="formula-dim">
            f: R<sup>{nIn}</sup> &rarr; R<sup>{nOut}</sup>
          </span>
        </div>
        <div className="formula-intractable">
          Closed-form not tractable for this subgraph
        </div>
      </div>
    );
  }

  const [nIn, nOut] = formula.dimensionality;

  return (
    <div className="formula-display">
      <div className="formula-header">
        <span className="formula-label">Formula</span>
        <span className="formula-dim">
          f: R<sup>{nIn}</sup> &rarr; R<sup>{nOut}</sup>
        </span>
        {formula.is_composed && (
          <button
            className="formula-toggle"
            onClick={handleToggle}
            title={expanded ? "Show composed form" : "Show expanded form"}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
        )}
      </div>
      <div className="formula-math" ref={mathRef} />
      {formula.is_composed && formula.children.length > 0 && !expanded && (
        <div className="formula-children">
          {formula.children.map((child) => (
            <div key={child.name} className="formula-child">
              <span className="formula-child-name">{child.name}</span>
              <span className="formula-child-dim">
                R<sup>{child.dimensionality[0]}</sup> &rarr; R<sup>{child.dimensionality[1]}</sup>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Add CSS for the toggle button and children list**

Add to `web/react-explorer/src/styles.css` (find the `.formula-display` section):

```css
.formula-toggle {
  font-size: 0.75rem;
  padding: 2px 8px;
  border: 1px solid #4b5563;
  border-radius: 4px;
  background: #1f2937;
  color: #d1d5db;
  cursor: pointer;
  margin-left: auto;
}

.formula-toggle:hover {
  background: #374151;
  border-color: #6b7280;
}

.formula-children {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid #374151;
}

.formula-child {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #9ca3af;
  background: #1f2937;
  padding: 2px 8px;
  border-radius: 4px;
}

.formula-child-name {
  color: #a78bfa;
  font-weight: 500;
}

.formula-child-dim {
  color: #6b7280;
}
```

- [ ] **Step 4: Build and verify**

Run:
```bash
cd web/react-explorer && npm run build
```
Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add web/react-explorer/src/api/client.ts web/react-explorer/src/components/FormulaDisplay.tsx web/react-explorer/src/styles.css
git commit -m "feat: add expand/collapse toggle for composed annotation formulas in FormulaDisplay"
```

---

## Implementation Notes

### Order of Execution

Tasks 1-3 (Chunk 1) must be completed first — they fix the fundamental connectivity bug.

Tasks 4-5 (Chunk 2) depend on Task 2 being complete (the collapse transform must produce correct structures for AnnotationFunction to work with).

Tasks 6-7 (Chunk 3) depend on Task 5 (the API needs the updated AnnotationFunction).

### Edge Cases to Watch

1. **Single-node annotations**: An annotation where entry == exit. Formula is just the identity. Should still work.

2. **Deeply nested compositions**: Parent -> child -> grandchild. The recursive `to_sympy(expand=True)` call handles this, but watch for stack depth with very deep nesting.

3. **Multi-output FUNCTION nodes**: When a child annotation has multiple exits, the `output_index` on connections determines which output feeds where. Both `__call__` and `to_sympy` use positional keys `"{fn_node}__out_{idx}"` to store/lookup individual outputs, keyed by the `output_index` on each connection. The `input_output_indices` list on regular-node steps captures this mapping.

4. **Intractable child formulas**: If a child's `to_sympy()` returns `None` (too complex), the parent's expanded form also returns `None`, but the collapsed form (using function names) should still work.

5. **`_build_engine` factoring**: Task 6 refactors `_build_model_state` by extracting `_build_engine` which returns the full `ModelStateEngine` (providing both `.current_state` and `.annotations`). The existing `_build_model_state` becomes a one-liner that delegates.

6. **Annotation ID vs Name resolution**: Evidence routes use synthetic `ann_{seq}` IDs while the `child_annotation_ids` field on composed annotations stores annotation **names** (e.g. "A1678"). The Task 6 formula endpoint bridges this by matching child operations by their `params["name"]`, not by synthetic ID. This is critical — comparing `ann_id in child_ann_names` would always fail.

7. **FUNCTION node input ordering**: When building computation steps for a FUNCTION node in `_build_from_structure`, the input connection order from `subgraph_connections` is arbitrary. Inputs must be reordered to match `meta.input_names` so the child AnnotationFunction receives values mapped to the correct symbols (x_0, x_1, ...).

8. **Exit node derivation after child collapse**: In the formula endpoint (Task 6), after collapsing children, the parent annotation's original exit_nodes reference node IDs that no longer exist (replaced by fn_ nodes). Use `_compute_effective_entries_exits()` from Chunk 1 to derive correct boundary nodes from the partially-collapsed structure.
