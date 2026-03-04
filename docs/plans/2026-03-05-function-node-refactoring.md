# Function Node Refactoring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace graph-surgery collapse with function-node term rewriting, fixing cycle bugs and unifying collapse logic server-side.

**Architecture:** Collapsed views are derived from the always-expanded canonical `NetworkStructure` via a pure transform function `collapse_structure()`. A new `NodeType.FUNCTION` represents collapsed annotations as multi-input, multi-output function nodes. A shared activation registry replaces hard-coded activation handling.

**Tech Stack:** Python (dataclasses, numpy, sympy, torch), FastAPI, React/TypeScript

**Design Doc:** `docs/plans/2026-03-05-function-node-refactoring-design.md`

---

### Task 1: Activation Registry

**Files:**
- Create: `explaneat/core/activations.py`
- Create: `tests/test_core/test_activations.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_activations.py
"""Tests for the activation registry."""

import numpy as np
import pytest

from explaneat.core.activations import get_numpy_activation, get_sympy_activation, ACTIVATIONS


class TestActivationRegistry:
    """Test that all neat-python activations are registered and correct."""

    def test_registry_has_all_neat_activations(self):
        """All neat-python activation names should be in the registry."""
        expected = {
            "sigmoid", "tanh", "sin", "gauss", "relu", "softplus",
            "identity", "clamped", "inv", "log", "exp", "abs",
            "hat", "square", "cube",
        }
        assert expected.issubset(set(ACTIVATIONS.keys()))

    def test_get_numpy_activation_sigmoid(self):
        fn = get_numpy_activation("sigmoid")
        x = np.array([-1.0, 0.0, 1.0])
        result = fn(x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(result, expected)

    def test_get_numpy_activation_relu(self):
        fn = get_numpy_activation("relu")
        x = np.array([-1.0, 0.0, 1.0])
        result = fn(x)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0])

    def test_get_numpy_activation_identity(self):
        fn = get_numpy_activation("identity")
        x = np.array([-1.0, 0.0, 1.0])
        result = fn(x)
        np.testing.assert_allclose(result, x)

    def test_get_numpy_activation_tanh(self):
        fn = get_numpy_activation("tanh")
        x = np.array([-1.0, 0.0, 1.0])
        result = fn(x)
        np.testing.assert_allclose(result, np.tanh(x))

    def test_get_numpy_activation_unknown_raises(self):
        with pytest.raises(KeyError):
            get_numpy_activation("nonexistent_activation")

    def test_get_sympy_activation_sigmoid(self):
        import sympy
        fn = get_sympy_activation("sigmoid")
        x = sympy.Symbol("x")
        expr = fn(x)
        # Should be equivalent to 1/(1+exp(-x))
        assert expr.subs(x, 0) == sympy.Rational(1, 2)

    def test_get_sympy_activation_relu(self):
        import sympy
        fn = get_sympy_activation("relu")
        x = sympy.Symbol("x")
        expr = fn(x)
        # ReLU(-1) = 0, ReLU(1) = 1
        assert float(expr.subs(x, -1)) == 0.0
        assert float(expr.subs(x, 1)) == 1.0

    def test_get_sympy_activation_identity(self):
        import sympy
        fn = get_sympy_activation("identity")
        x = sympy.Symbol("x")
        expr = fn(x)
        assert expr == x

    def test_numpy_sympy_agree(self):
        """Numpy and sympy implementations should produce the same results."""
        import sympy
        x_sym = sympy.Symbol("x")
        test_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

        for name in ["sigmoid", "tanh", "relu", "identity", "abs", "square", "cube"]:
            np_fn = get_numpy_activation(name)
            sym_fn = get_sympy_activation(name)
            sym_expr = sym_fn(x_sym)

            for val in test_vals:
                np_result = float(np_fn(np.array([val]))[0])
                sym_result = float(sym_expr.subs(x_sym, val))
                assert abs(np_result - sym_result) < 1e-6, (
                    f"{name}({val}): numpy={np_result}, sympy={sym_result}"
                )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_activations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'explaneat.core.activations'`

**Step 3: Write the activation registry**

```python
# explaneat/core/activations.py
"""
Shared activation function registry.

Maps activation name strings (matching neat-python's ActivationFunctionSet)
to numpy and sympy implementations. Used by StructureNetwork and
AnnotationFunction to avoid hard-coded activation handling.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class ActivationPair:
    """Numpy and sympy implementations of an activation function."""
    numpy_fn: Callable[[np.ndarray], np.ndarray]
    sympy_fn: Callable  # sympy.Expr -> sympy.Expr


def _make_sympy_fns():
    """Build sympy activation functions (lazy to avoid import at module level)."""
    import sympy

    return {
        "sigmoid": lambda x: 1 / (1 + sympy.exp(-x)),
        "tanh": lambda x: sympy.tanh(x),
        "sin": lambda x: sympy.sin(x),
        "gauss": lambda x: sympy.exp(-x**2),
        "relu": lambda x: sympy.Piecewise((x, x > 0), (0, True)),
        "softplus": lambda x: sympy.log(1 + sympy.exp(x)),
        "identity": lambda x: x,
        "clamped": lambda x: sympy.Piecewise(
            (sympy.Rational(-1), x < -1), (x, (x >= -1) & (x <= 1)), (sympy.Integer(1), True)
        ),
        "inv": lambda x: sympy.Piecewise((0, sympy.Eq(x, 0)), (1 / x, True)),
        "log": lambda x: sympy.Piecewise(
            (0, x <= 0), (sympy.log(x), True)
        ),
        "exp": lambda x: sympy.exp(sympy.Piecewise((x, x < 60), (60, True))),
        "abs": lambda x: sympy.Abs(x),
        "hat": lambda x: sympy.Piecewise(
            (0, x < 0), (x, (x >= 0) & (x < 1)), (2 - x, (x >= 1) & (x < 2)), (0, True)
        ),
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }


_NUMPY_ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))),
    "tanh": lambda x: np.tanh(x),
    "sin": lambda x: np.sin(x),
    "gauss": lambda x: np.exp(-x**2),
    "relu": lambda x: np.maximum(0, x),
    "softplus": lambda x: np.log1p(np.exp(np.clip(x, -60, 60))),
    "identity": lambda x: x,
    "clamped": lambda x: np.clip(x, -1.0, 1.0),
    "inv": lambda x: np.where(np.abs(x) < 1e-12, 0.0, 1.0 / x),
    "log": lambda x: np.where(x > 0, np.log(x), 0.0),
    "exp": lambda x: np.exp(np.clip(x, -60, 60)),
    "abs": lambda x: np.abs(x),
    "hat": lambda x: np.clip(1.0 - np.abs(x - 1.0), 0.0, 1.0) * (x >= 0) * (x < 2),
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
}

# Lazy-initialized sympy functions
_SYMPY_ACTIVATIONS = None


def get_numpy_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Get the numpy implementation of an activation function by name."""
    if name not in _NUMPY_ACTIVATIONS:
        raise KeyError(f"Unknown activation function: {name!r}. Available: {sorted(_NUMPY_ACTIVATIONS.keys())}")
    return _NUMPY_ACTIVATIONS[name]


def get_sympy_activation(name: str) -> Callable:
    """Get the sympy implementation of an activation function by name."""
    global _SYMPY_ACTIVATIONS
    if _SYMPY_ACTIVATIONS is None:
        _SYMPY_ACTIVATIONS = _make_sympy_fns()
    if name not in _SYMPY_ACTIVATIONS:
        raise KeyError(f"Unknown activation function: {name!r}. Available: {sorted(_SYMPY_ACTIVATIONS.keys())}")
    return _SYMPY_ACTIVATIONS[name]


# Public registry for introspection
ACTIVATIONS = set(_NUMPY_ACTIVATIONS.keys())
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_activations.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add explaneat/core/activations.py tests/test_core/test_activations.py
git commit -m "feat: add shared activation function registry for numpy and sympy"
```

---

### Task 2: Data Model Changes (NodeType.FUNCTION, FunctionNodeMetadata, output_index)

**Files:**
- Modify: `explaneat/core/genome_network.py:13-45`
- Create: `tests/test_core/test_genome_network_function_node.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_genome_network_function_node.py
"""Tests for function node data model additions."""

import pytest
from explaneat.core.genome_network import (
    NodeType,
    NetworkNode,
    NetworkConnection,
    NetworkStructure,
    FunctionNodeMetadata,
)


class TestFunctionNodeDataModel:
    """Test the FUNCTION node type and related data structures."""

    def test_node_type_has_function(self):
        assert NodeType.FUNCTION == "function"
        assert NodeType.FUNCTION.value == "function"

    def test_function_node_metadata_creation(self):
        meta = FunctionNodeMetadata(
            annotation_name="drug_interaction",
            annotation_id="ann_123",
            hypothesis="Additive effect of drugs A and B",
            n_inputs=2,
            n_outputs=1,
            input_names=["5", "6"],
            output_names=["10"],
            formula_latex="y_0 = \\sigma(w_1 x_0 + w_2 x_1)",
            subgraph_nodes=["5", "6", "7", "10"],
            subgraph_connections=[("5", "7"), ("6", "7"), ("7", "10")],
        )
        assert meta.n_inputs == 2
        assert meta.n_outputs == 1
        assert meta.hypothesis == "Additive effect of drugs A and B"

    def test_network_node_with_function_metadata(self):
        meta = FunctionNodeMetadata(
            annotation_name="F",
            annotation_id="ann_1",
            hypothesis="test",
            n_inputs=2,
            n_outputs=1,
            input_names=["a", "b"],
            output_names=["x"],
            formula_latex=None,
            subgraph_nodes=["a", "b", "c", "x"],
            subgraph_connections=[("a", "c"), ("b", "c"), ("c", "x")],
        )
        node = NetworkNode(
            id="F_ann_1",
            type=NodeType.FUNCTION,
            function_metadata=meta,
        )
        assert node.type == NodeType.FUNCTION
        assert node.function_metadata.annotation_name == "F"
        assert node.bias is None  # function nodes don't use bias directly

    def test_network_node_without_function_metadata(self):
        """Regular nodes should still work with function_metadata=None."""
        node = NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="relu")
        assert node.function_metadata is None

    def test_connection_with_output_index(self):
        conn = NetworkConnection(
            from_node="F_ann",
            to_node="downstream",
            weight=1.0,
            enabled=True,
            output_index=0,
        )
        assert conn.output_index == 0

    def test_connection_without_output_index(self):
        """Regular connections should still work with output_index=None."""
        conn = NetworkConnection(
            from_node="5",
            to_node="6",
            weight=0.5,
            enabled=True,
        )
        assert conn.output_index is None

    def test_network_structure_validates_with_function_nodes(self):
        meta = FunctionNodeMetadata(
            annotation_name="F",
            annotation_id="ann_1",
            hypothesis="test",
            n_inputs=1,
            n_outputs=1,
            input_names=["a"],
            output_names=["x"],
            formula_latex=None,
            subgraph_nodes=["a", "mid", "x"],
            subgraph_connections=[("a", "mid"), ("mid", "x")],
        )
        structure = NetworkStructure(
            nodes=[
                NetworkNode(id="-1", type=NodeType.INPUT),
                NetworkNode(id="a", type=NodeType.HIDDEN),
                NetworkNode(id="F_ann", type=NodeType.FUNCTION, function_metadata=meta),
                NetworkNode(id="0", type=NodeType.OUTPUT),
            ],
            connections=[
                NetworkConnection(from_node="-1", to_node="a", weight=1.0, enabled=True),
                NetworkConnection(from_node="a", to_node="F_ann", weight=1.0, enabled=True),
                NetworkConnection(from_node="F_ann", to_node="0", weight=1.0, enabled=True, output_index=0),
            ],
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        result = structure.validate()
        assert result["is_valid"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_genome_network_function_node.py -v`
Expected: FAIL — `ImportError: cannot import name 'FunctionNodeMetadata'`

**Step 3: Update the data model**

In `explaneat/core/genome_network.py`:

Add `FUNCTION = "function"` to the `NodeType` enum (after line 17).

Add `FunctionNodeMetadata` dataclass (before `NetworkNode`).

Add `function_metadata: Optional[FunctionNodeMetadata] = None` to `NetworkNode`.

Add `output_index: Optional[int] = None` to `NetworkConnection`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_genome_network_function_node.py -v`
Expected: All PASS

**Step 5: Run existing tests to check for regressions**

Run: `uv run pytest tests/test_core/ -v`
Expected: All PASS — existing code doesn't use FUNCTION or output_index, so no breakage.

**Step 6: Commit**

```bash
git add explaneat/core/genome_network.py tests/test_core/test_genome_network_function_node.py
git commit -m "feat: add NodeType.FUNCTION, FunctionNodeMetadata, and output_index to data model"
```

---

### Task 3: Collapse Transform

**Files:**
- Create: `explaneat/core/collapse_transform.py`
- Create: `tests/test_core/test_collapse_transform.py`

**Depends on:** Tasks 1, 2

**Step 1: Write the failing tests**

```python
# tests/test_core/test_collapse_transform.py
"""Tests for the collapse_structure transform."""

import pytest
from explaneat.core.genome_network import (
    NetworkStructure, NetworkNode, NetworkConnection, NodeType, FunctionNodeMetadata,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.collapse_transform import collapse_structure


def _make_linear_network():
    """Build: IN(-1) -> A(5) -> B(6) -> OUT(0). Simple chain."""
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="6", weight=0.5, enabled=True),
            NetworkConnection(from_node="6", to_node="0", weight=0.8, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _make_diamond_network():
    """Build: IN(-1) -> A(5) -> C(7) -> OUT(0)
                    \\-> B(6) -> C(7)
    Diamond with fan-out at input, fan-in at C."""
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="7", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="6", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="7", weight=0.5, enabled=True),
            NetworkConnection(from_node="6", to_node="7", weight=0.3, enabled=True),
            NetworkConnection(from_node="7", to_node="0", weight=0.8, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _make_multi_exit_network():
    """Build: IN(-1) -> A(5) -> B(6) -> OUT(0)
                             \\-> C(7) -> OUT(1)
    Annotation with 1 entry (5), 2 exits (6, 7)."""
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="7", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            NetworkNode(id="1", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="6", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="7", weight=0.3, enabled=True),
            NetworkConnection(from_node="6", to_node="0", weight=0.8, enabled=True),
            NetworkConnection(from_node="7", to_node="1", weight=0.9, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0", "1"],
    )


class TestCollapseStructureBasic:
    """Test basic collapse behavior."""

    def test_collapse_linear_annotation(self):
        """Collapse {5, 6} in linear chain. Entry=5, exit=6."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test function",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})

        # Should have: IN(-1), entry(5), F_node, OUT(0)
        node_ids = {n.id for n in result.nodes}
        assert "-1" in node_ids  # input preserved
        assert "5" in node_ids   # entry preserved
        assert "0" in node_ids   # output preserved
        assert "6" not in node_ids  # exit (internal) removed

        # Should have exactly one FUNCTION node
        fn_nodes = [n for n in result.nodes if n.type == NodeType.FUNCTION]
        assert len(fn_nodes) == 1
        fn_node = fn_nodes[0]
        assert fn_node.function_metadata.annotation_name == "F"
        assert fn_node.function_metadata.n_inputs == 1
        assert fn_node.function_metadata.n_outputs == 1
        assert fn_node.function_metadata.hypothesis == "test function"

    def test_collapse_preserves_entry_nodes(self):
        """Entry nodes must remain in the collapsed graph."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})
        node_ids = {n.id for n in result.nodes}
        assert "5" in node_ids

    def test_collapse_multi_exit_annotation(self):
        """Collapse annotation with 2 exits uses output_index."""
        structure = _make_multi_exit_network()
        ann = AnnotationData(
            name="G",
            hypothesis="two outputs",
            entry_nodes=["5"],
            exit_nodes=["6", "7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("5", "7")],
        )
        result = collapse_structure(structure, [ann], {"G"})

        fn_nodes = [n for n in result.nodes if n.type == NodeType.FUNCTION]
        assert len(fn_nodes) == 1
        assert fn_nodes[0].function_metadata.n_outputs == 2

        # Connections from function node should have output_index set
        fn_id = fn_nodes[0].id
        fn_conns = [c for c in result.connections if c.from_node == fn_id]
        assert len(fn_conns) == 2
        indices = {c.output_index for c in fn_conns}
        assert indices == {0, 1}

    def test_collapse_empty_set_returns_unchanged(self):
        """Collapsing no annotations returns the structure unchanged."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], set())
        assert len(result.nodes) == len(structure.nodes)
        assert len(result.connections) == len(structure.connections)


class TestCollapseStructureCycleFreedom:
    """Test that collapse never introduces cycles."""

    def test_no_cycles_linear(self):
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})
        assert not _has_cycle(result)

    def test_no_cycles_diamond(self):
        structure = _make_diamond_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5", "6"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "7"), ("6", "7")],
        )
        result = collapse_structure(structure, [ann], {"F"})
        assert not _has_cycle(result)

    def test_no_cycles_multi_exit(self):
        structure = _make_multi_exit_network()
        ann = AnnotationData(
            name="G",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6", "7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("5", "7")],
        )
        result = collapse_structure(structure, [ann], {"G"})
        assert not _has_cycle(result)


class TestCollapseStructureConnections:
    """Test that connections are correctly rewired."""

    def test_entry_to_function_node_connection(self):
        """Entry nodes that fed internal nodes should connect to function node."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})
        fn_nodes = [n for n in result.nodes if n.type == NodeType.FUNCTION]
        fn_id = fn_nodes[0].id

        # Should have connection from entry(5) to function node
        entry_to_fn = [c for c in result.connections if c.from_node == "5" and c.to_node == fn_id]
        assert len(entry_to_fn) == 1

    def test_function_node_to_downstream_connection(self):
        """Function node should connect to nodes that exit nodes fed."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})
        fn_nodes = [n for n in result.nodes if n.type == NodeType.FUNCTION]
        fn_id = fn_nodes[0].id

        # Should have connection from function node to output(0)
        fn_to_out = [c for c in result.connections if c.from_node == fn_id and c.to_node == "0"]
        assert len(fn_to_out) == 1
        assert fn_to_out[0].weight == 0.8  # original exit->output weight

    def test_external_connections_preserved(self):
        """Connections not involving the annotation should be unchanged."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})

        # Input -> entry connection should be preserved
        in_to_entry = [c for c in result.connections if c.from_node == "-1" and c.to_node == "5"]
        assert len(in_to_entry) == 1
        assert in_to_entry[0].weight == 1.0

    def test_internal_connections_removed(self):
        """Connections between internal nodes should be removed."""
        structure = _make_linear_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["6"],
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        result = collapse_structure(structure, [ann], {"F"})

        # Original 5->6 connection should be gone
        internal = [c for c in result.connections if c.from_node == "5" and c.to_node == "6"]
        assert len(internal) == 0


def _has_cycle(structure: NetworkStructure) -> bool:
    """Check if a NetworkStructure has cycles (DFS-based)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    nodes = {n.id for n in structure.nodes}
    color = {n: WHITE for n in nodes}
    adj = {n: [] for n in nodes}
    for c in structure.connections:
        if c.enabled and c.from_node in adj:
            adj[c.from_node].append(c.to_node)

    def dfs(node):
        color[node] = GRAY
        for nb in adj.get(node, []):
            if nb not in color:
                continue
            if color[nb] == GRAY:
                return True
            if color[nb] == WHITE and dfs(nb):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in nodes)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_collapse_transform.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'explaneat.core.collapse_transform'`

**Step 3: Implement the collapse transform**

```python
# explaneat/core/collapse_transform.py
"""
Pure collapse transform: produces derived NetworkStructure views
with collapsed annotations replaced by function nodes.

This is the core of the function-node model — it performs term rewriting
on the DAG, replacing annotation subgraphs with named multi-input,
multi-output function nodes. This is provably cycle-free because it
substitutes subexpressions with named functions, preserving topological order.

See docs/theoretical_framework.md for the mathematical treatment.
See docs/plans/2026-03-05-function-node-refactoring-design.md for the design.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from .genome_network import (
    FunctionNodeMetadata,
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from ..core.model_state import AnnotationData


def collapse_structure(
    structure: NetworkStructure,
    annotations: List[AnnotationData],
    collapsed_ids: Set[str],
) -> NetworkStructure:
    """
    Produce a derived NetworkStructure with collapsed annotations
    replaced by function nodes.

    Args:
        structure: The fully-expanded model state (not mutated)
        annotations: All annotations from ModelStateEngine
        collapsed_ids: Which annotation names to collapse

    Returns:
        New NetworkStructure with FUNCTION nodes replacing collapsed subgraphs
    """
    if not collapsed_ids:
        return deepcopy(structure)

    # Build annotation lookup by name
    ann_by_name: Dict[str, AnnotationData] = {a.name: a for a in annotations}

    # Filter to requested annotations that actually exist
    to_collapse = [
        ann_by_name[name] for name in collapsed_ids if name in ann_by_name
    ]
    if not to_collapse:
        return deepcopy(structure)

    # Order: children before parents (leaf annotations first)
    # For now, sort by subgraph size (smaller = more likely to be children)
    to_collapse.sort(key=lambda a: len(a.subgraph_nodes))

    # Work on a copy
    nodes = list(structure.nodes)
    connections = list(structure.connections)
    input_node_ids = list(structure.input_node_ids)
    output_node_ids = list(structure.output_node_ids)

    for ann in to_collapse:
        nodes, connections, input_node_ids, output_node_ids = _collapse_one(
            nodes, connections, input_node_ids, output_node_ids, ann, structure,
        )

    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=input_node_ids,
        output_node_ids=output_node_ids,
        metadata={**structure.metadata, "collapsed_annotations": list(collapsed_ids)},
    )


def _collapse_one(
    nodes: List[NetworkNode],
    connections: List[NetworkConnection],
    input_node_ids: List[str],
    output_node_ids: List[str],
    ann: AnnotationData,
    original_structure: NetworkStructure,
) -> Tuple[List[NetworkNode], List[NetworkConnection], List[str], List[str]]:
    """Collapse a single annotation into a function node."""
    entry_set = set(ann.entry_nodes)
    exit_set = set(ann.exit_nodes)
    subgraph_set = set(ann.subgraph_nodes)
    internal_nodes = subgraph_set - entry_set  # intermediate + exit nodes

    # Generate function node ID
    fn_node_id = f"fn_{ann.name}"

    # Try to compute LaTeX formula (best-effort)
    formula_latex = _compute_formula(ann, original_structure)

    # Create function node
    fn_metadata = FunctionNodeMetadata(
        annotation_name=ann.name,
        annotation_id=ann.name,  # using name as ID for now
        hypothesis=ann.hypothesis,
        n_inputs=len(ann.entry_nodes),
        n_outputs=len(ann.exit_nodes),
        input_names=list(ann.entry_nodes),
        output_names=list(ann.exit_nodes),
        formula_latex=formula_latex,
        subgraph_nodes=list(ann.subgraph_nodes),
        subgraph_connections=list(ann.subgraph_connections),
    )
    fn_node = NetworkNode(
        id=fn_node_id,
        type=NodeType.FUNCTION,
        function_metadata=fn_metadata,
    )

    # Build new node list: remove internal nodes, add function node
    new_nodes = [n for n in nodes if n.id not in internal_nodes]
    new_nodes.append(fn_node)

    # Build new connection list
    new_connections = []
    seen_edges: Set[Tuple[str, str, Optional[int]]] = set()

    # Map exit nodes to their output index
    exit_to_index = {exit_id: i for i, exit_id in enumerate(ann.exit_nodes)}

    for conn in connections:
        from_in = conn.from_node in subgraph_set
        to_in = conn.to_node in subgraph_set
        from_internal = conn.from_node in internal_nodes
        to_internal = conn.to_node in internal_nodes

        if from_internal and to_internal:
            # Internal connection — drop
            continue
        elif from_internal and not to_in:
            # Exit -> external: reroute from function node
            # Only if from_node is an exit node
            if conn.from_node in exit_set:
                out_idx = exit_to_index[conn.from_node]
                edge_key = (fn_node_id, conn.to_node, out_idx)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    new_connections.append(NetworkConnection(
                        from_node=fn_node_id,
                        to_node=conn.to_node,
                        weight=conn.weight,
                        enabled=conn.enabled,
                        output_index=out_idx,
                    ))
        elif not from_in and to_internal:
            # External -> internal: should only happen for entry nodes
            # (preconditions guarantee this), but entry nodes aren't internal
            # So this case shouldn't occur if preconditions hold.
            # If it does (entry -> exit within subgraph), drop it.
            continue
        elif from_in and not from_internal and to_internal:
            # Entry -> internal: reroute to function node
            edge_key = (conn.from_node, fn_node_id, None)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                new_connections.append(NetworkConnection(
                    from_node=conn.from_node,
                    to_node=fn_node_id,
                    weight=conn.weight,
                    enabled=conn.enabled,
                ))
        else:
            # Not involving internal nodes — preserve
            new_connections.append(NetworkConnection(
                from_node=conn.from_node,
                to_node=conn.to_node,
                weight=conn.weight,
                enabled=conn.enabled,
                innovation=conn.innovation,
                output_index=conn.output_index,
            ))

    # Update input/output node IDs (internal nodes that were inputs/outputs get removed)
    new_input_ids = [nid for nid in input_node_ids if nid not in internal_nodes]
    new_output_ids = [nid for nid in output_node_ids if nid not in internal_nodes]

    return new_nodes, new_connections, new_input_ids, new_output_ids


def _compute_formula(
    ann: AnnotationData, structure: NetworkStructure
) -> Optional[str]:
    """Best-effort LaTeX formula extraction. Returns None on failure."""
    try:
        from ..analysis.annotation_function import AnnotationFunction
        af = AnnotationFunction.from_structure(ann, structure)
        return af.to_latex()
    except Exception:
        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_collapse_transform.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add explaneat/core/collapse_transform.py tests/test_core/test_collapse_transform.py
git commit -m "feat: add collapse_structure transform producing function nodes from annotations"
```

---

### Task 4: Wire Activation Registry into AnnotationFunction

**Files:**
- Modify: `explaneat/analysis/annotation_function.py:127-134,295-300,349-354`
- Test: `tests/test_analysis/test_annotation_function.py` (existing tests should still pass)

**Step 1: Run existing tests as baseline**

Run: `uv run pytest tests/test_analysis/test_annotation_function.py -v`
Expected: All PASS (establish baseline)

**Step 2: Update `_build_from_structure` to use registry**

In `explaneat/analysis/annotation_function.py`, replace the activation assignment block (lines ~127-134):

```python
# OLD:
if node_obj and node_obj.activation == "identity":
    activation = "identity"
elif is_output_node:
    activation = "sigmoid"
else:
    activation = "relu"

# NEW:
if node_obj and node_obj.activation:
    activation = node_obj.activation
elif is_output_node:
    activation = "sigmoid"
else:
    activation = "relu"
```

This preserves existing behavior (identity/sigmoid/relu) but now also passes through any activation the node actually has (tanh, abs, etc.).

**Step 3: Update `__call__` to use registry**

Replace the activation dispatch (lines ~295-300):

```python
# OLD:
if step["activation"] == "relu":
    activations[step["node"]] = np.maximum(0, z)
elif step["activation"] == "identity":
    activations[step["node"]] = z
else:  # sigmoid
    activations[step["node"]] = 1.0 / (1.0 + np.exp(-z))

# NEW:
from ..core.activations import get_numpy_activation
act_fn = get_numpy_activation(step["activation"])
activations[step["node"]] = act_fn(z)
```

**Step 4: Update `to_sympy` to use registry**

Replace the activation dispatch (lines ~349-354):

```python
# OLD:
if step["activation"] == "relu":
    expr = sympy.Piecewise((expr, expr > 0), (0, True))
elif step["activation"] == "identity":
    pass
else:  # sigmoid
    expr = 1 / (1 + sympy.exp(-expr))

# NEW:
from ..core.activations import get_sympy_activation
act_sym_fn = get_sympy_activation(step["activation"])
expr = act_sym_fn(expr)
```

**Step 5: Run tests to verify no regression**

Run: `uv run pytest tests/test_analysis/test_annotation_function.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add explaneat/analysis/annotation_function.py
git commit -m "refactor: use activation registry in AnnotationFunction"
```

---

### Task 5: Wire Activation Registry into StructureNetwork

**Files:**
- Modify: `explaneat/core/structure_network.py:130-137,254-260`
- Test: `tests/test_core/test_structure_network.py` (existing tests should still pass)

**Step 1: Run existing tests as baseline**

Run: `uv run pytest tests/test_core/test_structure_network.py -v`
Expected: All PASS

**Step 2: Update `_build` activation assignment (lines ~130-137)**

```python
# OLD:
if nid in input_ids:
    act = "input"
elif nid in output_ids:
    act = "sigmoid"
elif node_obj and node_obj.activation == "identity":
    act = "identity"
else:
    act = "relu"

# NEW:
if nid in input_ids:
    act = "input"
elif node_obj and node_obj.activation:
    act = node_obj.activation
elif nid in output_ids:
    act = "sigmoid"
else:
    act = "relu"
```

**Step 3: Update `forward` activation dispatch (lines ~254-260)**

```python
# OLD:
if act == "relu":
    output[:, j] = torch.relu(z[:, j])
elif act == "sigmoid":
    output[:, j] = torch.sigmoid(z[:, j])
else:  # identity / input / unknown
    output[:, j] = z[:, j]

# NEW:
if act == "input":
    output[:, j] = z[:, j]
else:
    from .activations import get_numpy_activation
    np_fn = get_numpy_activation(act)
    output[:, j] = torch.from_numpy(
        np_fn(z[:, j].detach().numpy())
    ).to(z.dtype)
```

Note: `StructureNetwork` uses torch tensors, so we convert numpy↔torch. An alternative is adding a `get_torch_activation` to the registry; defer that to a follow-up if perf matters.

**Step 4: Run tests to verify no regression**

Run: `uv run pytest tests/test_core/test_structure_network.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add explaneat/core/structure_network.py
git commit -m "refactor: use activation registry in StructureNetwork"
```

---

### Task 6: API — Add `collapsed` Parameter to Model-State Endpoint

**Files:**
- Modify: `explaneat/api/routes/operations.py:159-175`
- Modify: `explaneat/api/schemas.py:15-52`

**Step 1: Update schemas to support function nodes**

In `explaneat/api/schemas.py`, update `NodeSchema` to accept `"function"` type and add optional function metadata:

```python
class FunctionNodeMetadataSchema(BaseModel):
    """Schema for function node metadata."""
    annotation_name: str
    annotation_id: str
    hypothesis: str
    n_inputs: int
    n_outputs: int
    input_names: List[str]
    output_names: List[str]
    formula_latex: Optional[str] = None

class NodeSchema(BaseModel):
    id: str
    type: Literal["input", "hidden", "output", "identity", "function"]
    bias: Optional[float] = None
    activation: Optional[str] = None
    response: Optional[float] = None
    aggregation: Optional[str] = None
    function_metadata: Optional[FunctionNodeMetadataSchema] = None
    model_config = ConfigDict(from_attributes=True)

class ConnectionSchema(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    weight: float
    enabled: bool = True
    output_index: Optional[int] = None
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class ModelMetadata(BaseModel):
    input_nodes: List[str]
    output_nodes: List[str]
    is_original: bool = True
    collapsed_annotations: List[str] = []
```

**Step 2: Update the model-state endpoint**

In `explaneat/api/routes/operations.py`, update `get_current_model` (line 159):

```python
@router.get("/model", response_model=ModelStateResponse)
async def get_current_model(
    genome_id: UUID = Path(..., description="The genome ID"),
    collapsed: Optional[str] = Query(None, description="Comma-separated annotation names to collapse"),
    db: Session = Depends(get_db),
):
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)
    current_state = engine.current_state
    is_original = len(engine.operations) == 0

    # Apply collapse transform if requested
    if collapsed:
        from ...core.collapse_transform import collapse_structure
        collapsed_ids = {name.strip() for name in collapsed.split(",") if name.strip()}
        current_state = collapse_structure(current_state, engine.annotations, collapsed_ids)

    return _network_to_response(current_state, is_original=is_original)
```

**Step 3: Update `_network_to_response` to handle function nodes**

In `explaneat/api/routes/operations.py`, update `_network_to_response` (line 104):

```python
def _network_to_response(network, is_original: bool = False) -> ModelStateResponse:
    nodes = []
    for node in network.nodes:
        schema_kwargs = dict(
            id=node.id,
            type=node.type.value if hasattr(node.type, "value") else node.type,
            bias=node.bias,
            activation=node.activation,
            response=node.response,
            aggregation=node.aggregation,
        )
        if node.function_metadata:
            schema_kwargs["function_metadata"] = FunctionNodeMetadataSchema(
                annotation_name=node.function_metadata.annotation_name,
                annotation_id=node.function_metadata.annotation_id,
                hypothesis=node.function_metadata.hypothesis,
                n_inputs=node.function_metadata.n_inputs,
                n_outputs=node.function_metadata.n_outputs,
                input_names=node.function_metadata.input_names,
                output_names=node.function_metadata.output_names,
                formula_latex=node.function_metadata.formula_latex,
            )
        nodes.append(NodeSchema(**schema_kwargs))

    connections = [
        ConnectionSchema(
            **{
                "from": conn.from_node,
                "to": conn.to_node,
                "weight": conn.weight,
                "enabled": conn.enabled,
                "output_index": conn.output_index,
            }
        )
        for conn in network.connections
    ]

    metadata = ModelMetadata(
        input_nodes=network.input_node_ids,
        output_nodes=network.output_node_ids,
        is_original=is_original or network.metadata.get("is_original", True),
        collapsed_annotations=network.metadata.get("collapsed_annotations", []),
    )

    return ModelStateResponse(nodes=nodes, connections=connections, metadata=metadata)
```

**Step 4: Run the API and smoke test**

Run: `uv run python -m explaneat api` and verify:
- `GET /api/genomes/{id}/operations/model` still works without `collapsed` param
- `GET /api/genomes/{id}/operations/model?collapsed=ann1` returns function nodes (if annotations exist)

**Step 5: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/operations.py
git commit -m "feat: add collapsed parameter to model-state endpoint, serialize function nodes"
```

---

### Task 7: Frontend — Remove useCollapsedView, Consume Server-Computed Views

**Files:**
- Delete: `web/react-explorer/src/hooks/useCollapsedView.ts`
- Modify: `web/react-explorer/src/components/GenomeExplorer.tsx` (remove useCollapsedView usage)
- Modify: `web/react-explorer/src/components/OperationsPanel.tsx` (collapse toggles re-fetch)
- Modify: `web/react-explorer/src/api/client.ts` (add collapsed param to fetch)

**Note:** This task is frontend-only. The exact changes depend on how the React components currently import and use `useCollapsedView`. The key principle:

1. **Remove** the `useCollapsedView` import and all its usage
2. **Add** a `collapsedAnnotations: Set<string>` state to `GenomeExplorer`
3. **Pass** the collapsed set as a query parameter when fetching model state
4. **Render** function nodes with a distinct visual (e.g., rounded rectangle, different color, show annotation name)
5. The `NetworkViewer` already receives nodes/connections — it just needs to handle `type: "function"` for styling

**Step 1: Update API client**

Add `collapsed` parameter to the model-state fetch function in `client.ts`.

**Step 2: Update GenomeExplorer**

Replace `useCollapsedView` with `collapsedAnnotations` state. When toggling collapse, re-fetch model state with `?collapsed=name1,name2`.

**Step 3: Update OperationsPanel**

Collapse/expand buttons call a callback that updates `collapsedAnnotations` state in the parent, triggering re-fetch.

**Step 4: Remove useCollapsedView.ts**

Delete the file.

**Step 5: Build and test**

Run: `cd web/react-explorer && npm run build`
Expected: Build succeeds with no TypeScript errors.

**Step 6: Commit**

```bash
git add -A web/react-explorer/src/
git commit -m "refactor: remove client-side collapse, use server-computed collapsed views"
```

---

### Task 8: Integration Test — Collapse + Forward Pass Equivalence

**Files:**
- Create: `tests/test_core/test_collapse_integration.py`

**Step 1: Write integration test**

```python
# tests/test_core/test_collapse_integration.py
"""Integration test: collapsed forward pass matches expanded forward pass."""

import numpy as np
import pytest

from explaneat.core.genome_network import (
    NetworkStructure, NetworkNode, NetworkConnection, NodeType,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.collapse_transform import collapse_structure
from explaneat.core.structure_network import StructureNetwork


def _make_annotated_network():
    """Network with annotation {5,6,7}: entry=5, exit=7.
    IN(-1) ->w=1.0-> 5 ->w=0.5-> 6 ->w=0.8-> 7 ->w=1.0-> OUT(0)
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="7", type=NodeType.HIDDEN, bias=-0.2, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="6", weight=0.5, enabled=True),
            NetworkConnection(from_node="6", to_node="7", weight=0.8, enabled=True),
            NetworkConnection(from_node="7", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


class TestCollapseForwardPassEquivalence:
    """The collapsed network should compute the same function as the expanded one."""

    def test_expanded_and_collapsed_produce_same_output(self):
        structure = _make_annotated_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("6", "7")],
        )

        # Forward pass on expanded network
        expanded_net = StructureNetwork(structure)
        x = np.array([[0.5], [1.0], [-0.3]])
        import torch
        x_torch = torch.tensor(x, dtype=torch.float32)
        expanded_out = expanded_net.forward(x_torch).detach().numpy()

        # Forward pass on collapsed network
        collapsed = collapse_structure(structure, [ann], {"F"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x_torch).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-5)
```

**Note:** This test will only pass once `StructureNetwork` supports `FUNCTION` nodes (Task 5 does activation registry; a follow-up sub-task within Task 8 adds FUNCTION node handling to `StructureNetwork.forward`). If this test fails after Task 5, it indicates we need to add FUNCTION node forward-pass logic to StructureNetwork — write that code, then re-run.

**Step 2: Run test**

Run: `uv run pytest tests/test_core/test_collapse_integration.py -v`
Expected: May fail initially if StructureNetwork doesn't handle FUNCTION nodes yet. Implement the handling, then re-run until PASS.

**Step 3: Commit**

```bash
git add tests/test_core/test_collapse_integration.py explaneat/core/structure_network.py
git commit -m "feat: StructureNetwork handles FUNCTION nodes, integration test passes"
```

---

### Task 9: Deprecate CollapseValidator.collapse() and expand()

**Files:**
- Modify: `explaneat/analysis/collapse_validator.py:275-408`

**Step 1: Add deprecation warnings**

Add `warnings.warn("Use collapse_transform.collapse_structure instead", DeprecationWarning)` to the top of `CollapseValidator.collapse()` and `CollapseValidator.expand()`.

Do NOT delete them yet — existing tests reference them, and we want to verify the new path works first in production.

**Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: All PASS (with deprecation warnings visible)

**Step 3: Commit**

```bash
git add explaneat/analysis/collapse_validator.py
git commit -m "deprecate: mark CollapseValidator.collapse/expand as deprecated in favor of collapse_transform"
```
