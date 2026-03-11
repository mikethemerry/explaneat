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
    Two-child composed topology:

        in1 --> A --\\
                     --> C --> out1
        in2 --> B --/

    child1: entries=[in1], exits=[A], subgraph=[in1, A]
    child2: entries=[in2], exits=[B], subgraph=[in2, B]
    parent: compositional over child1 + child2 + C

    After collapsing child1 and child2:
    - fn_child1 replaces A
    - fn_child2 replaces B
    - Parent annotation covers [in1, fn_child1, in2, fn_child2, C]
    """
    nodes = [
        _make_node("in1", NodeType.INPUT),
        _make_node("in2", NodeType.INPUT),
        _make_node("A", bias=0.1, activation="relu"),
        _make_node("B", bias=-0.2, activation="relu"),
        _make_node("C", bias=0.3, activation="relu"),
        _make_node("out1", NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        _make_conn("in1", "A", weight=0.5),
        _make_conn("in2", "B", weight=0.8),
        _make_conn("A", "C", weight=1.0),
        _make_conn("B", "C", weight=0.6),
        _make_conn("C", "out1", weight=1.0),
    ]
    structure = NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["in1", "in2"],
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
        entry_nodes=["in2"], exit_nodes=["B"],
        subgraph_nodes=["in2", "B"],
        subgraph_connections=[("in2", "B")],
        parent_annotation_id="parent",
    )
    parent = AnnotationData(
        name="parent", hypothesis="test",
        entry_nodes=["in1", "in2"], exit_nodes=["C"],
        subgraph_nodes=[],
        subgraph_connections=[],
    )
    return structure, [child1, child2, parent]


def _compute_expected_output(in1_val, in2_val):
    """Compute expected output for the composed network manually.

    A = relu(0.5 * in1 + 0.1)
    B = relu(0.8 * in2 - 0.2)
    C = relu(1.0 * A + 0.6 * B + 0.3)
    """
    A = max(0, 0.5 * in1_val + 0.1)
    B = max(0, 0.8 * in2_val - 0.2)
    C = max(0, 1.0 * A + 0.6 * B + 0.3)
    return C


class TestAnnotationFunctionWithFunctionNodes:
    """Test AnnotationFunction on a structure containing collapsed fn_ nodes."""

    def test_expanded_formula_from_uncollapsed_structure(self):
        """Fully-expanded formula from the original (uncollapsed) structure."""
        structure, annotations = _composed_network_and_annotations()
        # Build the parent annotation with full subgraph on the uncollapsed structure
        parent_with_subgraph = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"], exit_nodes=["C"],
            subgraph_nodes=["in1", "in2", "A", "B", "C"],
            subgraph_connections=[("in1", "A"), ("in2", "B"), ("A", "C"), ("B", "C")],
        )
        af = AnnotationFunction.from_structure(parent_with_subgraph, structure)
        # Test with in1=1.0, in2=1.0
        result = af(np.array([1.0, 1.0]))
        expected = _compute_expected_output(1.0, 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_collapsed_formula_references_child_names(self):
        """Collapsed formula should use child function names, not primitives."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        # Build parent annotation referencing fn_ nodes in the collapsed structure
        # Find the connections within the parent subgraph
        subgraph_nodes = {"in1", "fn_child1", "in2", "fn_child2", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        latex = af.to_latex(expand=False)
        assert latex is not None
        # sympy renders function names like "child1" as "\operatorname{child}_{1}"
        # or similar LaTeX forms, so check for the base name "child"
        assert "child" in latex

    def test_expanded_formula_substitutes_child_expressions(self):
        """Expanded formula should inline child expressions to primitives."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        subgraph_nodes = {"in1", "fn_child1", "in2", "fn_child2", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        latex_expanded = af.to_latex(expand=True)
        assert latex_expanded is not None
        # Expanded form should NOT reference child function names.
        # sympy renders "child1" as "\operatorname{child}_{1}" so check
        # for any form of the child function reference.
        assert "operatorname{child" not in latex_expanded

    def test_evaluation_with_function_nodes_matches_expanded(self):
        """Numerical evaluation through fn_ nodes matches the expanded version."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        subgraph_nodes = {"in1", "fn_child1", "in2", "fn_child2", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        result = af(np.array([1.0, 1.0]))
        expected = _compute_expected_output(1.0, 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_to_latex_default_is_expanded(self):
        """Default to_latex() (no expand param) returns expanded form."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        subgraph_nodes = {"in1", "fn_child1", "in2", "fn_child2", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        default_latex = af.to_latex()
        expanded_latex = af.to_latex(expand=True)
        assert default_latex == expanded_latex

    def test_to_sympy_expand_false_uses_function_symbols(self):
        """to_sympy(expand=False) should use sympy Function objects for children."""
        import sympy
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        subgraph_nodes = {"in1", "fn_child1", "in2", "fn_child2", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        exprs = af.to_sympy(expand=False)
        assert exprs is not None
        # The expression should contain applied sympy.Function references
        expr_str = str(exprs["y_0"])
        assert "child1" in expr_str or "child2" in expr_str

    def test_to_sympy_expand_true_no_function_symbols(self):
        """to_sympy(expand=True) should inline all child expressions."""
        structure, annotations = _composed_network_and_annotations()
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )
        subgraph_nodes = {"in1", "fn_child1", "in2", "fn_child2", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        parent_ann = AnnotationData(
            name="parent", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
        exprs = af.to_sympy(expand=True)
        assert exprs is not None
        expr_str = str(exprs["y_0"])
        # Should not contain child function names
        assert "child1" not in expr_str
        assert "child2" not in expr_str
        # Should contain primitive elements (x_0, x_1 are the input symbols)
        assert "x_0" in expr_str or "x_1" in expr_str

    def test_single_child_function_node(self):
        """Test with a single FUNCTION node in the annotation subgraph."""
        structure, annotations = _composed_network_and_annotations()
        # Collapse only child1
        partially_collapsed = collapse_structure(
            structure, annotations, {"child1"}
        )
        # Build annotation that includes fn_child1 + B + C
        subgraph_nodes = {"in1", "fn_child1", "in2", "B", "C"}
        subgraph_conns = [
            (c.from_node, c.to_node)
            for c in partially_collapsed.connections
            if c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
        ]
        ann = AnnotationData(
            name="mixed", hypothesis="test",
            entry_nodes=["in1", "in2"],
            exit_nodes=["C"],
            subgraph_nodes=list(subgraph_nodes),
            subgraph_connections=subgraph_conns,
        )
        af = AnnotationFunction.from_structure(ann, partially_collapsed)

        # Evaluation should match expanded
        result = af(np.array([1.0, 1.0]))
        expected = _compute_expected_output(1.0, 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

        # expand=False latex should reference the child function name
        latex_collapsed = af.to_latex(expand=False)
        assert latex_collapsed is not None
        # sympy renders "child1" as "\operatorname{child}_{1}" in LaTeX
        assert "child" in latex_collapsed

        # expand=True latex should NOT reference the child function name
        latex_expanded = af.to_latex(expand=True)
        assert latex_expanded is not None
        assert "child" not in latex_expanded
