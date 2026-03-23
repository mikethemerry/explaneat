"""Tests for AnnotationFunction with NetworkStructure (structure mode)."""

import math

import numpy as np
import pytest

from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.analysis.annotation_function import AnnotationFunction


def make_annotated_network():
    """Network with identity node and standard hidden nodes."""
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(
                id="identity_5",
                type=NodeType.HIDDEN,
                bias=0.0,
                activation="identity",
            ),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.5, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-2", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="5", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=2.0, enabled=True),
            NetworkConnection(
                from_node="-2", to_node="identity_5", weight=0.8, enabled=True
            ),
            NetworkConnection(
                from_node="identity_5", to_node="0", weight=1.0, enabled=True
            ),
        ],
        input_node_ids=["-2", "-1"],
        output_node_ids=["0"],
    )


class TestAnnotationFunctionStructureMode:
    def test_dimensionality(self):
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)
        assert ann_fn.dimensionality == (1, 1)

    def test_identity_to_output_subgraph(self):
        """Annotation: identity_5 -> 0 (sigmoid output)."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)

        # f(0.8) = sigmoid(0.8*1.0 - 0.5) = sigmoid(0.3)
        x = np.array([[0.8]])
        y = ann_fn(x)
        expected = 1.0 / (1.0 + math.exp(-0.3))
        np.testing.assert_almost_equal(y[0, 0], expected)

    def test_input_to_hidden_subgraph(self):
        """Annotation: -2, -1 -> 5 (relu hidden)."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["-2", "-1"],
            "exit_nodes": ["5"],
            "subgraph_nodes": ["-2", "-1", "5"],
            "subgraph_connections": [("-2", "5"), ("-1", "5")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)
        assert ann_fn.dimensionality == (2, 1)

        # f(1.0, 2.0) = relu(1.0*1.0 + 2.0*0.5 + 0.1) = 2.1
        x = np.array([[1.0, 2.0]])
        y = ann_fn(x)
        np.testing.assert_almost_equal(y[0, 0], 2.1)

    def test_identity_activation_in_function(self):
        """Identity nodes should not apply relu or sigmoid."""
        structure = make_annotated_network()
        # Annotation where identity_5 is an internal (non-entry) node
        # But identity_5 only has one input (-2), so make a subgraph
        # that includes -2 -> identity_5 -> 0
        annotation = {
            "entry_nodes": ["-2"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["-2", "identity_5", "0"],
            "subgraph_connections": [("-2", "identity_5"), ("identity_5", "0")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)

        # f(1.0) = sigmoid(identity(1.0*0.8 + 0.0)*1.0 - 0.5)
        #        = sigmoid(0.8 - 0.5) = sigmoid(0.3)
        x = np.array([[1.0]])
        y = ann_fn(x)
        expected = 1.0 / (1.0 + math.exp(-0.3))
        np.testing.assert_almost_equal(y[0, 0], expected)

    def test_single_sample_squeeze(self):
        """Single-sample input (1D) should return 1D output."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)

        x = np.array([0.8])
        y = ann_fn(x)
        assert y.ndim == 1
        assert len(y) == 1

    def test_batch_evaluation(self):
        """Batch evaluation should return (n_samples, n_outputs)."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)

        x = np.array([[0.5], [1.0], [2.0]])
        y = ann_fn(x)
        assert y.shape == (3, 1)

    def test_latex_output(self):
        """Should produce LaTeX for small subgraphs."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }
        ann_fn = AnnotationFunction.from_structure(annotation, structure)
        latex = ann_fn.to_latex()
        # Should produce some LaTeX (depends on sympy availability)
        if latex is not None:
            assert "y_0" in latex

    def test_latex_uses_display_names(self):
        """Sympy symbols should use display_name when set on entry nodes."""
        nodes = [
            NetworkNode(id="-1", type=NodeType.INPUT, display_name="sepalLen"),
            NetworkNode(id="-2", type=NodeType.INPUT, display_name="petalWid"),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ]
        conns = [
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
        ]
        structure = NetworkStructure(
            nodes=nodes, connections=conns,
            input_node_ids=["-1", "-2"], output_node_ids=["0"],
        )
        annotation = {
            "name": "test",
            "hypothesis": "test",
            "entry_nodes": ["-1", "-2"],
            "exit_nodes": ["5"],
            "subgraph_nodes": ["-1", "-2", "5"],
            "subgraph_connections": [("-1", "5"), ("-2", "5")],
        }
        af = AnnotationFunction.from_structure(annotation, structure)
        latex = af.to_latex()
        assert latex is not None
        assert "sepalLen" in latex
        assert "petalWid" in latex
        # Should NOT contain the raw node IDs as symbols
        assert "x_{-1}" not in latex
        assert "x_{-2}" not in latex
