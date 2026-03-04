"""Integration test: collapsed forward pass matches expanded forward pass.

Verifies that replacing an annotation subgraph with a FUNCTION node
produces identical outputs to running the full expanded network.
"""

import numpy as np
import torch
import pytest

from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.collapse_transform import collapse_structure
from explaneat.core.structure_network import StructureNetwork


def _make_annotated_network():
    """Network with annotation {5,6,7}: entry=5, exit=7.

    IN(-1) ->w=1.0-> 5 ->w=0.5-> 6 ->w=0.8-> 7 ->w=1.0-> OUT(0)

    Node 5: relu, bias=0.0
    Node 6: relu, bias=0.1
    Node 7: relu, bias=-0.2
    Node 0: sigmoid, bias=0.0
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


def _make_diamond_annotated_network():
    """Network with diamond annotation: entry=5, exit=8.

    IN(-1) ->w=1.0-> 5 ->w=0.5-> 6 ->w=0.8-> 8 ->w=1.0-> OUT(0)
                        ->w=0.3-> 7 ->w=0.6->

    Node 5: relu, bias=0.0
    Node 6: relu, bias=0.1
    Node 7: relu, bias=0.2
    Node 8: relu, bias=-0.1
    Node 0: sigmoid, bias=0.0
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="7", type=NodeType.HIDDEN, bias=0.2, activation="relu"),
            NetworkNode(id="8", type=NodeType.HIDDEN, bias=-0.1, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="6", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="7", weight=0.3, enabled=True),
            NetworkConnection(from_node="6", to_node="8", weight=0.8, enabled=True),
            NetworkConnection(from_node="7", to_node="8", weight=0.6, enabled=True),
            NetworkConnection(from_node="8", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _make_multi_exit_annotated_network():
    """Network with multi-exit annotation: entry=5, exits={6,7}.

    IN(-1) ->w=1.0-> 5 ->w=0.5-> 6 ->w=0.8-> OUT(0)
                        ->w=0.3-> 7 ->w=0.6-> OUT(1)

    Node 5: relu, bias=0.0
    Node 6: relu, bias=0.1
    Node 7: relu, bias=0.2
    Node 0: sigmoid, bias=0.0
    Node 1: sigmoid, bias=0.0
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="7", type=NodeType.HIDDEN, bias=0.2, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            NetworkNode(id="1", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="6", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="7", weight=0.3, enabled=True),
            NetworkConnection(from_node="6", to_node="0", weight=0.8, enabled=True),
            NetworkConnection(from_node="7", to_node="1", weight=0.6, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0", "1"],
    )


def _make_bypass_annotated_network():
    """Network with bypass (skip connection around the annotation).

    IN(-1) ->w=1.0-> 5 ->w=0.5-> 6 ->w=0.8-> 7 ->w=1.0-> OUT(0)
    IN(-1) ->w=0.3-----------------------------------> OUT(0)

    Node 5: relu, bias=0.0
    Node 6: relu, bias=0.1
    Node 7: relu, bias=-0.2
    Node 0: sigmoid, bias=0.0
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
            NetworkConnection(from_node="-1", to_node="0", weight=0.3, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _make_multi_input_annotated_network():
    """Network with two inputs feeding annotation: entry={5,8}, exit=7.

    IN(-1) ->w=1.0-> 5 ->w=0.5-> 6 ->w=0.8-> 7 ->w=1.0-> OUT(0)
    IN(-2) ->w=0.7-> 8 ->w=0.4-> 6

    Annotation covers {5, 8, 6, 7}: entries={5, 8}, exit={7}

    Node 5: relu, bias=0.0
    Node 8: relu, bias=0.0
    Node 6: relu, bias=0.1
    Node 7: relu, bias=-0.2
    Node 0: sigmoid, bias=0.0
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="8", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="7", type=NodeType.HIDDEN, bias=-0.2, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="8", weight=0.7, enabled=True),
            NetworkConnection(from_node="5", to_node="6", weight=0.5, enabled=True),
            NetworkConnection(from_node="8", to_node="6", weight=0.4, enabled=True),
            NetworkConnection(from_node="6", to_node="7", weight=0.8, enabled=True),
            NetworkConnection(from_node="7", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1", "-2"],
        output_node_ids=["0"],
    )


class TestCollapseForwardPassEquivalence:
    """The collapsed network should compute the same function as the expanded one."""

    def test_linear_chain_same_output(self):
        """Simple linear chain annotation: expanded == collapsed."""
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
        x = torch.tensor([[0.5], [1.0], [-0.3]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        # Forward pass on collapsed network
        collapsed = collapse_structure(structure, [ann], {"F"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_diamond_annotation_same_output(self):
        """Diamond-shaped annotation: expanded == collapsed."""
        structure = _make_diamond_annotated_network()
        ann = AnnotationData(
            name="D",
            hypothesis="diamond",
            entry_nodes=["5"],
            exit_nodes=["8"],
            subgraph_nodes=["5", "6", "7", "8"],
            subgraph_connections=[("5", "6"), ("5", "7"), ("6", "8"), ("7", "8")],
        )

        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[0.5], [1.0], [-0.3], [2.0]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(structure, [ann], {"D"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_multi_exit_annotation_same_output(self):
        """Multi-exit annotation with output_index: expanded == collapsed."""
        structure = _make_multi_exit_annotated_network()
        ann = AnnotationData(
            name="M",
            hypothesis="multi-exit",
            entry_nodes=["5"],
            exit_nodes=["6", "7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("5", "7")],
        )

        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[0.5], [1.0], [-0.3], [2.0]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(structure, [ann], {"M"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_bypass_connection_same_output(self):
        """Annotation with external bypass connection: expanded == collapsed."""
        structure = _make_bypass_annotated_network()
        ann = AnnotationData(
            name="B",
            hypothesis="bypass",
            entry_nodes=["5"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("6", "7")],
        )

        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[0.5], [1.0], [-0.3], [2.0]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(structure, [ann], {"B"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_multi_input_annotation_same_output(self):
        """Annotation with multiple entry nodes: expanded == collapsed."""
        structure = _make_multi_input_annotated_network()
        ann = AnnotationData(
            name="MI",
            hypothesis="multi-input",
            entry_nodes=["5", "8"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "8", "6", "7"],
            subgraph_connections=[("5", "6"), ("8", "6"), ("6", "7")],
        )

        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[0.5, 1.0], [1.0, -0.5], [-0.3, 2.0]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(structure, [ann], {"MI"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_negative_inputs_same_output(self):
        """Verify equivalence with negative inputs (relu clipping)."""
        structure = _make_annotated_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("6", "7")],
        )

        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[-5.0], [-1.0], [0.0]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(structure, [ann], {"F"})
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_function_node_activation_extraction(self):
        """Can extract activations from function node after forward pass."""
        structure = _make_annotated_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("6", "7")],
        )

        collapsed = collapse_structure(structure, [ann], {"F"})
        collapsed_net = StructureNetwork(collapsed)
        x = torch.tensor([[0.5], [1.0]], dtype=torch.float64)
        collapsed_net.forward(x)

        # The function node should have extractable activations
        fn_act = collapsed_net.get_node_activation("fn_F")
        assert fn_act.shape == (2,)

    def test_collapsed_structure_builds_without_error(self):
        """StructureNetwork can be constructed from a collapsed structure."""
        structure = _make_annotated_network()
        ann = AnnotationData(
            name="F",
            hypothesis="test",
            entry_nodes=["5"],
            exit_nodes=["7"],
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "6"), ("6", "7")],
        )

        collapsed = collapse_structure(structure, [ann], {"F"})
        # Should not raise
        net = StructureNetwork(collapsed)
        assert "fn_F" in net.node_info
