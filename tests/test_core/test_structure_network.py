"""Tests for StructureNetwork - forward pass through NetworkStructure objects."""

import math

import numpy as np
import pytest
import torch

from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.structure_network import StructureNetwork


# =============================================================================
# Fixtures: reusable network topologies
# =============================================================================


def make_simple_network():
    """Two inputs, one hidden, one output.

    -2 --1.0--> 5 --2.0--> 0
    -1 --0.5--> 5

    Node 5: relu, bias=0.1
    Node 0: sigmoid, bias=-0.5
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.5, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-2", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="5", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=2.0, enabled=True),
        ],
        input_node_ids=["-2", "-1"],
        output_node_ids=["0"],
    )


def make_identity_node_network():
    """Network with an identity node intercepting one input to node 0.

    -2 --1.0--> 5 --2.0--> 0
    -1 --0.5--> 5
    -2 --0.8--> identity_5 --1.0--> 0

    identity_5: bias=0, activation=identity
    """
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


def make_split_node_network():
    """Network with split nodes (5_a, 5_b replace node 5).

    -2 --1.0--> 5_a --2.0--> 0
    -1 --0.5--> 5_a
    -2 --1.0--> 5_b --3.0--> 6
    -1 --0.5--> 5_b
    6 --1.5--> 0

    5_a, 5_b: relu, bias=0.1
    6: relu, bias=0.2
    0: sigmoid, bias=-0.5
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5_a", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="5_b", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.2, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.5, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-2", to_node="5_a", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="5_a", weight=0.5, enabled=True),
            NetworkConnection(from_node="5_a", to_node="0", weight=2.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="5_b", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="5_b", weight=0.5, enabled=True),
            NetworkConnection(from_node="5_b", to_node="6", weight=3.0, enabled=True),
            NetworkConnection(from_node="6", to_node="0", weight=1.5, enabled=True),
        ],
        input_node_ids=["-2", "-1"],
        output_node_ids=["0"],
    )


def make_skip_connection_network():
    """Network with a skip connection from input directly to output.

    -1 --1.0--> 5 --2.0--> 0
    -1 --0.3--> 0   (skip connection)

    Node 5: relu, bias=0.0
    Node 0: sigmoid, bias=0.0
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=2.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="0", weight=0.3, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


# =============================================================================
# Tests: StructureNetwork construction
# =============================================================================


class TestStructureNetworkBuild:
    def test_simple_network_layers(self):
        net = StructureNetwork(make_simple_network())
        assert len(net._layer_order) == 3  # input, hidden, output

    def test_node_info_contains_all_nodes(self):
        net = StructureNetwork(make_simple_network())
        assert set(net.node_info.keys()) == {"-2", "-1", "5", "0"}

    def test_input_nodes_preserve_order(self):
        net = StructureNetwork(make_simple_network())
        assert net.node_info["-2"]["layer_index"] == 0
        assert net.node_info["-1"]["layer_index"] == 1

    def test_identity_node_activation(self):
        net = StructureNetwork(make_identity_node_network())
        assert net.node_info["identity_5"]["activation"] == "identity"
        assert net.node_info["5"]["activation"] == "relu"
        assert net.node_info["0"]["activation"] == "sigmoid"

    def test_split_nodes_recognized(self):
        net = StructureNetwork(make_split_node_network())
        assert "5_a" in net.node_info
        assert "5_b" in net.node_info
        assert net.node_info["5_a"]["activation"] == "relu"
        assert net.node_info["5_b"]["activation"] == "relu"

    def test_disabled_connections_ignored(self):
        structure = make_simple_network()
        # Disable the connection from -1 to 5
        for c in structure.connections:
            if c.from_node == "-1" and c.to_node == "5":
                c.enabled = False
        net = StructureNetwork(structure)
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)
        # Node 5 should only get input from -2 (weight 1.0), not from -1
        # relu(1.0*1.0 + 0.1) = 1.1
        assert abs(net.get_node_activation("5")[0] - 1.1) < 1e-10


# =============================================================================
# Tests: Forward pass correctness
# =============================================================================


class TestForwardPass:
    def test_simple_forward(self):
        """Verify forward pass matches manual computation."""
        net = StructureNetwork(make_simple_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        out = net.forward(X)

        # Node 5: relu(1.0*1.0 + 2.0*0.5 + 0.1) = 2.1
        # Node 0: sigmoid(2.1*2.0 - 0.5) = sigmoid(3.7)
        expected_0 = 1.0 / (1.0 + math.exp(-3.7))
        assert abs(out[0, 0].item() - expected_0) < 1e-10

    def test_identity_node_passthrough(self):
        """Identity nodes should pass through input without activation."""
        net = StructureNetwork(make_identity_node_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)

        # identity_5: identity(1.0*0.8 + 0.0) = 0.8
        assert abs(net.get_node_activation("identity_5")[0] - 0.8) < 1e-10

    def test_identity_node_full_output(self):
        """Full output should combine hidden and identity paths."""
        net = StructureNetwork(make_identity_node_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)

        node_5 = max(0, 1.0 * 1.0 + 2.0 * 0.5 + 0.1)  # 2.1
        identity_5 = 1.0 * 0.8  # 0.8
        expected_0 = 1.0 / (1.0 + math.exp(-(node_5 * 2.0 + identity_5 * 1.0 - 0.5)))
        assert abs(net.get_node_activation("0")[0] - expected_0) < 1e-10

    def test_split_nodes_same_inputs(self):
        """Split nodes should receive the same inputs as each other."""
        net = StructureNetwork(make_split_node_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)

        # Both 5_a and 5_b get same inputs and have same bias
        # relu(1.0*1.0 + 2.0*0.5 + 0.1) = 2.1
        assert abs(net.get_node_activation("5_a")[0] - 2.1) < 1e-10
        assert abs(net.get_node_activation("5_b")[0] - 2.1) < 1e-10

    def test_split_nodes_different_outputs(self):
        """Split nodes route through different output paths."""
        net = StructureNetwork(make_split_node_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)

        val_5 = 2.1
        # Node 6: relu(2.1*3.0 + 0.2) = 6.5
        node_6 = max(0, val_5 * 3.0 + 0.2)
        # Node 0: sigmoid(2.1*2.0 + 6.5*1.5 - 0.5) = sigmoid(13.45)
        expected_0 = 1.0 / (1.0 + math.exp(-(val_5 * 2.0 + node_6 * 1.5 - 0.5)))
        assert abs(net.get_node_activation("0")[0] - expected_0) < 1e-10

    def test_skip_connection(self):
        """Network with skip connections from input layer to output."""
        net = StructureNetwork(make_skip_connection_network())
        X = torch.tensor([[2.0]], dtype=torch.float64)
        net.forward(X)

        # Node 5: relu(2.0*1.0 + 0.0) = 2.0
        # Node 0: sigmoid(2.0*2.0 + 2.0*0.3 + 0.0) = sigmoid(4.6)
        expected_0 = 1.0 / (1.0 + math.exp(-4.6))
        assert abs(net.get_node_activation("0")[0] - expected_0) < 1e-10

    def test_batch_forward(self):
        """Forward pass handles multiple samples correctly."""
        net = StructureNetwork(make_simple_network())
        X = torch.tensor([[1.0, 2.0], [0.0, 0.0], [-1.0, 3.0]], dtype=torch.float64)
        out = net.forward(X)
        assert out.shape == (3, 1)

        # Each sample should be independent
        for i in range(3):
            single_X = torch.tensor([X[i].tolist()], dtype=torch.float64)
            net2 = StructureNetwork(make_simple_network())
            single_out = net2.forward(single_X)
            assert abs(out[i, 0].item() - single_out[0, 0].item()) < 1e-10


# =============================================================================
# Tests: Node activation extraction
# =============================================================================


class TestNodeActivationExtraction:
    def test_input_node_activations(self):
        net = StructureNetwork(make_simple_network())
        X = torch.tensor([[3.0, 7.0]], dtype=torch.float64)
        net.forward(X)
        np.testing.assert_array_almost_equal(
            net.get_node_activation("-2"), [3.0]
        )
        np.testing.assert_array_almost_equal(
            net.get_node_activation("-1"), [7.0]
        )

    def test_hidden_node_activation(self):
        net = StructureNetwork(make_simple_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)
        # relu(1.0 + 1.0 + 0.1) = 2.1
        np.testing.assert_almost_equal(
            net.get_node_activation("5")[0], 2.1
        )

    def test_unknown_node_raises(self):
        net = StructureNetwork(make_simple_network())
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        net.forward(X)
        with pytest.raises(ValueError, match="not found"):
            net.get_node_activation("nonexistent")

    def test_no_forward_raises(self):
        net = StructureNetwork(make_simple_network())
        with pytest.raises(RuntimeError, match="forward"):
            net.get_node_activation("-2")

    def test_relu_clips_negative(self):
        """ReLU nodes should clip negative pre-activations to zero."""
        net = StructureNetwork(make_simple_network())
        # With inputs that produce negative pre-activation at node 5:
        # relu(-10*1.0 + 0*0.5 + 0.1) = relu(-9.9) = 0
        X = torch.tensor([[-10.0, 0.0]], dtype=torch.float64)
        net.forward(X)
        assert net.get_node_activation("5")[0] == 0.0


# =============================================================================
# Tests: Partial input connectivity (pruned phenotype)
# =============================================================================


class TestPartialInputs:
    def test_unused_inputs_pruned(self):
        """Input nodes with no path to output should be pruned from the network."""
        # 4 input nodes but only 2 are connected to hidden/output
        structure = NetworkStructure(
            nodes=[
                NetworkNode(id="-4", type=NodeType.INPUT),
                NetworkNode(id="-3", type=NodeType.INPUT),
                NetworkNode(id="-2", type=NodeType.INPUT),
                NetworkNode(id="-1", type=NodeType.INPUT),
                NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
                NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            ],
            connections=[
                # Only -2 and -1 connect to the network
                NetworkConnection(from_node="-2", to_node="5", weight=1.0, enabled=True),
                NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
                NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
            ],
            input_node_ids=["-4", "-3", "-2", "-1"],
            output_node_ids=["0"],
        )
        net = StructureNetwork(structure)

        # Only connected inputs should be in node_info
        assert "-2" in net.node_info
        assert "-1" in net.node_info
        assert "-4" not in net.node_info
        assert "-3" not in net.node_info

        # Forward pass with 4-column input tensor should still work
        X = torch.tensor([[10.0, 20.0, 3.0, 4.0]], dtype=torch.float64)
        net.forward(X)

        # Node 5: relu(3.0*1.0 + 4.0*1.0) = 7.0  (columns 2 and 3 = inputs -2 and -1)
        np.testing.assert_almost_equal(net.get_node_activation("5")[0], 7.0)

    def test_many_inputs_few_connected(self):
        """Simulates a 32-feature dataset where only some inputs are used."""
        n_inputs = 32
        n_connected = 5
        input_ids = [str(-i - 1) for i in range(n_inputs)]

        nodes = [NetworkNode(id=nid, type=NodeType.INPUT) for nid in input_ids]
        nodes.append(NetworkNode(id="H", type=NodeType.HIDDEN, bias=0.0, activation="relu"))
        nodes.append(NetworkNode(id="O", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"))

        connections = []
        for i in range(n_connected):
            connections.append(
                NetworkConnection(
                    from_node=input_ids[i], to_node="H", weight=1.0, enabled=True
                )
            )
        connections.append(
            NetworkConnection(from_node="H", to_node="O", weight=1.0, enabled=True)
        )

        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=input_ids,
            output_node_ids=["O"],
        )
        net = StructureNetwork(structure)

        # Should not crash with a 32-column input
        X = torch.ones((2, n_inputs), dtype=torch.float64)
        out = net.forward(X)
        assert out.shape == (2, 1)
        # H = relu(5 * 1.0) = 5.0
        np.testing.assert_almost_equal(net.get_node_activation("H")[0], 5.0)


# =============================================================================
# Tests: Split input nodes (input nodes split by apply_split_node)
# =============================================================================


def make_split_input_network():
    """Network where input -2 was split into -2_a and -2_b.

    Simulates what apply_split_node does: the original -2 is replaced
    by -2_a and -2_b (both type INPUT), but input_node_ids still lists -2.

    -2_a --1.0--> 5 --2.0--> 0
    -1  --0.5--> 5
    -2_b --0.8--> 6 --1.0--> 0

    Node 5: relu, bias=0.1
    Node 6: relu, bias=0.0
    Node 0: sigmoid, bias=-0.5
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-2_a", type=NodeType.INPUT),
            NetworkNode(id="-2_b", type=NodeType.INPUT),
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.5, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-2_a", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="5", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=2.0, enabled=True),
            NetworkConnection(from_node="-2_b", to_node="6", weight=0.8, enabled=True),
            NetworkConnection(from_node="6", to_node="0", weight=1.0, enabled=True),
        ],
        # input_node_ids still lists the ORIGINAL -2, not the splits
        input_node_ids=["-2", "-1"],
        output_node_ids=["0"],
    )


class TestSplitInputNodes:
    def test_split_input_nodes_recognized(self):
        """Split input nodes should be in node_info even though not in input_node_ids."""
        net = StructureNetwork(make_split_input_network())
        assert "-2_a" in net.node_info
        assert "-2_b" in net.node_info
        assert "-1" in net.node_info

    def test_split_input_nodes_mapped_to_base_column(self):
        """Split inputs -2_a and -2_b should both read from column 0 (same as -2)."""
        net = StructureNetwork(make_split_input_network())
        # -2 is at index 0 in input_node_ids, so -2_a and -2_b should map to col 0
        assert net._input_col_map["-2_a"] == 0
        assert net._input_col_map["-2_b"] == 0
        assert net._input_col_map["-1"] == 1

    def test_split_input_forward_pass(self):
        """Forward pass should work correctly with split input nodes."""
        net = StructureNetwork(make_split_input_network())
        # x[0] = 3.0 (mapped to -2_a and -2_b), x[1] = 4.0 (mapped to -1)
        X = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        net.forward(X)

        # -2_a reads col 0 = 3.0, -2_b reads col 0 = 3.0
        np.testing.assert_almost_equal(net.get_node_activation("-2_a")[0], 3.0)
        np.testing.assert_almost_equal(net.get_node_activation("-2_b")[0], 3.0)

        # Node 5: relu(3.0*1.0 + 4.0*0.5 + 0.1) = relu(5.1) = 5.1
        np.testing.assert_almost_equal(net.get_node_activation("5")[0], 5.1)

        # Node 6: relu(3.0*0.8 + 0.0) = relu(2.4) = 2.4
        np.testing.assert_almost_equal(net.get_node_activation("6")[0], 2.4)

    def test_split_input_batch(self):
        """Batch forward pass with split input nodes."""
        net = StructureNetwork(make_split_input_network())
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        out = net.forward(X)
        assert out.shape == (2, 1)

    def test_multiple_split_inputs(self):
        """Multiple input nodes split simultaneously."""
        structure = NetworkStructure(
            nodes=[
                NetworkNode(id="-2_a", type=NodeType.INPUT),
                NetworkNode(id="-2_b", type=NodeType.INPUT),
                NetworkNode(id="-1_a", type=NodeType.INPUT),
                NetworkNode(id="-1_b", type=NodeType.INPUT),
                NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
                NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            ],
            connections=[
                NetworkConnection(from_node="-2_a", to_node="5", weight=1.0, enabled=True),
                NetworkConnection(from_node="-1_a", to_node="5", weight=1.0, enabled=True),
                NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
            ],
            input_node_ids=["-2", "-1"],
            output_node_ids=["0"],
        )
        net = StructureNetwork(structure)

        # -2_a maps to col 0, -1_a maps to col 1
        X = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        net.forward(X)
        np.testing.assert_almost_equal(net.get_node_activation("-2_a")[0], 2.0)
        np.testing.assert_almost_equal(net.get_node_activation("-1_a")[0], 3.0)
        # 5: relu(2.0 + 3.0) = 5.0
        np.testing.assert_almost_equal(net.get_node_activation("5")[0], 5.0)
