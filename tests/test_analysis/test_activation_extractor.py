"""Tests for ActivationExtractor with NetworkStructure (structure mode)."""

import math

import numpy as np
import pytest

from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.analysis.activation_extractor import ActivationExtractor


def make_annotated_network():
    """Network with an identity node, simulating a post-operation state.

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


class TestActivationExtractorStructureMode:
    def test_extract_identity_node(self):
        """Should extract activations from identity nodes."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }

        X = np.array([[1.0, 2.0]])
        extractor = ActivationExtractor.from_structure(structure)
        entry_acts, exit_acts = extractor.extract(X, annotation)

        assert entry_acts.shape == (1, 1)
        assert exit_acts.shape == (1, 1)
        # identity_5: identity(1.0*0.8) = 0.8
        np.testing.assert_almost_equal(entry_acts[0, 0], 0.8)

    def test_extract_regular_nodes(self):
        """Should extract activations from regular integer-ID nodes."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["-2", "-1"],
            "exit_nodes": ["5"],
            "subgraph_nodes": ["-2", "-1", "5"],
            "subgraph_connections": [("-2", "5"), ("-1", "5")],
        }

        X = np.array([[1.0, 2.0]])
        extractor = ActivationExtractor.from_structure(structure)
        entry_acts, exit_acts = extractor.extract(X, annotation)

        assert entry_acts.shape == (1, 2)
        assert exit_acts.shape == (1, 1)
        # Node 5: relu(1.0*1.0 + 2.0*0.5 + 0.1) = 2.1
        np.testing.assert_almost_equal(exit_acts[0, 0], 2.1)

    def test_extract_batch(self):
        """Should handle multiple samples."""
        structure = make_annotated_network()
        annotation = {
            "entry_nodes": ["identity_5"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["identity_5", "0"],
            "subgraph_connections": [("identity_5", "0")],
        }

        X = np.array([[1.0, 2.0], [0.5, 0.0], [2.0, 1.0]])
        extractor = ActivationExtractor.from_structure(structure)
        entry_acts, exit_acts = extractor.extract(X, annotation)

        assert entry_acts.shape == (3, 1)
        assert exit_acts.shape == (3, 1)

    def test_extract_split_nodes(self):
        """Should extract activations from split nodes like 5_a."""
        structure = NetworkStructure(
            nodes=[
                NetworkNode(id="-1", type=NodeType.INPUT),
                NetworkNode(
                    id="5_a", type=NodeType.HIDDEN, bias=0.0, activation="relu"
                ),
                NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            ],
            connections=[
                NetworkConnection(
                    from_node="-1", to_node="5_a", weight=2.0, enabled=True
                ),
                NetworkConnection(
                    from_node="5_a", to_node="0", weight=1.0, enabled=True
                ),
            ],
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        annotation = {
            "entry_nodes": ["-1"],
            "exit_nodes": ["5_a"],
            "subgraph_nodes": ["-1", "5_a"],
            "subgraph_connections": [("-1", "5_a")],
        }

        X = np.array([[3.0]])
        extractor = ActivationExtractor.from_structure(structure)
        entry_acts, exit_acts = extractor.extract(X, annotation)

        # 5_a: relu(3.0*2.0) = 6.0
        np.testing.assert_almost_equal(exit_acts[0, 0], 6.0)
