"""Tests for composition annotations with child_annotation_ids.

Composition annotations reference children's boundary nodes as entry/exit
and only add junction nodes (uncovered) in their own subgraph_nodes.
"""

import pytest

from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)
from explaneat.core.model_state import ModelStateEngine, AnnotationData
from explaneat.core.operations import OperationError


def _make_diamond_network():
    """Two-input diamond: -1 -> 3, -2 -> 4, 3 -> 5, 4 -> 5, 5 -> 0."""
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="3", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="4", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="3", weight=1.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="4", weight=1.0, enabled=True),
            NetworkConnection(from_node="3", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="4", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1", "-2"],
        output_node_ids=["0"],
    )


class TestCompositionAnnotation:
    """Test that composition annotations relax overlap for children's boundary nodes."""

    def test_composition_annotation_allowed(self):
        """Composition C1 with junction node 5 should succeed when children own boundary nodes."""
        net = _make_diamond_network()
        engine = ModelStateEngine(net)

        # Leaf A1: covers -1 and 3
        engine.add_operation("annotate", {
            "name": "A1",
            "hypothesis": "left branch",
            "entry_nodes": ["-1"],
            "exit_nodes": ["3"],
            "subgraph_nodes": ["-1", "3"],
            "subgraph_connections": [["-1", "3"]],
        })

        # Leaf A2: covers -2 and 4
        engine.add_operation("annotate", {
            "name": "A2",
            "hypothesis": "right branch",
            "entry_nodes": ["-2"],
            "exit_nodes": ["4"],
            "subgraph_nodes": ["-2", "4"],
            "subgraph_connections": [["-2", "4"]],
        })

        # Composition C1: junction node 5, entries from children's exits
        engine.add_operation("annotate", {
            "name": "C1",
            "hypothesis": "combined",
            "entry_nodes": ["3", "4"],
            "exit_nodes": ["5"],
            "subgraph_nodes": ["5"],
            "subgraph_connections": [["3", "5"], ["4", "5"]],
            "child_annotation_ids": ["A1", "A2"],
        })

        # Verify composition was created
        annotations = engine.annotations
        assert len(annotations) == 3
        assert annotations[2].name == "C1"

        # Verify children got parent set
        assert annotations[0].parent_annotation_id == "C1"
        assert annotations[1].parent_annotation_id == "C1"

    def test_composition_rejects_overlap_in_junction(self):
        """Composition should fail if its junction node is already covered by a non-child.

        Network: -1 -> 3 -> 5 -> 6 -> 0, -2 -> 4 -> 6
        A1 covers the chain [-1, 3, 5] (valid: single path, exit=5)
        A2 covers [-2, 4] (valid: single path, exit=4)
        C1 tries to use junction 5 as a subgraph_node, but 5 is already in A1.
        """
        net = NetworkStructure(
            nodes=[
                NetworkNode(id="-1", type=NodeType.INPUT),
                NetworkNode(id="-2", type=NodeType.INPUT),
                NetworkNode(id="3", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
                NetworkNode(id="4", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
                NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
                NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
                NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            ],
            connections=[
                NetworkConnection(from_node="-1", to_node="3", weight=1.0, enabled=True),
                NetworkConnection(from_node="3", to_node="5", weight=1.0, enabled=True),
                NetworkConnection(from_node="5", to_node="6", weight=1.0, enabled=True),
                NetworkConnection(from_node="-2", to_node="4", weight=1.0, enabled=True),
                NetworkConnection(from_node="4", to_node="6", weight=1.0, enabled=True),
                NetworkConnection(from_node="6", to_node="0", weight=1.0, enabled=True),
            ],
            input_node_ids=["-1", "-2"],
            output_node_ids=["0"],
        )
        engine = ModelStateEngine(net)

        # A1 covers the full left chain including node 5
        engine.add_operation("annotate", {
            "name": "A1",
            "hypothesis": "left chain",
            "entry_nodes": ["-1"],
            "exit_nodes": ["5"],
            "subgraph_nodes": ["-1", "3", "5"],
            "subgraph_connections": [["-1", "3"], ["3", "5"]],
        })

        # A2 covers right branch
        engine.add_operation("annotate", {
            "name": "A2",
            "hypothesis": "right branch",
            "entry_nodes": ["-2"],
            "exit_nodes": ["4"],
            "subgraph_nodes": ["-2", "4"],
            "subgraph_connections": [["-2", "4"]],
        })

        # Composition C1 tries to claim node 5 as its own junction node,
        # but 5 is already covered by A1 (which is a child — however 5 is
        # in A1's subgraph_nodes, so it's covered). The composition's own
        # subgraph_nodes must be uncovered.
        with pytest.raises(OperationError, match="already covered"):
            engine.add_operation("annotate", {
                "name": "C1",
                "hypothesis": "combined",
                "entry_nodes": ["5", "4"],
                "exit_nodes": ["6"],
                "subgraph_nodes": ["5", "6"],
                "subgraph_connections": [["5", "6"], ["4", "6"]],
                "child_annotation_ids": ["A1", "A2"],
            })

    def test_leaf_annotation_still_rejects_overlap(self):
        """Non-composition annotation must still reject overlap strictly."""
        net = _make_diamond_network()
        engine = ModelStateEngine(net)

        # A1 covers -1 and 3
        engine.add_operation("annotate", {
            "name": "A1",
            "hypothesis": "left branch",
            "entry_nodes": ["-1"],
            "exit_nodes": ["3"],
            "subgraph_nodes": ["-1", "3"],
            "subgraph_connections": [["-1", "3"]],
        })

        # A2 (non-composition) tries to cover -1 again
        with pytest.raises(OperationError, match="already covered"):
            engine.add_operation("annotate", {
                "name": "A2",
                "hypothesis": "overlapping",
                "entry_nodes": ["-1"],
                "exit_nodes": ["4"],
                "subgraph_nodes": ["-1", "4"],
                "subgraph_connections": [["-1", "4"]],
            })
