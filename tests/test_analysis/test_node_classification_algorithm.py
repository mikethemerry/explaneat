"""
Tests for node classification algorithm.

Tests the algorithm that classifies nodes within a proposed coverage
as entry, intermediate, or exit based on their external connections.
"""

import pytest
from typing import List, Set

from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)
from explaneat.analysis.node_classification import (
    classify_node,
    classify_coverage,
    validate_annotation_params,
    get_subgraph_connections,
    auto_detect_entry_exit,
    NodeRole,
    NodeClassificationDetail,
    ClassificationResult,
)


def create_simple_network() -> NetworkStructure:
    """
    Create a simple network for testing:

    i0 -> h1 -> o0
    i1 -> h1 -> o1
    """
    nodes = [
        NetworkNode(id="i0", type=NodeType.INPUT),
        NetworkNode(id="i1", type=NodeType.INPUT),
        NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        NetworkNode(id="o1", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
        NetworkConnection(from_node="i1", to_node="h1", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="o0", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="o1", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["i0", "i1"],
        output_node_ids=["o0", "o1"],
    )


def create_chain_network() -> NetworkStructure:
    """
    Create a simple chain network:

    i0 -> h1 -> h2 -> h3 -> o0
    """
    nodes = [
        NetworkNode(id="i0", type=NodeType.INPUT),
        NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="h2", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="h3", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="h2", weight=1.0, enabled=True),
        NetworkConnection(from_node="h2", to_node="h3", weight=1.0, enabled=True),
        NetworkConnection(from_node="h3", to_node="o0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["i0"],
        output_node_ids=["o0"],
    )


def create_multi_path_network() -> NetworkStructure:
    """
    Create a network with multiple paths:

    i0 -> h1 -> h3 -> o0
    i1 -> h2 -> h3 -> o0
    """
    nodes = [
        NetworkNode(id="i0", type=NodeType.INPUT),
        NetworkNode(id="i1", type=NodeType.INPUT),
        NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="h2", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="h3", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
        NetworkConnection(from_node="i1", to_node="h2", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="h3", weight=1.0, enabled=True),
        NetworkConnection(from_node="h2", to_node="h3", weight=1.0, enabled=True),
        NetworkConnection(from_node="h3", to_node="o0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["i0", "i1"],
        output_node_ids=["o0"],
    )


@pytest.mark.unit
class TestClassifyNode:
    """Test classify_node function."""

    def test_entry_node_classification(self):
        """Node with external input and no external output is ENTRY."""
        network = create_chain_network()
        # Coverage: h1, h2 (middle nodes)
        # h1 has external input (from i0) and internal output (to h2)
        coverage = {"h1", "h2"}

        detail = classify_node(network, "h1", coverage)

        assert detail.role == NodeRole.ENTRY
        assert detail.node_id == "h1"
        assert len(detail.external_inputs) == 1
        assert len(detail.external_outputs) == 0
        assert len(detail.internal_outputs) == 1

    def test_exit_node_classification(self):
        """Node with external output and no external input is EXIT."""
        network = create_chain_network()
        # Coverage: h1, h2 (middle nodes)
        # h2 has internal input (from h1) and external output (to h3)
        coverage = {"h1", "h2"}

        detail = classify_node(network, "h2", coverage)

        assert detail.role == NodeRole.EXIT
        assert detail.node_id == "h2"
        assert len(detail.external_inputs) == 0
        assert len(detail.external_outputs) == 1
        assert len(detail.internal_inputs) == 1

    def test_intermediate_node_classification(self):
        """Node with no external I/O is INTERMEDIATE."""
        network = create_chain_network()
        # Coverage: h1, h2, h3 (all middle nodes)
        # h2 has internal input (from h1) and internal output (to h3)
        coverage = {"h1", "h2", "h3"}

        detail = classify_node(network, "h2", coverage)

        assert detail.role == NodeRole.INTERMEDIATE
        assert detail.node_id == "h2"
        assert len(detail.external_inputs) == 0
        assert len(detail.external_outputs) == 0
        assert len(detail.internal_inputs) == 1
        assert len(detail.internal_outputs) == 1

    def test_input_node_is_entry(self):
        """Network input nodes are always classified as ENTRY."""
        network = create_chain_network()
        # Coverage includes input node
        coverage = {"i0", "h1"}

        detail = classify_node(network, "i0", coverage)

        # Input nodes have implicit "external input" from the environment
        assert detail.role == NodeRole.ENTRY
        assert detail.is_input_node is True

    def test_node_with_both_external_input_and_output_is_entry(self):
        """Node with both external I/O is classified as ENTRY (but flagged)."""
        # This is a violation case - the algorithm classifies as ENTRY
        # but this should be flagged
        nodes = [
            NetworkNode(id="i0", type=NodeType.INPUT),
            NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            NetworkNode(id="o1", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ]
        connections = [
            NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
            NetworkConnection(from_node="h1", to_node="o0", weight=1.0, enabled=True),
            NetworkConnection(from_node="h1", to_node="o1", weight=1.0, enabled=True),
        ]
        network = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["i0"],
            output_node_ids=["o0", "o1"],
        )

        # Coverage: h1 only - it has external input (i0) and external outputs (o0, o1)
        coverage = {"h1"}

        detail = classify_node(network, "h1", coverage)

        # Classified as ENTRY but has external outputs (violation)
        assert detail.role == NodeRole.ENTRY
        assert len(detail.external_inputs) == 1
        assert len(detail.external_outputs) == 2

    def test_disabled_connections_ignored(self):
        """Disabled connections should not affect classification."""
        nodes = [
            NetworkNode(id="i0", type=NodeType.INPUT),
            NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ]
        connections = [
            NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
            NetworkConnection(from_node="h1", to_node="o0", weight=1.0, enabled=False),  # Disabled
        ]
        network = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["i0"],
            output_node_ids=["o0"],
        )

        coverage = {"h1"}

        detail = classify_node(network, "h1", coverage)

        # h1 only has external input (disabled output doesn't count)
        assert detail.role == NodeRole.ENTRY
        assert len(detail.external_outputs) == 0


@pytest.mark.unit
class TestClassifyCoverage:
    """Test classify_coverage function."""

    def test_classifies_all_nodes(self):
        """Should classify all nodes in coverage."""
        network = create_chain_network()
        coverage = ["h1", "h2", "h3"]

        result = classify_coverage(network, coverage)

        assert len(result.details) == 3
        assert set(result.coverage) == {"h1", "h2", "h3"}

    def test_entry_intermediate_exit_grouping(self):
        """Should correctly group nodes by role."""
        network = create_chain_network()
        # h1: entry (external input from i0)
        # h2: intermediate (internal I/O only)
        # h3: exit (external output to o0)
        coverage = ["h1", "h2", "h3"]

        result = classify_coverage(network, coverage)

        assert "h1" in result.entry_nodes
        assert "h2" in result.intermediate_nodes
        assert "h3" in result.exit_nodes

    def test_valid_coverage_has_no_violations(self):
        """Coverage with no entry nodes having external outputs is valid."""
        network = create_chain_network()
        coverage = ["h1", "h2", "h3"]

        result = classify_coverage(network, coverage)

        assert result.valid is True
        assert len(result.violations) == 0

    def test_invalid_coverage_has_violations(self):
        """Entry node with external outputs creates violation."""
        nodes = [
            NetworkNode(id="i0", type=NodeType.INPUT),
            NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="h2", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ]
        connections = [
            NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
            NetworkConnection(from_node="h1", to_node="h2", weight=1.0, enabled=True),
            NetworkConnection(from_node="h1", to_node="o0", weight=1.0, enabled=True),  # External output
            NetworkConnection(from_node="h2", to_node="o0", weight=1.0, enabled=True),
        ]
        network = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["i0"],
            output_node_ids=["o0"],
        )

        # h1 has external input (i0) AND external output (o0)
        coverage = ["h1", "h2"]

        result = classify_coverage(network, coverage)

        assert result.valid is False
        assert len(result.violations) == 1
        assert result.violations[0]["node_id"] == "h1"
        assert result.violations[0]["reason"] == "has_external_input_and_output"

    def test_multiple_entry_nodes(self):
        """Multiple nodes can be entry nodes."""
        network = create_multi_path_network()
        # h1: entry (external input from i0)
        # h2: entry (external input from i1)
        # h3: exit (external output to o0)
        coverage = ["h1", "h2", "h3"]

        result = classify_coverage(network, coverage)

        assert len(result.entry_nodes) == 2
        assert "h1" in result.entry_nodes
        assert "h2" in result.entry_nodes
        assert "h3" in result.exit_nodes


@pytest.mark.unit
class TestValidateAnnotationParams:
    """Test validate_annotation_params function."""

    def test_valid_params_return_no_errors(self):
        """Valid annotation params should return empty error list."""
        network = create_chain_network()

        errors = validate_annotation_params(
            model=network,
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3")],
        )

        assert len(errors) == 0

    def test_nonexistent_node_returns_error(self):
        """Referencing nonexistent node should return error."""
        network = create_chain_network()

        errors = validate_annotation_params(
            model=network,
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h99"],  # h99 doesn't exist
            subgraph_connections=[("h1", "h2")],
        )

        assert any("h99" in e and "does not exist" in e for e in errors)

    def test_entry_not_subset_of_subgraph_returns_error(self):
        """Entry nodes not in subgraph should return error."""
        network = create_chain_network()

        errors = validate_annotation_params(
            model=network,
            entry_nodes=["h1", "h0"],  # h0 not in subgraph
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3")],
        )

        assert any("subset" in e.lower() for e in errors)

    def test_exit_not_subset_of_subgraph_returns_error(self):
        """Exit nodes not in subgraph should return error."""
        network = create_chain_network()

        errors = validate_annotation_params(
            model=network,
            entry_nodes=["h1"],
            exit_nodes=["h3", "h99"],  # h99 not in subgraph
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3")],
        )

        assert any("subset" in e.lower() for e in errors)

    def test_nonexistent_connection_returns_error(self):
        """Connection that doesn't exist should return error."""
        network = create_chain_network()

        errors = validate_annotation_params(
            model=network,
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3")],  # h1->h3 doesn't exist
        )

        assert any("does not exist" in e for e in errors)

    def test_missing_entry_nodes_returns_error(self):
        """Missing computed entry nodes should return error."""
        network = create_chain_network()

        # h1 should be entry but we don't declare it
        errors = validate_annotation_params(
            model=network,
            entry_nodes=[],  # Missing h1
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3")],
        )

        assert any("Missing entry nodes" in e for e in errors)

    def test_extra_entry_nodes_returns_error(self):
        """Declaring entry nodes that aren't entries should return error."""
        network = create_chain_network()

        # h2 is not an entry but we declare it
        errors = validate_annotation_params(
            model=network,
            entry_nodes=["h1", "h2"],  # h2 is not entry
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3")],
        )

        assert any("not entries" in e for e in errors)


@pytest.mark.unit
class TestGetSubgraphConnections:
    """Test get_subgraph_connections function."""

    def test_returns_internal_connections_only(self):
        """Should only return connections where both endpoints are in coverage."""
        network = create_chain_network()
        coverage = {"h1", "h2", "h3"}

        connections = get_subgraph_connections(network, coverage)

        assert len(connections) == 2
        assert ("h1", "h2") in connections
        assert ("h2", "h3") in connections
        # Should NOT include (i0, h1) or (h3, o0)
        assert ("i0", "h1") not in connections
        assert ("h3", "o0") not in connections

    def test_empty_coverage_returns_empty(self):
        """Empty coverage should return empty list."""
        network = create_chain_network()

        connections = get_subgraph_connections(network, set())

        assert len(connections) == 0

    def test_single_node_coverage_returns_empty(self):
        """Single node coverage has no internal connections."""
        network = create_chain_network()
        coverage = {"h1"}

        connections = get_subgraph_connections(network, coverage)

        # No connection has both endpoints in a single-node coverage
        assert len(connections) == 0

    def test_disabled_connections_ignored(self):
        """Disabled connections should not be returned."""
        nodes = [
            NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="h2", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        ]
        connections = [
            NetworkConnection(from_node="h1", to_node="h2", weight=1.0, enabled=False),
        ]
        network = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=[],
            output_node_ids=[],
        )

        coverage = {"h1", "h2"}
        result = get_subgraph_connections(network, coverage)

        assert len(result) == 0


@pytest.mark.unit
class TestAutoDetectEntryExit:
    """Test auto_detect_entry_exit function."""

    def test_detects_entry_and_exit(self):
        """Should correctly identify entry and exit nodes."""
        network = create_chain_network()
        coverage = ["h1", "h2", "h3"]

        entry_nodes, exit_nodes = auto_detect_entry_exit(network, coverage)

        assert "h1" in entry_nodes
        assert "h3" in exit_nodes
        # h2 should not be in either
        assert "h2" not in entry_nodes
        assert "h2" not in exit_nodes

    def test_multiple_entry_points(self):
        """Should detect multiple entry points."""
        network = create_multi_path_network()
        coverage = ["h1", "h2", "h3"]

        entry_nodes, exit_nodes = auto_detect_entry_exit(network, coverage)

        assert len(entry_nodes) == 2
        assert "h1" in entry_nodes
        assert "h2" in entry_nodes
        assert "h3" in exit_nodes

    def test_single_node_coverage(self):
        """Single node is both entry and exit if it has external I/O."""
        network = create_simple_network()
        # h1 has external inputs and outputs
        coverage = ["h1"]

        entry_nodes, exit_nodes = auto_detect_entry_exit(network, coverage)

        # h1 classified as entry (because it has external input)
        # Note: this is actually a violation case
        assert "h1" in entry_nodes
