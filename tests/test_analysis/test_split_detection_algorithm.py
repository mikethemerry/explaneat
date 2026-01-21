"""
Tests for split detection algorithm.

Tests the algorithm that identifies nodes requiring splits before
an annotation can be created.
"""

import pytest
from typing import List, Set

from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)
from explaneat.analysis.split_detection import (
    detect_required_splits,
    suggest_split_operations,
    analyze_coverage_for_splits,
    iterative_split_resolution,
    ViolationDetail,
    SplitDetectionResult,
)


def create_simple_network() -> NetworkStructure:
    """
    Create a simple network for testing:

    i0 -> h1 -> o0
    i1 -> h1 -> o1

    Where h1 has connections from both inputs and to both outputs.
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


def create_branching_network() -> NetworkStructure:
    """
    Create a network with branching paths:

    i0 -> h1 -> h2 -> o0
          h1 -> o1

    h1 connects to both h2 (internal) and o1 (external for coverage {h1, h2}).
    """
    nodes = [
        NetworkNode(id="i0", type=NodeType.INPUT),
        NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="h2", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        NetworkNode(id="o1", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="h2", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="o1", weight=1.0, enabled=True),
        NetworkConnection(from_node="h2", to_node="o0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["i0"],
        output_node_ids=["o0", "o1"],
    )


def create_chain_network() -> NetworkStructure:
    """
    Create a simple chain network:

    i0 -> h1 -> h2 -> o0
    """
    nodes = [
        NetworkNode(id="i0", type=NodeType.INPUT),
        NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="h2", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
        NetworkConnection(from_node="h1", to_node="h2", weight=1.0, enabled=True),
        NetworkConnection(from_node="h2", to_node="o0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["i0"],
        output_node_ids=["o0"],
    )


@pytest.mark.unit
class TestDetectRequiredSplits:
    """Test detect_required_splits function."""

    def test_no_violations_when_coverage_is_full(self):
        """No violations when all nodes are covered."""
        network = create_simple_network()
        coverage = {"i0", "i1", "h1", "o0", "o1"}

        violations = detect_required_splits(network, coverage)

        assert len(violations) == 0

    def test_no_violations_for_entry_only(self):
        """No violations when node only has external inputs (entry node)."""
        network = create_simple_network()
        # h1 has external inputs from i0, i1 and external outputs to o0, o1
        # But if we include h1 and both outputs, h1 only has external INPUT
        coverage = {"h1", "o0", "o1"}

        violations = detect_required_splits(network, coverage)

        # h1 has external inputs (from i0, i1) but no external outputs
        # (o0 and o1 are in coverage)
        assert len(violations) == 0

    def test_no_violations_for_exit_only(self):
        """No violations when node only has external outputs (exit node)."""
        network = create_simple_network()
        # If coverage is just h1, then h1 has both external inputs AND outputs
        # Let's test with i0 and h1 - h1 has external output only (not from outside)
        coverage = {"i0", "h1"}

        violations = detect_required_splits(network, coverage)

        # h1 has internal input (from i0) and external outputs (to o0, o1)
        # i0 is input node with no external inputs in coverage
        # Since h1 has no external input (i0 is in coverage), no violation
        assert len(violations) == 0

    def test_violation_detected_with_external_input_and_output(self):
        """Violation when node has both external input AND external output."""
        network = create_branching_network()
        # Coverage: h1 and h2
        # h1: external input from i0, internal output to h2, external output to o1
        # h1 has BOTH external input AND external output -> violation
        coverage = {"h1", "h2"}

        violations = detect_required_splits(network, coverage)

        assert len(violations) == 1
        assert violations[0].node_id == "h1"
        assert violations[0].reason == "has_external_input_and_output"

    def test_violation_details_include_connections(self):
        """Violation includes external input and output connection details."""
        network = create_branching_network()
        coverage = {"h1", "h2"}

        violations = detect_required_splits(network, coverage)

        assert len(violations) == 1
        v = violations[0]

        # External inputs: from i0 to h1
        assert len(v.external_inputs) == 1
        assert ("i0", "h1") in v.external_inputs

        # External outputs: from h1 to o1
        assert len(v.external_outputs) == 1
        assert ("h1", "o1") in v.external_outputs

        # Internal outputs: from h1 to h2
        assert len(v.internal_outputs) == 1
        assert ("h1", "h2") in v.internal_outputs

    def test_no_violation_for_chain_coverage(self):
        """No violation when chain is fully covered internally."""
        network = create_chain_network()
        # Coverage: h1 and h2 (middle nodes)
        # h1: external input from i0, internal output to h2
        # h2: internal input from h1, external output to o0
        # Neither has BOTH external input AND output
        coverage = {"h1", "h2"}

        violations = detect_required_splits(network, coverage)

        # h1 has external input but internal output only
        # h2 has internal input but external output
        # Neither has both -> no violations
        assert len(violations) == 0

    def test_disabled_connections_ignored(self):
        """Disabled connections should not cause violations."""
        nodes = [
            NetworkNode(id="i0", type=NodeType.INPUT),
            NetworkNode(id="h1", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
            NetworkNode(id="o0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
            NetworkNode(id="o1", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ]
        connections = [
            NetworkConnection(from_node="i0", to_node="h1", weight=1.0, enabled=True),
            NetworkConnection(from_node="h1", to_node="o0", weight=1.0, enabled=True),
            # This connection is DISABLED
            NetworkConnection(from_node="h1", to_node="o1", weight=1.0, enabled=False),
        ]
        network = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["i0"],
            output_node_ids=["o0", "o1"],
        )

        # Coverage: h1 only
        coverage = {"h1"}

        violations = detect_required_splits(network, coverage)

        # h1 has external input (i0) and external output (o0)
        # But o1 connection is disabled, so only one external output
        assert len(violations) == 1


@pytest.mark.unit
class TestSuggestSplitOperations:
    """Test suggest_split_operations function."""

    def test_suggests_split_for_each_violation(self):
        """Should suggest one split operation per violation."""
        violations = [
            ViolationDetail(
                node_id="h1",
                reason="has_external_input_and_output",
                external_inputs=[("i0", "h1")],
                external_outputs=[("h1", "o1")],
                internal_outputs=[("h1", "h2")],
            ),
            ViolationDetail(
                node_id="h3",
                reason="has_external_input_and_output",
                external_inputs=[("i1", "h3")],
                external_outputs=[("h3", "o2")],
                internal_outputs=[],
            ),
        ]

        operations = suggest_split_operations(violations)

        assert len(operations) == 2
        assert operations[0]["type"] == "split_node"
        assert operations[0]["params"]["node_id"] == "h1"
        assert operations[1]["type"] == "split_node"
        assert operations[1]["params"]["node_id"] == "h3"

    def test_empty_violations_returns_empty_operations(self):
        """No violations should return no operations."""
        violations = []

        operations = suggest_split_operations(violations)

        assert len(operations) == 0


@pytest.mark.unit
class TestAnalyzeCoverageForSplits:
    """Test analyze_coverage_for_splits function."""

    def test_returns_complete_result(self):
        """Should return a complete SplitDetectionResult."""
        network = create_branching_network()
        coverage = ["h1", "h2"]

        result = analyze_coverage_for_splits(network, coverage)

        assert isinstance(result, SplitDetectionResult)
        assert result.proposed_coverage == coverage
        assert len(result.violations) == 1
        assert len(result.suggested_operations) == 1
        assert result.adjusted_coverage is None  # Not computed without actual splits

    def test_no_violations_result(self):
        """Result with no violations has empty lists."""
        network = create_chain_network()
        coverage = ["h1", "h2"]

        result = analyze_coverage_for_splits(network, coverage)

        assert result.violations == []
        assert result.suggested_operations == []


@pytest.mark.unit
class TestIterativeSplitResolution:
    """Test iterative_split_resolution function."""

    def test_iterative_resolution_removes_violating_nodes(self):
        """Violating nodes should be removed from coverage."""
        network = create_branching_network()
        coverage = {"h1", "h2"}

        final_coverage, operations = iterative_split_resolution(network, coverage)

        # h1 violates, should be split
        assert "h1" not in final_coverage
        # Since h1 has internal outputs (to h2), a placeholder should be added
        assert "h1_internal" in final_coverage
        # h2 should still be in coverage
        assert "h2" in final_coverage

        # Should have one split operation
        assert len(operations) == 1
        assert operations[0]["params"]["node_id"] == "h1"

    def test_iterative_resolution_with_no_violations(self):
        """Coverage with no violations should remain unchanged."""
        network = create_chain_network()
        coverage = {"h1", "h2"}

        final_coverage, operations = iterative_split_resolution(network, coverage)

        assert final_coverage == coverage
        assert len(operations) == 0

    def test_iterative_resolution_max_iterations(self):
        """Should stop after max_iterations."""
        network = create_branching_network()
        coverage = {"h1", "h2"}

        # With max_iterations=1, should only do one round
        final_coverage, operations = iterative_split_resolution(
            network, coverage, max_iterations=1
        )

        # Should have processed at least one violation
        assert len(operations) >= 1
