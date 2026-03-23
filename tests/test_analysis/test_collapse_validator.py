"""
Tests for collapse validation.

Tests the collapse operation, preconditions, round-trip identity,
composition property, and fix suggestions.

Uses graph topologies inspired by the Beyond Intuition paper examples.
"""

import pytest

from explaneat.analysis.collapse_validator import (
    CollapseGraph,
    CollapseAnnotation,
    CollapseValidator,
    PreconditionType,
    FixType,
)


# =============================================================================
# Fixtures: Common graph topologies
# =============================================================================


def make_linear_regression_graph():
    """
    Linear regression: y = ax + by + c

    Graph:
        x(-1) --w_a--> H(1) ---> O(0)
        y(-2) --w_b--> H(1)
        bias(-3) --w_c--> O(0)

    Simplified as:
        -1 -> 1 -> 0
        -2 -> 1
        -3 -> 0
    """
    return CollapseGraph.from_sets(
        nodes={"-1", "-2", "-3", "1", "0"},
        edges={("-1", "1"), ("-2", "1"), ("1", "0"), ("-3", "0")},
        input_nodes={"-1", "-2", "-3"},
        output_nodes={"0"},
    )


def make_diamond_graph():
    """
    Diamond topology:
        -1 -> A -> C -> 0
        -1 -> B -> C -> 0

    Nodes: -1 (input), A, B, C, 0 (output)
    """
    return CollapseGraph.from_sets(
        nodes={"-1", "A", "B", "C", "0"},
        edges={("-1", "A"), ("-1", "B"), ("A", "C"), ("B", "C"), ("C", "0")},
        input_nodes={"-1"},
        output_nodes={"0"},
    )


def make_branching_graph():
    """
    Branching graph (entry has external output):
        X -> Y -> Z -> Out
        X -> W (external)

    Annotation covers {X, Y, Z}, entry={X}, exit={Z}
    X has external output to W — violates Precondition 2
    """
    return CollapseGraph.from_sets(
        nodes={"X", "Y", "Z", "W", "Out"},
        edges={("X", "Y"), ("Y", "Z"), ("Z", "Out"), ("X", "W")},
        input_nodes={"X"},
        output_nodes={"Out"},
    )


def make_external_input_to_exit_graph():
    """
    Graph where exit has external input (violates Precondition 3):
        X -> Y -> Z -> Out
        W -> Z (external input to exit)

    Annotation covers {X, Y, Z}, entry={X}, exit={Z}
    """
    return CollapseGraph.from_sets(
        nodes={"X", "Y", "Z", "W", "Out"},
        edges={("X", "Y"), ("Y", "Z"), ("Z", "Out"), ("W", "Z")},
        input_nodes={"X"},
        output_nodes={"Out"},
    )


def make_composition_graph():
    """
    Composition: g(x, y, z) = f(x, y) + cz

    With identity node F for composition:
        x(-1) -> H1 -> F -> O(0)
        y(-2) -> H1
        F -> O(0)
        z(-3) -> O(0)

    f covers: entry={-1, -2}, exit={F}, nodes={-1, -2, H1, F}
    g covers: entry={F, -3}, exit={0}, nodes={F, -3, 0}
    """
    return CollapseGraph.from_sets(
        nodes={"-1", "-2", "-3", "H1", "F", "0"},
        edges={("-1", "H1"), ("-2", "H1"), ("H1", "F"), ("F", "0"), ("-3", "0")},
        input_nodes={"-1", "-2", "-3"},
        output_nodes={"0"},
    )


# =============================================================================
# Test Preconditions
# =============================================================================


@pytest.mark.unit
class TestCollapsePreconditions:
    """Test the three collapse preconditions."""

    def test_valid_annotation_passes_all_preconditions(self):
        """A well-formed annotation with clean boundaries passes all checks."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_precondition1_entry_only_ingress_violation(self):
        """External input to non-entry node violates Precondition 1."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C", "W"},
            edges={("A", "B"), ("B", "C"), ("W", "B")},
            input_nodes={"A", "W"},
            output_nodes={"C"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"A"},
            exit_nodes={"C"},
            subgraph_nodes={"A", "B", "C"},
            subgraph_edges={("A", "B"), ("B", "C")},
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert not result.is_valid
        assert len(result.entry_ingress_violations) > 0
        assert result.entry_ingress_violations[0].node == "B"

    def test_precondition2_exit_only_egress_violation(self):
        """Non-exit node with external output violates Precondition 2."""
        graph = make_branching_graph()
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"X"},
            exit_nodes={"Z"},
            subgraph_nodes={"X", "Y", "Z"},
            subgraph_edges={("X", "Y"), ("Y", "Z")},
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert not result.is_valid
        # X is entry, has external output to W -> violates precondition 2
        egress_violations = result.exit_egress_violations
        assert len(egress_violations) > 0
        assert any(v.node == "X" for v in egress_violations)

    def test_precondition3_pure_exits_violation(self):
        """Exit node with external input violates Precondition 3."""
        graph = make_external_input_to_exit_graph()
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"X"},
            exit_nodes={"Z"},
            subgraph_nodes={"X", "Y", "Z"},
            subgraph_edges={("X", "Y"), ("Y", "Z")},
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert not result.is_valid
        pure_exit_violations = result.pure_exit_violations
        assert len(pure_exit_violations) > 0
        assert pure_exit_violations[0].node == "Z"

    def test_multiple_precondition_violations(self):
        """Multiple preconditions can be violated simultaneously."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C", "W1", "W2"},
            edges={
                ("A", "B"),
                ("B", "C"),
                ("A", "W1"),  # entry has external output (P2)
                ("W2", "C"),  # exit has external input (P3)
            },
            input_nodes={"A", "W2"},
            output_nodes=set(),
        )
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"A"},
            exit_nodes={"C"},
            subgraph_nodes={"A", "B", "C"},
            subgraph_edges={("A", "B"), ("B", "C")},
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert not result.is_valid
        assert len(result.exit_egress_violations) > 0
        assert len(result.pure_exit_violations) > 0

    def test_single_node_annotation_valid(self):
        """Annotation with a single node (entry=exit) can be valid."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C"},
            edges={("A", "B"), ("B", "C")},
            input_nodes={"A"},
            output_nodes={"C"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="single",
            entry_nodes={"B"},
            exit_nodes={"B"},
            subgraph_nodes={"B"},
            subgraph_edges=set(),
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        # B has external input A and external output C
        # B is entry, so A->B is ok (P1)
        # B is exit, so B->C is ok (P2)
        # But A is external and B is exit: P3 violated (A->B, A not in V_A)
        assert not result.is_valid

    def test_no_external_connections_valid(self):
        """Annotation with no external connections is trivially valid."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B"},
            edges={("A", "B")},
            input_nodes={"A"},
            output_nodes={"B"},
        )
        # Annotation covers the entire graph
        annotation = CollapseAnnotation.from_sets(
            id="full",
            entry_nodes={"A"},
            exit_nodes={"B"},
            subgraph_nodes={"A", "B"},
            subgraph_edges={("A", "B")},
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert result.is_valid


# =============================================================================
# Test Collapse Operation
# =============================================================================


@pytest.mark.unit
class TestCollapseOperation:
    """Test the collapse operation itself."""

    def test_collapse_linear_annotation(self):
        """Collapse a linear annotation: entry nodes stay, internals replaced."""
        graph = make_linear_regression_graph()
        # Annotation: -1, -2 are entries, 1 is exit
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # Entry nodes -1, -2 should be preserved
        assert "-1" in result.graph.nodes
        assert "-2" in result.graph.nodes
        # Internal node 1 (exit) should be removed
        assert "1" not in result.graph.nodes
        # Annotation node should exist
        assert result.annotation_node_id in result.graph.nodes
        # Original non-annotation nodes preserved
        assert "-3" in result.graph.nodes
        assert "0" in result.graph.nodes

    def test_collapse_preserves_external_edges(self):
        """Edges not involving the annotation are preserved."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # -3 -> 0 should be preserved (completely external to annotation)
        assert ("-3", "0") in result.graph.edges

    def test_collapse_creates_entry_to_annotation_edges(self):
        """Entry nodes get edges to the annotation node."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # Both entries should connect to annotation node
        assert ("-1", result.annotation_node_id) in result.graph.edges
        assert ("-2", result.annotation_node_id) in result.graph.edges

    def test_collapse_creates_annotation_to_external_edges(self):
        """Annotation node gets edges to where exit nodes connected externally."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # Exit node 1 connected to 0: annotation node should connect to 0
        assert (result.annotation_node_id, "0") in result.graph.edges

    def test_collapse_removes_internal_edges(self):
        """Internal edges (between subgraph nodes) are removed."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # Original internal edges should not exist
        assert ("-1", "1") not in result.graph.edges
        assert ("-2", "1") not in result.graph.edges
        # But entry-to-annotation edges should
        assert ("-1", result.annotation_node_id) in result.graph.edges

    def test_collapse_raises_on_precondition_violation(self):
        """Collapse raises ValueError when preconditions are violated."""
        graph = make_branching_graph()
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"X"},
            exit_nodes={"Z"},
            subgraph_nodes={"X", "Y", "Z"},
            subgraph_edges={("X", "Y"), ("Y", "Z")},
        )

        with pytest.raises(ValueError, match="preconditions violated"):
            CollapseValidator.collapse(graph, annotation)

    def test_collapse_diamond_topology(self):
        """Collapse an annotation within a diamond graph."""
        graph = make_diamond_graph()
        # Annotate the A branch: entry={-1}, exit={A}, internal={}
        # Wait — if entry={-1} and exit={A}, internal={A}\{-1}={A}
        # But A has no external inputs (only -1 which is entry)
        # But -1 is entry, has external output to B: P2 violation

        # Better: annotate the whole diamond interior
        annotation = CollapseAnnotation.from_sets(
            id="diamond",
            entry_nodes={"-1"},
            exit_nodes={"C"},
            subgraph_nodes={"-1", "A", "B", "C"},
            subgraph_edges={("-1", "A"), ("-1", "B"), ("A", "C"), ("B", "C")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # -1 (entry) preserved, A, B, C (internal) removed
        assert "-1" in result.graph.nodes
        assert "A" not in result.graph.nodes
        assert "B" not in result.graph.nodes
        assert "C" not in result.graph.nodes
        # Annotation node and output preserved
        assert result.annotation_node_id in result.graph.nodes
        assert "0" in result.graph.nodes
        # Edges: -1 -> a_A, a_A -> 0
        assert ("-1", result.annotation_node_id) in result.graph.edges
        assert (result.annotation_node_id, "0") in result.graph.edges

    def test_collapse_multi_exit(self):
        """Collapse annotation with multiple exit nodes."""
        graph = CollapseGraph.from_sets(
            nodes={"I", "H", "E1", "E2", "O1", "O2"},
            edges={("I", "H"), ("H", "E1"), ("H", "E2"), ("E1", "O1"), ("E2", "O2")},
            input_nodes={"I"},
            output_nodes={"O1", "O2"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="multi_exit",
            entry_nodes={"I"},
            exit_nodes={"E1", "E2"},
            subgraph_nodes={"I", "H", "E1", "E2"},
            subgraph_edges={("I", "H"), ("H", "E1"), ("H", "E2")},
        )

        result = CollapseValidator.collapse(graph, annotation)

        # Entry I preserved, H, E1, E2 internal (removed)
        assert "I" in result.graph.nodes
        assert "H" not in result.graph.nodes
        assert "E1" not in result.graph.nodes
        assert "E2" not in result.graph.nodes
        # a_A -> O1 and a_A -> O2
        assert (result.annotation_node_id, "O1") in result.graph.edges
        assert (result.annotation_node_id, "O2") in result.graph.edges

    def test_collapse_no_cycle_on_dag(self):
        """Collapsing a valid annotation on a DAG produces no cycles."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)
        assert not CollapseValidator._has_cycle(result.graph)


# =============================================================================
# Test Round Trip
# =============================================================================


@pytest.mark.unit
class TestRoundTrip:
    """Test collapse + expand = identity."""

    def test_round_trip_linear(self):
        """Round-trip on linear regression graph."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.validate_round_trip(graph, annotation)
        assert result.is_valid, f"Round-trip failed: {result.errors}"

    def test_round_trip_diamond(self):
        """Round-trip on diamond graph."""
        graph = make_diamond_graph()
        annotation = CollapseAnnotation.from_sets(
            id="diamond",
            entry_nodes={"-1"},
            exit_nodes={"C"},
            subgraph_nodes={"-1", "A", "B", "C"},
            subgraph_edges={("-1", "A"), ("-1", "B"), ("A", "C"), ("B", "C")},
        )

        result = CollapseValidator.validate_round_trip(graph, annotation)
        assert result.is_valid, f"Round-trip failed: {result.errors}"

    def test_round_trip_multi_exit(self):
        """Round-trip on multi-exit annotation."""
        graph = CollapseGraph.from_sets(
            nodes={"I", "H", "E1", "E2", "O1", "O2"},
            edges={("I", "H"), ("H", "E1"), ("H", "E2"), ("E1", "O1"), ("E2", "O2")},
            input_nodes={"I"},
            output_nodes={"O1", "O2"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="multi_exit",
            entry_nodes={"I"},
            exit_nodes={"E1", "E2"},
            subgraph_nodes={"I", "H", "E1", "E2"},
            subgraph_edges={("I", "H"), ("H", "E1"), ("H", "E2")},
        )

        result = CollapseValidator.validate_round_trip(graph, annotation)
        assert result.is_valid, f"Round-trip failed: {result.errors}"

    def test_round_trip_full_graph_annotation(self):
        """Round-trip when annotation covers the entire graph."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B"},
            edges={("A", "B")},
            input_nodes={"A"},
            output_nodes={"B"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="full",
            entry_nodes={"A"},
            exit_nodes={"B"},
            subgraph_nodes={"A", "B"},
            subgraph_edges={("A", "B")},
        )

        result = CollapseValidator.validate_round_trip(graph, annotation)
        assert result.is_valid, f"Round-trip failed: {result.errors}"


# =============================================================================
# Test Composition
# =============================================================================


@pytest.mark.unit
class TestComposition:
    """Test hierarchical composition of annotations."""

    def test_composition_with_identity_node(self):
        """
        Test composition: g(x,y,z) = f(x,y) + cz with identity node F.

        f: entry={-1,-2}, exit={F}, nodes={-1,-2,H1,F}
        g: entry={F,-3}, exit={0}, nodes={F,-3,0}
        """
        graph = make_composition_graph()

        f_ann = CollapseAnnotation.from_sets(
            id="f",
            entry_nodes={"-1", "-2"},
            exit_nodes={"F"},
            subgraph_nodes={"-1", "-2", "H1", "F"},
            subgraph_edges={("-1", "H1"), ("-2", "H1"), ("H1", "F")},
        )

        g_ann = CollapseAnnotation.from_sets(
            id="g",
            entry_nodes={"F", "-3"},
            exit_nodes={"0"},
            subgraph_nodes={"F", "-3", "0"},
            subgraph_edges={("F", "0"), ("-3", "0")},
        )

        # Both should be individually collapsible
        assert CollapseValidator.validate_collapsible(graph, f_ann).is_valid
        assert CollapseValidator.validate_collapsible(graph, g_ann).is_valid

        # Collapse f first
        f_result = CollapseValidator.collapse(graph, f_ann)
        collapsed_after_f = f_result.graph

        # After collapsing f: -1, -2 stay, H1 and F removed, a_f added
        assert "-1" in collapsed_after_f.nodes
        assert "-2" in collapsed_after_f.nodes
        assert "H1" not in collapsed_after_f.nodes
        assert "F" not in collapsed_after_f.nodes
        assert "a_f" in collapsed_after_f.nodes
        # a_f -> 0 (F's external edge)
        assert ("a_f", "0") in collapsed_after_f.edges

    def test_collapse_child_then_parent(self):
        """Collapse child f, then parent g. Result should match direct parent collapse."""
        graph = make_composition_graph()

        f_ann = CollapseAnnotation.from_sets(
            id="f",
            entry_nodes={"-1", "-2"},
            exit_nodes={"F"},
            subgraph_nodes={"-1", "-2", "H1", "F"},
            subgraph_edges={("-1", "H1"), ("-2", "H1"), ("H1", "F")},
        )

        # g is a parent that includes f's nodes plus its own
        g_parent = CollapseAnnotation.from_sets(
            id="g",
            entry_nodes={"-1", "-2", "-3"},
            exit_nodes={"0"},
            subgraph_nodes={"-1", "-2", "-3", "H1", "F", "0"},
            subgraph_edges={
                ("-1", "H1"),
                ("-2", "H1"),
                ("H1", "F"),
                ("F", "0"),
                ("-3", "0"),
            },
        )

        result = CollapseValidator.validate_composition(
            graph, g_parent, [f_ann]
        )
        assert result.is_valid, f"Composition failed: {result.errors}"

    def test_sibling_annotations_no_overlap(self):
        """Two sibling annotations with no overlapping nodes."""
        graph = CollapseGraph.from_sets(
            nodes={"-1", "-2", "A", "B", "F1", "F2", "0"},
            edges={
                ("-1", "A"),
                ("A", "F1"),
                ("-2", "B"),
                ("B", "F2"),
                ("F1", "0"),
                ("F2", "0"),
            },
            input_nodes={"-1", "-2"},
            output_nodes={"0"},
        )

        ann1 = CollapseAnnotation.from_sets(
            id="left",
            entry_nodes={"-1"},
            exit_nodes={"F1"},
            subgraph_nodes={"-1", "A", "F1"},
            subgraph_edges={("-1", "A"), ("A", "F1")},
        )

        ann2 = CollapseAnnotation.from_sets(
            id="right",
            entry_nodes={"-2"},
            exit_nodes={"F2"},
            subgraph_nodes={"-2", "B", "F2"},
            subgraph_edges={("-2", "B"), ("B", "F2")},
        )

        # Both should be collapsible
        assert CollapseValidator.validate_collapsible(graph, ann1).is_valid
        assert CollapseValidator.validate_collapsible(graph, ann2).is_valid

        # Collapse both sequentially
        r1 = CollapseValidator.collapse(graph, ann1)
        r2 = CollapseValidator.collapse(r1.graph, ann2)

        # Both annotation nodes should exist
        assert "a_left" in r2.graph.nodes
        assert "a_right" in r2.graph.nodes
        # Entries preserved
        assert "-1" in r2.graph.nodes
        assert "-2" in r2.graph.nodes
        # Internals removed
        assert "A" not in r2.graph.nodes
        assert "B" not in r2.graph.nodes


# =============================================================================
# Test Collapsed Graph Validation
# =============================================================================


@pytest.mark.unit
class TestCollapsedGraphValidation:
    """Test validation of collapsed graphs."""

    def test_valid_collapsed_graph(self):
        """A correctly collapsed graph passes validation."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        result = CollapseValidator.collapse(graph, annotation)
        validation = CollapseValidator.validate_collapsed_graph(
            graph, result.graph, annotation
        )
        assert validation.is_valid, f"Validation failed: {validation.errors}"

    def test_invalid_collapsed_graph_missing_annotation_node(self):
        """Collapsed graph missing annotation node is invalid."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        # Create a bad collapsed graph (missing annotation node)
        bad_collapsed = CollapseGraph.from_sets(
            nodes={"-1", "-2", "-3", "0"},
            edges={("-3", "0")},
            input_nodes={"-1", "-2", "-3"},
            output_nodes={"0"},
        )

        validation = CollapseValidator.validate_collapsed_graph(
            graph, bad_collapsed, annotation
        )
        assert not validation.is_valid
        assert any("not in collapsed graph" in e for e in validation.errors)

    def test_invalid_collapsed_graph_internal_node_still_present(self):
        """Collapsed graph with internal nodes still present is invalid."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        # Bad collapsed: internal node 1 still present
        bad_collapsed = CollapseGraph.from_sets(
            nodes={"-1", "-2", "-3", "1", "0", "a_weighted_sum"},
            edges={("-1", "a_weighted_sum"), ("-2", "a_weighted_sum"), ("a_weighted_sum", "0"), ("-3", "0")},
            input_nodes={"-1", "-2", "-3"},
            output_nodes={"0"},
        )

        validation = CollapseValidator.validate_collapsed_graph(
            graph, bad_collapsed, annotation
        )
        assert not validation.is_valid
        assert any("still in collapsed graph" in e for e in validation.errors)

    def test_invalid_collapsed_graph_entry_removed(self):
        """Collapsed graph with entry node removed is invalid."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        # Bad collapsed: entry -1 removed
        bad_collapsed = CollapseGraph.from_sets(
            nodes={"-2", "-3", "0", "a_weighted_sum"},
            edges={("-2", "a_weighted_sum"), ("a_weighted_sum", "0"), ("-3", "0")},
            input_nodes={"-2", "-3"},
            output_nodes={"0"},
        )

        validation = CollapseValidator.validate_collapsed_graph(
            graph, bad_collapsed, annotation
        )
        assert not validation.is_valid
        assert any("Entry node" in e for e in validation.errors)

    def test_cycle_detection(self):
        """Detect cycles in a manually constructed graph."""
        cyclic_graph = CollapseGraph.from_sets(
            nodes={"A", "B"},
            edges={("A", "B"), ("B", "A")},
            input_nodes=set(),
            output_nodes=set(),
        )
        assert CollapseValidator._has_cycle(cyclic_graph)

        acyclic_graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C"},
            edges={("A", "B"), ("B", "C")},
            input_nodes={"A"},
            output_nodes={"C"},
        )
        assert not CollapseValidator._has_cycle(acyclic_graph)


# =============================================================================
# Test Fix Suggestions
# =============================================================================


@pytest.mark.unit
class TestSuggestFixes:
    """Test fix suggestion logic."""

    def test_no_fixes_for_valid_annotation(self):
        """No fixes suggested for a valid annotation."""
        graph = make_linear_regression_graph()
        annotation = CollapseAnnotation.from_sets(
            id="weighted_sum",
            entry_nodes={"-1", "-2"},
            exit_nodes={"1"},
            subgraph_nodes={"-1", "-2", "1"},
            subgraph_edges={("-1", "1"), ("-2", "1")},
        )

        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        assert len(fixes) == 0

    def test_suggest_expand_for_external_input_to_intermediate(self):
        """Suggest expand selection for Precondition 1 violation."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C", "W"},
            edges={("A", "B"), ("B", "C"), ("W", "B")},
            input_nodes={"A", "W"},
            output_nodes={"C"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"A"},
            exit_nodes={"C"},
            subgraph_nodes={"A", "B", "C"},
            subgraph_edges={("A", "B"), ("B", "C")},
        )

        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        expand_fixes = [f for f in fixes if f.fix_type == FixType.EXPAND_SELECTION]
        assert len(expand_fixes) > 0
        assert expand_fixes[0].node == "B"

    def test_suggest_split_for_external_output_from_entry(self):
        """Suggest split for Precondition 2 violation."""
        graph = make_branching_graph()
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"X"},
            exit_nodes={"Z"},
            subgraph_nodes={"X", "Y", "Z"},
            subgraph_edges={("X", "Y"), ("Y", "Z")},
        )

        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        split_fixes = [f for f in fixes if f.fix_type == FixType.SPLIT_NODE]
        assert len(split_fixes) > 0
        assert split_fixes[0].node == "X"

    def test_suggest_identity_for_external_input_to_exit(self):
        """Suggest identity node for Precondition 3 violation."""
        graph = make_external_input_to_exit_graph()
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"X"},
            exit_nodes={"Z"},
            subgraph_nodes={"X", "Y", "Z"},
            subgraph_edges={("X", "Y"), ("Y", "Z")},
        )

        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        identity_fixes = [f for f in fixes if f.fix_type == FixType.IDENTITY_NODE]
        assert len(identity_fixes) > 0
        assert identity_fixes[0].node == "Z"
        assert identity_fixes[0].details["external_source"] == "W"

    def test_suggest_multiple_fix_types(self):
        """Multiple fix types suggested for combined violations."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C", "W1", "W2"},
            edges={
                ("A", "B"),
                ("B", "C"),
                ("A", "W1"),  # P2: entry with external output
                ("W2", "C"),  # P3: exit with external input
            },
            input_nodes={"A", "W2"},
            output_nodes=set(),
        )
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"A"},
            exit_nodes={"C"},
            subgraph_nodes={"A", "B", "C"},
            subgraph_edges={("A", "B"), ("B", "C")},
        )

        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        fix_types = {f.fix_type for f in fixes}
        assert FixType.SPLIT_NODE in fix_types
        assert FixType.IDENTITY_NODE in fix_types

    def test_deduplicate_fixes(self):
        """Multiple violations on the same node don't produce duplicate fixes."""
        graph = CollapseGraph.from_sets(
            nodes={"A", "B", "C", "W1", "W2"},
            edges={
                ("A", "B"),
                ("B", "C"),
                ("A", "W1"),  # P2 violation for A
                ("A", "W2"),  # Another P2 violation for A
            },
            input_nodes={"A"},
            output_nodes={"C"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="test",
            entry_nodes={"A"},
            exit_nodes={"C"},
            subgraph_nodes={"A", "B", "C"},
            subgraph_edges={("A", "B"), ("B", "C")},
        )

        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        split_fixes = [f for f in fixes if f.fix_type == FixType.SPLIT_NODE]
        # Should be deduplicated to one fix per node
        assert len(split_fixes) == 1
        assert split_fixes[0].node == "A"


# =============================================================================
# Test the concrete bug scenario from the plan
# =============================================================================


@pytest.mark.unit
class TestConcreteBugScenario:
    """
    Test the specific bug scenario described in the plan:
    absorbing entry nodes causes cycles.
    """

    def test_absorbing_entry_would_cause_cycle(self):
        """
        The bug: X(entry), Y(intermediate), Z(exit), W(external)
        Edges: X->Y, Y->Z, Z->Out, X->W, W->Z

        If X is absorbed: a_A->W (from X->W) and W->a_A (from W->Z) => cycle!
        Correct: X stays, precondition check catches W->Z as P3 violation.
        """
        graph = CollapseGraph.from_sets(
            nodes={"X", "Y", "Z", "W", "Out"},
            edges={("X", "Y"), ("Y", "Z"), ("Z", "Out"), ("X", "W"), ("W", "Z")},
            input_nodes={"X"},
            output_nodes={"Out"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="bug",
            entry_nodes={"X"},
            exit_nodes={"Z"},
            subgraph_nodes={"X", "Y", "Z"},
            subgraph_edges={("X", "Y"), ("Y", "Z")},
        )

        # Should NOT be collapsible due to:
        # P2: X (entry) has external output to W
        # P3: Z (exit) has external input from W
        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert not result.is_valid

        # Should suggest: split X, identity for Z
        fixes = CollapseValidator.suggest_fixes(graph, annotation)
        fix_types = {f.fix_type for f in fixes}
        assert FixType.SPLIT_NODE in fix_types  # for X -> W
        assert FixType.IDENTITY_NODE in fix_types  # for W -> Z

    def test_fixed_bug_scenario_with_identity_node(self):
        """
        After adding identity node F:
        - F replaces Z as exit, Z leaves annotation
        - X is split: X_internal (entry), X_external (exit for W connection)

        Graph becomes:
            X -> Y -> F -> Z -> Out
            X -> W -> Z

        Annotation: entry={X}, exit={F}, nodes={X, Y, F}
        But X still has external output to W (P2), so needs split.

        After split:
            X_int -> Y -> F (annotation: entry={X_int}, exit={F})
            X_ext -> W -> Z -> Out

        This is now collapsible.
        """
        # After identity node and split
        graph = CollapseGraph.from_sets(
            nodes={"X_int", "X_ext", "Y", "F", "Z", "W", "Out"},
            edges={
                ("X_int", "Y"),
                ("Y", "F"),
                ("F", "Z"),
                ("X_ext", "W"),
                ("W", "Z"),
                ("Z", "Out"),
            },
            input_nodes={"X_int", "X_ext"},
            output_nodes={"Out"},
        )
        annotation = CollapseAnnotation.from_sets(
            id="fixed",
            entry_nodes={"X_int"},
            exit_nodes={"F"},
            subgraph_nodes={"X_int", "Y", "F"},
            subgraph_edges={("X_int", "Y"), ("Y", "F")},
        )

        # Should now be collapsible
        result = CollapseValidator.validate_collapsible(graph, annotation)
        assert result.is_valid, f"Should be valid: {[v.message for v in result.violations]}"

        # Collapse should produce no cycles
        collapse_result = CollapseValidator.collapse(graph, annotation)
        assert not CollapseValidator._has_cycle(collapse_result.graph)

        # Entry X_int stays, Y and F replaced by a_fixed
        assert "X_int" in collapse_result.graph.nodes
        assert "Y" not in collapse_result.graph.nodes
        assert "F" not in collapse_result.graph.nodes
        assert "a_fixed" in collapse_result.graph.nodes
        # a_fixed -> Z (F's external connection)
        assert ("a_fixed", "Z") in collapse_result.graph.edges
