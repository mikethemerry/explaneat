"""Tests for collapse_structure transform.

Verifies that collapse_structure produces a derived NetworkStructure where
collapsed annotations are replaced by function nodes, without cycles or
dangling references.
"""

import pytest
from typing import Set, List, Dict, Tuple

from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
    FunctionNodeMetadata,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.collapse_transform import collapse_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(id: str, type: NodeType = NodeType.HIDDEN, **kwargs) -> NetworkNode:
    """Helper to create a node with minimal boilerplate."""
    return NetworkNode(id=id, type=type, **kwargs)


def _make_conn(
    from_node: str, to_node: str, weight: float = 1.0, enabled: bool = True, **kwargs
) -> NetworkConnection:
    """Helper to create a connection with minimal boilerplate."""
    return NetworkConnection(
        from_node=from_node, to_node=to_node, weight=weight, enabled=enabled, **kwargs
    )


def _make_annotation(
    name: str,
    entry_nodes: List[str],
    exit_nodes: List[str],
    subgraph_nodes: List[str],
    subgraph_connections: List[Tuple[str, str]],
    hypothesis: str = "test hypothesis",
) -> AnnotationData:
    """Helper to create an AnnotationData."""
    return AnnotationData(
        name=name,
        hypothesis=hypothesis,
        entry_nodes=entry_nodes,
        exit_nodes=exit_nodes,
        subgraph_nodes=subgraph_nodes,
        subgraph_connections=subgraph_connections,
    )


def _has_cycle(structure: NetworkStructure) -> bool:
    """Check if a NetworkStructure has any cycles using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    node_ids = {n.id for n in structure.nodes}
    color: Dict[str, int] = {nid: WHITE for nid in node_ids}

    # Build adjacency list from enabled connections
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for conn in structure.connections:
        if conn.enabled and conn.from_node in adj:
            adj[conn.from_node].append(conn.to_node)

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbor in adj.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                return True  # back edge => cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for nid in node_ids:
        if color[nid] == WHITE:
            if dfs(nid):
                return True
    return False


def _node_ids(structure: NetworkStructure) -> Set[str]:
    """Get set of node IDs in a structure."""
    return {n.id for n in structure.nodes}


def _conn_tuples(structure: NetworkStructure) -> Set[Tuple[str, str]]:
    """Get set of (from, to) tuples for all connections."""
    return {(c.from_node, c.to_node) for c in structure.connections}


# ---------------------------------------------------------------------------
# Fixtures: hand-built NetworkStructures
# ---------------------------------------------------------------------------


def _linear_chain_structure() -> NetworkStructure:
    """
    Linear chain: in1 -> h1 -> h2 -> out1

    Annotation covers {h1, h2} with entry={h1}, exit={h2}.
    h1 is entry (preserved as node), h2 is internal (collapsed).
    """
    nodes = [
        _make_node("-1", NodeType.INPUT),
        _make_node("h1", NodeType.HIDDEN, bias=0.1, activation="relu"),
        _make_node("h2", NodeType.HIDDEN, bias=0.2, activation="relu"),
        _make_node("0", NodeType.OUTPUT),
    ]
    connections = [
        _make_conn("-1", "h1", weight=0.5),
        _make_conn("h1", "h2", weight=0.8),
        _make_conn("h2", "0", weight=0.3),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _diamond_structure() -> NetworkStructure:
    """
    Diamond: in1 -> h1 -> h3 -> out1
                  -> h2 ->

    Annotation covers {h1, h2, h3} with entry={h1}, exit={h3}.
    """
    nodes = [
        _make_node("-1", NodeType.INPUT),
        _make_node("h1", NodeType.HIDDEN, bias=0.0),
        _make_node("h2", NodeType.HIDDEN, bias=0.1),
        _make_node("h3", NodeType.HIDDEN, bias=0.2),
        _make_node("0", NodeType.OUTPUT),
    ]
    connections = [
        _make_conn("-1", "h1", weight=1.0),
        _make_conn("h1", "h2", weight=0.5),
        _make_conn("h1", "h3", weight=0.6),
        _make_conn("h2", "h3", weight=0.7),
        _make_conn("h3", "0", weight=0.9),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _multi_exit_structure() -> NetworkStructure:
    """
    Multi-exit: in1 -> h1 -> h2 -> out1
                            -> h3 -> out2

    Annotation covers {h1, h2, h3} with entry={h1}, exit={h2, h3}.
    """
    nodes = [
        _make_node("-1", NodeType.INPUT),
        _make_node("h1", NodeType.HIDDEN),
        _make_node("h2", NodeType.HIDDEN),
        _make_node("h3", NodeType.HIDDEN),
        _make_node("0", NodeType.OUTPUT),
        _make_node("1", NodeType.OUTPUT),
    ]
    connections = [
        _make_conn("-1", "h1", weight=1.0),
        _make_conn("h1", "h2", weight=0.5),
        _make_conn("h1", "h3", weight=0.6),
        _make_conn("h2", "0", weight=0.7),
        _make_conn("h3", "1", weight=0.8),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1"],
        output_node_ids=["0", "1"],
    )


def _structure_with_external_bypass() -> NetworkStructure:
    """
    Structure with external bypass connection:
    in1 -> h1 -> h2 -> out1
    in1 ----------------> out1  (bypass)

    Annotation covers {h1, h2} with entry={h1}, exit={h2}.
    The bypass in1->out1 is external and should be preserved.
    """
    nodes = [
        _make_node("-1", NodeType.INPUT),
        _make_node("h1", NodeType.HIDDEN),
        _make_node("h2", NodeType.HIDDEN),
        _make_node("0", NodeType.OUTPUT),
    ]
    connections = [
        _make_conn("-1", "h1", weight=1.0),
        _make_conn("h1", "h2", weight=0.5),
        _make_conn("h2", "0", weight=0.7),
        _make_conn("-1", "0", weight=0.3),  # bypass
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def _compositional_structure() -> NetworkStructure:
    """
    Structure for compositional (parent with children) annotations:

        in1 -> A -> C -> E -> out1
        in2 -> B -> D -/

    Child1 covers {A, C} with entry={A}, exit={C}.
    Child2 covers {B, D} with entry={B}, exit={D}.
    Parent covers the whole thing with entry={A, B}, exit={E},
    but has subgraph_nodes=[] (relies on children).
    """
    nodes = [
        _make_node("-1", NodeType.INPUT),
        _make_node("-2", NodeType.INPUT),
        _make_node("A", NodeType.HIDDEN, bias=0.1, activation="sigmoid"),
        _make_node("B", NodeType.HIDDEN, bias=0.2, activation="relu"),
        _make_node("C", NodeType.HIDDEN, bias=0.3, activation="sigmoid"),
        _make_node("D", NodeType.HIDDEN, bias=0.4, activation="relu"),
        _make_node("E", NodeType.HIDDEN, bias=0.5, activation="sigmoid"),
        _make_node("0", NodeType.OUTPUT),
    ]
    connections = [
        _make_conn("-1", "A", weight=1.0),
        _make_conn("-2", "B", weight=1.0),
        _make_conn("A", "C", weight=0.5),
        _make_conn("B", "D", weight=0.6),
        _make_conn("C", "E", weight=0.7),
        _make_conn("D", "E", weight=0.8),
        _make_conn("E", "0", weight=0.9),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1", "-2"],
        output_node_ids=["0"],
    )


def _compositional_annotations():
    """Create child1, child2, parent annotations for _compositional_structure."""
    child1 = _make_annotation(
        "child1",
        entry_nodes=["A"],
        exit_nodes=["C"],
        subgraph_nodes=["A", "C"],
        subgraph_connections=[("A", "C")],
        hypothesis="First pathway",
    )
    child1.parent_annotation_id = "parent"

    child2 = _make_annotation(
        "child2",
        entry_nodes=["B"],
        exit_nodes=["D"],
        subgraph_nodes=["B", "D"],
        subgraph_connections=[("B", "D")],
        hypothesis="Second pathway",
    )
    child2.parent_annotation_id = "parent"

    parent = _make_annotation(
        "parent",
        entry_nodes=["A", "B"],
        exit_nodes=["E"],
        subgraph_nodes=[],  # Empty! Compositional annotation.
        subgraph_connections=[],
        hypothesis="Combined pathways",
    )
    return child1, child2, parent


def _nested_annotation_structure() -> NetworkStructure:
    """
    Structure for nested annotations:
    in1 -> h1 -> h2 -> h3 -> h4 -> out1

    Child annotation: {h2, h3} with entry={h2}, exit={h3}
    Parent annotation: {h1, h2, h3, h4} with entry={h1}, exit={h4}
    """
    nodes = [
        _make_node("-1", NodeType.INPUT),
        _make_node("h1", NodeType.HIDDEN),
        _make_node("h2", NodeType.HIDDEN),
        _make_node("h3", NodeType.HIDDEN),
        _make_node("h4", NodeType.HIDDEN),
        _make_node("0", NodeType.OUTPUT),
    ]
    connections = [
        _make_conn("-1", "h1", weight=1.0),
        _make_conn("h1", "h2", weight=0.5),
        _make_conn("h2", "h3", weight=0.6),
        _make_conn("h3", "h4", weight=0.7),
        _make_conn("h4", "0", weight=0.8),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCollapseStructureBasic:
    """Basic collapse tests: linear chain, entry preservation, multi-exit, empty set."""

    def test_empty_collapse_set_returns_deepcopy(self):
        """When collapsed_ids is empty, return a deepcopy unchanged."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann1", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], set())
        # Should be structurally identical
        assert _node_ids(result) == _node_ids(structure)
        assert _conn_tuples(result) == _conn_tuples(structure)
        # But a different object (deepcopy)
        assert result is not structure
        assert result.nodes is not structure.nodes

    def test_linear_chain_collapse(self):
        """Collapse a linear chain annotation: internal nodes removed, fn node added."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "relu_chain",
            entry_nodes=["h1"],
            exit_nodes=["h2"],
            subgraph_nodes=["h1", "h2"],
            subgraph_connections=[("h1", "h2")],
        )
        result = collapse_structure(structure, [annotation], {"relu_chain"})

        # Function node should exist
        fn_node = result.get_node_by_id("fn_relu_chain")
        assert fn_node is not None
        assert fn_node.type == NodeType.FUNCTION
        assert fn_node.function_metadata is not None
        assert fn_node.function_metadata.annotation_name == "relu_chain"

        # Internal nodes (h2) should be removed; entry (h1) is preserved
        node_ids = _node_ids(result)
        assert "h1" in node_ids  # entry preserved
        assert "h2" not in node_ids  # internal removed
        assert "fn_relu_chain" in node_ids

        # Input and output nodes preserved
        assert "-1" in node_ids
        assert "0" in node_ids

    def test_entry_nodes_preserved(self):
        """Entry nodes must remain in the structure after collapse."""
        structure = _diamond_structure()
        annotation = _make_annotation(
            "diamond",
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3"), ("h2", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"diamond"})
        node_ids = _node_ids(result)
        assert "h1" in node_ids  # entry preserved
        assert "h2" not in node_ids  # intermediate removed
        assert "h3" not in node_ids  # exit (internal) removed

    def test_multi_exit_with_output_index(self):
        """Multi-exit annotations produce connections with output_index."""
        structure = _multi_exit_structure()
        annotation = _make_annotation(
            "multi",
            entry_nodes=["h1"],
            exit_nodes=["h2", "h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"multi"})

        # Find connections from function node
        fn_conns = [c for c in result.connections if c.from_node == "fn_multi"]
        assert len(fn_conns) >= 2

        # Check output_index is set based on exit_nodes order
        conn_to_0 = [c for c in fn_conns if c.to_node == "0"]
        conn_to_1 = [c for c in fn_conns if c.to_node == "1"]
        assert len(conn_to_0) == 1
        assert len(conn_to_1) == 1
        # h2 is exit_nodes[0], h3 is exit_nodes[1]
        assert conn_to_0[0].output_index == 0  # h2->0 maps to index 0
        assert conn_to_1[0].output_index == 1  # h3->1 maps to index 1

    def test_function_node_metadata_populated(self):
        """FunctionNodeMetadata should have correct n_inputs, n_outputs, etc."""
        structure = _multi_exit_structure()
        annotation = _make_annotation(
            "multi",
            entry_nodes=["h1"],
            exit_nodes=["h2", "h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3")],
            hypothesis="Splits input into two channels",
        )
        result = collapse_structure(structure, [annotation], {"multi"})
        fn_node = result.get_node_by_id("fn_multi")
        meta = fn_node.function_metadata
        assert meta.n_inputs == 1
        assert meta.n_outputs == 2
        assert meta.input_names == ["h1"]
        assert meta.output_names == ["h2", "h3"]
        assert meta.hypothesis == "Splits input into two channels"
        assert set(meta.subgraph_nodes) == {"h1", "h2", "h3"}

    def test_structure_validates_after_collapse(self):
        """The resulting structure should pass validate()."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})
        validation = result.validate()
        assert validation["is_valid"] is True, f"Errors: {validation['errors']}"

    def test_collapse_does_not_mutate_input(self):
        """collapse_structure must not modify the input structure."""
        structure = _linear_chain_structure()
        original_node_ids = _node_ids(structure)
        original_conn_tuples = _conn_tuples(structure)
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        collapse_structure(structure, [annotation], {"ann"})
        # Input should be unchanged
        assert _node_ids(structure) == original_node_ids
        assert _conn_tuples(structure) == original_conn_tuples


class TestCollapseStructureCycleFreedom:
    """Verify that collapse never introduces cycles."""

    def test_linear_chain_no_cycle(self):
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})
        assert not _has_cycle(result), "Cycle detected after collapsing linear chain"

    def test_diamond_no_cycle(self):
        structure = _diamond_structure()
        annotation = _make_annotation(
            "ann",
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3"), ("h2", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"ann"})
        assert not _has_cycle(result), "Cycle detected after collapsing diamond"

    def test_multi_exit_no_cycle(self):
        structure = _multi_exit_structure()
        annotation = _make_annotation(
            "ann",
            entry_nodes=["h1"],
            exit_nodes=["h2", "h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"ann"})
        assert not _has_cycle(result), "Cycle detected after collapsing multi-exit"

    def test_bypass_no_cycle(self):
        structure = _structure_with_external_bypass()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})
        assert not _has_cycle(result), "Cycle detected after collapsing with bypass"

    def test_nested_collapse_no_cycle(self):
        """Collapsing child then parent (via children-first ordering) has no cycle."""
        structure = _nested_annotation_structure()
        child = _make_annotation(
            "child",
            entry_nodes=["h2"],
            exit_nodes=["h3"],
            subgraph_nodes=["h2", "h3"],
            subgraph_connections=[("h2", "h3")],
        )
        parent = _make_annotation(
            "parent",
            entry_nodes=["h1"],
            exit_nodes=["h4"],
            subgraph_nodes=["h1", "h2", "h3", "h4"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3"), ("h3", "h4")],
        )
        result = collapse_structure(
            structure, [child, parent], {"child", "parent"}
        )
        assert not _has_cycle(result), "Cycle detected after nested collapse"


class TestCollapseStructureConnections:
    """Test connection rewiring details."""

    def test_entry_to_function_node_connections(self):
        """Entry nodes should have connections to the function node."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        # h1 -> fn_ann should exist
        entry_to_fn = [
            c for c in result.connections
            if c.from_node == "h1" and c.to_node == "fn_ann"
        ]
        assert len(entry_to_fn) == 1

    def test_function_node_to_downstream_connections(self):
        """Function node should connect to downstream nodes with original weight."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        # fn_ann -> 0 should exist with weight from h2->0 (0.3)
        fn_to_out = [
            c for c in result.connections
            if c.from_node == "fn_ann" and c.to_node == "0"
        ]
        assert len(fn_to_out) == 1
        assert fn_to_out[0].weight == pytest.approx(0.3)

    def test_external_connections_preserved(self):
        """Connections not involving annotation internal nodes should be preserved."""
        structure = _structure_with_external_bypass()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        conn_set = _conn_tuples(result)
        # The bypass connection in1->out1 should still exist
        assert ("-1", "0") in conn_set
        # The upstream connection in1->h1 should still exist
        assert ("-1", "h1") in conn_set

    def test_internal_connections_removed(self):
        """Connections between internal nodes should be dropped."""
        structure = _diamond_structure()
        annotation = _make_annotation(
            "ann",
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3"), ("h2", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        conn_set = _conn_tuples(result)
        # All internal connections should be gone
        assert ("h1", "h2") not in conn_set
        assert ("h2", "h3") not in conn_set
        # h1->h3 is also internal (both are subgraph nodes, h3 is internal)
        assert ("h1", "h3") not in conn_set

    def test_output_index_on_exit_connections(self):
        """Exit->external connections should get output_index based on exit position."""
        structure = _multi_exit_structure()
        annotation = _make_annotation(
            "ann",
            entry_nodes=["h1"],
            exit_nodes=["h2", "h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        fn_conns = {
            c.to_node: c for c in result.connections if c.from_node == "fn_ann"
        }
        # h2 is exit_nodes[0] -> output_index=0, connects to "0"
        assert fn_conns["0"].output_index == 0
        # h3 is exit_nodes[1] -> output_index=1, connects to "1"
        assert fn_conns["1"].output_index == 1

    def test_deduplication_of_entry_to_fn_connections(self):
        """Even if an entry node has multiple connections to internal nodes,
        only one entry->fn connection should be created."""
        structure = _diamond_structure()
        annotation = _make_annotation(
            "ann",
            entry_nodes=["h1"],
            exit_nodes=["h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3"), ("h2", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        entry_to_fn = [
            c for c in result.connections
            if c.from_node == "h1" and c.to_node == "fn_ann"
        ]
        # Should have exactly one, even though h1 connects to both h2 and h3
        assert len(entry_to_fn) == 1

    def test_deduplication_of_exit_to_external_connections(self):
        """If multiple exit nodes connect to the same target, each should
        produce a distinct connection with the right output_index."""
        # Build a structure where two exit nodes both connect to the same output
        nodes = [
            _make_node("-1", NodeType.INPUT),
            _make_node("h1", NodeType.HIDDEN),
            _make_node("h2", NodeType.HIDDEN),
            _make_node("h3", NodeType.HIDDEN),
            _make_node("0", NodeType.OUTPUT),
        ]
        connections = [
            _make_conn("-1", "h1", weight=1.0),
            _make_conn("h1", "h2", weight=0.5),
            _make_conn("h1", "h3", weight=0.6),
            _make_conn("h2", "0", weight=0.7),
            _make_conn("h3", "0", weight=0.8),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        annotation = _make_annotation(
            "ann",
            entry_nodes=["h1"],
            exit_nodes=["h2", "h3"],
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h2"), ("h1", "h3")],
        )
        result = collapse_structure(structure, [annotation], {"ann"})

        fn_conns = [
            c for c in result.connections if c.from_node == "fn_ann" and c.to_node == "0"
        ]
        # Two connections to same target but with different output_index
        assert len(fn_conns) == 2
        indices = {c.output_index for c in fn_conns}
        assert indices == {0, 1}

    def test_children_before_parents_ordering(self):
        """When collapsing both child and parent, child should be collapsed first."""
        structure = _nested_annotation_structure()
        child = _make_annotation(
            "child",
            entry_nodes=["h2"],
            exit_nodes=["h3"],
            subgraph_nodes=["h2", "h3"],
            subgraph_connections=[("h2", "h3")],
        )
        parent = _make_annotation(
            "parent",
            entry_nodes=["h1"],
            exit_nodes=["h4"],
            subgraph_nodes=["h1", "h2", "h3", "h4"],
            subgraph_connections=[("h1", "h2"), ("h2", "h3"), ("h3", "h4")],
        )
        # Even if parent is listed first, child should be collapsed first
        result = collapse_structure(
            structure, [parent, child], {"child", "parent"}
        )

        node_ids = _node_ids(result)
        # After full collapse: h1 (entry of parent) preserved, fn_parent exists
        # child's fn node is inside parent's subgraph, so it becomes internal
        # and gets replaced by parent's collapse
        assert "fn_parent" in node_ids
        assert "h1" in node_ids
        # Internal nodes of parent (h2, h3, h4) should be removed
        assert "h2" not in node_ids
        assert "h3" not in node_ids
        assert "h4" not in node_ids

    def test_collapse_annotation_not_in_list_ignored(self):
        """If collapsed_ids references an annotation name not in the list, skip it."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        # "nonexistent" is in collapsed_ids but has no matching annotation
        result = collapse_structure(structure, [annotation], {"nonexistent"})
        # Structure should be unchanged (like empty collapse set)
        assert _node_ids(result) == _node_ids(structure)

    def test_formula_latex_none_on_failure(self):
        """If AnnotationFunction.from_structure fails, formula_latex should be None."""
        structure = _linear_chain_structure()
        annotation = _make_annotation(
            "ann", ["h1"], ["h2"], ["h1", "h2"], [("h1", "h2")]
        )
        result = collapse_structure(structure, [annotation], {"ann"})
        fn_node = result.get_node_by_id("fn_ann")
        # For our simple test fixtures, formula extraction may or may not work.
        # The key invariant is that it's either a string or None, never an exception.
        assert fn_node.function_metadata.formula_latex is None or isinstance(
            fn_node.function_metadata.formula_latex, str
        )

    def test_single_node_annotation(self):
        """An annotation with a single node (entry=exit): internal_nodes is empty,
        so the entry/exit node is preserved with its connections unchanged.
        The function node is added as a metadata marker with no connections."""
        nodes = [
            _make_node("-1", NodeType.INPUT),
            _make_node("h1", NodeType.HIDDEN, bias=0.5, activation="relu"),
            _make_node("0", NodeType.OUTPUT),
        ]
        connections = [
            _make_conn("-1", "h1", weight=1.0),
            _make_conn("h1", "0", weight=0.7),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        # Single node: h1 is both entry and exit
        annotation = _make_annotation(
            "single",
            entry_nodes=["h1"],
            exit_nodes=["h1"],
            subgraph_nodes=["h1"],
            subgraph_connections=[],
        )
        result = collapse_structure(structure, [annotation], {"single"})

        node_ids = _node_ids(result)
        assert "fn_single" in node_ids
        # h1 is entry, so it's preserved
        assert "h1" in node_ids
        # Since internal_nodes is empty, h1's original connections are preserved
        assert ("-1", "h1") in _conn_tuples(result)
        assert ("h1", "0") in _conn_tuples(result)
        # The function node has no connections (marker only)
        fn_out = [c for c in result.connections if c.from_node == "fn_single"]
        fn_in = [c for c in result.connections if c.to_node == "fn_single"]
        assert len(fn_out) == 0
        assert len(fn_in) == 0
        # Structure still validates
        validation = result.validate()
        assert validation["is_valid"] is True


class TestCollapseWithMissingEntryInSubgraph:
    """Regression tests for entry nodes omitted from subgraph_nodes.

    When entry_nodes lists nodes not present in subgraph_nodes, collapse
    must still create entry -> fn connections for ALL entry nodes.
    """

    def test_entry_not_in_subgraph_still_connects_to_fn(self):
        """If entry nodes are not in subgraph_nodes, they should still get
        connections to the function node.

        Reproduces: annotation M covers a->c and b->c, but subgraph_nodes
        only contains [c]. Both a->fn_M and b->fn_M must exist.
        """
        nodes = [
            _make_node("-1", NodeType.INPUT),
            _make_node("-2", NodeType.INPUT),
            _make_node("c", NodeType.HIDDEN, bias=0.1, activation="relu"),
            _make_node("0", NodeType.OUTPUT),
        ]
        connections = [
            _make_conn("-1", "c", weight=0.5),
            _make_conn("-2", "c", weight=0.7),
            _make_conn("c", "0", weight=0.9),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1", "-2"],
            output_node_ids=["0"],
        )
        # Entry nodes [-1, -2] are NOT in subgraph_nodes (only [c])
        annotation = _make_annotation(
            "M",
            entry_nodes=["-1", "-2"],
            exit_nodes=["c"],
            subgraph_nodes=["c"],  # Missing entry nodes!
            subgraph_connections=[],
        )
        result = collapse_structure(structure, [annotation], {"M"})

        conn_set = _conn_tuples(result)
        # Both entry nodes must connect to fn_M
        assert ("-1", "fn_M") in conn_set, "Missing connection -1 -> fn_M"
        assert ("-2", "fn_M") in conn_set, "Missing connection -2 -> fn_M"
        # fn_M must connect to output
        assert ("fn_M", "0") in conn_set
        assert not _has_cycle(result)

    def test_partial_entry_in_subgraph(self):
        """If only some entry nodes are in subgraph_nodes, ALL entries must
        still get connections to the function node.

        Reproduces: annotation M has entry_nodes=[a, b], but subgraph_nodes
        only contains [b, c]. Connection a->fn_M must still exist.
        """
        nodes = [
            _make_node("-1", NodeType.INPUT),
            _make_node("-2", NodeType.INPUT),
            _make_node("c", NodeType.HIDDEN, bias=0.1, activation="relu"),
            _make_node("0", NodeType.OUTPUT),
        ]
        connections = [
            _make_conn("-1", "c", weight=0.5),
            _make_conn("-2", "c", weight=0.7),
            _make_conn("c", "0", weight=0.9),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1", "-2"],
            output_node_ids=["0"],
        )
        # Only -2 is in subgraph_nodes, -1 is missing
        annotation = _make_annotation(
            "M",
            entry_nodes=["-1", "-2"],
            exit_nodes=["c"],
            subgraph_nodes=["-2", "c"],  # -1 missing from subgraph!
            subgraph_connections=[("-2", "c")],
        )
        result = collapse_structure(structure, [annotation], {"M"})

        conn_set = _conn_tuples(result)
        assert ("-1", "fn_M") in conn_set, "Missing connection -1 -> fn_M"
        assert ("-2", "fn_M") in conn_set, "Missing connection -2 -> fn_M"
        assert ("fn_M", "0") in conn_set
        assert not _has_cycle(result)

    def test_exit_not_in_subgraph_still_reroutes(self):
        """If exit nodes are not in subgraph_nodes, their outgoing connections
        should still be rerouted through the function node."""
        nodes = [
            _make_node("-1", NodeType.INPUT),
            _make_node("h1", NodeType.HIDDEN),
            _make_node("h2", NodeType.HIDDEN),
            _make_node("0", NodeType.OUTPUT),
        ]
        connections = [
            _make_conn("-1", "h1", weight=1.0),
            _make_conn("h1", "h2", weight=0.5),
            _make_conn("h2", "0", weight=0.7),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        # exit node h2 missing from subgraph_nodes
        annotation = _make_annotation(
            "M",
            entry_nodes=["h1"],
            exit_nodes=["h2"],
            subgraph_nodes=["h1"],  # h2 missing!
            subgraph_connections=[],
        )
        result = collapse_structure(structure, [annotation], {"M"})

        conn_set = _conn_tuples(result)
        assert ("h1", "fn_M") in conn_set
        assert ("fn_M", "0") in conn_set
        assert not _has_cycle(result)


class TestCompositionalAnnotationCollapse:
    """Test collapse of compositional annotations (parent with children).

    Compositional annotations have empty subgraph_nodes and rely on their
    children's subgraphs. The effective subgraph is the union of all
    descendants' subgraphs plus the parent's entry/exit nodes.
    """

    def test_collapse_all_three_produces_single_fn_parent(self):
        """Collapsing children + parent produces a single fn_parent node."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(
            structure, annotations, {"child1", "child2", "parent"}
        )

        node_ids = _node_ids(result)
        # fn_parent should exist
        assert "fn_parent" in node_ids
        # Entry nodes A, B should be preserved
        assert "A" in node_ids
        assert "B" in node_ids
        # Internal nodes should be removed
        assert "C" not in node_ids
        assert "D" not in node_ids
        assert "E" not in node_ids
        # Child fn nodes should also be removed (internal to parent)
        assert "fn_child1" not in node_ids
        assert "fn_child2" not in node_ids

    def test_collapse_all_three_fn_parent_is_connected(self):
        """fn_parent should have incoming connections from entries and
        outgoing connections to external nodes."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(
            structure, annotations, {"child1", "child2", "parent"}
        )

        conn_set = _conn_tuples(result)
        # Entry -> fn_parent connections
        assert ("A", "fn_parent") in conn_set
        assert ("B", "fn_parent") in conn_set
        # fn_parent -> output connection (rerouted from E -> out1)
        assert ("fn_parent", "0") in conn_set

    def test_collapse_all_three_no_cycle(self):
        """Collapsing children + parent should not introduce cycles."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(
            structure, annotations, {"child1", "child2", "parent"}
        )
        assert not _has_cycle(result)

    def test_collapse_only_parent_subsumes_children(self):
        """Collapsing only the parent (not children) should collapse
        the entire effective subgraph at once."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(structure, annotations, {"parent"})

        node_ids = _node_ids(result)
        assert "fn_parent" in node_ids
        # Entry nodes preserved
        assert "A" in node_ids
        assert "B" in node_ids
        # All internal nodes removed
        assert "C" not in node_ids
        assert "D" not in node_ids
        assert "E" not in node_ids
        # No child fn nodes (children weren't collapsed separately)
        assert "fn_child1" not in node_ids
        assert "fn_child2" not in node_ids

    def test_collapse_only_parent_is_connected(self):
        """When only the parent is collapsed, fn_parent should be connected."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(structure, annotations, {"parent"})

        conn_set = _conn_tuples(result)
        assert ("A", "fn_parent") in conn_set
        assert ("B", "fn_parent") in conn_set
        assert ("fn_parent", "0") in conn_set

    def test_collapse_only_children_leaves_parent_intact(self):
        """Collapsing only children without the parent should not touch
        the parent's exit nodes or create fn_parent."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(
            structure, annotations, {"child1", "child2"}
        )

        node_ids = _node_ids(result)
        # Child fn nodes exist
        assert "fn_child1" in node_ids
        assert "fn_child2" in node_ids
        # No parent fn node
        assert "fn_parent" not in node_ids
        # E is still there (not part of any collapsed annotation)
        assert "E" in node_ids
        # Entry nodes preserved
        assert "A" in node_ids
        assert "B" in node_ids

    def test_compositional_metadata_has_effective_subgraph(self):
        """The function node metadata should contain the effective subgraph
        (not the empty one from the annotation)."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(
            structure, annotations, {"child1", "child2", "parent"}
        )

        fn_node = result.get_node_by_id("fn_parent")
        meta = fn_node.function_metadata
        # Should have non-empty subgraph
        assert len(meta.subgraph_nodes) > 0
        assert meta.n_inputs == 2
        assert meta.n_outputs == 1
        assert set(meta.input_names) == {"A", "B"}
        assert meta.output_names == ["E"]

    def test_compositional_output_index(self):
        """fn_parent -> out1 should carry the correct output_index."""
        structure = _compositional_structure()
        child1, child2, parent = _compositional_annotations()
        annotations = [child1, child2, parent]

        result = collapse_structure(
            structure, annotations, {"child1", "child2", "parent"}
        )

        fn_out = [c for c in result.connections if c.from_node == "fn_parent"]
        assert len(fn_out) == 1
        # E is exit_nodes[0], so output_index=0
        assert fn_out[0].output_index == 0
        assert fn_out[0].to_node == "0"
