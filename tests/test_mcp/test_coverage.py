"""Tests for MCP coverage tools."""

import json
from unittest.mock import patch, MagicMock

import pytest

from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import AnnotationData


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.coverage import register, classify_nodes, detect_splits, get_coverage
    server = create_server()
    register(server)
    assert callable(classify_nodes)
    assert callable(detect_splits)
    assert callable(get_coverage)


def _make_network():
    """Build a small network: 2 inputs -> 2 hidden -> 1 output.

    Topology:
        -1 ---> 3 ---> 0
        -2 ---> 5 ---> 0
        -1 ---> 5       (cross-connection)
    """
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT, bias=0.0, activation="identity", response=1.0, aggregation="sum"),
        NetworkNode(id="-2", type=NodeType.INPUT, bias=0.0, activation="identity", response=1.0, aggregation="sum"),
        NetworkNode(id="3", type=NodeType.HIDDEN, bias=0.1, activation="sigmoid", response=1.0, aggregation="sum"),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.2, activation="sigmoid", response=1.0, aggregation="sum"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid", response=1.0, aggregation="sum"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="3", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-1", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="3", to_node="0", weight=1.0, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1", "-2"],
        output_node_ids=["0"],
    )


def _mock_engine(network, annotations):
    """Create a mock ModelStateEngine with given network and annotations."""
    engine = MagicMock()
    engine.current_state = network
    engine.annotations = annotations
    return engine


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_get_coverage_no_annotations(mock_build_engine, mock_get_db):
    """With no annotations, structural coverage is 0."""
    from mcp_server.tools.coverage import get_coverage

    network = _make_network()
    mock_build_engine.return_value = _mock_engine(network, [])
    mock_get_db.return_value = MagicMock()

    result = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))

    assert result["structural_coverage"] == 0.0
    assert result["compositional_coverage"] == 1.0
    assert result["annotations_count"] == 0
    assert result["leaf_count"] == 0
    assert result["composition_count"] == 0
    # Candidate nodes = all non-output = {-1, -2, 3, 5}
    assert result["node_coverage"]["total_candidate"] == 4
    assert sorted(result["node_coverage"]["uncovered"]) == sorted(["-1", "-2", "3", "5"])
    assert result["node_coverage"]["covered"] == []


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_get_coverage_single_annotation_full_coverage(mock_build_engine, mock_get_db):
    """An annotation covering node 3 with all its outgoing edges should cover it."""
    from mcp_server.tools.coverage import get_coverage

    network = _make_network()
    # Annotation covers node 3: entry=-1, exit=3, subgraph={-1, 3}, edges={(-1,3), (3,0)}
    # Node 3 has outgoing edge (3, 0). For coverage, (3, 0) must be in subgraph_connections.
    ann = AnnotationData(
        name="A1",
        hypothesis="test",
        entry_nodes=["-1"],
        exit_nodes=["3"],
        subgraph_nodes=["-1", "3"],
        subgraph_connections=[("-1", "3"), ("3", "0")],
    )
    mock_build_engine.return_value = _mock_engine(network, [ann])
    mock_get_db.return_value = MagicMock()

    result = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))

    assert result["annotations_count"] == 1
    assert result["leaf_count"] == 1
    # Node 3 should be covered (all outgoing edges in subgraph)
    # Node -1 has outgoing edges (-1,3) and (-1,5). (-1,5) is NOT in subgraph, so -1 is NOT covered.
    assert "3" in result["node_coverage"]["covered"]
    assert "-1" not in result["node_coverage"]["covered"]
    assert result["node_coverage"]["by_annotation"]["A1"] == ["3"]


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_get_coverage_node_not_covered_if_external_edges(mock_build_engine, mock_get_db):
    """A node with outgoing edges outside the annotation is NOT covered (paper Def 10)."""
    from mcp_server.tools.coverage import get_coverage

    network = _make_network()
    # Annotation includes node 5, but only edge (5, 0) — however node -1 has edge to 5 AND to 3.
    # Node 5's outgoing: (5, 0). If (5, 0) is in subgraph_connections, 5 is covered.
    # Node -1's outgoing: (-1, 3) and (-1, 5). Only (-1, 5) in subgraph, so -1 not covered.
    ann = AnnotationData(
        name="A2",
        hypothesis="test",
        entry_nodes=["-1", "-2"],
        exit_nodes=["5"],
        subgraph_nodes=["-1", "-2", "5"],
        subgraph_connections=[("-1", "5"), ("-2", "5"), ("5", "0")],
    )
    mock_build_engine.return_value = _mock_engine(network, [ann])
    mock_get_db.return_value = MagicMock()

    result = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))

    # Node 5 covered (outgoing (5,0) is in subgraph)
    assert "5" in result["node_coverage"]["covered"]
    # Node -2 covered (outgoing (-2,5) is in subgraph, that's its only edge)
    assert "-2" in result["node_coverage"]["covered"]
    # Node -1 NOT covered (outgoing includes (-1,3) which is not in subgraph)
    assert "-1" not in result["node_coverage"]["covered"]


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_get_coverage_composition_annotations(mock_build_engine, mock_get_db):
    """Composition annotations are counted separately from leaves."""
    from mcp_server.tools.coverage import get_coverage

    network = _make_network()
    child1 = AnnotationData(
        name="child1",
        hypothesis="part 1",
        entry_nodes=["-1"],
        exit_nodes=["3"],
        subgraph_nodes=["-1", "3"],
        subgraph_connections=[("-1", "3"), ("3", "0")],
        parent_annotation_id="parent",
    )
    child2 = AnnotationData(
        name="child2",
        hypothesis="part 2",
        entry_nodes=["-1", "-2"],
        exit_nodes=["5"],
        subgraph_nodes=["-1", "-2", "5"],
        subgraph_connections=[("-1", "5"), ("-2", "5"), ("5", "0")],
        parent_annotation_id="parent",
    )
    parent = AnnotationData(
        name="parent",
        hypothesis="combined",
        entry_nodes=["-1", "-2"],
        exit_nodes=["3", "5"],
        subgraph_nodes=["-1", "-2", "3", "5"],
        subgraph_connections=[("-1", "3"), ("-2", "5"), ("-1", "5"), ("3", "0"), ("5", "0")],
    )
    mock_build_engine.return_value = _mock_engine(network, [child1, child2, parent])
    mock_get_db.return_value = MagicMock()

    result = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))

    assert result["annotations_count"] == 3
    assert result["composition_count"] == 1  # parent has children
    # Leaf annotations are those with no parent AND no children
    # child1 and child2 have parent_annotation_id set, so they're not "root leaves"
    # parent has children, so it's a composition
    # The leaf_anns logic: no parent_annotation_id AND no other annotation has this as parent
    # child1: parent_annotation_id="parent" -> excluded
    # child2: parent_annotation_id="parent" -> excluded
    # parent: parent_annotation_id=None, but child1/child2 have parent="parent" -> composition, not leaf
    # So leaf_count = 0, and we fall back to using all ann_dicts
    assert result["leaf_count"] == 0


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_get_coverage_edge_coverage(mock_build_engine, mock_get_db):
    """Edge coverage tracks which edges are covered."""
    from mcp_server.tools.coverage import get_coverage

    network = _make_network()
    ann = AnnotationData(
        name="A1",
        hypothesis="test",
        entry_nodes=["-2"],
        exit_nodes=["5"],
        subgraph_nodes=["-2", "5"],
        subgraph_connections=[("-2", "5"), ("5", "0")],
    )
    mock_build_engine.return_value = _mock_engine(network, [ann])
    mock_get_db.return_value = MagicMock()

    result = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))

    assert result["edge_coverage"]["total"] == 5
    # Some edges should be covered
    assert len(result["edge_coverage"]["covered"]) > 0
    assert len(result["edge_coverage"]["uncovered"]) > 0


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_get_coverage_returns_valid_json(mock_build_engine, mock_get_db):
    """Result is always valid JSON with all expected keys."""
    from mcp_server.tools.coverage import get_coverage

    network = _make_network()
    mock_build_engine.return_value = _mock_engine(network, [])
    mock_get_db.return_value = MagicMock()

    result = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))

    expected_keys = {
        "structural_coverage", "compositional_coverage",
        "node_coverage", "edge_coverage",
        "annotations_count", "leaf_count", "composition_count",
    }
    assert expected_keys == set(result.keys())
    assert set(result["node_coverage"].keys()) == {"covered", "uncovered", "total_candidate", "by_annotation"}
    assert set(result["edge_coverage"].keys()) == {"covered", "uncovered", "total"}
