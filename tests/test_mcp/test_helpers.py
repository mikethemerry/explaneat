"""Tests for mcp_server.helpers module."""

import uuid

import pytest


def test_all_functions_exist():
    """All expected helper functions are importable."""
    from mcp_server.helpers import (
        _to_uuid,
        build_engine,
        build_model_state,
        find_annotation_in_operations,
        load_genome_and_config,
        load_split_data,
        serialize_network,
    )

    # Verify they are callable
    assert callable(_to_uuid)
    assert callable(build_engine)
    assert callable(build_model_state)
    assert callable(find_annotation_in_operations)
    assert callable(load_genome_and_config)
    assert callable(load_split_data)
    assert callable(serialize_network)


def test_to_uuid_valid():
    """_to_uuid converts a valid string to UUID."""
    from mcp_server.helpers import _to_uuid

    test_id = "12345678-1234-5678-1234-567812345678"
    result = _to_uuid(test_id)
    assert isinstance(result, uuid.UUID)
    assert str(result) == test_id


def test_to_uuid_invalid():
    """_to_uuid raises ValueError for invalid strings."""
    from mcp_server.helpers import _to_uuid

    with pytest.raises(ValueError):
        _to_uuid("not-a-uuid")


def test_serialize_network_structure():
    """serialize_network converts NetworkStructure to dict."""
    from mcp_server.helpers import serialize_network
    from explaneat.core.genome_network import (
        NetworkConnection,
        NetworkNode,
        NetworkStructure,
        NodeType,
    )

    ns = NetworkStructure(
        nodes=[
            NetworkNode(
                id="-1",
                type=NodeType.INPUT,
                bias=0.0,
                activation="identity",
                response=1.0,
                aggregation="sum",
            ),
            NetworkNode(
                id="0",
                type=NodeType.OUTPUT,
                bias=0.5,
                activation="sigmoid",
                response=1.0,
                aggregation="sum",
            ),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )
    result = serialize_network(ns)
    assert "nodes" in result
    assert "connections" in result
    assert "input_node_ids" in result
    assert "output_node_ids" in result
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["id"] == "-1"
    assert result["nodes"][0]["type"] == "input"
    assert result["nodes"][1]["type"] == "output"
    assert result["connections"][0]["weight"] == 1.0
    assert result["connections"][0]["from_node"] == "-1"
    assert result["connections"][0]["to_node"] == "0"
    assert result["input_node_ids"] == ["-1"]
    assert result["output_node_ids"] == ["0"]


def test_serialize_network_with_display_name():
    """serialize_network includes display_name when present."""
    from mcp_server.helpers import serialize_network
    from explaneat.core.genome_network import (
        NetworkNode,
        NetworkStructure,
        NodeType,
    )

    ns = NetworkStructure(
        nodes=[
            NetworkNode(
                id="-1",
                type=NodeType.INPUT,
                bias=0.0,
                activation="identity",
                response=1.0,
                aggregation="sum",
                display_name="feature_x",
            ),
        ],
        connections=[],
        input_node_ids=["-1"],
        output_node_ids=[],
    )
    result = serialize_network(ns)
    assert result["nodes"][0]["display_name"] == "feature_x"


def test_serialize_network_hidden_node_type():
    """serialize_network maps NodeType.HIDDEN to 'hidden'."""
    from mcp_server.helpers import serialize_network
    from explaneat.core.genome_network import (
        NetworkNode,
        NetworkStructure,
        NodeType,
    )

    ns = NetworkStructure(
        nodes=[
            NetworkNode(
                id="5",
                type=NodeType.HIDDEN,
                bias=0.1,
                activation="relu",
                response=1.0,
                aggregation="sum",
            ),
        ],
        connections=[],
        input_node_ids=[],
        output_node_ids=[],
    )
    result = serialize_network(ns)
    assert result["nodes"][0]["type"] == "hidden"
