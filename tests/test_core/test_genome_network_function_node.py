"""Tests for function node data model additions to genome_network.py.

Verifies:
- NodeType.FUNCTION enum value
- FunctionNodeMetadata dataclass
- NetworkNode with function_metadata field
- NetworkConnection with output_index field
- Backwards compatibility (new fields default to None)
- NetworkStructure.validate() works with function nodes
"""

import pytest
from explaneat.core.genome_network import (
    NodeType,
    FunctionNodeMetadata,
    NetworkNode,
    NetworkConnection,
    NetworkStructure,
)


class TestNodeTypeFunction:
    """Test NodeType.FUNCTION enum value."""

    def test_function_enum_value(self):
        assert NodeType.FUNCTION == "function"

    def test_function_is_string_enum(self):
        assert isinstance(NodeType.FUNCTION, str)
        assert isinstance(NodeType.FUNCTION, NodeType)

    def test_existing_enum_values_unchanged(self):
        """Ensure existing enum values are not broken."""
        assert NodeType.INPUT == "input"
        assert NodeType.OUTPUT == "output"
        assert NodeType.HIDDEN == "hidden"


class TestFunctionNodeMetadata:
    """Test FunctionNodeMetadata dataclass."""

    def test_creation_with_all_fields(self):
        metadata = FunctionNodeMetadata(
            annotation_name="relu_gate",
            annotation_id="ann-001",
            hypothesis="This subgraph implements a ReLU gating mechanism",
            n_inputs=2,
            n_outputs=1,
            input_names=["x1", "x2"],
            output_names=["y"],
            formula_latex=r"\max(0, w_1 x_1 + w_2 x_2)",
            subgraph_nodes=["5", "6", "7"],
            subgraph_connections=[("5", "7"), ("6", "7")],
        )
        assert metadata.annotation_name == "relu_gate"
        assert metadata.annotation_id == "ann-001"
        assert metadata.hypothesis == "This subgraph implements a ReLU gating mechanism"
        assert metadata.n_inputs == 2
        assert metadata.n_outputs == 1
        assert metadata.input_names == ["x1", "x2"]
        assert metadata.output_names == ["y"]
        assert metadata.formula_latex == r"\max(0, w_1 x_1 + w_2 x_2)"
        assert metadata.subgraph_nodes == ["5", "6", "7"]
        assert metadata.subgraph_connections == [("5", "7"), ("6", "7")]

    def test_formula_latex_optional_none(self):
        metadata = FunctionNodeMetadata(
            annotation_name="unknown",
            annotation_id="ann-002",
            hypothesis="Unknown function",
            n_inputs=1,
            n_outputs=1,
            input_names=["x"],
            output_names=["y"],
            formula_latex=None,
            subgraph_nodes=["10"],
            subgraph_connections=[],
        )
        assert metadata.formula_latex is None

    def test_empty_subgraph(self):
        metadata = FunctionNodeMetadata(
            annotation_name="trivial",
            annotation_id="ann-003",
            hypothesis="Trivial pass-through",
            n_inputs=1,
            n_outputs=1,
            input_names=["in"],
            output_names=["out"],
            formula_latex=None,
            subgraph_nodes=[],
            subgraph_connections=[],
        )
        assert metadata.subgraph_nodes == []
        assert metadata.subgraph_connections == []

    def test_multiple_outputs(self):
        metadata = FunctionNodeMetadata(
            annotation_name="split_fn",
            annotation_id="ann-004",
            hypothesis="Splits input into two outputs",
            n_inputs=1,
            n_outputs=2,
            input_names=["x"],
            output_names=["y1", "y2"],
            formula_latex=None,
            subgraph_nodes=["a", "b", "c"],
            subgraph_connections=[("a", "b"), ("a", "c")],
        )
        assert metadata.n_outputs == 2
        assert len(metadata.output_names) == 2


class TestNetworkNodeFunctionMetadata:
    """Test NetworkNode with function_metadata field."""

    def test_node_with_function_metadata(self):
        metadata = FunctionNodeMetadata(
            annotation_name="gate",
            annotation_id="ann-010",
            hypothesis="Gating function",
            n_inputs=2,
            n_outputs=1,
            input_names=["a", "b"],
            output_names=["out"],
            formula_latex=r"a \cdot b",
            subgraph_nodes=["5", "6"],
            subgraph_connections=[("5", "6")],
        )
        node = NetworkNode(
            id="fn_gate",
            type=NodeType.FUNCTION,
            function_metadata=metadata,
        )
        assert node.function_metadata is not None
        assert node.function_metadata.annotation_name == "gate"
        assert node.type == NodeType.FUNCTION

    def test_node_without_function_metadata_backwards_compat(self):
        """Existing code creating NetworkNode without function_metadata should still work."""
        node = NetworkNode(
            id="5",
            type=NodeType.HIDDEN,
            bias=0.1,
            activation="sigmoid",
            response=1.0,
            aggregation="sum",
        )
        assert node.function_metadata is None

    def test_node_default_function_metadata_is_none(self):
        node = NetworkNode(id="1", type=NodeType.INPUT)
        assert node.function_metadata is None

    def test_function_node_with_all_fields(self):
        """A function node can also have bias, activation, etc."""
        metadata = FunctionNodeMetadata(
            annotation_name="test",
            annotation_id="ann-020",
            hypothesis="test",
            n_inputs=1,
            n_outputs=1,
            input_names=["x"],
            output_names=["y"],
            formula_latex=None,
            subgraph_nodes=[],
            subgraph_connections=[],
        )
        node = NetworkNode(
            id="fn_1",
            type=NodeType.FUNCTION,
            bias=0.5,
            activation="relu",
            response=1.0,
            aggregation="sum",
            function_metadata=metadata,
        )
        assert node.bias == 0.5
        assert node.function_metadata is not None


class TestNetworkConnectionOutputIndex:
    """Test NetworkConnection with output_index field."""

    def test_connection_with_output_index(self):
        conn = NetworkConnection(
            from_node="fn_gate",
            to_node="0",
            weight=1.0,
            enabled=True,
            output_index=0,
        )
        assert conn.output_index == 0

    def test_connection_with_higher_output_index(self):
        conn = NetworkConnection(
            from_node="fn_gate",
            to_node="1",
            weight=0.5,
            enabled=True,
            output_index=2,
        )
        assert conn.output_index == 2

    def test_connection_without_output_index_backwards_compat(self):
        """Existing code creating NetworkConnection without output_index should still work."""
        conn = NetworkConnection(
            from_node="5",
            to_node="6",
            weight=0.75,
            enabled=True,
            innovation=42,
        )
        assert conn.output_index is None

    def test_connection_default_output_index_is_none(self):
        conn = NetworkConnection(
            from_node="a",
            to_node="b",
            weight=1.0,
            enabled=True,
        )
        assert conn.output_index is None

    def test_connection_with_both_innovation_and_output_index(self):
        conn = NetworkConnection(
            from_node="fn_1",
            to_node="0",
            weight=1.0,
            enabled=True,
            innovation=99,
            output_index=1,
        )
        assert conn.innovation == 99
        assert conn.output_index == 1


class TestNetworkStructureWithFunctionNodes:
    """Test NetworkStructure.validate() works with function nodes."""

    def _make_simple_structure_with_function_node(self):
        """Helper: input -> function_node -> output."""
        fn_metadata = FunctionNodeMetadata(
            annotation_name="add",
            annotation_id="ann-100",
            hypothesis="Adds two inputs",
            n_inputs=2,
            n_outputs=1,
            input_names=["x1", "x2"],
            output_names=["sum"],
            formula_latex="x_1 + x_2",
            subgraph_nodes=["h1", "h2", "h3"],
            subgraph_connections=[("h1", "h3"), ("h2", "h3")],
        )
        nodes = [
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="fn_add", type=NodeType.FUNCTION, function_metadata=fn_metadata),
            NetworkNode(id="0", type=NodeType.OUTPUT),
        ]
        connections = [
            NetworkConnection(from_node="-1", to_node="fn_add", weight=1.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="fn_add", weight=1.0, enabled=True),
            NetworkConnection(from_node="fn_add", to_node="0", weight=1.0, enabled=True, output_index=0),
        ]
        return NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1", "-2"],
            output_node_ids=["0"],
        )

    def test_validate_with_function_node_is_valid(self):
        structure = self._make_simple_structure_with_function_node()
        result = structure.validate()
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_get_node_by_id_finds_function_node(self):
        structure = self._make_simple_structure_with_function_node()
        fn_node = structure.get_node_by_id("fn_add")
        assert fn_node is not None
        assert fn_node.type == NodeType.FUNCTION
        assert fn_node.function_metadata is not None
        assert fn_node.function_metadata.annotation_name == "add"

    def test_get_connections_from_function_node(self):
        structure = self._make_simple_structure_with_function_node()
        conns = structure.get_connections_from("fn_add")
        assert len(conns) == 1
        assert conns[0].to_node == "0"
        assert conns[0].output_index == 0

    def test_get_connections_to_function_node(self):
        structure = self._make_simple_structure_with_function_node()
        conns = structure.get_connections_to("fn_add")
        assert len(conns) == 2

    def test_validate_detects_invalid_function_node_connection(self):
        """validate() should catch connections to non-existent function nodes."""
        nodes = [
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="0", type=NodeType.OUTPUT),
        ]
        connections = [
            NetworkConnection(from_node="-1", to_node="fn_missing", weight=1.0, enabled=True),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        result = structure.validate()
        assert result["is_valid"] is False
        assert any("fn_missing" in err for err in result["errors"])

    def test_mixed_structure_regular_and_function_nodes(self):
        """A structure with both regular hidden nodes and function nodes should validate."""
        fn_metadata = FunctionNodeMetadata(
            annotation_name="sub",
            annotation_id="ann-200",
            hypothesis="Subtraction",
            n_inputs=1,
            n_outputs=1,
            input_names=["x"],
            output_names=["y"],
            formula_latex=None,
            subgraph_nodes=["old_h"],
            subgraph_connections=[],
        )
        nodes = [
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.1, activation="sigmoid"),
            NetworkNode(id="fn_sub", type=NodeType.FUNCTION, function_metadata=fn_metadata),
            NetworkNode(id="0", type=NodeType.OUTPUT),
        ]
        connections = [
            NetworkConnection(from_node="-1", to_node="5", weight=0.5, enabled=True),
            NetworkConnection(from_node="5", to_node="fn_sub", weight=1.0, enabled=True),
            NetworkConnection(from_node="fn_sub", to_node="0", weight=1.0, enabled=True, output_index=0),
        ]
        structure = NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=["-1"],
            output_node_ids=["0"],
        )
        result = structure.validate()
        assert result["is_valid"] is True
