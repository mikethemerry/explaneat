"""Tests for operation notes field."""
from explaneat.core.model_state import ModelStateEngine, Operation
from explaneat.core.genome_network import (
    NetworkStructure, NetworkNode, NetworkConnection, NodeType,
)


def _simple_structure():
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes, connections=connections,
        input_node_ids=["-1"], output_node_ids=["0"],
    )


class TestOperationNotes:
    def test_add_operation_with_notes(self):
        engine = ModelStateEngine(_simple_structure())
        op = engine.add_operation(
            "rename_node",
            {"node_id": "-1", "display_name": "age"},
            notes="Mapped from dataset feature column 0",
        )
        assert op.notes == "Mapped from dataset feature column 0"

    def test_notes_round_trip_serialization(self):
        engine = ModelStateEngine(_simple_structure())
        engine.add_operation(
            "rename_node",
            {"node_id": "-1", "display_name": "age"},
            notes="Dataset column mapping",
        )
        data = engine.to_dict()
        assert data["operations"][0]["notes"] == "Dataset column mapping"

        engine2 = ModelStateEngine(_simple_structure())
        engine2.load_operations(data)
        assert engine2.operations[0].notes == "Dataset column mapping"

    def test_notes_default_none(self):
        engine = ModelStateEngine(_simple_structure())
        op = engine.add_operation("rename_node", {"node_id": "-1", "display_name": "age"})
        assert op.notes is None

    def test_notes_omitted_from_serialization_when_none(self):
        engine = ModelStateEngine(_simple_structure())
        engine.add_operation("rename_node", {"node_id": "-1", "display_name": "age"})
        data = engine.to_dict()
        assert "notes" not in data["operations"][0]
