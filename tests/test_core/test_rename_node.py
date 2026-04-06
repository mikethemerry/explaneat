"""Tests for rename_node operation."""
import pytest
from explaneat.core.genome_network import (
    NetworkNode,
    NetworkConnection,
    NetworkStructure,
    NodeType,
)
from explaneat.core.operations import apply_rename_node, apply_split_node, validate_operation
from explaneat.core.model_state import ModelStateEngine


def _simple_structure():
    """3 inputs -> 1 hidden -> 1 output."""
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="-3", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.1, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="-3", to_node="5", weight=-0.3, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1", "-2", "-3"],
        output_node_ids=["0"],
    )


class TestApplyRenameNode:
    def test_sets_display_name(self):
        structure = _simple_structure()
        apply_rename_node(structure, "-1", "sepalLength", set())
        node = structure.get_node_by_id("-1")
        assert node.display_name == "sepalLength"

    def test_result_contains_node_id_and_display_name(self):
        structure = _simple_structure()
        result = apply_rename_node(structure, "-2", "petalWidth", set())
        assert result["node_id"] == "-2"
        assert result["display_name"] == "petalWidth"

    def test_rename_covered_node_succeeds(self):
        structure = _simple_structure()
        result = apply_rename_node(structure, "-1", "sepalLength", {"-1"})
        assert result["node_id"] == "-1"
        assert result["display_name"] == "sepalLength"
        node = structure.get_node_by_id("-1")
        assert node.display_name == "sepalLength"

    def test_rename_nonexistent_node_raises(self):
        structure = _simple_structure()
        with pytest.raises(Exception, match="not found"):
            apply_rename_node(structure, "999", "foo", set())

    def test_clear_display_name_with_none(self):
        structure = _simple_structure()
        apply_rename_node(structure, "-1", "sepalLength", set())
        apply_rename_node(structure, "-1", None, set())
        node = structure.get_node_by_id("-1")
        assert node.display_name is None

    def test_display_name_validation_no_spaces(self):
        structure = _simple_structure()
        with pytest.raises(Exception, match="spaces"):
            apply_rename_node(structure, "-1", "sepal length", set())

    def test_display_name_validation_no_empty(self):
        structure = _simple_structure()
        with pytest.raises(Exception, match="empty"):
            apply_rename_node(structure, "-1", "", set())

    def test_display_label_property(self):
        structure = _simple_structure()
        node = structure.get_node_by_id("-1")
        assert node.display_label == "-1"
        apply_rename_node(structure, "-1", "sepalLength", set())
        assert node.display_label == "sepalLength"

    def test_get_display_map(self):
        structure = _simple_structure()
        apply_rename_node(structure, "-1", "sepalLength", set())
        dm = structure.get_display_map()
        assert dm["-1"] == "sepalLength"
        assert dm["-2"] == "-2"  # No display_name, falls back to id


class TestRenameNodeValidation:
    def test_validate_passes_for_valid_rename(self):
        structure = _simple_structure()
        errors = validate_operation(
            structure, "rename_node",
            {"node_id": "-1", "display_name": "sepalLength"},
            set(), set(),
        )
        assert errors == []

    def test_validate_fails_for_missing_node(self):
        structure = _simple_structure()
        errors = validate_operation(
            structure, "rename_node",
            {"node_id": "999", "display_name": "foo"},
            set(), set(),
        )
        assert len(errors) > 0

    def test_validate_passes_for_covered_node(self):
        structure = _simple_structure()
        errors = validate_operation(
            structure, "rename_node",
            {"node_id": "-1", "display_name": "foo"},
            {"-1"}, set(),
        )
        assert errors == []

    def test_validate_fails_for_empty_name(self):
        structure = _simple_structure()
        errors = validate_operation(
            structure, "rename_node",
            {"node_id": "-1", "display_name": ""},
            set(), set(),
        )
        assert len(errors) > 0

    def test_validate_fails_for_spaces(self):
        structure = _simple_structure()
        errors = validate_operation(
            structure, "rename_node",
            {"node_id": "-1", "display_name": "sepal length"},
            set(), set(),
        )
        assert len(errors) > 0

    def test_validate_passes_for_none_display_name(self):
        structure = _simple_structure()
        errors = validate_operation(
            structure, "rename_node",
            {"node_id": "-1", "display_name": None},
            set(), set(),
        )
        assert errors == []


class TestRenameNodeInEngine:
    def test_rename_via_engine(self):
        structure = _simple_structure()
        engine = ModelStateEngine(structure)
        engine.add_operation("rename_node", {"node_id": "-1", "display_name": "sepalLength"})
        node = engine.current_state.get_node_by_id("-1")
        assert node.display_name == "sepalLength"

    def test_undo_rename_clears_display_name(self):
        structure = _simple_structure()
        engine = ModelStateEngine(structure)
        op = engine.add_operation("rename_node", {"node_id": "-1", "display_name": "sepalLength"})
        engine.remove_operation(op.seq)
        node = engine.current_state.get_node_by_id("-1")
        assert node.display_name is None

    def test_rename_persists_through_replay(self):
        structure = _simple_structure()
        engine = ModelStateEngine(structure)
        engine.add_operation("rename_node", {"node_id": "-1", "display_name": "sepalLength"})
        # Serialize and reload
        data = engine.to_dict()
        engine2 = ModelStateEngine(structure)
        engine2.load_operations(data)
        node = engine2.current_state.get_node_by_id("-1")
        assert node.display_name == "sepalLength"


def _splittable_structure():
    """Structure where node '5' has 2 outgoing connections (splittable)."""
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="relu"),
        NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.2, activation="relu"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.1, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="5", to_node="6", weight=0.8, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
        NetworkConnection(from_node="6", to_node="0", weight=0.3, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-1", "-2"],
        output_node_ids=["0"],
    )


class TestSplitDisplayNameInheritance:
    def test_split_inherits_display_name(self):
        structure = _splittable_structure()
        apply_rename_node(structure, "5", "hiddenUnit", set())
        result = apply_split_node(structure, "5", set())
        for nid in result["created_nodes"]:
            node = structure.get_node_by_id(nid)
            assert node.display_name is not None
            assert node.display_name.startswith("hiddenUnit_")

    def test_split_without_display_name_stays_none(self):
        structure = _splittable_structure()
        result = apply_split_node(structure, "5", set())
        for nid in result["created_nodes"]:
            node = structure.get_node_by_id(nid)
            assert node.display_name is None

    def test_split_inheritance_via_engine(self):
        structure = _splittable_structure()
        engine = ModelStateEngine(structure)
        engine.add_operation("rename_node", {"node_id": "5", "display_name": "hub"})
        engine.add_operation("split_node", {"node_id": "5"})
        state = engine.current_state
        # Node "5" should be gone, replaced by "5_a" and "5_b"
        assert state.get_node_by_id("5") is None
        for nid in ["5_a", "5_b"]:
            node = state.get_node_by_id(nid)
            assert node is not None
            assert node.display_name is not None
            assert node.display_name.startswith("hub_")
