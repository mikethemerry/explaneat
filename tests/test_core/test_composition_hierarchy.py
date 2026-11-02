"""Multi-level composition validation and annotation references.

Covers:
- descendants at any depth count as internal when validating a composition
  (grandchild exits may be read out, grandchild entries may be listed);
- child_annotation_ids / rename_annotation accept either annotation names
  or the synthetic ``ann_<seq>`` ids reported by the MCP tools.
"""

import pytest

from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import ModelStateEngine
from explaneat.core.operations import OperationError


def _three_level_network(grandchild_exit_feeds_output: bool = False):
    """-1,-2 -> 3 ; 3,-3 -> 4 ; 4 -> 0  (optionally 3 -> 0 as well).

    Leaf A (entries -1,-2, exit 3); composition B over A adds glue node 4
    (fed by -3); root R over B adds the output 0.
    """
    conns = [
        NetworkConnection(from_node="-1", to_node="3", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="3", weight=-1.0, enabled=True),
        NetworkConnection(from_node="3", to_node="4", weight=2.0, enabled=True),
        NetworkConnection(from_node="-3", to_node="4", weight=0.5, enabled=True),
        NetworkConnection(from_node="4", to_node="0", weight=1.5, enabled=True),
    ]
    if grandchild_exit_feeds_output:
        conns.append(
            NetworkConnection(from_node="3", to_node="0", weight=-0.7, enabled=True)
        )
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="-3", type=NodeType.INPUT),
            NetworkNode(id="3", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="4", type=NodeType.HIDDEN, bias=-0.2, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.3, activation="sigmoid"),
        ],
        connections=conns,
        input_node_ids=["-1", "-2", "-3"],
        output_node_ids=["0"],
    )


LEAF_A = {
    "name": "A",
    "hypothesis": "leaf",
    "entry_nodes": ["-1", "-2"],
    "exit_nodes": ["3"],
    "subgraph_nodes": ["-1", "-2", "3"],
    "subgraph_connections": [["-1", "3"], ["-2", "3"]],
}


def _comp_b(children=("A",)):
    return {
        "name": "B",
        "hypothesis": "mid",
        "entry_nodes": ["-1", "-2", "-3"],
        "exit_nodes": ["4"],
        "subgraph_nodes": ["-3", "4"],
        "subgraph_connections": [["3", "4"], ["-3", "4"]],
        "child_annotation_ids": list(children),
    }


def _root_r(extra_conns=(), children=("B",)):
    return {
        "name": "R",
        "hypothesis": "root",
        # -1, -2 are grandchild (A) entries; -3 is a child (B) entry
        "entry_nodes": ["-1", "-2", "-3"],
        "exit_nodes": ["0"],
        "subgraph_nodes": ["0"],
        "subgraph_connections": [["4", "0"], *extra_conns],
        "child_annotation_ids": list(children),
    }


class TestDescendantsCountAsInternal:
    def test_root_may_list_grandchild_entries(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))
        engine.add_operation("annotate", _comp_b())
        errors = engine.validate_operation("annotate", _root_r())
        assert errors == []
        engine.add_operation("annotate", _root_r())
        assert [a.name for a in engine.annotations] == ["A", "B", "R"]

    def test_root_may_read_grandchild_exit(self):
        """Output 0 reads grandchild A's exit 3 directly: not a P3 violation."""
        engine = ModelStateEngine(_three_level_network(grandchild_exit_feeds_output=True))
        engine.add_operation("annotate", dict(LEAF_A))
        engine.add_operation("annotate", _comp_b())
        errors = engine.validate_operation("annotate", _root_r(extra_conns=[["3", "0"]]))
        assert errors == [], errors

    def test_non_descendant_still_external(self):
        """Nodes of an annotation that is not a descendant stay external."""
        engine = ModelStateEngine(_three_level_network(grandchild_exit_feeds_output=True))
        engine.add_operation("annotate", dict(LEAF_A))
        engine.add_operation("annotate", _comp_b())
        # R claims only A as child: B's exit 4 is outside -> P3 on output 0
        errors = engine.validate_operation(
            "annotate",
            {**_root_r(extra_conns=[["3", "0"]]), "child_annotation_ids": ["A"]},
        )
        assert any("P3" in e and "4" in e for e in errors), errors


class TestAnnotationReferencesByIdOrName:
    def test_children_by_ann_seq_id(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))  # seq 0 -> ann_0
        errors = engine.validate_operation("annotate", _comp_b(children=("ann_0",)))
        assert errors == [], errors
        op = engine.add_operation("annotate", _comp_b(children=("ann_0",)))
        # stored canonically as names so every consumer (hierarchy, coverage,
        # get_annotations) keeps working
        assert op.params["child_annotation_ids"] == ["A"]
        assert engine.annotations[0].parent_annotation_id == "B"

    def test_children_by_id_three_levels(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))  # ann_0
        engine.add_operation("annotate", _comp_b(children=("ann_0",)))  # ann_1
        engine.add_operation("annotate", _root_r(children=("ann_1",)))
        assert engine.annotations[1].parent_annotation_id == "R"

    def test_unknown_child_reports_not_found_only(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))
        errors = engine.validate_operation("annotate", _comp_b(children=("ann_9",)))
        assert any("not found" in e for e in errors)

    def test_caller_params_not_mutated(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))
        params = _comp_b(children=("ann_0",))
        engine.add_operation("annotate", params)
        assert params["child_annotation_ids"] == ["ann_0"]


class TestRenameAnnotation:
    def test_rename_by_name(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))
        engine.add_operation("rename_annotation", {"annotation_id": "A", "display_name": "leafA"})
        assert engine.annotations[0].display_name == "leafA"

    def test_rename_by_ann_id(self):
        engine = ModelStateEngine(_three_level_network())
        engine.add_operation("annotate", dict(LEAF_A))
        op = engine.add_operation(
            "rename_annotation", {"annotation_id": "ann_0", "display_name": "leafA"}
        )
        assert op.params["annotation_id"] == "A"
        assert engine.annotations[0].display_name == "leafA"

    def test_rename_unknown(self):
        engine = ModelStateEngine(_three_level_network())
        with pytest.raises(OperationError, match="not found"):
            engine.add_operation(
                "rename_annotation", {"annotation_id": "nope", "display_name": "x"}
            )
