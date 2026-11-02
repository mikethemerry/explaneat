"""Input fan-out across annotations must be flagged before annotate fails.

Network:  -1 -> 0 (direct edge)      -1 -> 5 -> 0 (hidden gate)
          -2 -> 5

Input -1 feeds both the output's direct-edge group and the hidden gate 5.
Whichever annotation takes -1 first leaves the other one unable to cover it
("already covered"), so -1 must be split first.
"""

import json
from unittest.mock import MagicMock, patch

from explaneat.analysis.node_classification import classify_coverage
from explaneat.analysis.split_detection import analyze_coverage_for_splits
from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import ModelStateEngine


def _net():
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.0, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="0", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1", "-2"],
        output_node_ids=["0"],
    )


def test_detect_splits_flags_input_feeding_outside_coverage():
    result = analyze_coverage_for_splits(_net(), ["-1", "-2", "5"])
    flagged = {v.node_id: v for v in result.violations}
    assert "-1" in flagged
    assert ("-1", "0") in flagged["-1"].external_outputs
    assert {"type": "split_node", "params": {"node_id": "-1"}} in result.suggested_operations


def test_detect_splits_no_flag_when_input_fully_inside():
    result = analyze_coverage_for_splits(_net(), ["-1", "-2", "5", "0"])
    assert result.violations == []


def test_classify_flags_input_fanout():
    result = classify_coverage(_net(), ["-1", "-2", "5"])
    assert not result.valid
    assert any(v["node_id"] == "-1" for v in result.violations)


def test_existing_annotation_makes_node_unavailable():
    """After the gate is annotated (taking -1), the direct-edge coverage
    containing -1 must be reported as blocked, not silently accepted."""
    engine = ModelStateEngine(_net())
    engine.add_operation("annotate", {
        "name": "gate", "hypothesis": "",
        "entry_nodes": ["-1", "-2"], "exit_nodes": ["5", "-1"],
        "subgraph_nodes": ["-1", "-2", "5"],
        "subgraph_connections": [["-1", "5"], ["-2", "5"]],
    })
    result = analyze_coverage_for_splits(
        engine.current_state, ["-1", "0"], annotations=engine.annotations,
    )
    assert result.already_covered == [{"node_id": "-1", "annotation": "gate"}]
    # covered nodes cannot be split, so no split suggestion for them
    assert {"type": "split_node", "params": {"node_id": "-1"}} not in result.suggested_operations


@patch("mcp_server.tools.coverage.get_db")
@patch("mcp_server.tools.coverage.build_engine")
def test_mcp_detect_splits_reports_fanout_and_covered(mock_build_engine, mock_get_db):
    from mcp_server.tools.coverage import classify_nodes, detect_splits

    engine = ModelStateEngine(_net())
    mock_build_engine.return_value = engine
    mock_get_db.return_value = MagicMock()

    out = json.loads(detect_splits("00000000-0000-0000-0000-000000000001", '["-1", "-2", "5"]'))
    assert any(v["node_id"] == "-1" for v in out["violations"])
    assert out["already_covered"] == []

    engine.add_operation("annotate", {
        "name": "gate", "hypothesis": "",
        "entry_nodes": ["-1", "-2"], "exit_nodes": ["5", "-1"],
        "subgraph_nodes": ["-1", "-2", "5"],
        "subgraph_connections": [["-1", "5"], ["-2", "5"]],
    })
    out = json.loads(detect_splits("00000000-0000-0000-0000-000000000001", '["-1", "0"]'))
    assert out["already_covered"] == [{"node_id": "-1", "annotation": "gate"}]
    cls = json.loads(classify_nodes("00000000-0000-0000-0000-000000000001", '["-1", "0"]'))
    assert cls["valid"] is False
    assert cls["already_covered"] == [{"node_id": "-1", "annotation": "gate"}]
