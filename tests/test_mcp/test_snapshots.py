"""Tests for MCP snapshot tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.snapshots import register, save_snapshot, update_narrative, list_evidence, add_evidence
    server = create_server()
    register(server)
    assert callable(save_snapshot)
    assert callable(update_narrative)
    assert callable(list_evidence)
    assert callable(add_evidence)


def test_add_evidence_tool_exists():
    from mcp_server.tools.snapshots import add_evidence
    assert callable(add_evidence)


def test_evidence_record_schema():
    from explaneat.api.schemas import EvidenceRecord
    record = EvidenceRecord(
        type="analytical_formula",
        payload={"latex": "\\sigma(x)", "tractable": True},
        narrative="Simple sigmoid",
    )
    assert record.type == "analytical_formula"
    assert record.payload["latex"] == "\\sigma(x)"
    assert record.timestamp is not None


def test_evidence_record_schema_defaults():
    from explaneat.api.schemas import EvidenceRecord
    record = EvidenceRecord(
        type="visualization",
        payload={"viz_type": "line"},
    )
    assert record.type == "visualization"
    assert record.narrative is None
    assert record.timestamp is not None


def test_evidence_record_schema_with_all_fields():
    from datetime import datetime, timezone
    from explaneat.api.schemas import EvidenceRecord
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    record = EvidenceRecord(
        type="shap_importance",
        payload={"feature_names": ["x1", "x2"], "values": [0.5, 0.3]},
        narrative="SHAP analysis shows x1 is most important",
        timestamp=ts,
    )
    assert record.type == "shap_importance"
    assert record.timestamp == ts
    assert record.narrative == "SHAP analysis shows x1 is most important"
    assert record.payload["values"] == [0.5, 0.3]
