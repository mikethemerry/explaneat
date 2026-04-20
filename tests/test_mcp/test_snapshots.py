"""Tests for MCP snapshot tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.snapshots import register, save_snapshot, update_narrative, list_evidence
    server = create_server()
    register(server)
    assert callable(save_snapshot)
    assert callable(update_narrative)
    assert callable(list_evidence)
