"""Tests for MCP coverage tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.coverage import register, classify_nodes, detect_splits, get_coverage
    server = create_server()
    register(server)
    assert callable(classify_nodes)
    assert callable(detect_splits)
    assert callable(get_coverage)
