"""Tests for MCP model structure tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.models import register, get_phenotype, get_model_state, get_node_info

    server = create_server()
    register(server)
    assert callable(get_phenotype)
    assert callable(get_model_state)
    assert callable(get_node_info)
