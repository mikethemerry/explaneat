"""Tests for MCP dataset tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.datasets import register, list_datasets, get_dataset, get_dataset_splits
    server = create_server()
    register(server)
    assert callable(list_datasets)
    assert callable(get_dataset)
    assert callable(get_dataset_splits)
