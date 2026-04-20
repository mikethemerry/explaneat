"""Tests for MCP operations tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.operations import (
        register,
        list_operations,
        apply_operation,
        validate_operation,
        undo_operation,
        get_annotations,
    )

    server = create_server()
    register(server)
    assert callable(list_operations)
    assert callable(apply_operation)
    assert callable(validate_operation)
    assert callable(undo_operation)
    assert callable(get_annotations)
