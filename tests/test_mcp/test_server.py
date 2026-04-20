"""Test MCP server initialization."""
import pytest


def test_server_creates():
    """Server object can be created."""
    from mcp_server.server import create_server
    server = create_server()
    assert server is not None
    assert server.name == "explaneat"


def test_server_has_db():
    """Server initializes database connection."""
    from mcp_server.server import create_server, get_db
    create_server()
    db = get_db()
    assert db is not None
