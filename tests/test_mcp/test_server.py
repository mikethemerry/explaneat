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


def test_analysis_tools_registered():
    """The explanation-tool MCP tools are registered."""
    import asyncio

    from mcp_server.server import create_server
    from mcp_server.tools import register_all

    server = create_server()
    register_all(server)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"check_explanation", "linearise_annotations", "unit_activity",
            "suggest_candidates", "category_offsets"} <= names
