"""MCP server setup and database initialization."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from explaneat.db.base import Database

_db: Database | None = None


def get_db() -> Database:
    """Get the shared Database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools registered."""
    server = FastMCP("explaneat")
    get_db()
    return server
