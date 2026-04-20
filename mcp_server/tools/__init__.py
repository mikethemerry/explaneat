"""Tool registration."""
from mcp.server.fastmcp import FastMCP


def register_all(mcp: FastMCP) -> None:
    """Register all tool modules with the server."""
    from mcp_server.tools import experiments
    from mcp_server.tools import models
    from mcp_server.tools import operations
    from mcp_server.tools import evidence
    from mcp_server.tools import coverage
    from mcp_server.tools import datasets
    from mcp_server.tools import snapshots

    for module in [experiments, models, operations, evidence, coverage, datasets, snapshots]:
        module.register(mcp)
