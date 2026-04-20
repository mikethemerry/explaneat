"""Entry point: python -m mcp_server"""
from mcp_server.server import create_server
from mcp_server.tools import register_all

mcp = create_server()
register_all(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
