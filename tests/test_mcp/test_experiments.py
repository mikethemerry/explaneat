"""Tests for experiment and genome discovery MCP tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.experiments import (
        register,
        list_experiments,
        get_experiment,
        get_best_genome,
        list_genomes,
        get_genome,
    )

    server = create_server()
    register(server)

    assert callable(list_experiments)
    assert callable(get_experiment)
    assert callable(get_best_genome)
    assert callable(list_genomes)
    assert callable(get_genome)
