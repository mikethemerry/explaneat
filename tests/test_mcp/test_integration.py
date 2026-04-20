"""Integration test — verify all tools register and server can start."""
import pytest


def test_all_tools_register():
    """Create server, register all tools, verify all 29 are importable."""
    from mcp_server.server import create_server
    from mcp_server.tools import register_all

    server = create_server()
    register_all(server)

    # Import all 29 tools
    from mcp_server.tools.experiments import list_experiments, get_experiment, get_best_genome, list_genomes, get_genome
    from mcp_server.tools.models import get_phenotype, get_model_state, get_node_info
    from mcp_server.tools.operations import list_operations, apply_operation, validate_operation, undo_operation, get_annotations
    from mcp_server.tools.evidence import get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution
    from mcp_server.tools.coverage import classify_nodes, detect_splits, get_coverage
    from mcp_server.tools.datasets import list_datasets, get_dataset, get_dataset_splits
    from mcp_server.tools.snapshots import save_snapshot, update_narrative, list_evidence

    all_tools = [
        list_experiments, get_experiment, get_best_genome, list_genomes, get_genome,
        get_phenotype, get_model_state, get_node_info,
        list_operations, apply_operation, validate_operation, undo_operation, get_annotations,
        get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution,
        classify_nodes, detect_splits, get_coverage,
        list_datasets, get_dataset, get_dataset_splits,
        save_snapshot, update_narrative, list_evidence,
    ]
    assert len(all_tools) == 29
    for tool in all_tools:
        assert callable(tool)


def test_server_entry_point():
    """The server module can be created without errors."""
    from mcp_server.server import create_server
    server = create_server()
    assert server.name == "explaneat"
