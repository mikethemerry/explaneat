"""Tests for MCP evidence tools."""


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.evidence import (
        register,
        get_formula,
        compute_viz_data,
        render_visualization,
        get_viz_summary,
        compute_shap,
        compute_performance,
        get_input_distribution,
    )

    server = create_server()
    register(server)
    assert callable(get_formula)
    assert callable(compute_viz_data)
    assert callable(render_visualization)
    assert callable(get_viz_summary)
    assert callable(compute_shap)
    assert callable(compute_performance)
    assert callable(get_input_distribution)


def test_rendering_module_importable():
    from mcp_server.rendering import render_to_png, _RENDERERS

    # All expected renderers present
    expected = [
        "line", "heatmap", "pca_scatter", "feature_output_scatter",
        "sensitivity", "edge_influence", "output_distribution",
        "activation_profile", "ice", "partial_dependence", "histogram",
        "scatter2d",
    ]
    for viz_type in expected:
        assert viz_type in _RENDERERS, f"Missing renderer for {viz_type}"


def test_render_line_plot():
    """Test that the line renderer produces a base64 PNG."""
    from mcp_server.rendering import render_to_png

    data = {
        "grid_x": [0.0, 0.5, 1.0],
        "grid_y": [0.0, 0.25, 1.0],
        "scatter_x": [0.1, 0.5, 0.9],
        "scatter_y": [0.05, 0.3, 0.85],
        "x_label": "input",
        "y_label": "output",
    }
    result = render_to_png("line", data, title="Test Line")
    assert isinstance(result, str)
    assert len(result) > 100  # Should be non-trivial base64


def test_render_heatmap():
    """Test heatmap renderer."""
    import numpy as np
    from mcp_server.rendering import render_to_png

    data = {
        "x_range": [0.0, 0.5, 1.0],
        "y_range": [0.0, 0.5, 1.0],
        "z_grid": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "x_label": "x0",
        "y_label": "x1",
        "z_label": "output",
    }
    result = render_to_png("heatmap", data)
    assert isinstance(result, str)
    assert len(result) > 100


def test_render_bar():
    """Test bar renderer (sensitivity)."""
    from mcp_server.rendering import render_to_png

    data = {
        "input_labels": ["x0", "x1", "x2"],
        "sensitivities": {"y_0": [0.5, 0.3, 0.1]},
    }
    result = render_to_png("sensitivity", data)
    assert isinstance(result, str)
    assert len(result) > 100


def test_render_distribution():
    """Test distribution renderer."""
    from mcp_server.rendering import render_to_png

    data = {
        "bin_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
        "counts": [10, 20, 15, 5],
        "x_label": "value",
        "stats": {"mean": 0.4, "std": 0.2, "min": 0.0, "max": 1.0},
    }
    result = render_to_png("output_distribution", data)
    assert isinstance(result, str)
    assert len(result) > 100


def test_render_ice():
    """Test ICE renderer."""
    from mcp_server.rendering import render_to_png

    data = {
        "grid_x": [0.0, 0.5, 1.0],
        "ice_curves": [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
        "pd_curve": [0.15, 0.35, 0.55],
        "x_label": "x",
        "y_label": "y",
    }
    result = render_to_png("ice", data)
    assert isinstance(result, str)
    assert len(result) > 100


def test_json_default_numpy():
    """Test that numpy types serialize correctly."""
    import json
    import numpy as np
    from mcp_server.tools.evidence import _json_default

    assert json.dumps(np.float64(3.14), default=_json_default) == "3.14"
    assert json.dumps(np.int64(42), default=_json_default) == "42"
    assert json.dumps(np.bool_(True), default=_json_default) == "true"
    arr = np.array([1, 2, 3])
    assert json.loads(json.dumps(arr, default=_json_default)) == [1, 2, 3]
