"""Matplotlib-based renderer for producing PNG images from viz_data output."""

import base64
import io
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def render_to_png(viz_type: str, data: dict, title: str = "") -> str:
    """Render viz_data output to a base64-encoded PNG string.

    Args:
        viz_type: The visualization type (line, heatmap, scatter, etc.)
        data: The dict returned by a viz_data compute function.
        title: Optional plot title.

    Returns:
        Base64-encoded PNG string.
    """
    renderer = _RENDERERS.get(viz_type)
    if renderer is None:
        raise ValueError(f"No renderer for viz_type '{viz_type}'")

    fig, ax = plt.subplots(figsize=(8, 5))
    renderer(ax, data)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _render_line(ax, data: Dict[str, Any]):
    """Line plot with scatter overlay."""
    ax.plot(data["grid_x"], data["grid_y"], color="steelblue", linewidth=2, label="Function")
    ax.scatter(
        data["scatter_x"], data["scatter_y"],
        alpha=0.4, s=12, color="coral", label="Data",
    )
    ax.set_xlabel(data.get("x_label", "x"))
    ax.set_ylabel(data.get("y_label", "y"))
    ax.legend(fontsize=9)


def _render_heatmap(ax, data: Dict[str, Any]):
    """2D heatmap with colorbar."""
    z_grid = np.array(data["z_grid"])
    extent = [
        data["x_range"][0], data["x_range"][-1],
        data["y_range"][0], data["y_range"][-1],
    ]
    im = ax.imshow(
        z_grid, origin="lower", aspect="auto", extent=extent, cmap="viridis",
    )
    ax.set_xlabel(data.get("x_label", "x_0"))
    ax.set_ylabel(data.get("y_label", "x_1"))
    plt.colorbar(im, ax=ax, label=data.get("z_label", "output"))


def _render_scatter(ax, data: Dict[str, Any]):
    """Scatter plot for PCA or feature-output."""
    # PCA scatter uses pca_x/pca_y/color_values
    if "pca_x" in data:
        sc = ax.scatter(
            data["pca_x"], data["pca_y"],
            c=data["color_values"], cmap="viridis", alpha=0.6, s=15,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label=data.get("color_label", "output"))
    # Feature-output scatter uses scatter_x/scatter_y + pd overlay
    elif "scatter_x" in data and "pd_x" in data:
        ax.scatter(data["scatter_x"], data["scatter_y"], alpha=0.3, s=10, color="gray", label="Data")
        ax.plot(data["pd_x"], data["pd_y"], color="steelblue", linewidth=2, label="PD")
        ax.set_xlabel(data.get("x_label", "x"))
        ax.set_ylabel(data.get("y_label", "y"))
        ax.legend(fontsize=9)
    # Generic scatter (scatter_2d)
    elif "x_values" in data:
        ax.scatter(data["x_values"], data["y_values"], alpha=0.4, s=12)
        ax.set_xlabel(data.get("x_label", "x"))
        ax.set_ylabel(data.get("y_label", "y"))


def _render_bar(ax, data: Dict[str, Any]):
    """Bar chart for sensitivity and edge influence."""
    if "input_labels" in data and "sensitivities" in data:
        # Sensitivity: take first output's scores
        labels = data["input_labels"]
        sensitivities = data["sensitivities"]
        first_key = next(iter(sensitivities))
        scores = sensitivities[first_key]
        y_pos = range(len(labels))
        ax.barh(y_pos, scores, color="steelblue")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Sensitivity")
    elif "edges" in data:
        # Edge influence
        edges = data["edges"][:20]  # top 20
        labels = [f"{e.get('from_label', e['from'])} -> {e.get('to_label', e['to'])}" for e in edges]
        values = [e["normalized_influence"] for e in edges]
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color="teal")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Normalized Influence")


def _render_distribution(ax, data: Dict[str, Any]):
    """Histogram distribution."""
    bin_edges = np.array(data["bin_edges"])
    counts = np.array(data["counts"])
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts, width=widths, align="edge", color="steelblue", alpha=0.7)
    ax.set_xlabel(data.get("x_label", "Value"))
    ax.set_ylabel("Count")
    if "stats" in data:
        stats = data["stats"]
        ax.axvline(stats.get("mean", 0), color="red", linestyle="--", label=f"mean={stats['mean']:.3f}")
        ax.legend(fontsize=8)


def _render_ice(ax, data: Dict[str, Any]):
    """ICE curves with PDP overlay."""
    grid_x = data["grid_x"]
    for curve in data["ice_curves"]:
        ax.plot(grid_x, curve, color="lightblue", alpha=0.3, linewidth=0.5)
    ax.plot(grid_x, data["pd_curve"], color="darkblue", linewidth=2, label="PD (mean)")
    ax.set_xlabel(data.get("x_label", "x"))
    ax.set_ylabel(data.get("y_label", "y"))
    ax.legend(fontsize=9)


def _render_partial_dependence(ax, data: Dict[str, Any]):
    """Partial dependence plot (1D case)."""
    if data.get("type") == "1d":
        ax.plot(data["grid_x"], data["grid_y"], color="steelblue", linewidth=2)
        ax.set_xlabel(data.get("x_label", "x"))
        ax.set_ylabel(data.get("y_label", "y"))
    else:
        # 2D case renders as heatmap
        _render_heatmap(ax, data)


# Map viz_types to renderer functions
_RENDERERS = {
    "line": _render_line,
    "heatmap": _render_heatmap,
    "pca_scatter": _render_scatter,
    "feature_output_scatter": _render_scatter,
    "scatter2d": _render_scatter,
    "sensitivity": _render_bar,
    "edge_influence": _render_bar,
    "output_distribution": _render_distribution,
    "activation_profile": _render_distribution,
    "histogram": _render_distribution,
    "ice": _render_ice,
    "partial_dependence": _render_partial_dependence,
}
