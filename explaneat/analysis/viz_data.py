"""Compute visualization-ready data for D3/Observable Plot rendering."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def suggest_viz_types(n_in: int, n_out: int) -> List[str]:
    """Suggest appropriate visualization types based on annotation dimensionality.

    All viz types work for any dimensionality thanks to fix-at-median and
    output dimension selection. Order reflects what's most natural for the
    given shape.
    """
    suggestions = []

    # Line plot: always available (varies 1 input, fixes others at median)
    suggestions.append("line")

    # Heatmap: available when n_in >= 2 (varies 2 inputs, fixes others)
    if n_in >= 2:
        suggestions.append("heatmap")

    # Partial dependence: useful for n_in >= 2 (explicit fix-at-median framing)
    if n_in >= 2:
        suggestions.append("partial_dependence")

    # PCA scatter: useful for n_in >= 2 to see overall input-output structure
    if n_in >= 2:
        suggestions.append("pca_scatter")

    # Sensitivity: always available
    suggestions.append("sensitivity")

    return suggestions


def compute_line_plot(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dim: int = 0,
    output_dim: int = 0,
    grid_points: int = 200,
) -> Dict[str, Any]:
    """Compute line plot data: function curve + actual data scatter.

    Args:
        fn: Callable (n_samples, n_in) -> (n_samples, n_out)
        entry_acts: Actual entry activations (n_samples, n_in)
        exit_acts: Actual exit activations (n_samples, n_out)
        input_dim: Which input dimension to use for x-axis
        output_dim: Which output dimension to use for y-axis
        grid_points: Number of grid points for the function curve

    Returns:
        JSON-serializable dict with grid_x, grid_y, scatter_x, scatter_y
    """
    n_in = entry_acts.shape[1]
    x_min = float(entry_acts[:, input_dim].min())
    x_max = float(entry_acts[:, input_dim].max())
    margin = (x_max - x_min) * 0.05
    grid_x = np.linspace(x_min - margin, x_max + margin, grid_points)

    # Build full input: fix other dims at median
    medians = np.median(entry_acts, axis=0)
    grid_input = np.tile(medians, (grid_points, 1))
    grid_input[:, input_dim] = grid_x

    grid_output = fn(grid_input)
    if grid_output.ndim == 1:
        grid_y = grid_output
    else:
        grid_y = grid_output[:, output_dim]

    return {
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "scatter_x": entry_acts[:, input_dim].tolist(),
        "scatter_y": exit_acts[:, output_dim].tolist(),
        "x_label": f"x_{input_dim}",
        "y_label": f"y_{output_dim}",
    }


def compute_heatmap(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dims: Tuple[int, int] = (0, 1),
    output_dim: int = 0,
    grid_size: int = 50,
) -> Dict[str, Any]:
    """Compute heatmap data for 2-input functions.

    Returns:
        Dict with x_range, y_range, z_grid (grid_size x grid_size),
        and scatter points.
    """
    n_in = entry_acts.shape[1]
    dim_x, dim_y = input_dims

    x_min, x_max = float(entry_acts[:, dim_x].min()), float(entry_acts[:, dim_x].max())
    y_min, y_max = float(entry_acts[:, dim_y].min()), float(entry_acts[:, dim_y].max())
    mx = (x_max - x_min) * 0.05
    my = (y_max - y_min) * 0.05

    x_range = np.linspace(x_min - mx, x_max + mx, grid_size)
    y_range = np.linspace(y_min - my, y_max + my, grid_size)
    xx, yy = np.meshgrid(x_range, y_range)

    # Build full grid input
    medians = np.median(entry_acts, axis=0)
    flat = np.tile(medians, (grid_size * grid_size, 1))
    flat[:, dim_x] = xx.ravel()
    flat[:, dim_y] = yy.ravel()

    z = fn(flat)
    if z.ndim > 1:
        z = z[:, output_dim]
    z_grid = z.reshape(grid_size, grid_size)

    return {
        "x_range": x_range.tolist(),
        "y_range": y_range.tolist(),
        "z_grid": z_grid.tolist(),
        "scatter_x": entry_acts[:, dim_x].tolist(),
        "scatter_y": entry_acts[:, dim_y].tolist(),
        "scatter_z": exit_acts[:, output_dim].tolist(),
        "x_label": f"x_{dim_x}",
        "y_label": f"x_{dim_y}",
        "z_label": f"y_{output_dim}",
    }


def compute_partial_dependence(
    fn: Callable,
    entry_acts: np.ndarray,
    vary_dims: List[int],
    output_dim: int = 0,
    fix_values: str = "median",
    grid_points: int = 100,
) -> Dict[str, Any]:
    """Compute partial dependence plot data.

    Args:
        fn: Callable function
        entry_acts: Entry activations (n_samples, n_in)
        vary_dims: Which dimensions to vary (1 or 2)
        output_dim: Which output dimension
        fix_values: 'median' or 'mean' for fixed dimensions
        grid_points: Grid resolution

    Returns:
        Dict with grid data for 1D or 2D partial dependence
    """
    if fix_values == "median":
        fix = np.median(entry_acts, axis=0)
    else:
        fix = np.mean(entry_acts, axis=0)

    if len(vary_dims) == 1:
        dim = vary_dims[0]
        x_min, x_max = float(entry_acts[:, dim].min()), float(entry_acts[:, dim].max())
        grid_x = np.linspace(x_min, x_max, grid_points)
        grid_input = np.tile(fix, (grid_points, 1))
        grid_input[:, dim] = grid_x
        y = fn(grid_input)
        if y.ndim > 1:
            y = y[:, output_dim]
        return {
            "type": "1d",
            "grid_x": grid_x.tolist(),
            "grid_y": y.tolist(),
            "x_label": f"x_{dim}",
            "y_label": f"y_{output_dim}",
        }
    elif len(vary_dims) == 2:
        return compute_heatmap(
            fn, entry_acts, np.zeros((len(entry_acts), 1)),
            input_dims=(vary_dims[0], vary_dims[1]),
            output_dim=output_dim,
            grid_size=min(grid_points, 50),
        )
    else:
        raise ValueError("vary_dims must have 1 or 2 elements")


def compute_pca_scatter(
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    output_dim: int = 0,
) -> Dict[str, Any]:
    """Compute 2D PCA projection of entry activations colored by exit value.

    Returns:
        Dict with pca_x, pca_y, color_values, explained_variance
    """
    from sklearn.decomposition import PCA

    n_components = min(2, entry_acts.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(entry_acts)

    color_values = exit_acts[:, output_dim] if exit_acts.ndim > 1 else exit_acts

    result: Dict[str, Any] = {
        "pca_x": projected[:, 0].tolist(),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "color_values": color_values.tolist(),
        "color_label": f"y_{output_dim}",
    }
    if n_components >= 2:
        result["pca_y"] = projected[:, 1].tolist()
    else:
        result["pca_y"] = np.zeros(len(projected)).tolist()

    return result


def compute_sensitivity(
    fn: Callable,
    entry_acts: np.ndarray,
    perturbation: float = 0.01,
) -> Dict[str, Any]:
    """Compute per-input sensitivity scores.

    Measures how much each input affects the output by perturbing
    each dimension independently.

    Returns:
        Dict with input_labels and sensitivity_scores per output
    """
    n_in = entry_acts.shape[1]
    base_output = fn(entry_acts)
    if base_output.ndim == 1:
        base_output = base_output.reshape(-1, 1)
    n_out = base_output.shape[1]

    sensitivities = {}
    for out_dim in range(n_out):
        scores = []
        for in_dim in range(n_in):
            perturbed = entry_acts.copy()
            perturbed[:, in_dim] += perturbation
            perturbed_output = fn(perturbed)
            if perturbed_output.ndim == 1:
                perturbed_output = perturbed_output.reshape(-1, 1)
            diff = np.abs(perturbed_output[:, out_dim] - base_output[:, out_dim])
            scores.append(float(np.mean(diff) / perturbation))
        sensitivities[f"y_{out_dim}"] = scores

    return {
        "input_labels": [f"x_{i}" for i in range(n_in)],
        "sensitivities": sensitivities,
    }
