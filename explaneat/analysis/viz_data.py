"""Compute visualization-ready data for D3/Observable Plot rendering."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

    # ICE plot: always available (per-sample curves for one feature)
    suggestions.append("ice")

    # Feature vs output scatter: always available
    suggestions.append("feature_output_scatter")

    # Output distribution: always available
    suggestions.append("output_distribution")

    # Activation profile: always available (single node)
    suggestions.append("activation_profile")

    # Edge influence: always available
    suggestions.append("edge_influence")

    # Regime map: always available (uses ReLU nodes)
    suggestions.append("regime_map")

    return suggestions


def compute_line_plot(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dim: int = 0,
    output_dim: int = 0,
    grid_points: int = 200,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute line plot data: function curve + actual data scatter.

    Args:
        fn: Callable (n_samples, n_in) -> (n_samples, n_out)
        entry_acts: Actual entry activations (n_samples, n_in)
        exit_acts: Actual exit activations (n_samples, n_out)
        input_dim: Which input dimension to use for x-axis
        output_dim: Which output dimension to use for y-axis
        grid_points: Number of grid points for the function curve
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

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

    x_label = entry_names[input_dim] if entry_names and input_dim < len(entry_names) else f"x_{input_dim}"
    y_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"

    return {
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "scatter_x": entry_acts[:, input_dim].tolist(),
        "scatter_y": exit_acts[:, output_dim].tolist(),
        "x_label": x_label,
        "y_label": y_label,
    }


def compute_heatmap(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dims: Tuple[int, int] = (0, 1),
    output_dim: int = 0,
    grid_size: int = 50,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute heatmap data for 2-input functions.

    Args:
        fn: Callable function
        entry_acts: Entry activations (n_samples, n_in)
        exit_acts: Exit activations (n_samples, n_out)
        input_dims: Which two input dimensions to vary (x, y)
        output_dim: Which output dimension for the color (z) axis
        grid_size: Grid resolution
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

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

    x_label = entry_names[dim_x] if entry_names and dim_x < len(entry_names) else f"x_{dim_x}"
    y_label = entry_names[dim_y] if entry_names and dim_y < len(entry_names) else f"x_{dim_y}"
    z_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"

    return {
        "x_range": x_range.tolist(),
        "y_range": y_range.tolist(),
        "z_grid": z_grid.tolist(),
        "scatter_x": entry_acts[:, dim_x].tolist(),
        "scatter_y": entry_acts[:, dim_y].tolist(),
        "scatter_z": exit_acts[:, output_dim].tolist(),
        "x_label": x_label,
        "y_label": y_label,
        "z_label": z_label,
    }


def compute_partial_dependence(
    fn: Callable,
    entry_acts: np.ndarray,
    vary_dims: List[int],
    output_dim: int = 0,
    fix_values: str = "median",
    grid_points: int = 100,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute partial dependence plot data.

    Args:
        fn: Callable function
        entry_acts: Entry activations (n_samples, n_in)
        vary_dims: Which dimensions to vary (1 or 2)
        output_dim: Which output dimension
        fix_values: 'median' or 'mean' for fixed dimensions
        grid_points: Grid resolution
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

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
        x_label = entry_names[dim] if entry_names and dim < len(entry_names) else f"x_{dim}"
        y_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"
        return {
            "type": "1d",
            "grid_x": grid_x.tolist(),
            "grid_y": y.tolist(),
            "x_label": x_label,
            "y_label": y_label,
        }
    elif len(vary_dims) == 2:
        return compute_heatmap(
            fn, entry_acts, np.zeros((len(entry_acts), 1)),
            input_dims=(vary_dims[0], vary_dims[1]),
            output_dim=output_dim,
            grid_size=min(grid_points, 50),
            entry_names=entry_names,
            exit_names=exit_names,
        )
    else:
        raise ValueError("vary_dims must have 1 or 2 elements")


def compute_pca_scatter(
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    output_dim: int = 0,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute 2D PCA projection of entry activations colored by exit value.

    Args:
        entry_acts: Entry activations (n_samples, n_in)
        exit_acts: Exit activations (n_samples, n_out)
        output_dim: Which output dimension to use for color encoding
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

    Returns:
        Dict with pca_x, pca_y, color_values, explained_variance
    """
    from sklearn.decomposition import PCA

    n_components = min(2, entry_acts.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(entry_acts)

    color_values = exit_acts[:, output_dim] if exit_acts.ndim > 1 else exit_acts

    color_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"

    result: Dict[str, Any] = {
        "pca_x": projected[:, 0].tolist(),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "color_values": color_values.tolist(),
        "color_label": color_label,
    }
    if n_components >= 2:
        result["pca_y"] = projected[:, 1].tolist()
    else:
        result["pca_y"] = np.zeros(len(projected)).tolist()

    return result


def compute_histogram(
    values: np.ndarray,
    num_bins: int = 30,
    label: str = "x",
) -> Dict[str, Any]:
    """Compute histogram data for a single feature.

    Returns:
        Dict with bin_edges, counts, x_label, and summary stats.
    """
    counts, bin_edges = np.histogram(values, bins=num_bins)

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "x_label": label,
        "stats": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": int(len(values)),
        },
    }


def compute_scatter_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
) -> Dict[str, Any]:
    """Compute scatter plot data for two features.

    Returns:
        Dict with x_values, y_values, labels, and per-feature stats.
    """
    return {
        "x_values": x_values.tolist(),
        "y_values": y_values.tolist(),
        "x_label": x_label,
        "y_label": y_label,
        "x_stats": {
            "mean": float(np.mean(x_values)),
            "std": float(np.std(x_values)),
            "min": float(np.min(x_values)),
            "max": float(np.max(x_values)),
            "median": float(np.median(x_values)),
        },
        "y_stats": {
            "mean": float(np.mean(y_values)),
            "std": float(np.std(y_values)),
            "min": float(np.min(y_values)),
            "max": float(np.max(y_values)),
            "median": float(np.median(y_values)),
        },
    }


def compute_sensitivity(
    fn: Callable,
    entry_acts: np.ndarray,
    perturbation: float = 0.01,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute per-input sensitivity scores.

    Measures how much each input affects the output by perturbing
    each dimension independently.

    Args:
        fn: Callable function
        entry_acts: Entry activations (n_samples, n_in)
        perturbation: Perturbation magnitude for finite differences
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

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
        out_label = exit_names[out_dim] if exit_names and out_dim < len(exit_names) else f"y_{out_dim}"
        scores = []
        for in_dim in range(n_in):
            perturbed = entry_acts.copy()
            perturbed[:, in_dim] += perturbation
            perturbed_output = fn(perturbed)
            if perturbed_output.ndim == 1:
                perturbed_output = perturbed_output.reshape(-1, 1)
            diff = np.abs(perturbed_output[:, out_dim] - base_output[:, out_dim])
            scores.append(float(np.mean(diff) / perturbation))
        sensitivities[out_label] = scores

    return {
        "input_labels": [entry_names[i] if entry_names and i < len(entry_names) else f"x_{i}" for i in range(n_in)],
        "sensitivities": sensitivities,
    }


def compute_ice_plot(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dim: int = 0,
    output_dim: int = 0,
    grid_points: int = 100,
    max_ice_samples: int = 50,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Individual Conditional Expectation (ICE) plot data.

    Each ICE curve shows how the prediction changes for one sample as
    a single feature varies across its range, with all other features
    held at that sample's actual values.

    Args:
        fn: Callable (n_samples, n_in) -> (n_samples, n_out)
        entry_acts: Entry activations (n_samples, n_in)
        exit_acts: Exit activations (n_samples, n_out)
        input_dim: Which input dimension to vary
        output_dim: Which output dimension for y-axis
        grid_points: Number of grid points per curve
        max_ice_samples: Maximum number of individual curves
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

    Returns:
        Dict with grid_x, ice_curves (list of lists), pd_curve (average)
    """
    n_samples = entry_acts.shape[0]
    x_min = float(entry_acts[:, input_dim].min())
    x_max = float(entry_acts[:, input_dim].max())
    margin = (x_max - x_min) * 0.05
    grid_x = np.linspace(x_min - margin, x_max + margin, grid_points)

    # Subsample if needed
    if n_samples > max_ice_samples:
        indices = np.random.choice(n_samples, max_ice_samples, replace=False)
    else:
        indices = np.arange(n_samples)

    ice_curves = []
    for idx in indices:
        sample = np.tile(entry_acts[idx], (grid_points, 1))
        sample[:, input_dim] = grid_x
        out = fn(sample)
        if out.ndim == 1:
            curve = out
        else:
            curve = out[:, output_dim]
        ice_curves.append(curve.tolist())

    # Partial dependence = average of ICE curves
    pd_curve = np.mean(ice_curves, axis=0).tolist()

    x_label = entry_names[input_dim] if entry_names and input_dim < len(entry_names) else f"x_{input_dim}"
    y_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"

    return {
        "grid_x": grid_x.tolist(),
        "ice_curves": ice_curves,
        "pd_curve": pd_curve,
        "x_label": x_label,
        "y_label": y_label,
    }


def compute_feature_output_scatter(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dim: int = 0,
    output_dim: int = 0,
    grid_points: int = 100,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute scatter of one input feature vs model output, with PD overlay.

    Args:
        fn: Callable (n_samples, n_in) -> (n_samples, n_out)
        entry_acts: Entry activations (n_samples, n_in)
        exit_acts: Exit activations (n_samples, n_out)
        input_dim: Which input dimension for x-axis
        output_dim: Which output dimension for y-axis
        grid_points: Number of grid points for PD curve
        entry_names: Optional list of names for entry (input) dimensions
        exit_names: Optional list of names for exit (output) dimensions

    Returns:
        Dict with scatter_x, scatter_y, pd_x, pd_y
    """
    scatter_x = entry_acts[:, input_dim]
    if exit_acts.ndim == 1:
        scatter_y = exit_acts
    else:
        scatter_y = exit_acts[:, output_dim]

    # PD curve: fix others at median, vary input_dim
    x_min = float(scatter_x.min())
    x_max = float(scatter_x.max())
    margin = (x_max - x_min) * 0.05
    pd_x = np.linspace(x_min - margin, x_max + margin, grid_points)

    medians = np.median(entry_acts, axis=0)
    grid_input = np.tile(medians, (grid_points, 1))
    grid_input[:, input_dim] = pd_x

    pd_out = fn(grid_input)
    if pd_out.ndim == 1:
        pd_y = pd_out
    else:
        pd_y = pd_out[:, output_dim]

    x_label = entry_names[input_dim] if entry_names and input_dim < len(entry_names) else f"x_{input_dim}"
    y_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"

    return {
        "scatter_x": scatter_x.tolist(),
        "scatter_y": scatter_y.tolist(),
        "pd_x": pd_x.tolist(),
        "pd_y": pd_y.tolist(),
        "x_label": x_label,
        "y_label": y_label,
    }


def compute_output_distribution(
    exit_acts: np.ndarray,
    output_dim: int = 0,
    num_bins: int = 30,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute histogram of model output values.

    Thin wrapper around compute_histogram for the exit_acts column.

    Args:
        exit_acts: Exit activations (n_samples, n_out)
        output_dim: Which output dimension to histogram
        num_bins: Number of histogram bins
        exit_names: Optional list of names for exit (output) dimensions

    Returns:
        Dict with bin_edges, counts, x_label, stats
    """
    if exit_acts.ndim == 1:
        values = exit_acts
    else:
        values = exit_acts[:, output_dim]

    label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"
    return compute_histogram(values, num_bins=num_bins, label=label)


def compute_activation_profile(
    activations: np.ndarray,
    activation_fn: str = "relu",
    n_bins: int = 50,
) -> Dict[str, Any]:
    """Compute activation distribution profile for a single node.

    Args:
        activations: shape (n_samples,) — one node's activation values.
        activation_fn: the node's activation function name (e.g. "relu", "sigmoid").
        n_bins: number of histogram bins.

    Returns:
        Dict with bin_edges, counts, stats including activation_rate and zero_fraction.
    """
    counts, bin_edges = np.histogram(activations, bins=n_bins)

    zero_count = int(np.sum(activations == 0.0))
    total = len(activations)

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "x_label": "Activation value",
        "stats": {
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "min": float(np.min(activations)),
            "max": float(np.max(activations)),
            "median": float(np.median(activations)),
            "count": total,
            "activation_rate": float(np.mean(activations > 0)),
            "zero_fraction": float(zero_count / total) if total > 0 else 0.0,
        },
    }


def compute_edge_influence(
    connections: Sequence,
    node_activations: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute per-edge influence scores based on weight * source activations.

    Args:
        connections: list of NetworkConnection objects (or dicts with
            from_node, to_node, weight attributes/keys).
        node_activations: {node_id: activation_array} mapping.

    Returns:
        Dict with sorted list of edge info dicts including influence and
        normalized_influence scores.
    """
    edges = []
    max_influence = 0.0

    for conn in connections:
        # Support both object attributes and dict keys
        if isinstance(conn, dict):
            from_node = conn["from_node"]
            to_node = conn["to_node"]
            weight = conn["weight"]
        else:
            from_node = conn.from_node
            to_node = conn.to_node
            weight = conn.weight

        if from_node not in node_activations:
            continue

        source_acts = node_activations[from_node]
        weighted = weight * source_acts
        influence = float(np.var(weighted))
        mean_contribution = float(np.mean(weighted))

        if influence > max_influence:
            max_influence = influence

        edges.append({
            "from": from_node,
            "to": to_node,
            "weight": float(weight),
            "influence": influence,
            "mean_contribution": mean_contribution,
        })

    # Normalize influence to [0, 1]
    for edge in edges:
        edge["normalized_influence"] = (
            edge["influence"] / max_influence if max_influence > 0 else 0.0
        )

    # Sort by influence descending
    edges.sort(key=lambda e: e["influence"], reverse=True)

    return {"edges": edges}


def compute_regime_map(
    node_activations: Dict[str, np.ndarray],
    relu_node_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute ReLU activation regime map.

    Groups samples by their binary ReLU on/off pattern and computes
    per-regime statistics.

    Args:
        node_activations: {node_id: activation_array} for all nodes.
        relu_node_ids: which nodes to use for regime detection.
        y_true: true labels, shape (n_samples,).
        y_pred: model predictions, shape (n_samples,) or (n_samples, 1).

    Returns:
        Dict with regime list sorted by count, plus summary info.
    """
    y_pred_flat = y_pred.ravel()
    y_true_flat = y_true.ravel()
    n_samples = len(y_true_flat)

    # Filter to relu nodes that exist in activations
    valid_relu_ids = [nid for nid in relu_node_ids if nid in node_activations]

    if not valid_relu_ids:
        return {
            "regimes": [],
            "relu_nodes": [],
            "total_samples": n_samples,
            "num_regimes": 0,
        }

    # Build binary pattern per sample
    # Pattern key: tuple of booleans for each relu node
    from collections import defaultdict
    regime_groups: Dict[tuple, List[int]] = defaultdict(list)

    for i in range(n_samples):
        pattern = tuple(
            bool(node_activations[nid][i] > 0) for nid in valid_relu_ids
        )
        regime_groups[pattern].append(i)

    regimes = []
    for pattern, indices in regime_groups.items():
        indices_arr = np.array(indices)
        count = len(indices)
        preds = y_pred_flat[indices_arr]
        trues = y_true_flat[indices_arr]

        # Classification accuracy: pred > 0.5 matches y_true
        predicted_classes = (preds > 0.5).astype(int)
        true_classes = trues.astype(int)
        accuracy = float(np.mean(predicted_classes == true_classes))

        # Class distribution
        unique, counts = np.unique(true_classes, return_counts=True)
        class_dist = {str(int(u)): int(c) for u, c in zip(unique, counts)}

        pattern_dict = {
            nid: bool(v) for nid, v in zip(valid_relu_ids, pattern)
        }

        regimes.append({
            "pattern": pattern_dict,
            "count": count,
            "fraction": float(count / n_samples) if n_samples > 0 else 0.0,
            "mean_prediction": float(np.mean(preds)),
            "accuracy": accuracy,
            "class_distribution": class_dist,
        })

    # Sort by count descending
    regimes.sort(key=lambda r: r["count"], reverse=True)

    return {
        "regimes": regimes,
        "relu_nodes": valid_relu_ids,
        "total_samples": n_samples,
        "num_regimes": len(regimes),
    }
