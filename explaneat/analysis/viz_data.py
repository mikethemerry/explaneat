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

    # Pairs plot: pairwise interactions when there are 3 or more inputs
    if n_in >= 3:
        suggestions.append("pairs_plot")

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
    zero_count = int(np.sum(activations == 0.0))
    total = len(activations)
    positive_mask = activations > 0

    # For ReLU-like activations, separate the zero-clamped spike from the
    # positive distribution so the dead zone is clearly visible.
    if activation_fn in ("relu", "clamped") and zero_count > 0 and positive_mask.any():
        pos_vals = activations[positive_mask]
        pos_counts, pos_edges = np.histogram(pos_vals, bins=n_bins)
        # Prepend a zero bin: [small_negative, 0] containing the zero count
        epsilon = pos_edges[1] - pos_edges[0]  # match bin width
        bin_edges = np.concatenate([[-epsilon], pos_edges])
        counts = np.concatenate([[zero_count], pos_counts])
    else:
        counts, bin_edges = np.histogram(activations, bins=n_bins)

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
            "activation_rate": float(np.mean(positive_mask)),
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


# ---------------------------------------------------------------------------
# Pairs plot: k x k grid of 1D / 2D partial dependence and the data
# ---------------------------------------------------------------------------

PAIRS_MAX_INPUTS = 8
PAIRS_DEFAULT_INPUTS = 5
_DISCRETE_MAX_LEVELS = 10
_EVAL_CHUNK_ROWS = 200_000


def _eval_out(fn: Callable, x: np.ndarray, output_dim: int) -> np.ndarray:
    """fn on x in row chunks, the chosen output column as a 1-D array."""
    parts = []
    for s in range(0, len(x), _EVAL_CHUNK_ROWS):
        y = np.asarray(fn(x[s:s + _EVAL_CHUNK_ROWS]), dtype=float)
        if y.ndim > 1:
            y = y[:, output_dim]
        parts.append(y.ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def _input_grid(col: np.ndarray, grid_size: int) -> Tuple[str, np.ndarray, np.ndarray]:
    """(kind, grid values, weight of each grid value in the data).

    ``binary`` (2 values) and ``discrete`` (3..10 values) inputs are evaluated at the
    values the data takes, never in between; ``continuous`` ones on an even grid over
    the data's range. The weights are the share of rows nearest each grid value.
    """
    levels = np.unique(col)
    if len(levels) <= 2:
        kind = "binary" if len(levels) == 2 else "constant"
    elif len(levels) <= _DISCRETE_MAX_LEVELS:
        kind = "discrete"
    else:
        kind = "continuous"
    if kind != "continuous":
        counts = np.array([np.sum(col == v) for v in levels], dtype=float)
        return kind, levels.astype(float), counts / max(counts.sum(), 1.0)
    grid = np.linspace(float(col.min()), float(col.max()), grid_size)
    step = grid[1] - grid[0]
    idx = np.clip(np.rint((col - grid[0]) / step).astype(int), 0, grid_size - 1)
    counts = np.bincount(idx, minlength=grid_size).astype(float)
    return kind, grid, counts / max(counts.sum(), 1.0)


def _wcenter(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return v - float(np.sum(v * w) / max(np.sum(w), 1e-300))


def _additive_fit(f2: np.ndarray, w: np.ndarray, iters: int = 500, tol: float = 1e-12) -> np.ndarray:
    """Weighted least-squares fit of ``r[a] + c[b]`` to the surface ``f2`` (backfitting).

    Alternating weighted row / column means: exact projection in the limit, no BLAS
    (tiny dense solves are slow under some threaded BLAS builds).
    """
    w = np.asarray(w, dtype=float)
    wr = w.sum(axis=1)
    wc = w.sum(axis=0)
    r = np.zeros(f2.shape[0])
    c = np.zeros(f2.shape[1])
    prev = np.inf
    for _ in range(iters):
        r = np.divide((w * (f2 - c[None, :])).sum(axis=1), wr, out=np.zeros_like(r), where=wr > 0)
        c = np.divide((w * (f2 - r[:, None])).sum(axis=0), wc, out=np.zeros_like(c), where=wc > 0)
        sse = float(np.sum(w * (f2 - r[:, None] - c[None, :]) ** 2))
        if np.isfinite(prev) and prev - sse <= tol * max(prev, 1e-300):
            break
        prev = sse
    return r[:, None] + c[None, :]


def pair_interaction(f2: np.ndarray, f1_i: np.ndarray, f1_j: np.ndarray,
                     w_i: np.ndarray, w_j: np.ndarray,
                     w_joint: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Interaction strength of a pair from its PD surfaces on a grid.

    ``f2[a, b]`` is the 2-D PD at (grid_i[a], grid_j[b]); ``f1_i``/``f1_j`` the 1-D PDs.
    Cells are weighted by where the data lies: ``w_joint`` (the share of rows nearest
    each cell) when given, else ``w_i ⊗ w_j``. Cells no row is near get weight 0, so
    combinations the data cannot take (two levels of one category, a split copy
    against itself) do not count.

    - ``h`` (shown): ``H² = Σ w R² / Σ w F2c²``, where ``R`` is what is left of the
      surface after the best (weighted least-squares) sum of one effect per input and
      ``F2c`` the surface minus its weighted mean: the share of the surface's variation
      that is not one effect of each input added. 0 for an additive function, at most 1.
    - ``h_pd``: the same with the computed 1-D PDs as the effects
      (``F2c − F1c_i − F1c_j``), Friedman's form; with the others held at the median the
      1-D PDs are slices through the median, so ``h_pd`` can exceed 1.
    - ``interaction_sd``: the weighted SD of ``R`` in the exit's units.
    """
    w = np.outer(w_i, w_j) if w_joint is None else np.asarray(w_joint, dtype=float)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return {"h2": 0.0, "h": 0.0, "h_pd": 0.0, "interaction_sd": 0.0}
    w = w / wsum
    c2 = _wcenter(f2, w)
    den = float(np.sum(w * c2 ** 2))
    resid = f2 - _additive_fit(f2, w)
    num = float(np.sum(w * resid ** 2))
    h2 = min(max(num / den, 0.0), 1.0) if den > 1e-24 else 0.0
    r_pd = c2 - _wcenter(f1_i, w.sum(axis=1))[:, None] - _wcenter(f1_j, w.sum(axis=0))[None, :]
    h2_pd = float(np.sum(w * r_pd ** 2)) / den if den > 1e-24 else 0.0
    return {
        "h2": float(h2), "h": float(np.sqrt(h2)),
        "h_pd": float(np.sqrt(max(h2_pd, 0.0))),
        "interaction_sd": float(np.sqrt(max(num, 0.0))),
    }


def _grid_index(col: np.ndarray, grid: np.ndarray, kind: str) -> np.ndarray:
    """Index of the grid value nearest each row's value."""
    if kind != "continuous":
        return np.abs(col[:, None] - grid[None, :]).argmin(axis=1)
    step = grid[1] - grid[0]
    return np.clip(np.rint((col - grid[0]) / step).astype(int), 0, len(grid) - 1)


def compute_pairs(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    dims: Optional[Sequence[int]] = None,
    output_dim: int = 0,
    grid_size: int = 25,
    max_scatter: int = 2000,
    fix: str = "median",
    max_background: int = 200,
    seed: int = 42,
    center: Optional[float] = None,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Pairs plot over a few inputs of ``fn`` (an annotation or the whole model).

    - ``ranking``: every input's sensitivity (mean |Δexit| / Δx for a 0.01 step, the
      ``compute_sensitivity`` measure, on the scatter rows); default ``dims`` are the top
      ``PAIRS_DEFAULT_INPUTS`` inputs that vary; at most ``PAIRS_MAX_INPUTS``.
    - ``inputs[k]``: kind, grid, data weights, histogram, the 1-D PD over the grid and
      the scatter rows' values.
    - ``pairs``: for each i < j the 2-D PD over grid_i × grid_j and its interaction
      (``pair_interaction``), sorted by H, strongest first.
    - ``fix``: ``"median"`` holds the other inputs at their median over all rows (as the
      other viz types do); ``"average"`` averages over ``max_background`` seeded rows
      (classic partial dependence).
    """
    entry_acts = np.asarray(entry_acts, dtype=float)
    exit_acts = np.asarray(exit_acts, dtype=float)
    if exit_acts.ndim == 1:
        exit_acts = exit_acts.reshape(-1, 1)
    n_rows, n_in = entry_acts.shape
    if n_in < 2:
        raise ValueError("A pairs plot needs at least 2 inputs")
    names = [entry_names[i] if entry_names and i < len(entry_names) else f"x_{i}" for i in range(n_in)]
    out_name = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"

    rng = np.random.RandomState(seed)
    rows = np.sort(rng.choice(n_rows, size=max_scatter, replace=False)) if n_rows > max_scatter else np.arange(n_rows)
    S = entry_acts[rows]
    medians = np.median(entry_acts, axis=0)

    # sensitivity ranking, one vectorised call
    h = 0.01
    base = _eval_out(fn, S, output_dim)
    sens = np.zeros(n_in)
    per = max(1, int(2e7 // max(1, len(S) * n_in)))  # bound the perturbed copies' memory
    for s0 in range(0, n_in, per):
        ds = np.arange(s0, min(n_in, s0 + per))
        pert = np.repeat(S[None, :, :], len(ds), axis=0)
        pert[np.arange(len(ds)), :, ds] += h
        moved = _eval_out(fn, pert.reshape(-1, n_in), output_dim).reshape(len(ds), len(S))
        sens[ds] = np.mean(np.abs(moved - base[None, :]), axis=1) / h
    varies = np.ptp(entry_acts, axis=0) > 0
    order = sorted(range(n_in), key=lambda i: (-(sens[i] if varies[i] else -1.0), i))
    ranking = [{"dim": int(i), "name": names[i], "sensitivity": float(sens[i]), "varies": bool(varies[i])}
               for i in order]

    if dims is None or len(dims) == 0:
        dims = [i for i in order if varies[i]][:PAIRS_DEFAULT_INPUTS]
        if len(dims) < 2:
            dims = order[:max(2, min(PAIRS_DEFAULT_INPUTS, n_in))]
    dims = [int(d) for d in dims if 0 <= int(d) < n_in]
    dims = list(dict.fromkeys(dims))[:PAIRS_MAX_INPUTS]
    if len(dims) < 2:
        raise ValueError("Choose at least 2 inputs for a pairs plot")

    if fix == "average":
        n_bg = min(max_background, len(S))
        B = S[np.sort(rng.choice(len(S), size=n_bg, replace=False))] if len(S) > n_bg else S
    else:
        fix = "median"
        B = medians[None, :]
    n_b = len(B)

    def pd_eval(assign: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """PD at each point of a (flat) grid: assign = [(dim, values per point), ...]."""
        n_pts = len(assign[0][1])
        x = np.repeat(B[None, :, :], n_pts, axis=0)  # (pts, n_b, n_in)
        for d, vals in assign:
            x[:, :, d] = np.asarray(vals)[:, None]
        return _eval_out(fn, x.reshape(-1, n_in), output_dim).reshape(n_pts, n_b).mean(axis=1)

    if fix == "median":
        reference = float(_eval_out(fn, medians[None, :], output_dim)[0])
    else:
        reference = float(np.mean(_eval_out(fn, B, output_dim)))

    inputs = []
    for d in dims:
        col = entry_acts[:, d]
        kind, grid, weights = _input_grid(col, grid_size)
        if kind == "continuous":
            counts, edges = np.histogram(col, bins=20, range=(float(grid[0]), float(grid[-1])))
            hist = {"edges": edges.tolist(), "counts": counts.tolist()}
        else:
            hist = {"levels": grid.tolist(), "counts": [int(np.sum(col == v)) for v in grid]}
        inputs.append({
            "dim": int(d),
            "name": names[d],
            "kind": kind,
            "grid": grid.tolist(),
            "weights": weights.tolist(),
            "median": float(medians[d]),
            "min": float(col.min()),
            "max": float(col.max()),
            "hist": hist,
            "pd": pd_eval([(d, grid)]).tolist(),
            "scatter": S[:, d].tolist(),
            "sensitivity": float(sens[d]),
        })

    cell_idx = [_grid_index(entry_acts[:, d], np.asarray(inp["grid"]), inp["kind"])
                for d, inp in zip(dims, inputs)]
    pairs = []
    for a in range(len(dims)):
        for b in range(a + 1, len(dims)):
            gi, gj = np.asarray(inputs[a]["grid"]), np.asarray(inputs[b]["grid"])
            ii, jj = np.meshgrid(gi, gj, indexing="ij")
            f2 = pd_eval([(dims[a], ii.ravel()), (dims[b], jj.ravel())]).reshape(len(gi), len(gj))
            counts = np.zeros((len(gi), len(gj)), dtype=int)
            np.add.at(counts, (cell_idx[a], cell_idx[b]), 1)
            inter = pair_interaction(f2, np.asarray(inputs[a]["pd"]), np.asarray(inputs[b]["pd"]),
                                     np.asarray(inputs[a]["weights"]), np.asarray(inputs[b]["weights"]),
                                     w_joint=counts)
            same = bool(np.array_equal(entry_acts[:, dims[a]], entry_acts[:, dims[b]]))
            pairs.append({"a": a, "b": b, "z": f2.tolist(), "counts": counts.tolist(),
                          "same_input": same, **inter})
    # strongest first; a split copy against itself (identical columns) last
    pairs.sort(key=lambda p: (p["same_input"], -p["h"], p["a"], p["b"]))
    for rank, p in enumerate(pairs):
        p["rank"] = rank + 1

    colour = exit_acts[rows, output_dim]
    if center is None:
        center = reference
    all_vals = [colour] + [np.asarray(p["z"]).ravel() for p in pairs] + [np.asarray(i["pd"]) for i in inputs]
    allv = np.concatenate(all_vals)
    vmin, vmax = float(np.min(allv)), float(np.max(allv))

    return {
        "dims": dims,
        "inputs": inputs,
        "pairs": pairs,
        "color": colour.tolist(),
        "exit_label": out_name,
        "output_dim": int(output_dim),
        # diverging around the centre, each side spanning its own data (an exit that is
        # never below the centre, e.g. a ReLU at 0, gets the full scale above it)
        "color_domain": [min(vmin, float(center)), float(center), max(vmax, float(center))],
        "value_range": [vmin, vmax],
        "reference": reference,
        "ranking": ranking,
        "fix": fix,
        "n_rows": int(n_rows),
        "n_scatter": int(len(rows)),
        "n_background": int(n_b) if fix == "average" else None,
        "grid_size": int(grid_size),
        "max_inputs": PAIRS_MAX_INPUTS,
    }
