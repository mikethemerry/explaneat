"""MCP tools for evidence gathering: formulas, visualizations, SHAP, performance."""

import json
from typing import Any, Dict, List, Optional

import numpy as np

from mcp.server.fastmcp import FastMCP

from ..server import get_db
from ..helpers import (
    _to_uuid,
    build_engine,
    build_model_state,
    find_annotation_in_operations,
    load_split_data,
)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _compute_node_subgraph(model_state, node_id: str) -> Dict[str, Any]:
    """Compute a virtual annotation dict for a single node.

    BFS backward from node_id to find ancestor input nodes, then forward
    to find all nodes on paths to node_id.
    """
    predecessors: Dict[str, List[str]] = {}
    successors: Dict[str, List[str]] = {}
    for conn in model_state.connections:
        if not conn.enabled:
            continue
        predecessors.setdefault(conn.to_node, []).append(conn.from_node)
        successors.setdefault(conn.from_node, []).append(conn.to_node)

    # BFS backward
    backward_reachable: set = set()
    queue = [node_id]
    while queue:
        current = queue.pop(0)
        if current in backward_reachable:
            continue
        backward_reachable.add(current)
        for pred in predecessors.get(current, []):
            if pred not in backward_reachable:
                queue.append(pred)

    input_ids = set(model_state.input_node_ids)
    entry_nodes = sorted(n for n in backward_reachable if n in input_ids)

    if not entry_nodes:
        raise ValueError(f"Node '{node_id}' has no reachable input nodes")

    # BFS forward from entry nodes
    forward_reachable: set = set()
    queue = list(entry_nodes)
    while queue:
        current = queue.pop(0)
        if current in forward_reachable:
            continue
        forward_reachable.add(current)
        for succ in successors.get(current, []):
            if succ not in forward_reachable:
                queue.append(succ)

    subgraph_nodes = sorted(backward_reachable & forward_reachable)

    return {
        "id": f"__node_{node_id}__",
        "entry_nodes": entry_nodes,
        "exit_nodes": [node_id],
        "subgraph_nodes": subgraph_nodes,
        "subgraph_connections": [],
        "name": f"Node {node_id}",
    }


def _resolve_annotation(session, genome_id, model_state, annotation_id, node_id):
    """Resolve an annotation dict from annotation_id or node_id."""
    if node_id:
        return _compute_node_subgraph(model_state, node_id)
    elif annotation_id:
        return find_annotation_in_operations(session, genome_id, annotation_id)
    else:
        raise ValueError("Provide annotation_id or node_id")


def _build_whole_model_context(session, genome_id: str, X: np.ndarray, num_classes=None):
    """Build whole-model predict function and activations.

    Returns (fn, entry_acts, exit_acts, entry_names, exit_names, n_in, n_out).
    """
    from explaneat.core.structure_network import StructureNetwork
    import torch

    model_state = build_model_state(session, genome_id)
    struct_net = StructureNetwork(model_state)

    is_binary = (num_classes is not None and num_classes == 2
                 and len(model_state.output_node_ids) == 1)
    if is_binary:
        struct_net.override_output_activation("sigmoid")

    entry_names = [str(nid) for nid in model_state.input_node_ids]
    exit_names = [str(nid) for nid in model_state.output_node_ids]

    def model_predict(x):
        x = torch.as_tensor(x, dtype=torch.float64)
        out = struct_net.forward(x).detach().numpy()
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.ravel()
        return out

    exit_acts = model_predict(X)
    if exit_acts.ndim == 1:
        exit_acts = exit_acts.reshape(-1, 1)

    n_in = X.shape[1]
    n_out = exit_acts.shape[1]

    return model_predict, X, exit_acts, entry_names, exit_names, n_in, n_out


def _do_compute_viz_data(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 200,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Core implementation for computing viz data (shared by multiple tools)."""
    from explaneat.analysis.annotation_function import AnnotationFunction
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.analysis import viz_data as vd
    from explaneat.core.structure_network import StructureNetwork
    import torch

    db = get_db()
    params = params or {}

    with db.session_scope() as session:
        X, y, feature_names, class_names, num_classes = load_split_data(
            session, dataset_split_id, split, sample_fraction, max_samples,
        )

        is_whole_model = not annotation_id and not node_id

        if is_whole_model:
            fn, entry_acts, exit_acts, entry_names, exit_names, n_in, n_out = \
                _build_whole_model_context(session, genome_id, X, num_classes)
        else:
            model_state = build_model_state(session, genome_id)
            annotation = _resolve_annotation(
                session, genome_id, model_state, annotation_id, node_id,
            )
            extractor = ActivationExtractor.from_structure(model_state)
            entry_acts, exit_acts = extractor.extract(X, annotation)

            ann_fn = AnnotationFunction.from_structure(annotation, model_state)
            fn = ann_fn
            n_in, n_out = ann_fn.dimensionality

            entry_names = [str(n) for n in annotation["entry_nodes"]]
            exit_names = [str(n) for n in annotation["exit_nodes"]]

        suggested = vd.suggest_viz_types(n_in, n_out)

        if viz_type == "line":
            data = vd.compute_line_plot(
                fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", output_index),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "heatmap":
            data = vd.compute_heatmap(
                fn, entry_acts, exit_acts,
                input_dims=tuple(params.get("input_dims", [0, 1])),
                output_dim=params.get("output_dim", output_index),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "partial_dependence":
            data = vd.compute_partial_dependence(
                fn, entry_acts,
                vary_dims=params.get("vary_dims", [0]),
                output_dim=params.get("output_dim", output_index),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "pca_scatter":
            data = vd.compute_pca_scatter(
                entry_acts, exit_acts,
                output_dim=params.get("output_dim", output_index),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "sensitivity":
            data = vd.compute_sensitivity(
                fn, entry_acts,
                perturbation=params.get("perturbation", 0.01),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "ice":
            data = vd.compute_ice_plot(
                fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", output_index),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "feature_output_scatter":
            data = vd.compute_feature_output_scatter(
                fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", output_index),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "output_distribution":
            data = vd.compute_output_distribution(
                exit_acts,
                output_dim=params.get("output_dim", output_index),
                exit_names=exit_names,
            )
        elif viz_type == "activation_profile":
            target_node = node_id
            if not target_node:
                raise ValueError("activation_profile requires node_id")
            ms = build_model_state(session, genome_id) if is_whole_model else model_state
            all_node_ids = [n.id for n in ms.nodes]
            struct_net = StructureNetwork(ms)
            struct_net.override_output_activation("sigmoid")
            struct_net.override_hidden_activation("relu")
            x_tensor = torch.as_tensor(X, dtype=torch.float64)
            struct_net.forward(x_tensor)
            node_acts = {}
            for nid in all_node_ids:
                try:
                    node_acts[nid] = struct_net.get_node_activation(nid)
                except (ValueError, RuntimeError):
                    continue
            if target_node not in node_acts:
                raise ValueError(f"Node '{target_node}' not found or has no activations")
            act_fn = struct_net.node_info.get(target_node, {}).get("activation", "relu")
            data = vd.compute_activation_profile(
                node_acts[target_node],
                activation_fn=act_fn,
                n_bins=params.get("n_bins", 50),
            )
        elif viz_type == "edge_influence":
            ms = build_model_state(session, genome_id) if is_whole_model else model_state
            if is_whole_model:
                connections = [c for c in ms.connections if c.enabled]
                all_node_ids = [n.id for n in ms.nodes]
            else:
                subgraph_nodes = set(annotation.get("subgraph_nodes", []))
                connections = [
                    c for c in ms.connections
                    if c.enabled and c.from_node in subgraph_nodes and c.to_node in subgraph_nodes
                ]
                all_node_ids = list(subgraph_nodes)
            source_ids = list({c.from_node for c in connections})
            struct_net = StructureNetwork(ms)
            struct_net.override_output_activation("sigmoid")
            struct_net.override_hidden_activation("relu")
            x_tensor = torch.as_tensor(X, dtype=torch.float64)
            struct_net.forward(x_tensor)
            node_acts = {}
            for nid in source_ids:
                try:
                    node_acts[nid] = struct_net.get_node_activation(nid)
                except (ValueError, RuntimeError):
                    continue
            data = vd.compute_edge_influence(connections, node_acts)
        elif viz_type == "regime_map":
            ms = build_model_state(session, genome_id) if is_whole_model else model_state
            if is_whole_model:
                subgraph_nodes_set = {n.id for n in ms.nodes}
            else:
                subgraph_nodes_set = set(annotation.get("subgraph_nodes", []))
            all_node_ids = list(subgraph_nodes_set)
            struct_net = StructureNetwork(ms)
            struct_net.override_output_activation("sigmoid")
            struct_net.override_hidden_activation("relu")
            x_tensor = torch.as_tensor(X, dtype=torch.float64)
            struct_net.forward(x_tensor)
            node_acts = {}
            for nid in all_node_ids:
                try:
                    node_acts[nid] = struct_net.get_node_activation(nid)
                except (ValueError, RuntimeError):
                    continue
            input_ids = set(ms.input_node_ids)
            output_ids_set = set(ms.output_node_ids)
            relu_node_ids = [
                nid for nid in all_node_ids
                if nid not in input_ids and nid not in output_ids_set and nid in node_acts
            ]
            # Get y_pred
            output_ids = ms.output_node_ids
            if output_ids[0] in node_acts:
                y_pred = node_acts[output_ids[0]]
            else:
                y_pred = struct_net.forward(x_tensor).detach().numpy().ravel()
            data = vd.compute_regime_map(node_acts, relu_node_ids, y, y_pred)
        else:
            raise ValueError(f"Unknown viz_type: {viz_type}")

        return {
            "viz_type": viz_type,
            "data": data,
            "entry_names": entry_names,
            "exit_names": exit_names,
            "dimensionality": [n_in, n_out],
            "suggested_viz_types": suggested,
        }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def get_formula(
    genome_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    force: bool = False,
) -> str:
    """Get the closed-form mathematical formula for an annotation or node.

    Returns LaTeX representation of the annotation's function, dimensionality,
    and composition information.

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3"). Provide this OR node_id.
        node_id: ID of a single node to get the formula for. Provide this OR annotation_id.
        force: If True, compute formula even for intractable expressions.
    """
    from explaneat.analysis.annotation_function import AnnotationFunction

    if not annotation_id and not node_id:
        return json.dumps({"error": "Provide annotation_id or node_id"})

    db = get_db()
    try:
        with db.session_scope() as session:
            model_state = build_model_state(session, genome_id)
            annotation = _resolve_annotation(session, genome_id, model_state, annotation_id, node_id)

            ann_fn = AnnotationFunction.from_structure(annotation, model_state)
            n_in, n_out = ann_fn.dimensionality

            child_ann_ids = annotation.get("child_annotation_ids", [])
            is_composed = len(child_ann_ids) > 0

            latex = ann_fn.to_latex(force=force)
            latex_collapsed = None
            if is_composed:
                latex_collapsed = ann_fn.to_latex(expand=False, force=force)

            result = {
                "latex": latex,
                "latex_collapsed": latex_collapsed,
                "tractable": latex is not None,
                "is_composed": is_composed,
                "child_annotation_ids": child_ann_ids,
                "dimensionality": [n_in, n_out],
            }
            return json.dumps(result, indent=2, default=_json_default)
    except Exception as e:
        return json.dumps({"error": str(e)})


def compute_viz_data(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 200,
    params: Optional[str] = None,
) -> str:
    """Compute visualization data for an annotation, node, or whole model.

    Produces JSON data suitable for rendering as a chart. Supports many viz
    types: line, heatmap, partial_dependence, pca_scatter, sensitivity, ice,
    feature_output_scatter, output_distribution, activation_profile,
    edge_influence, regime_map.

    Args:
        genome_id: UUID of the genome.
        viz_type: Type of visualization to compute.
        dataset_split_id: UUID of the dataset split to use for data.
        annotation_id: Annotation ID (optional, for annotation-level viz).
        node_id: Node ID (optional, for node-level viz).
        split: Which split to use: "train", "test", "val", or "both".
        output_index: Which output dimension to visualize (default 0).
        sample_fraction: Fraction of data to sample (0-1).
        max_samples: Maximum number of samples.
        params: JSON string of extra params (e.g. {"input_dim": 1, "input_dims": [0,2]}).
    """
    extra_params = json.loads(params) if params else {}
    try:
        result = _do_compute_viz_data(
            genome_id=genome_id,
            viz_type=viz_type,
            dataset_split_id=dataset_split_id,
            annotation_id=annotation_id,
            node_id=node_id,
            split=split,
            output_index=output_index,
            sample_fraction=sample_fraction,
            max_samples=max_samples,
            params=extra_params,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps(result, indent=2, default=_json_default)


def render_visualization(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 200,
    params: Optional[str] = None,
    title: str = "",
) -> list:
    """Compute visualization data and render it as a PNG image.

    Returns a list containing an image (base64 PNG) and text metadata.
    Use this when you want to SEE the visualization rather than just get data.

    Args:
        genome_id: UUID of the genome.
        viz_type: Type of visualization to compute.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional).
        node_id: Node ID (optional).
        split: Which split: "train", "test", "val", or "both".
        output_index: Which output dimension (default 0).
        sample_fraction: Fraction of data to sample (0-1).
        max_samples: Maximum number of samples.
        params: JSON string of extra params.
        title: Optional title for the plot.
    """
    from mcp.types import ImageContent, TextContent
    from ..rendering import render_to_png

    extra_params = json.loads(params) if params else {}
    result = _do_compute_viz_data(
        genome_id=genome_id,
        viz_type=viz_type,
        dataset_split_id=dataset_split_id,
        annotation_id=annotation_id,
        node_id=node_id,
        split=split,
        output_index=output_index,
        sample_fraction=sample_fraction,
        max_samples=max_samples,
        params=extra_params,
    )

    png_b64 = render_to_png(viz_type, result["data"], title=title)

    metadata = {
        "viz_type": viz_type,
        "dimensionality": result["dimensionality"],
        "entry_names": result["entry_names"],
        "exit_names": result["exit_names"],
        "suggested_viz_types": result["suggested_viz_types"],
    }

    return [
        ImageContent(type="image", data=png_b64, mimeType="image/png"),
        TextContent(type="text", text=json.dumps(metadata, indent=2, default=_json_default)),
    ]


def get_viz_summary(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 200,
    params: Optional[str] = None,
) -> str:
    """Compute visualization data and return summary statistics (no image).

    Returns ranges, means, correlations, and point counts from the computed data.
    Useful for quick analysis without rendering.

    Args:
        genome_id: UUID of the genome.
        viz_type: Type of visualization.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional).
        node_id: Node ID (optional).
        split: Which split: "train", "test", "val", or "both".
        output_index: Which output dimension (default 0).
        sample_fraction: Fraction of data to sample (0-1).
        max_samples: Maximum number of samples.
        params: JSON string of extra params.
    """
    extra_params = json.loads(params) if params else {}
    result = _do_compute_viz_data(
        genome_id=genome_id,
        viz_type=viz_type,
        dataset_split_id=dataset_split_id,
        annotation_id=annotation_id,
        node_id=node_id,
        split=split,
        output_index=output_index,
        sample_fraction=sample_fraction,
        max_samples=max_samples,
        params=extra_params,
    )

    data = result["data"]
    summary: Dict[str, Any] = {
        "viz_type": viz_type,
        "dimensionality": result["dimensionality"],
        "entry_names": result["entry_names"],
        "exit_names": result["exit_names"],
    }

    # Extract stats based on viz_type
    if "stats" in data:
        summary["stats"] = data["stats"]
    if "grid_x" in data:
        gx = np.array(data["grid_x"])
        summary["x_range"] = [float(gx.min()), float(gx.max())]
        summary["x_points"] = len(gx)
    if "grid_y" in data:
        gy = np.array(data["grid_y"])
        summary["y_range"] = [float(np.min(gy)), float(np.max(gy))]
        summary["y_mean"] = float(np.mean(gy))
    if "scatter_x" in data:
        sx = np.array(data["scatter_x"])
        sy = np.array(data["scatter_y"])
        summary["scatter_points"] = len(sx)
        summary["scatter_x_range"] = [float(sx.min()), float(sx.max())]
        summary["scatter_y_range"] = [float(sy.min()), float(sy.max())]
        if len(sx) > 2:
            corr = float(np.corrcoef(sx, sy)[0, 1])
            summary["correlation"] = corr if not np.isnan(corr) else None
    if "sensitivities" in data:
        summary["sensitivities"] = data["sensitivities"]
        summary["input_labels"] = data["input_labels"]
    if "ice_curves" in data:
        summary["num_ice_curves"] = len(data["ice_curves"])
    if "regimes" in data:
        summary["num_regimes"] = data["num_regimes"]
        summary["total_samples"] = data["total_samples"]
    if "edges" in data:
        summary["num_edges"] = len(data["edges"])
        if data["edges"]:
            summary["top_edge"] = data["edges"][0]

    return json.dumps(summary, indent=2, default=_json_default)


def compute_shap(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    max_samples: int = 200,
) -> str:
    """Compute SHAP values to assess feature importance.

    For an annotation: uses the annotation function with entry activations as input.
    For whole model: uses StructureNetwork as the predict function.

    Returns feature_names, mean_abs_shap scores, and base_value.

    Args:
        genome_id: UUID of the genome.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional).
        node_id: Node ID (optional).
        split: Which split: "train", "test", "val", or "both".
        max_samples: Maximum samples for SHAP background (default 200).
    """
    from explaneat.analysis.shap_analysis import compute_shap_values
    from explaneat.analysis.annotation_function import AnnotationFunction
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.core.structure_network import StructureNetwork
    import torch

    db = get_db()
    with db.session_scope() as session:
        X, y, feature_names_ds, class_names, num_classes = load_split_data(
            session, dataset_split_id, split, 1.0, max_samples,
        )

        model_state = build_model_state(session, genome_id)

        if annotation_id or node_id:
            annotation = _resolve_annotation(
                session, genome_id, model_state, annotation_id, node_id,
            )
            ann_fn = AnnotationFunction.from_structure(annotation, model_state)
            extractor = ActivationExtractor.from_structure(model_state)
            entry_acts, _ = extractor.extract(X, annotation)

            feat_names = [str(n) for n in annotation["entry_nodes"]]
            predict_fn = ann_fn
            shap_input = entry_acts
        else:
            struct_net = StructureNetwork(model_state)
            is_binary = (num_classes is not None and num_classes == 2
                         and len(model_state.output_node_ids) == 1)
            if is_binary:
                struct_net.override_output_activation("sigmoid")

            feat_names = feature_names_ds or [str(nid) for nid in model_state.input_node_ids]

            def model_predict(x):
                x = torch.as_tensor(x, dtype=torch.float64)
                out = struct_net.forward(x).detach().numpy()
                if out.ndim == 2 and out.shape[1] == 1:
                    out = out.ravel()
                return out

            predict_fn = model_predict
            shap_input = X

    # Run SHAP (outside session since it's CPU-heavy)
    result = compute_shap_values(predict_fn, shap_input, feat_names, max_samples)

    output = {
        "feature_names": result["feature_names"],
        "mean_abs_shap": result["mean_abs_shap"],
        "base_value": result["base_value"],
    }
    if "outputs" in result:
        output["outputs"] = result["outputs"]

    return json.dumps(output, indent=2, default=_json_default)


def compute_performance(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    split: str = "test",
    max_samples: int = 1000,
) -> str:
    """Compute model performance metrics (MSE, RMSE, MAE, accuracy, etc.).

    For classification tasks, also computes accuracy, AUC-ROC, precision,
    recall, and F1.

    Args:
        genome_id: UUID of the genome.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional, for annotation-level perf).
        split: Which split: "train", "test", "val", or "both".
        max_samples: Maximum number of samples.
    """
    from explaneat.core.structure_network import StructureNetwork
    import torch

    db = get_db()
    with db.session_scope() as session:
        X, y, feature_names, class_names, num_classes = load_split_data(
            session, dataset_split_id, split, 1.0, max_samples,
        )

        # Infer num_classes
        if num_classes is None:
            y_check = y.ravel()
            if np.all(y_check == y_check.astype(int)):
                n_unique = len(np.unique(y_check.astype(int)))
                if n_unique <= 20:
                    num_classes = n_unique

        model_state = build_model_state(session, genome_id)
        struct_net = StructureNetwork(model_state)

        is_binary = (num_classes is not None and num_classes == 2
                     and len(model_state.output_node_ids) == 1)
        if is_binary:
            struct_net.override_output_activation("sigmoid")

        x_tensor = torch.as_tensor(X, dtype=torch.float64)
        predictions = struct_net.forward(x_tensor).detach().numpy()

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        y_flat = y.ravel()
        pred_flat = predictions.ravel() if predictions.ndim > 1 else predictions

        mse = float(np.mean((pred_flat - y_flat) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(pred_flat - y_flat)))

        # Correlation
        if len(y_flat) > 1:
            corr = float(np.corrcoef(pred_flat, y_flat)[0, 1])
            if np.isnan(corr):
                corr = None
        else:
            corr = None

        result: Dict[str, Any] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": corr,
            "n_samples": len(y_flat),
        }

        if num_classes is not None and num_classes >= 2:
            y_int = y_flat.astype(int)
            if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
                predicted_classes = (pred_flat > 0.5).astype(int)
            else:
                predicted_classes = np.argmax(predictions, axis=1).astype(int)
            result["accuracy"] = float(np.mean(predicted_classes == y_int))

            try:
                from sklearn.metrics import roc_auc_score
                pred_proba = np.clip(pred_flat, 1e-15, 1 - 1e-15)
                result["auc_roc"] = float(roc_auc_score(y_int, pred_proba))
            except Exception:
                pass

        return json.dumps(result, indent=2, default=_json_default)


def get_input_distribution(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    split: str = "test",
    max_samples: int = 1000,
) -> str:
    """Compute input feature distribution statistics.

    Returns per-feature summary stats (mean, std, min, max, quartiles) and
    a correlation matrix between features.

    Args:
        genome_id: UUID of the genome.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional, to use entry activations).
        split: Which split: "train", "test", "val", or "both".
        max_samples: Maximum number of samples.
    """
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.analysis.annotation_function import AnnotationFunction

    db = get_db()
    with db.session_scope() as session:
        X, y, feature_names, class_names, num_classes = load_split_data(
            session, dataset_split_id, split, 1.0, max_samples,
        )

        if annotation_id:
            model_state = build_model_state(session, genome_id)
            annotation = find_annotation_in_operations(session, genome_id, annotation_id)
            extractor = ActivationExtractor.from_structure(model_state)
            entry_acts, _ = extractor.extract(X, annotation)
            data = entry_acts
            names = [str(n) for n in annotation["entry_nodes"]]
        else:
            data = X
            names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

    n_features = data.shape[1]
    features = []
    for i in range(n_features):
        col = data[:, i]
        q25, q50, q75 = np.percentile(col, [25, 50, 75])
        features.append({
            "name": names[i] if i < len(names) else f"feature_{i}",
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
        })

    # Correlation matrix
    if n_features > 1:
        corr_matrix = np.corrcoef(data.T).tolist()
    else:
        corr_matrix = [[1.0]]

    result = {
        "features": features,
        "correlation_matrix": corr_matrix,
        "n_samples": data.shape[0],
        "n_features": n_features,
    }
    return json.dumps(result, indent=2, default=_json_default)


def register(mcp: FastMCP) -> None:
    """Register evidence tools with the MCP server."""
    mcp.tool()(get_formula)
    mcp.tool()(compute_viz_data)
    mcp.tool()(render_visualization)
    mcp.tool()(get_viz_summary)
    mcp.tool()(compute_shap)
    mcp.tool()(compute_performance)
    mcp.tool()(get_input_distribution)
