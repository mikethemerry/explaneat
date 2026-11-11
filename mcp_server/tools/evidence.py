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
    load_annotations,
    load_split_data,
    load_split_data_stamped,
    load_split_metadata,
    resolve_annotation_for_evaluation,
)

# Every evidence tool defaults to the FULL TRAINING split (no sampling, no train+test mixing).
# Sampling is opt-in (sample_fraction < 1 or max_samples); every result says which rows it used
# ("split": {split, n, n_split, sampled, rows_hash, label}) and carries a provenance id
# ("provenance": {"id": "q_...", ...}) whose stored values narratives reference as
# {{q:<id>.<path>|<fmt>}} (see explaneat/analysis/provenance.py).
DEFAULT_SPLIT = "train"


def _scope(annotation_id=None, node_id=None) -> str:
    if annotation_id:
        return f"annotation {annotation_id} (its function from entry to exit activations)"
    if node_id:
        return f"node {node_id} (its subgraph from the inputs)"
    return "whole model"


def stamp_and_record(genome_id: str, tool: str, arguments: Dict[str, Any], query: Dict[str, Any],
                     stamp: Optional[Dict[str, Any]], result: Dict[str, Any]) -> Dict[str, Any]:
    """Add the split stamp and a provenance record (stored in the evidence log) to a result."""
    from explaneat.analysis.provenance import record_query
    if stamp is not None:
        result["split"] = stamp
    db = get_db()
    try:
        with db.session_scope() as session:
            prov = record_query(session, genome_id, tool, arguments, {**query, **({"rows": stamp["label"]} if stamp else {})},
                                stamp, {k: v for k, v in result.items() if k != "provenance"})
    except Exception as e:  # never lose the result for want of a log entry
        prov = {"id": None, "stored": False, "warning": f"provenance not stored ({type(e).__name__})"}
    result["provenance"] = prov
    return result


VALID_VIZ_TYPES = (
    "line", "heatmap", "partial_dependence", "pca_scatter", "sensitivity",
    "ice", "feature_output_scatter", "output_distribution",
    "activation_profile", "edge_influence", "regime_map",
)

_VIZ_TYPE_ALIASES = {
    "line_plot": "line",
    "lineplot": "line",
    "pdp": "partial_dependence",
    "pca": "pca_scatter",
}


def _normalize_viz_type(viz_type: str) -> str:
    """Map aliases (e.g. 'line_plot') to canonical viz type names.

    Raises ValueError listing the valid names for unknown types.
    """
    key = (viz_type or "").strip().lower()
    key = _VIZ_TYPE_ALIASES.get(key, key)
    if key not in VALID_VIZ_TYPES:
        raise ValueError(
            f"Unknown viz_type: {viz_type!r}. Valid viz types: "
            f"{', '.join(VALID_VIZ_TYPES)} (aliases: "
            f"{', '.join(f'{a}->{c}' for a, c in _VIZ_TYPE_ALIASES.items())})"
        )
    return key


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
    """Resolve an annotation dict from annotation_id or node_id.

    Composition annotations are expanded over all their descendants so that
    their function includes the children's subgraphs.
    """
    if node_id:
        return _compute_node_subgraph(model_state, node_id)
    elif annotation_id:
        return resolve_annotation_for_evaluation(
            session, genome_id, annotation_id, model_state,
        )
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
        struct_net.override_output_activation(struct_net.model_output_activation)

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
    split: str = DEFAULT_SPLIT,
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Core implementation for computing viz data (shared by multiple tools)."""
    from explaneat.analysis.annotation_function import AnnotationFunction
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.analysis import viz_data as vd
    from explaneat.core.structure_network import StructureNetwork
    import torch

    viz_type = _normalize_viz_type(viz_type)
    db = get_db()
    params = params or {}

    with db.session_scope() as session:
        X, y, feature_names, class_names, num_classes, stamp = load_split_data_stamped(
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
            struct_net.override_output_activation(struct_net.model_output_activation)
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
            struct_net.override_output_activation(struct_net.model_output_activation)
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
            struct_net.override_output_activation(struct_net.model_output_activation)
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
            "split": stamp,
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

            child_ann_ids = annotation.get("child_annotation_ids", [])
            is_composed = len(child_ann_ids) > 0

            # ``annotation`` is already expanded over all descendants, so the
            # plain (fully expanded) formula includes every child's terms.
            ann_fn_expanded = AnnotationFunction.from_structure(annotation, model_state)
            n_in, n_out = ann_fn_expanded.dimensionality

            # Composition view: direct children (each expanded over its own
            # descendants) shown as named functions.
            ann_fn_composed = None
            if is_composed:
                from explaneat.analysis.composition import (
                    child_annotations as _children,
                    expand_annotation,
                )
                all_anns = load_annotations(session, genome_id)
                children = [
                    expand_annotation(c, all_anns, model_state)
                    for c in _children(annotation, all_anns)
                ]
                if children:
                    ann_fn_composed = AnnotationFunction.from_structure(
                        annotation, model_state, child_annotations=children
                    )

            # Smart default: auto-force for small tractable networks
            if not force and n_in <= 8:
                force = True

            # latex: fully expanded (all scalar nodes)
            latex = ann_fn_expanded.to_latex(force=force)

            # latex_collapsed: children shown as named functions
            latex_collapsed = None
            if ann_fn_composed is not None:
                latex_collapsed = ann_fn_composed.to_latex(expand=False, force=force)
                # Also try expanded form through children
                if latex_collapsed is None:
                    latex_collapsed = ann_fn_composed.to_latex(expand=True, force=force)

            result = {
                "latex": latex,
                "latex_collapsed": latex_collapsed,
                "tractable": latex is not None or latex_collapsed is not None,
                "is_composed": is_composed,
                "child_annotation_ids": child_ann_ids,
                "dimensionality": [n_in, n_out],
                "entry_nodes": [str(n) for n in annotation.get("entry_nodes", [])],
                "exit_nodes": [str(n) for n in annotation.get("exit_nodes", [])],
            }
            if annotation.get("added_entry_nodes"):
                result["added_entry_nodes"] = annotation["added_entry_nodes"]
            return json.dumps(result, indent=2, default=_json_default)
    except Exception as e:
        return json.dumps({"error": str(e)})


def compute_viz_data(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: Optional[int] = None,
    params: Optional[str] = None,
) -> str:
    """Compute visualization data for an annotation, node, or whole model.

    Produces JSON data suitable for rendering as a chart. Supports many viz
    types: line, heatmap, partial_dependence, pca_scatter, sensitivity, ice,
    feature_output_scatter, output_distribution, activation_profile,
    edge_influence, regime_map (aliases: line_plot/lineplot -> line,
    pdp -> partial_dependence, pca -> pca_scatter). Unknown types return an
    error listing the valid names.

    Args:
        genome_id: UUID of the genome.
        viz_type: Type of visualization to compute.
        dataset_split_id: UUID of the dataset split to use for data.
        annotation_id: Annotation ID (optional, for annotation-level viz).
        node_id: Node ID (optional, for node-level viz).
        split: Which split: "train" (default), "test", "val", or "both". Evidence for an
            explanation uses train.
        output_index: Which output dimension (default 0).
        sample_fraction: Fraction of the split to sample (default 1.0: all rows).
        max_samples: Optional cap on the rows (default None: all rows). Sampling is opt-in;
            the result's "split" block says which rows were used.
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
    result = stamp_and_record(
        genome_id, "compute_viz_data",
        {"viz_type": viz_type, "dataset_split_id": dataset_split_id, "annotation_id": annotation_id,
         "node_id": node_id, "split": split, "output_index": output_index, "sample_fraction": sample_fraction,
         "max_samples": max_samples, "params": extra_params},
        {"quantity": f"{result['viz_type']} visualisation data", "scope": _scope(annotation_id, node_id),
         "population": f"{split} split", "unit": "the exit activation (model output for the whole model)"},
        result.get("split"), result)
    return json.dumps(result, indent=2, default=_json_default)


def render_visualization(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: Optional[int] = None,
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
        split: Which split: "train" (default), "test", "val", or "both". Evidence for an
            explanation uses train.
        output_index: Which output dimension (default 0).
        sample_fraction: Fraction of the split to sample (default 1.0: all rows).
        max_samples: Optional cap on the rows (default None: all rows). Sampling is opt-in;
            the result's "split" block says which rows were used.
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

    viz_type = result["viz_type"]  # canonical name (aliases resolved)
    png_b64 = render_to_png(viz_type, result["data"], title=title)

    metadata = {
        "viz_type": viz_type,
        "dimensionality": result["dimensionality"],
        "entry_names": result["entry_names"],
        "exit_names": result["exit_names"],
        "suggested_viz_types": result["suggested_viz_types"],
        "split": result.get("split"),
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
    split: str = DEFAULT_SPLIT,
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: Optional[int] = None,
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
        split: Which split: "train" (default), "test", "val", or "both". Evidence for an
            explanation uses train.
        output_index: Which output dimension (default 0).
        sample_fraction: Fraction of the split to sample (default 1.0: all rows).
        max_samples: Optional cap on the rows (default None: all rows). Sampling is opt-in;
            the result's "split" block says which rows were used.
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
        "viz_type": result["viz_type"],
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

    summary = stamp_and_record(
        genome_id, "get_viz_summary",
        {"viz_type": viz_type, "dataset_split_id": dataset_split_id, "annotation_id": annotation_id,
         "node_id": node_id, "split": split, "output_index": output_index, "sample_fraction": sample_fraction,
         "max_samples": max_samples, "params": extra_params},
        {"quantity": f"summary of the {result['viz_type']} visualisation", "scope": _scope(annotation_id, node_id),
         "population": f"{split} split", "unit": "the exit activation (model output for the whole model)"},
        result.get("split"), summary)
    return json.dumps(summary, indent=2, default=_json_default)


def compute_shap(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
    max_samples: Optional[int] = None,
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
        split: Which split: "train" (default), "test", "val", or "both".
        max_samples: Optional cap on the rows explained and used as background (default None:
            the whole split; slow on large data, so pass e.g. 500 there and quote the n the
            result's "split" block reports).
    """
    from explaneat.analysis.shap_analysis import compute_shap_values
    from explaneat.analysis.annotation_function import AnnotationFunction
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.core.structure_network import StructureNetwork
    import torch

    db = get_db()
    with db.session_scope() as session:
        X, y, feature_names_ds, class_names, num_classes, stamp = load_split_data_stamped(
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
                struct_net.override_output_activation(struct_net.model_output_activation)

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
    result = compute_shap_values(predict_fn, shap_input, feat_names, max_samples or len(shap_input))

    output = {
        "feature_names": result["feature_names"],
        "mean_abs_shap": result["mean_abs_shap"],
        "base_value": result["base_value"],
    }
    if "outputs" in result:
        output["outputs"] = result["outputs"]

    output = stamp_and_record(
        genome_id, "compute_shap",
        {"dataset_split_id": dataset_split_id, "annotation_id": annotation_id, "node_id": node_id,
         "split": split, "max_samples": max_samples},
        {"quantity": "mean |SHAP value| per input", "scope": _scope(annotation_id, node_id),
         "population": f"{split} split", "unit": "the exit activation (model output for the whole model)"},
        stamp, output)
    return json.dumps(output, indent=2, default=_json_default)


def _multiclass_performance(outputs: np.ndarray, y: np.ndarray, output_activation: str) -> Dict[str, Any]:
    """Metrics for a k-output model: accuracy (argmax), and with identity
    outputs (logits) the softmax cross-entropy and one-vs-rest macro AUC."""
    y_int = np.asarray(y).ravel().astype(int)
    k = outputs.shape[1]
    result: Dict[str, Any] = {
        "n_samples": int(len(y_int)),
        "n_outputs": int(k),
        "accuracy": float(np.mean(outputs.argmax(axis=1) == y_int)),
        "per_class_accuracy": {
            str(c): float(np.mean(outputs[y_int == c].argmax(axis=1) == c))
            for c in range(k) if np.any(y_int == c)
        },
    }
    if output_activation == "identity":
        z = outputs - outputs.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        result["cross_entropy"] = float(-np.mean(logp[np.arange(len(y_int)), y_int]))
        try:
            from sklearn.metrics import roc_auc_score
            result["auc_roc_ovr_macro"] = float(roc_auc_score(
                y_int, np.exp(logp), multi_class="ovr", average="macro", labels=list(range(k))))
        except Exception:
            pass
    return result


def compute_performance(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
    max_samples: Optional[int] = None,
    horizons: Optional[List[float]] = None,
    calibration_horizon: Optional[float] = None,
) -> str:
    """Compute model performance metrics (MSE, RMSE, MAE, accuracy, etc.).

    For classification tasks, also computes accuracy, AUC-ROC, precision,
    recall, and F1.

    For survival datasets (``task_type: "survival"``, y = [N, 2] time/event)
    reports survival metrics instead: Harrell's and Uno's C on the log-risk,
    time-dependent AUC and IPCW Brier at ``horizons``, integrated Brier score,
    Cox loss vs its null, and calibration at ``calibration_horizon``
    (predicted vs observed risk by decile). The Breslow baseline is fitted on
    the split's training rows and the whole requested split is evaluated
    (``max_samples`` is ignored, since subsampling changes the risk sets).

    Args:
        genome_id: UUID of the genome.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional, for annotation-level perf).
        split: Which split: "train" (default), "test", "val", or "both".
        max_samples: Optional cap on the rows (default None: the whole split).
        horizons: Survival only: evaluation times (dataset time units).
        calibration_horizon: Survival only: horizon for the calibration table.
    """
    from explaneat.core.structure_network import StructureNetwork
    import torch

    db = get_db()
    with db.session_scope() as session:
        meta = load_split_metadata(session, dataset_split_id)
        if meta.get("task_type") == "survival":
            from explaneat.survival.evidence import (
                structure_log_risk_fn, survival_performance,
            )
            X_tr, y_tr, *_ = load_split_data(session, dataset_split_id, "train", 1.0, 10**9)
            X_ev, y_ev, *_ = load_split_data(session, dataset_split_id, split, 1.0, 10**9)
            model_state = build_model_state(session, genome_id)
            result = survival_performance(
                structure_log_risk_fn(model_state), X_tr, y_tr, X_ev, y_ev,
                meta=meta, horizons=horizons, calibration_horizon=calibration_horizon,
            )
            result["split_name"] = split
            from explaneat.analysis.provenance import split_stamp
            result = stamp_and_record(genome_id, "compute_performance",
                                      {"dataset_split_id": dataset_split_id, "split": split, "horizons": horizons,
                                       "calibration_horizon": calibration_horizon},
                                      {"quantity": "survival performance", "scope": "whole model",
                                       "population": f"{split} split", "unit": "metric"},
                                      split_stamp(split, len(y_ev), len(y_ev), False), result)
            return json.dumps(result, indent=2, default=_json_default)

        X, y, feature_names, class_names, num_classes, stamp = load_split_data_stamped(
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
            struct_net.override_output_activation(struct_net.model_output_activation)

        x_tensor = torch.as_tensor(X, dtype=torch.float64)
        predictions = struct_net.forward(x_tensor).detach().numpy()

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        if predictions.ndim == 2 and predictions.shape[1] > 1:
            # Multi-output (k-class logits, softmax-trained): the regression
            # metrics below assume one output per row, so report the
            # classification metrics only.
            mres = stamp_and_record(
                genome_id, "compute_performance", {"dataset_split_id": dataset_split_id, "split": split,
                                                   "max_samples": max_samples},
                {"quantity": "classification performance", "scope": "whole model", "population": f"{split} split",
                 "unit": "metric"}, stamp, _multiclass_performance(predictions, y, struct_net.model_output_activation))
            return json.dumps(mres, indent=2, default=_json_default)

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

    result = stamp_and_record(
        genome_id, "compute_performance", {"dataset_split_id": dataset_split_id, "split": split,
                                           "max_samples": max_samples, "annotation_id": annotation_id},
        {"quantity": "model performance (mse, accuracy, AUC)", "scope": "whole model", "population": f"{split} split",
         "unit": "metric"}, stamp, result)
    return json.dumps(result, indent=2, default=_json_default)


def get_input_distribution(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
    max_samples: Optional[int] = None,
) -> str:
    """Compute input feature distribution statistics.

    Returns per-feature summary stats (mean, std, min, max, quartiles) and
    a correlation matrix between features.

    Args:
        genome_id: UUID of the genome.
        dataset_split_id: UUID of the dataset split.
        annotation_id: Annotation ID (optional, to use entry activations).
        split: Which split: "train" (default), "test", "val", or "both".
        max_samples: Optional cap on the rows (default None: the whole split).
    """
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.analysis.annotation_function import AnnotationFunction

    db = get_db()
    with db.session_scope() as session:
        X, y, feature_names, class_names, num_classes, stamp = load_split_data_stamped(
            session, dataset_split_id, split, 1.0, max_samples,
        )

        if annotation_id:
            model_state = build_model_state(session, genome_id)
            annotation = resolve_annotation_for_evaluation(
                session, genome_id, annotation_id, model_state,
            )
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
    result = stamp_and_record(
        genome_id, "get_input_distribution", {"dataset_split_id": dataset_split_id, "annotation_id": annotation_id,
                                              "split": split, "max_samples": max_samples},
        {"quantity": "per-input distribution (standardised values)", "scope": _scope(annotation_id),
         "population": f"{split} split", "unit": "standardised (z) input values"}, stamp, result)
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
