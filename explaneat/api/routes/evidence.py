"""Evidence API routes for annotation visualization and evidence management."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Path

from ...db import db
from ...db.models import Dataset, DatasetSplit, Genome, Explanation
from ...db.dataset_utils import sample_dataset
from ...core.config_utils import load_neat_config
from ...core.genome_network import NetworkStructure
from ...core.model_state import ModelStateEngine
from ...analysis.annotation_function import AnnotationFunction
from ...analysis.activation_extractor import ActivationExtractor
from ...analysis.activation_cache import activation_cache
from ...analysis import viz_data as vd
from ..schemas import (
    VizDataRequest,
    VizDataResponse,
    FormulaResponse,
    ChildFormulaInfo,
    NodeEvidenceInfoResponse,
    SnapshotRequest,
    NarrativeUpdateRequest,
    EvidenceEntry,
    EvidenceListResponse,
    InputDistributionRequest,
    InputDistributionResponse,
    ShapRequest,
    ShapResponse,
    PerformanceRequest,
    PerformanceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_node_label(
    node_id: str,
    display_map: Dict[str, str],
    model_state: "NetworkStructure",
    fallback: Optional[str] = None,
) -> str:
    """Resolve a node ID to a human-readable label.

    Handles the case where a node was split away (e.g. "-7" was split into
    "-7_a", "-7_b") and no longer exists in the model.  In that case, finds
    the split variants' display names and joins them.
    """
    label = display_map.get(node_id)
    if label and label != node_id:
        return label

    # Node not in display map (probably split away) — look for variants
    prefix = f"{node_id}_"
    variant_names = []
    for node in model_state.nodes:
        if node.id.startswith(prefix):
            suffix = node.id[len(prefix):]
            if suffix.isalpha() and len(suffix) == 1:
                variant_names.append(display_map.get(node.id, node.id))

    if variant_names:
        # Derive the base concept from a named variant (strip the _a/_b suffix)
        for vn in variant_names:
            if vn != node_id and "_" in vn:
                base_name = vn.rsplit("_", 1)[0]
                return base_name
        # All variants are still numeric — join them
        return "/".join(variant_names)

    return fallback or node_id


def _load_genome_and_config(session, genome_id: str):
    """Load a genome DB record, NEAT genome, and config."""
    genome_db = session.query(Genome).filter_by(id=uuid.UUID(genome_id)).first()
    if not genome_db:
        raise HTTPException(status_code=404, detail="Genome not found")

    experiment = genome_db.population.experiment
    config = load_neat_config(
        experiment.neat_config_text or "",
        experiment.config_json,
    )
    neat_genome = genome_db.to_neat_genome(config)
    return genome_db, neat_genome, config


def _build_engine(session, genome_id: str) -> ModelStateEngine:
    """Build a ModelStateEngine for a genome with all operations replayed.

    Returns the engine (with .current_state and .annotations available).
    """
    from ...core.explaneat import ExplaNEAT

    genome_db, neat_genome, config = _load_genome_and_config(session, genome_id)
    explainer = ExplaNEAT(neat_genome, config)
    phenotype = explainer.get_phenotype_network()

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == uuid.UUID(genome_id))
        .first()
    )

    engine = ModelStateEngine(phenotype)
    if explanation and explanation.operations:
        engine.load_operations({"operations": explanation.operations})
    return engine


def _build_model_state(session, genome_id: str) -> NetworkStructure:
    """Build the current annotated model by replaying operations on the phenotype."""
    return _build_engine(session, genome_id).current_state


def _find_annotation_in_operations(session, genome_id: str, annotation_id: str) -> Dict[str, Any]:
    """Find an annotation from the explanation's operations array.

    Annotations are derived from 'annotate' operations and have synthetic IDs
    like 'ann_37' (from the operation sequence number), not database UUIDs.
    """
    genome_db = session.query(Genome).filter_by(id=uuid.UUID(genome_id)).first()
    if not genome_db:
        raise HTTPException(status_code=404, detail="Genome not found")

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == uuid.UUID(genome_id))
        .first()
    )
    if not explanation or not explanation.operations:
        raise HTTPException(status_code=404, detail="No explanation/operations found")

    for op in explanation.operations:
        if op.get("type") != "annotate":
            continue
        params = op.get("params", {})
        result = op.get("result", {})
        ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"

        if ann_id == annotation_id:
            return {
                "id": ann_id,
                "entry_nodes": params.get("entry_nodes", []),
                "exit_nodes": params.get("exit_nodes", []),
                "subgraph_nodes": params.get("subgraph_nodes", []),
                "subgraph_connections": params.get("subgraph_connections", []),
                "name": params.get("name"),
                "hypothesis": params.get("hypothesis"),
                "evidence": params.get("evidence") or {},
                "child_annotation_ids": params.get("child_annotation_ids", []),
                "parent_annotation_id": params.get("parent_annotation_id"),
            }

    raise HTTPException(status_code=404, detail=f"Annotation '{annotation_id}' not found")


def _unscale_values(values, feature_idx: int, scaler_params: dict):
    """Reverse StandardScaler: original = scaled * scale + mean."""
    if not scaler_params:
        return values
    mean = scaler_params["mean"][feature_idx]
    scale = scaler_params["scale"][feature_idx]
    if isinstance(values, (list, np.ndarray)):
        return [v * scale + mean for v in values]
    return values * scale + mean


def _group_onehot_features(feature_names: List[str]) -> Dict[str, List[int]]:
    """Map base feature name -> list of column indices for one-hot groups."""
    groups: Dict[str, List[int]] = {}
    for i, name in enumerate(feature_names):
        base = name.split(":")[0] if ":" in name else name
        groups.setdefault(base, []).append(i)
    return groups


def _get_source_feature_names(feature_names: List[str]) -> List[str]:
    """Collapse one-hot feature names back to original names."""
    seen = []
    for name in feature_names:
        base = name.split(":")[0] if ":" in name else name
        if base not in seen:
            seen.append(base)
    return seen


def _apply_source_view(viz_result: dict, session, split_id: str, entry_names: List[str]) -> dict:
    """Post-process viz data to show source (unscaled, ungrouped) values."""
    import copy
    result = copy.deepcopy(viz_result)

    # Get scaler params from split
    split = session.query(DatasetSplit).filter_by(id=uuid.UUID(split_id)).first()
    scaler_params = split.scaler_params if split else None

    # Get encoding config from dataset
    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first() if split else None
    has_encoding = dataset and dataset.encoding_config is not None

    # Replace entry_names with source names
    if has_encoding:
        source_names = _get_source_feature_names(entry_names)
        result["entry_names"] = source_names

    # Unscale axis values in the data if scaler was used
    if scaler_params and "data" in result:
        data = result["data"]
        # Line plots, PD plots, ICE plots, feature_output_scatter have x values
        # that correspond to a specific input dimension
        if isinstance(data, dict) and "x" in data and isinstance(data["x"], list):
            params = result.get("params", {})
            dim_idx = params.get("input_dim", 0)
            if dim_idx < len(scaler_params.get("mean", [])):
                data["x"] = _unscale_values(data["x"], dim_idx, scaler_params)

    return result


def _load_split_data(session, split_id: str, split_choice: str, sample_frac: float, max_samples: int):
    """Load X, y data and dataset metadata from a dataset split.

    Applies the scaler stored on the split (if any) so that evaluation
    uses the same preprocessing as training.

    Returns (X, y, feature_names, class_names, num_classes).
    """
    split = session.query(DatasetSplit).filter_by(id=uuid.UUID(split_id)).first()
    if not split:
        raise HTTPException(status_code=404, detail="Dataset split not found")

    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = dataset.get_data()
    if data is None:
        raise HTTPException(status_code=400, detail="Dataset has no stored data")

    X_full, y_full = data

    if split_choice == "train":
        indices = split.train_indices
    elif split_choice == "test":
        indices = split.test_indices
    elif split_choice == "val":
        indices = split.validation_indices
        if not indices:
            raise HTTPException(status_code=400, detail="No validation split available")
    else:  # both
        indices = (split.train_indices or []) + (split.test_indices or [])
        if split.validation_indices:
            indices += split.validation_indices

    if not indices:
        raise HTTPException(status_code=400, detail="No indices for requested split")

    X = X_full[indices]
    y = y_full[indices]

    # Apply scaler if the split was trained with one
    if split.scaler_type and split.scaler_params:
        if split.scaler_type == "StandardScaler":
            mean = np.array(split.scaler_params["mean"])
            scale = np.array(split.scaler_params["scale"])
            X = (X - mean) / scale
        elif split.scaler_type == "MinMaxScaler":
            data_min = np.array(split.scaler_params["data_min_"])
            scale = np.array(split.scaler_params["scale_"])
            X = X * scale + np.array(split.scaler_params["min_"])

    # Sample for visualization
    X, y = sample_dataset(X, y, fraction=sample_frac, max_samples=max_samples)

    return X, y, dataset.feature_names, dataset.class_names, dataset.num_classes


def _update_annotation_evidence(session, genome_id: str, annotation_id: str, evidence: Dict[str, Any]):
    """Update evidence on an annotation operation in the explanation's operations array."""
    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == uuid.UUID(genome_id))
        .first()
    )
    if not explanation or not explanation.operations:
        raise HTTPException(status_code=404, detail="No explanation found")

    operations = list(explanation.operations)
    for op in operations:
        if op.get("type") != "annotate":
            continue
        result = op.get("result", {})
        ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
        if ann_id == annotation_id:
            params = op.get("params", {})
            params["evidence"] = evidence
            op["params"] = params
            explanation.operations = operations
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(explanation, "operations")
            session.flush()
            return

    raise HTTPException(status_code=404, detail=f"Annotation '{annotation_id}' not found")


def _compute_node_subgraph(model_state: NetworkStructure, node_id: str) -> Dict[str, Any]:
    """Compute a virtual annotation dict for a single node.

    BFS backward from node_id to find all ancestor input nodes, then
    forward from those inputs to find all nodes on paths to node_id.
    """
    # Build adjacency maps from enabled connections
    predecessors: Dict[str, List[str]] = {}
    successors: Dict[str, List[str]] = {}
    for conn in model_state.connections:
        if not conn.enabled:
            continue
        predecessors.setdefault(conn.to_node, []).append(conn.from_node)
        successors.setdefault(conn.from_node, []).append(conn.to_node)

    # BFS backward from node_id
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

    # Entry nodes = network inputs that are backward-reachable
    input_ids = set(model_state.input_node_ids)
    entry_nodes = sorted(n for n in backward_reachable if n in input_ids)

    if not entry_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Node '{node_id}' has no reachable input nodes",
        )

    # BFS forward from entry_nodes
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

    # Subgraph = nodes on paths from inputs to node_id
    subgraph_nodes = sorted(backward_reachable & forward_reachable)

    # Display name
    display_map = model_state.get_display_map()
    display_name = display_map.get(node_id, node_id)

    return {
        "id": f"__node_{node_id}__",
        "entry_nodes": entry_nodes,
        "exit_nodes": [node_id],
        "subgraph_nodes": subgraph_nodes,
        "subgraph_connections": [],
        "name": f"Node {display_name}",
        "display_name": display_name,
    }


def _resolve_annotation(
    session, genome_id: str, model_state: NetworkStructure,
    annotation_id: Optional[str], node_id: Optional[str],
) -> Dict[str, Any]:
    """Resolve an annotation dict from either annotation_id or node_id."""
    if node_id:
        return _compute_node_subgraph(model_state, node_id)
    elif annotation_id:
        return _find_annotation_in_operations(session, genome_id, annotation_id)
    else:
        raise HTTPException(status_code=400, detail="Provide annotation_id or node_id")


@router.get("/node-info", response_model=NodeEvidenceInfoResponse)
async def get_node_evidence_info(
    node_id: str,
    genome_id: str = Path(...),
):
    """Get entry/exit/subgraph info for a single node (virtual annotation)."""
    with db.session_scope() as session:
        model_state = _build_model_state(session, genome_id)
        info = _compute_node_subgraph(model_state, node_id)
        return NodeEvidenceInfoResponse(
            node_id=node_id,
            entry_nodes=info["entry_nodes"],
            exit_nodes=info["exit_nodes"],
            subgraph_nodes=info["subgraph_nodes"],
            display_name=info["display_name"],
        )


def _build_whole_model_context(
    session, genome_id: str, X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
):
    """Build visualization context for the whole model (no annotation).

    Returns (fn, entry_acts, exit_acts, entry_names, exit_names, n_in, n_out).
    """
    from ...core.structure_network import StructureNetwork
    import torch

    model_state = _build_model_state(session, genome_id)
    struct_net = StructureNetwork(model_state)

    # Binary classification: force sigmoid on output to match NeuralNeat
    # training behaviour (NeuralNeat hardcodes sigmoid on output layer).
    is_binary = (num_classes is not None and num_classes == 2
                 and len(model_state.output_node_ids) == 1)
    if is_binary:
        struct_net.override_output_activation("sigmoid")

    display_map = model_state.get_display_map()

    # Build deduplicated feature names (handle split input nodes)
    seen_bases: set = set()
    entry_names: list = []
    feat_idx = 0
    for nid in model_state.input_node_ids:
        base = StructureNetwork._get_base_node_id(nid) or nid
        if base not in seen_bases:
            seen_bases.add(base)
            label = _resolve_node_label(base, display_map, model_state)
            # If _resolve_node_label returned the raw node ID, try dataset feature name
            if label == base and feature_names and feat_idx < len(feature_names):
                label = feature_names[feat_idx]
            entry_names.append(label)
            feat_idx += 1

    exit_names = [
        _resolve_node_label(n, display_map, model_state, fallback="output")
        for n in model_state.output_node_ids
    ]

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


def _compute_correctness(
    exit_acts: np.ndarray, y: np.ndarray, num_classes: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Compute classification correctness for whole-model predictions.

    Returns dict with correctness/predicted_class/true_class lists,
    or None for regression tasks.
    """
    if num_classes is None or num_classes < 2:
        return None

    n_out = exit_acts.shape[1]
    if n_out == 1:
        # Binary: class 1 if probability > 0.5
        predicted = (exit_acts[:, 0] > 0.5).astype(int)
    else:
        # Multi-class: argmax
        predicted = np.argmax(exit_acts, axis=1).astype(int)

    true_class = y.astype(int).ravel()
    correctness = (predicted == true_class).tolist()

    return {
        "correctness": correctness,
        "predicted_class": predicted.tolist(),
        "true_class": true_class.tolist(),
    }


@router.post("/viz-data", response_model=VizDataResponse)
async def compute_viz_data(
    request: VizDataRequest,
    genome_id: str = Path(...),
):
    """Compute visualization data for an annotation, single node, or whole model."""
    with db.session_scope() as session:
        X, y, ds_feature_names, ds_class_names, ds_num_classes = _load_split_data(
            session, request.dataset_split_id, request.split,
            request.sample_fraction, request.max_samples,
        )

        # Infer num_classes from data if not set in DB
        if ds_num_classes is None:
            y_check = y.ravel()
            if np.all(y_check == y_check.astype(int)):
                n_unique = len(np.unique(y_check.astype(int)))
                if n_unique <= 20:
                    ds_num_classes = n_unique

        is_whole_model = not request.annotation_id and not request.node_id
        correctness_info: Optional[Dict[str, Any]] = None

        if is_whole_model:
            fn, entry_acts, exit_acts, entry_names, exit_names, n_in, n_out = \
                _build_whole_model_context(
                    session, genome_id, X,
                    feature_names=ds_feature_names,
                    num_classes=ds_num_classes,
                )
            correctness_info = _compute_correctness(exit_acts, y, ds_num_classes)
        else:
            model_state = _build_model_state(session, genome_id)
            annotation = _resolve_annotation(
                session, genome_id, model_state,
                request.annotation_id, request.node_id,
            )

            cache_key = f"node:{request.node_id}" if request.node_id else request.annotation_id
            cached = activation_cache.get(genome_id, request.dataset_split_id, cache_key)
            if cached is not None:
                entry_acts, exit_acts = cached
            else:
                extractor = ActivationExtractor.from_structure(model_state)
                entry_acts, exit_acts = extractor.extract(X, annotation)
                activation_cache.put(
                    genome_id, request.dataset_split_id, cache_key,
                    entry_acts, exit_acts,
                )

            ann_fn = AnnotationFunction.from_structure(annotation, model_state)
            fn = ann_fn
            n_in, n_out = ann_fn.dimensionality

            display_map = model_state.get_display_map()
            entry_names = [
                _resolve_node_label(n, display_map, model_state)
                for n in annotation["entry_nodes"]
            ]

            # Fall back to dataset feature names for input entry nodes
            if ds_feature_names:
                from ...core.structure_network import StructureNetwork
                input_node_ids = model_state.input_node_ids
                # Map base input node ID -> feature column index
                seen_bases: set = set()
                base_to_feat_idx: dict = {}
                feat_idx = 0
                for nid in input_node_ids:
                    base = StructureNetwork._get_base_node_id(nid) or nid
                    if base not in seen_bases:
                        seen_bases.add(base)
                        base_to_feat_idx[base] = feat_idx
                        feat_idx += 1

                for i, entry_node in enumerate(annotation["entry_nodes"]):
                    if entry_names[i] == entry_node:
                        # Label is still raw node ID — try dataset feature name
                        base = StructureNetwork._get_base_node_id(entry_node) or entry_node
                        fidx = base_to_feat_idx.get(base)
                        if fidx is not None and fidx < len(ds_feature_names):
                            entry_names[i] = ds_feature_names[fidx]
            ann_name = annotation.get("name", "output")
            exit_nodes = annotation["exit_nodes"]
            exit_names = [
                _resolve_node_label(n, display_map, model_state,
                                    fallback=ann_name if len(exit_nodes) == 1
                                    else f"{ann_name}_{i}")
                for i, n in enumerate(exit_nodes)
            ]

        suggested = vd.suggest_viz_types(n_in, n_out)
        params = request.params or {}
        viz_type = request.viz_type

        if viz_type == "line":
            data = vd.compute_line_plot(
                fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", 0),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "heatmap":
            data = vd.compute_heatmap(
                fn, entry_acts, exit_acts,
                input_dims=tuple(params.get("input_dims", [0, 1])),
                output_dim=params.get("output_dim", 0),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "partial_dependence":
            data = vd.compute_partial_dependence(
                fn, entry_acts,
                vary_dims=params.get("vary_dims", [0]),
                output_dim=params.get("output_dim", 0),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "pca_scatter":
            data = vd.compute_pca_scatter(
                entry_acts, exit_acts,
                output_dim=params.get("output_dim", 0),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "sensitivity":
            data = vd.compute_sensitivity(
                fn, entry_acts,
                perturbation=params.get("perturbation", 0.01),
                entry_names=entry_names,
            )
        elif viz_type == "ice":
            data = vd.compute_ice_plot(
                fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", 0),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "feature_output_scatter":
            data = vd.compute_feature_output_scatter(
                fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", 0),
                entry_names=entry_names, exit_names=exit_names,
            )
        elif viz_type == "output_distribution":
            data = vd.compute_output_distribution(
                exit_acts,
                output_dim=params.get("output_dim", 0),
                exit_names=exit_names,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown viz_type: {viz_type}")

        response_kwargs: Dict[str, Any] = {
            "viz_type": viz_type,
            "data": data,
            "dimensionality": [n_in, n_out],
            "suggested_viz_types": suggested,
            "entry_names": entry_names,
            "exit_names": exit_names,
        }
        if correctness_info:
            response_kwargs.update(
                correctness=correctness_info["correctness"],
                predicted_class=correctness_info["predicted_class"],
                true_class=correctness_info["true_class"],
            )
        if ds_class_names:
            response_kwargs["class_names"] = ds_class_names
        if ds_num_classes is not None:
            response_kwargs["num_classes"] = ds_num_classes

        # Source view: reverse scaling on axis values, group one-hot features
        if request.view == "source":
            response_kwargs = _apply_source_view(
                response_kwargs, session, request.dataset_split_id, entry_names,
            )

        return VizDataResponse(**response_kwargs)


@router.get("/formula", response_model=FormulaResponse)
async def get_formula(
    genome_id: str = Path(...),
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
):
    """Get closed-form formula for an annotation subgraph or single node.

    For composed annotations (with children), returns both collapsed and
    expanded LaTeX representations.
    """
    if not annotation_id and not node_id:
        raise HTTPException(status_code=400, detail="Provide annotation_id or node_id")
    if annotation_id and node_id:
        raise HTTPException(status_code=400, detail="Provide only one of annotation_id or node_id")

    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        model_state = engine.current_state
        annotation = _resolve_annotation(
            session, genome_id, model_state, annotation_id, node_id,
        )

        ann_fn = AnnotationFunction.from_structure(annotation, model_state)
        n_in, n_out = ann_fn.dimensionality

        # child_annotation_ids stores child annotation NAMES (e.g. "A1678"),
        # not synthetic IDs (e.g. "ann_37").
        child_ann_names = annotation.get("child_annotation_ids", [])
        is_composed = len(child_ann_names) > 0

        if is_composed:
            children_info = []

            # Look up child formulas by matching annotation NAME
            explanation = (
                session.query(Explanation)
                .filter(Explanation.genome_id == uuid.UUID(genome_id))
                .first()
            )
            for op in (explanation.operations or []):
                if op.get("type") != "annotate":
                    continue
                params = op.get("params", {})
                op_name = params.get("name")
                if op_name not in child_ann_names:
                    continue
                result_data = op.get("result", {})
                child_ann_id = result_data.get("annotation_id") or f"ann_{op.get('seq', 0)}"
                child_ann = _find_annotation_in_operations(session, genome_id, child_ann_id)
                child_af = AnnotationFunction.from_structure(child_ann, model_state)
                child_latex = child_af.to_latex()
                cn_in, cn_out = child_af.dimensionality
                children_info.append(ChildFormulaInfo(
                    name=op_name,
                    latex=child_latex,
                    dimensionality=[cn_in, cn_out],
                ))

            # Build partially-collapsed structure (children collapsed, parent not)
            from ...core.collapse_transform import collapse_structure, compute_effective_entries_exits
            child_names_set = set(child_ann_names)
            partially_collapsed = collapse_structure(
                model_state, engine.annotations, child_names_set
            )

            # Build parent annotation dict for the partially-collapsed structure
            fn_node_ids = [f"fn_{name}" for name in child_ann_names]
            parent_ann = dict(annotation)
            parent_subgraph = (
                set(parent_ann.get("subgraph_nodes", []))
                | set(parent_ann.get("entry_nodes", []))
                | set(parent_ann.get("exit_nodes", []))
                | set(fn_node_ids)
            )
            existing_ids = {n.id for n in partially_collapsed.nodes}
            parent_subgraph = parent_subgraph & existing_ids
            parent_ann["subgraph_nodes"] = list(parent_subgraph)

            # Derive effective entries/exits from the partially-collapsed structure
            effective_entries, effective_exits = compute_effective_entries_exits(
                parent_subgraph, partially_collapsed
            )
            parent_ann["entry_nodes"] = sorted(effective_entries)
            parent_ann["exit_nodes"] = sorted(effective_exits)

            composed_af = AnnotationFunction.from_structure(parent_ann, partially_collapsed)
            latex_collapsed = composed_af.to_latex(expand=False)
            latex_expanded = composed_af.to_latex(expand=True)

            # Fall back to direct computation if composed path fails
            if latex_expanded is None:
                latex_expanded = ann_fn.to_latex()

            return FormulaResponse(
                latex=latex_expanded,
                latex_collapsed=latex_collapsed,
                latex_expanded=latex_expanded,
                tractable=latex_expanded is not None or latex_collapsed is not None,
                dimensionality=[n_in, n_out],
                is_composed=True,
                children=children_info,
            )
        else:
            latex = ann_fn.to_latex()
            return FormulaResponse(
                latex=latex,
                latex_collapsed=None,
                latex_expanded=latex,
                tractable=latex is not None,
                dimensionality=[n_in, n_out],
                is_composed=False,
                children=[],
            )


@router.post("/snapshot")
async def save_snapshot(
    request: SnapshotRequest,
    genome_id: str = Path(...),
):
    """Save a visualization snapshot as evidence on an annotation."""
    with db.session_scope() as session:
        annotation = _find_annotation_in_operations(session, genome_id, request.annotation_id)

        evidence = dict(annotation.get("evidence") or {})
        category = request.category
        if category not in evidence:
            evidence[category] = []

        evidence[category].append({
            "viz_config": request.viz_config,
            "svg_data": request.svg_data,
            "narrative": request.narrative,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        _update_annotation_evidence(session, genome_id, request.annotation_id, evidence)

        return {"status": "ok", "category": category, "count": len(evidence[category])}


@router.put("/narrative")
async def update_narrative(
    request: NarrativeUpdateRequest,
    genome_id: str = Path(...),
):
    """Update narrative text on a specific evidence entry."""
    with db.session_scope() as session:
        annotation = _find_annotation_in_operations(session, genome_id, request.annotation_id)

        evidence = dict(annotation.get("evidence") or {})

        # Find the entry across all categories
        idx = request.evidence_index
        for category in evidence:
            entries = evidence[category]
            if idx < len(entries):
                entries[idx]["narrative"] = request.narrative
                _update_annotation_evidence(session, genome_id, request.annotation_id, evidence)
                return {"status": "ok"}
            idx -= len(entries)

        raise HTTPException(status_code=404, detail="Evidence entry not found")


@router.get("", response_model=EvidenceListResponse)
async def list_evidence(
    annotation_id: str,
    genome_id: str = Path(...),
):
    """Get all evidence for an annotation."""
    with db.session_scope() as session:
        annotation = _find_annotation_in_operations(session, genome_id, annotation_id)

        evidence = annotation.get("evidence") or {}
        entries = []
        for category, items in evidence.items():
            if not isinstance(items, list):
                continue
            for item in items:
                entries.append(
                    EvidenceEntry(
                        viz_config=item.get("viz_config"),
                        svg_data=item.get("svg_data"),
                        narrative=item.get("narrative", ""),
                        category=category,
                        timestamp=item.get("timestamp"),
                    )
                )

        return EvidenceListResponse(
            annotation_id=annotation_id,
            entries=entries,
            total=len(entries),
        )


@router.post("/input-distribution", response_model=InputDistributionResponse)
async def compute_input_distribution(
    request: InputDistributionRequest,
    genome_id: str = Path(...),
):
    """Compute input feature distribution (histogram or scatter)."""
    if len(request.feature_indices) < 1 or len(request.feature_indices) > 2:
        raise HTTPException(
            status_code=400,
            detail="feature_indices must contain 1 or 2 indices",
        )

    with db.session_scope() as session:
        # Load data without sampling (distributions need all points)
        X, y, db_feature_names, _, _ = _load_split_data(
            session, request.dataset_split_id, request.split,
            sample_frac=1.0, max_samples=10000,
        )

        # Build feature name list for requested indices
        names = []
        for idx in request.feature_indices:
            if idx < 0 or idx >= X.shape[1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature index {idx} out of range (0-{X.shape[1]-1})",
                )
            if db_feature_names and idx < len(db_feature_names):
                names.append(db_feature_names[idx])
            else:
                names.append(f"feature_{idx}")

        if len(request.feature_indices) == 1:
            idx = request.feature_indices[0]
            data = vd.compute_histogram(
                X[:, idx], num_bins=request.num_bins, label=names[0],
            )
            viz_type = "histogram"
        else:
            idx0, idx1 = request.feature_indices
            data = vd.compute_scatter_2d(
                X[:, idx0], X[:, idx1],
                x_label=names[0], y_label=names[1],
            )
            viz_type = "scatter2d"

        return InputDistributionResponse(
            viz_type=viz_type,
            data=data,
            feature_names=names,
        )


@router.post("/performance", response_model=PerformanceResponse)
async def compute_performance(
    request: PerformanceRequest,
    genome_id: str = Path(...),
):
    """Compute model performance metrics (MSE, RMSE, MAE, accuracy).

    Optionally compute at a specific operation sequence number for
    before/after comparison.
    """
    from ...core.structure_network import StructureNetwork
    import torch

    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)

        if request.at_seq is not None:
            model_state = engine.get_state_at_seq(request.at_seq)
        else:
            model_state = engine.current_state

        X, y, _, ds_class_names, ds_num_classes = _load_split_data(
            session, request.dataset_split_id, request.split,
            request.sample_fraction, request.max_samples,
        )

        # Infer num_classes from data if not set in DB
        if ds_num_classes is None:
            y_flat_check = y.ravel()
            if np.all(y_flat_check == y_flat_check.astype(int)):
                n_unique = len(np.unique(y_flat_check.astype(int)))
                if n_unique <= 20:  # reasonable upper bound for classification
                    ds_num_classes = n_unique

        struct_net = StructureNetwork(model_state)

        # Binary classification: force sigmoid on output to match NeuralNeat
        # training behaviour (NeuralNeat hardcodes sigmoid on output layer).
        is_binary = (ds_num_classes is not None and ds_num_classes == 2
                     and len(model_state.output_node_ids) == 1)
        if is_binary:
            struct_net.override_output_activation("sigmoid")

        x_tensor = torch.as_tensor(X, dtype=torch.float64)
        predictions = struct_net.forward(x_tensor).detach().numpy()

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        y_flat = y.ravel()
        pred_flat = predictions.ravel() if predictions.ndim > 1 else predictions

        # For multi-output, compute MSE on flattened arrays
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            pred_flat = predictions.ravel()
            y_repeated = np.repeat(y_flat, predictions.shape[1])
            mse = float(np.mean((pred_flat - y_repeated) ** 2))
        else:
            mse = float(np.mean((pred_flat - y_flat) ** 2))

        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(pred_flat - y_flat))) if predictions.ndim == 1 else float(np.mean(np.abs(predictions.ravel() - np.repeat(y_flat, predictions.shape[1] if predictions.ndim > 1 else 1))))

        accuracy = None
        auc_roc = None
        precision = None
        recall = None
        f1 = None
        log_loss_val = None
        brier_score = None
        balanced_acc = None
        calibration = None

        if ds_num_classes is not None and ds_num_classes >= 2:
            y_int = y_flat.astype(int)

            if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
                # Binary classification: StructureNetwork outputs sigmoid probabilities
                predicted_classes = (pred_flat > 0.5).astype(int)
                pred_proba = np.clip(pred_flat, 1e-15, 1 - 1e-15)
            else:
                predicted_classes = np.argmax(predictions, axis=1).astype(int)
                pred_proba = predictions  # multi-class probabilities

            accuracy = float(np.mean(predicted_classes == y_int))

            from sklearn.metrics import (
                roc_auc_score, precision_recall_fscore_support,
                log_loss as sk_log_loss, brier_score_loss,
                balanced_accuracy_score,
            )
            from sklearn.calibration import calibration_curve

            try:
                balanced_acc = float(balanced_accuracy_score(y_int, predicted_classes))
            except Exception:
                pass

            try:
                prec, rec, f1_val, _ = precision_recall_fscore_support(
                    y_int, predicted_classes, average="weighted", zero_division=0,
                )
                precision = float(prec)
                recall = float(rec)
                f1 = float(f1_val)
            except Exception:
                pass

            is_binary = (predictions.ndim == 1 or
                         (predictions.ndim == 2 and predictions.shape[1] == 1))

            try:
                if is_binary:
                    auc_roc = float(roc_auc_score(y_int, pred_proba))
                else:
                    auc_roc = float(roc_auc_score(
                        y_int, pred_proba, multi_class="ovr", average="weighted",
                    ))
            except Exception:
                pass

            try:
                if is_binary:
                    log_loss_val = float(sk_log_loss(y_int, pred_proba))
                else:
                    log_loss_val = float(sk_log_loss(y_int, pred_proba))
            except Exception:
                pass

            try:
                if is_binary:
                    brier_score = float(brier_score_loss(y_int, pred_proba))
                    try:
                        fraction_pos, mean_pred = calibration_curve(
                            y_int, pred_proba, n_bins=10,
                        )
                        calibration = {
                            "bin_means": mean_pred.tolist(),
                            "fraction_positives": fraction_pos.tolist(),
                        }
                    except Exception:
                        pass
            except Exception:
                pass

        return PerformanceResponse(
            mse=mse,
            rmse=rmse,
            mae=mae,
            accuracy=accuracy,
            auc_roc=auc_roc,
            precision=precision,
            recall=recall,
            f1=f1,
            log_loss=log_loss_val,
            brier_score=brier_score,
            balanced_accuracy=balanced_acc,
            calibration=calibration,
            n_samples=len(y_flat),
            at_seq=request.at_seq,
            has_non_identity_ops=engine.has_non_identity_ops,
        )


@router.post("/shap", response_model=ShapResponse)
async def compute_shap(
    request: ShapRequest,
    genome_id: str = Path(...),
):
    """Compute SHAP values for the model or a specific annotation subgraph.

    The heavy SHAP computation runs in a thread pool so it doesn't block
    the async event loop (other requests can be served while SHAP runs).
    """
    import asyncio
    from ...analysis.shap_cache import get_cached_shap, save_shap_cache
    from ...analysis.shap_analysis import compute_shap_values

    # --- Phase 1: DB reads and setup (fast, inside session) ---------------
    with db.session_scope() as session:
        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == uuid.UUID(genome_id))
            .first()
        )
        ops_count = len(explanation.operations or []) if explanation else 0

        # Cache key: use node:X for node-level, annotation_id for annotations
        cache_key = f"node:{request.node_id}" if request.node_id else request.annotation_id

        # Try cache (skip on force_recompute)
        if not request.force_recompute:
            cached = get_cached_shap(
                session, genome_id, request.dataset_split_id,
                cache_key, request.split,
                request.max_samples, ops_count,
            )
            if cached is not None:
                logger.info("SHAP cache hit for genome=%s target=%s", genome_id, cache_key)
                return ShapResponse(**cached)

        logger.info("SHAP %s for genome=%s target=%s — computing",
                     "force recompute" if request.force_recompute else "cache miss",
                     genome_id, cache_key)

        engine = _build_engine(session, genome_id)
        model_state = engine.current_state
        X, y, _, _, ds_num_classes = _load_split_data(
            session, request.dataset_split_id, request.split,
            1.0, request.max_samples,
        )

        display_map = model_state.get_display_map()

        if request.annotation_id or request.node_id:
            annotation = _resolve_annotation(
                session, genome_id, model_state,
                request.annotation_id, request.node_id,
            )
            ann_fn = AnnotationFunction.from_structure(annotation, model_state)
            extractor = ActivationExtractor.from_structure(model_state)
            entry_acts, _ = extractor.extract(X, annotation)
            feature_names = [
                _resolve_node_label(n, display_map, model_state)
                for n in annotation["entry_nodes"]
            ]
            ann_name = annotation.get("name", "output")
            exit_nodes = annotation["exit_nodes"]
            output_names = [
                _resolve_node_label(n, display_map, model_state,
                                    fallback=ann_name if len(exit_nodes) == 1
                                    else f"{ann_name}_{i}")
                for i, n in enumerate(exit_nodes)
            ]
            predict_fn = ann_fn
            shap_input = entry_acts
            shap_output_names: Optional[List[str]] = output_names
        else:
            from ...core.structure_network import StructureNetwork
            struct_net = StructureNetwork(model_state)

            # Binary classification: force sigmoid on output to match
            # NeuralNeat training behaviour.
            is_binary = (ds_num_classes is not None and ds_num_classes == 2
                         and len(model_state.output_node_ids) == 1)
            if is_binary:
                struct_net.override_output_activation("sigmoid")

            seen_bases: set = set()
            feature_names: list = []
            for nid in model_state.input_node_ids:
                base = StructureNetwork._get_base_node_id(nid) or nid
                if base not in seen_bases:
                    seen_bases.add(base)
                    feature_names.append(
                        _resolve_node_label(base, display_map, model_state)
                    )

            def model_predict(x):
                import torch
                x = torch.as_tensor(x, dtype=torch.float64)
                out = struct_net.forward(x).detach().numpy()
                if out.ndim == 2 and out.shape[1] == 1:
                    out = out.ravel()
                return out

            predict_fn = model_predict
            shap_input = X
            shap_output_names = None

    # --- Phase 2: Heavy SHAP computation (in thread pool) -----------------
    result = await asyncio.to_thread(
        compute_shap_values,
        predict_fn, shap_input, feature_names, request.max_samples,
        shap_output_names,
    )

    # --- Phase 3: Cache the result ----------------------------------------
    with db.session_scope() as session:
        save_shap_cache(
            session, genome_id, request.dataset_split_id,
            cache_key, request.split,
            request.max_samples, ops_count, result,
        )

    return ShapResponse(
        feature_names=result["feature_names"],
        mean_abs_shap=result["mean_abs_shap"],
        base_value=result["base_value"],
        outputs=result.get("outputs"),
    )
