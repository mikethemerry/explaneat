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
    SnapshotRequest,
    NarrativeUpdateRequest,
    EvidenceEntry,
    EvidenceListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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


def _load_split_data(session, split_id: str, split_choice: str, sample_frac: float, max_samples: int):
    """Load X, y data from a dataset split."""
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
    else:  # both
        indices = (split.train_indices or []) + (split.test_indices or [])

    if not indices:
        raise HTTPException(status_code=400, detail="No indices for requested split")

    X = X_full[indices]
    y = y_full[indices]

    # Sample for visualization
    X, y = sample_dataset(X, y, fraction=sample_frac, max_samples=max_samples)

    return X, y


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


@router.post("/viz-data", response_model=VizDataResponse)
async def compute_viz_data(
    request: VizDataRequest,
    genome_id: str = Path(...),
):
    """Compute visualization data for an annotation."""
    with db.session_scope() as session:
        model_state = _build_model_state(session, genome_id)
        annotation = _find_annotation_in_operations(session, genome_id, request.annotation_id)
        X, y = _load_split_data(
            session, request.dataset_split_id, request.split,
            request.sample_fraction, request.max_samples,
        )

        # Check cache
        cached = activation_cache.get(genome_id, request.dataset_split_id, request.annotation_id)
        if cached is not None:
            entry_acts, exit_acts = cached
        else:
            extractor = ActivationExtractor.from_structure(model_state)
            entry_acts, exit_acts = extractor.extract(X, annotation)
            activation_cache.put(
                genome_id, request.dataset_split_id, request.annotation_id,
                entry_acts, exit_acts,
            )

        ann_fn = AnnotationFunction.from_structure(annotation, model_state)
        n_in, n_out = ann_fn.dimensionality
        suggested = vd.suggest_viz_types(n_in, n_out)

        params = request.params or {}
        viz_type = request.viz_type

        if viz_type == "line":
            data = vd.compute_line_plot(
                ann_fn, entry_acts, exit_acts,
                input_dim=params.get("input_dim", 0),
                output_dim=params.get("output_dim", 0),
            )
        elif viz_type == "heatmap":
            data = vd.compute_heatmap(
                ann_fn, entry_acts, exit_acts,
                input_dims=tuple(params.get("input_dims", [0, 1])),
                output_dim=params.get("output_dim", 0),
            )
        elif viz_type == "partial_dependence":
            data = vd.compute_partial_dependence(
                ann_fn, entry_acts,
                vary_dims=params.get("vary_dims", [0]),
                output_dim=params.get("output_dim", 0),
            )
        elif viz_type == "pca_scatter":
            data = vd.compute_pca_scatter(
                entry_acts, exit_acts,
                output_dim=params.get("output_dim", 0),
            )
        elif viz_type == "sensitivity":
            data = vd.compute_sensitivity(
                ann_fn, entry_acts,
                perturbation=params.get("perturbation", 0.01),
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown viz_type: {viz_type}")

        return VizDataResponse(
            viz_type=viz_type,
            data=data,
            dimensionality=[n_in, n_out],
            suggested_viz_types=suggested,
        )


@router.get("/formula", response_model=FormulaResponse)
async def get_formula(
    annotation_id: str,
    genome_id: str = Path(...),
):
    """Get closed-form formula for an annotation subgraph.

    For composed annotations (with children), returns both collapsed and
    expanded LaTeX representations.
    """
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        model_state = engine.current_state
        annotation = _find_annotation_in_operations(session, genome_id, annotation_id)

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
            from ...core.collapse_transform import collapse_structure, _compute_effective_entries_exits
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
            effective_entries, effective_exits = _compute_effective_entries_exits(
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
