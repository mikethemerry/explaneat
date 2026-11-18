"""
Genome-related API routes.
"""

import tempfile
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import neat

from ..dependencies import get_db
from ..schemas import (
    GenomeListItem,
    GenomeListResponse,
    GenomeDetail,
    ModelStateResponse,
    NodeSchema,
    ConnectionSchema,
    ModelMetadata,
    ExplanationResponse,
    ExplanationUpdateRequest,
    OperationResponse,
    AnnotationSummary,
    AnnotationListResponse,
)
from ...db import Genome, Population, Experiment, Explanation
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.genome_network import NetworkStructure


router = APIRouter()


def _get_neat_config(experiment: Experiment) -> neat.Config:
    """Load NEAT config from experiment's config text and JSON."""
    from ...core.config_utils import load_neat_config

    return load_neat_config(
        experiment.neat_config_text or "",
        experiment.config_json,
    )


def _network_to_response(network: NetworkStructure) -> ModelStateResponse:
    """Convert NetworkStructure to API response."""
    nodes = [
        NodeSchema(
            id=node.id,
            type=node.type.value if hasattr(node.type, "value") else node.type,
            bias=node.bias,
            activation=node.activation,
            response=node.response,
            aggregation=node.aggregation,
            display_name=getattr(node, "display_name", None),
        )
        for node in network.nodes
    ]

    connections = [
        ConnectionSchema(
            **{
                "from": conn.from_node,
                "to": conn.to_node,
                "weight": conn.weight,
                "enabled": conn.enabled,
            }
        )
        for conn in network.connections
    ]

    metadata = ModelMetadata(
        input_nodes=network.input_node_ids,
        output_nodes=network.output_node_ids,
        is_original=network.metadata.get("is_original", True),
    )

    return ModelStateResponse(
        nodes=nodes,
        connections=connections,
        metadata=metadata,
    )


@router.get("", response_model=GenomeListResponse)
async def list_genomes(
    db: Session = Depends(get_db),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    population_id: Optional[UUID] = None,
    min_fitness: Optional[float] = None,
):
    """
    List genomes with optional filtering.

    - **limit**: Maximum number of genomes to return (default 50, max 200)
    - **offset**: Number of genomes to skip
    - **population_id**: Filter by population
    - **min_fitness**: Filter by minimum fitness
    """
    query = db.query(Genome)

    if population_id:
        query = query.filter(Genome.population_id == population_id)
    if min_fitness is not None:
        query = query.filter(Genome.fitness >= min_fitness)

    total = query.count()
    genomes = query.order_by(Genome.fitness.desc().nullslast()).offset(offset).limit(limit).all()

    return GenomeListResponse(
        genomes=[
            GenomeListItem(
                id=str(g.id),
                genome_id=g.genome_id,
                fitness=g.fitness,
                num_nodes=g.num_nodes,
                num_connections=g.num_connections,
                population_id=str(g.population_id),
                created_at=g.created_at,
            )
            for g in genomes
        ],
        total=total,
    )


@router.get("/{genome_id}", response_model=GenomeDetail)
async def get_genome(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """Get detailed information about a specific genome."""
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    return GenomeDetail(
        id=str(genome.id),
        genome_id=genome.genome_id,
        fitness=genome.fitness,
        num_nodes=genome.num_nodes,
        num_connections=genome.num_connections,
        population_id=str(genome.population_id),
        created_at=genome.created_at,
        genome_data=genome.genome_data,
        network_depth=genome.network_depth,
        network_width=genome.network_width,
    )


@router.get("/{genome_id}/phenotype", response_model=ModelStateResponse)
async def get_genome_phenotype(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the phenotype (pruned network) for a genome.

    The phenotype contains only nodes and connections that are on
    at least one path from an input to an output.
    """
    genome_record = db.get(Genome, genome_id)
    if not genome_record:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # Get the population to access experiment config
    population = db.get(Population, genome_record.population_id)
    if not population:
        raise HTTPException(status_code=404, detail="Population not found")

    experiment = db.get(Experiment, population.experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Load NEAT config
    config = _get_neat_config(experiment)

    # Deserialize genome and create ExplaNEAT instance
    neat_genome = deserialize_genome(genome_record.genome_data, config)
    explaneat = ExplaNEAT(neat_genome, config)

    # Get phenotype network
    phenotype = explaneat.get_phenotype_network()
    phenotype.metadata["is_original"] = True

    return _network_to_response(phenotype)


def _operations_to_response(operations_data: list) -> list:
    """Convert operations JSON to response format."""
    from datetime import datetime

    result = []
    for op in operations_data or []:
        created_at = op.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not created_at:
            created_at = datetime.utcnow()

        result.append(OperationResponse(
            seq=op["seq"],
            type=op["type"],
            params=op["params"],
            result=op.get("result"),
            created_at=created_at,
        ))
    return result


@router.get("/{genome_id}/explanation", response_model=ExplanationResponse)
async def get_genome_explanation(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the explanation for a genome (404 if it has none yet).

    A GET never creates a row: the explanation is created by the first write
    (adding an operation, importing operations, PUT on this route).
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    if not explanation:
        raise HTTPException(status_code=404, detail="This genome has no explanation yet")

    # Load operations from the operations JSON column
    operations = _operations_to_response(explanation.operations)

    return ExplanationResponse(
        id=str(explanation.id),
        genome_id=str(explanation.genome_id),
        name=explanation.name,
        description=explanation.description,
        operations=operations,
        is_well_formed=explanation.is_well_formed,
        structural_coverage=explanation.structural_coverage,
        compositional_coverage=explanation.compositional_coverage,
        created_at=explanation.created_at,
        updated_at=explanation.updated_at,
    )


@router.put("/{genome_id}/explanation", response_model=ExplanationResponse)
async def update_genome_explanation(
    genome_id: UUID,
    update: ExplanationUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update explanation metadata (name, description)."""
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    from ...db.explanation_lock import lock_explanation

    explanation = lock_explanation(db, genome_id, create=True)

    if update.name is not None:
        explanation.name = update.name
    if update.description is not None:
        explanation.description = update.description

    db.commit()
    db.refresh(explanation)

    operations = _operations_to_response(explanation.operations)

    return ExplanationResponse(
        id=str(explanation.id),
        genome_id=str(explanation.genome_id),
        name=explanation.name,
        description=explanation.description,
        operations=operations,
        is_well_formed=explanation.is_well_formed,
        structural_coverage=explanation.structural_coverage,
        compositional_coverage=explanation.compositional_coverage,
        created_at=explanation.created_at,
        updated_at=explanation.updated_at,
    )


@router.delete("/{genome_id}/explanation")
async def delete_genome_explanation(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Delete the explanation for a genome.

    This removes all operations and resets to the original model.
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    if explanation:
        db.delete(explanation)
        db.commit()

    return {"status": "deleted"}


@router.get("/{genome_id}/annotations", response_model=AnnotationListResponse)
async def list_genome_annotations(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """
    List all annotations for a genome with hierarchy information.

    Annotations are derived from "annotate" operations in the explanation's
    operations array. Returns annotations with hierarchy info useful for
    collapsing/expanding in the UI.
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # Get explanation for this genome
    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    if not explanation or not explanation.operations:
        return AnnotationListResponse(annotations=[], total=0)

    # Extract annotations from "annotate" operations
    # First pass: collect all annotations with their child_annotation_ids
    raw_annotations = []
    for op in explanation.operations:
        if op.get("type") != "annotate":
            continue

        params = op.get("params", {})
        result = op.get("result", {})

        # Use annotation_id from result, or generate one from seq
        ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"

        raw_annotations.append({
            "id": ann_id,
            "name": params.get("name"),
            "entry_nodes": [str(n) for n in (params.get("entry_nodes") or [])],
            "exit_nodes": [str(n) for n in (params.get("exit_nodes") or [])],
            "subgraph_nodes": [str(n) for n in (params.get("subgraph_nodes") or [])],
            "child_annotation_ids": params.get("child_annotation_ids") or [],
        })

    # Build name -> [ids] mapping (handles duplicate names correctly)
    name_to_ids: dict[str, list[str]] = {}
    for ann in raw_annotations:
        if ann["name"]:
            name_to_ids.setdefault(ann["name"], []).append(ann["id"])

    # Resolve children_ids for each annotation (all matching names)
    for ann in raw_annotations:
        children_ids = []
        for child_name in ann["child_annotation_ids"]:
            children_ids.extend(name_to_ids.get(child_name, []))
        ann["_children_ids"] = children_ids

    # Derive parent_annotation_id by inverting children (single source of truth)
    child_to_parent_id: dict[str, str] = {}
    for ann in raw_annotations:
        for child_id in ann["_children_ids"]:
            if child_id in child_to_parent_id:
                import logging
                logging.getLogger(__name__).warning(
                    f"Annotation '{child_id}' claimed by both "
                    f"'{child_to_parent_id[child_id]}' and '{ann['id']}'"
                )
            else:
                child_to_parent_id[child_id] = ann["id"]

    # Scan for rename_annotation operations to get display names
    annotation_display_names: dict[str, str | None] = {}
    for op in explanation.operations:
        if op.get("type") == "rename_annotation":
            params = op.get("params", {})
            ann_id = params.get("annotation_id")
            if ann_id:
                annotation_display_names[ann_id] = params.get("display_name")

    # Build final annotation summaries
    annotation_summaries = []
    for ann in raw_annotations:
        parent_id = child_to_parent_id.get(ann["id"])
        children_ids = ann["_children_ids"]

        annotation_summaries.append(
            AnnotationSummary(
                id=ann["id"],
                name=ann["name"],
                display_name=annotation_display_names.get(ann["name"]),
                entry_nodes=ann["entry_nodes"],
                exit_nodes=ann["exit_nodes"],
                subgraph_nodes=ann["subgraph_nodes"],
                parent_annotation_id=parent_id,
                children_ids=children_ids,
                is_leaf=len(children_ids) == 0,
            )
        )

    return AnnotationListResponse(
        annotations=annotation_summaries,
        total=len(annotation_summaries),
    )


# =============================================================================
# Feature labels (readable names of the input nodes, from the dataset codebook)
# =============================================================================

_FEATURE_CB_CACHE: dict = {}


def _dataset_codebook(db: Session, experiment: Experiment):
    """(dataset, feature names, codebook) for an experiment; codebook checks run once per
    dataset/split on the training rows, as the plain-language layer does. Read-only."""
    from ...db.models import Dataset, DatasetSplit
    from ...analysis.feature_labels import codebook_for

    if not experiment.dataset_id:
        return None, [], None
    key = (str(experiment.dataset_id), str(experiment.split_id))
    if key in _FEATURE_CB_CACHE:
        return _FEATURE_CB_CACHE[key]
    ds = db.get(Dataset, experiment.dataset_id)
    if ds is None:
        return None, [], None
    names = [str(n) for n in (ds.feature_names or [])]
    raw = y = None
    try:
        import numpy as np
        X, Y = ds.get_data()
        rows = None
        if experiment.split_id:
            split = db.get(DatasetSplit, experiment.split_id)
            if split is not None and split.train_indices is not None:
                rows = np.asarray(split.train_indices, dtype=int)
        raw = np.asarray(X if rows is None else X[rows], dtype=float)
        y = np.asarray(Y if rows is None else Y[rows], dtype=float)
    except Exception:  # no data: decode without the checks
        raw = y = None
    cb = codebook_for(ds.name, names, raw, y)
    _FEATURE_CB_CACHE[key] = (ds, names, cb)
    return ds, names, cb


@router.get("/{genome_id}/feature-labels")
async def get_feature_labels(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """Readable labels of the genome's input nodes (the "feature key").

    One entry per dataset column: ``node_id`` (``-k`` is column ``k-1``), ``raw`` (the stored
    feature name), ``label`` (decoded meaning from the dataset codebook, the raw name when there
    is none), ``variable_label``, ``full_label``, ``one_means`` (what value 1 means for one-hot /
    binary columns), ``unit`` and ``used`` (the phenotype reads it). Read-only.
    """
    from ...analysis.feature_labels import feature_key

    genome_record = db.get(Genome, genome_id)
    if not genome_record:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")
    population = db.get(Population, genome_record.population_id)
    experiment = db.get(Experiment, population.experiment_id) if population else None
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    ds, names, cb = _dataset_codebook(db, experiment)

    used_cols = None
    try:
        config = _get_neat_config(experiment)
        phen = ExplaNEAT(deserialize_genome(genome_record.genome_data, config), config).get_phenotype_network()
        srcs = {str(c.from_node) for c in phen.connections if c.enabled}
        used_cols = {-int(n) - 1 for n in srcs if n.lstrip("-").isdigit() and int(n) < 0}
        if not names:
            n_in = len(phen.input_node_ids)
            names = [str(i) for i in range(n_in)]
    except Exception:
        pass
    features = feature_key(cb, names, used_cols)
    return {
        "genome_id": str(genome_id),
        "dataset": {"id": str(ds.id), "name": ds.name} if ds is not None else None,
        "codebook": {"id": (cb or {}).get("id"), "source": (cb or {}).get("source"),
                     "gaps": list((cb or {}).get("gaps") or [])},
        "features": features,
    }
