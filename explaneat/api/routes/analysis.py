"""
Analysis-related API routes.

These routes handle split detection, node classification, and coverage analysis.
"""

import tempfile
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
import neat

from ..dependencies import get_db
from ..schemas import (
    SplitDetectionRequest,
    SplitDetectionResponse,
    ClassifyNodesRequest,
    ClassifyNodesResponse,
    CoverageResponse,
    ViolationDetail,
    NodeClassification,
    OperationRequest,
    SplitNodeParams,
)
from ...db import Genome, Population, Experiment, Explanation
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.model_state import ModelStateEngine
from ...analysis.split_detection import (
    analyze_coverage_for_splits,
    detect_required_splits,
)
from ...analysis.node_classification import (
    classify_coverage,
    auto_detect_entry_exit,
)


router = APIRouter()


def _get_neat_config(experiment: Experiment) -> neat.Config:
    """Load NEAT config from experiment's config text and JSON."""
    from ...core.config_utils import load_neat_config

    return load_neat_config(
        experiment.neat_config_text or "",
        experiment.config_json,
    )


def _phenotype_for(genome, db: Session):
    population = db.get(Population, genome.population_id)
    experiment = db.get(Experiment, population.experiment_id)
    config = _get_neat_config(experiment)
    return ExplaNEAT(deserialize_genome(genome.genome_data, config), config).get_phenotype_network()


def _get_current_model(genome_id: UUID, db: Session):
    """
    Get the current model state (phenotype with operations applied).

    Returns:
        Tuple of (genome, explanation, model_state)
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # Get population and experiment for config
    population = db.get(Population, genome.population_id)
    if not population:
        raise HTTPException(status_code=404, detail="Population not found")

    experiment = db.get(Experiment, population.experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Load NEAT config and get phenotype
    config = _get_neat_config(experiment)
    neat_genome = deserialize_genome(genome.genome_data, config)
    explaneat = ExplaNEAT(neat_genome, config)
    phenotype = explaneat.get_phenotype_network()

    # Get explanation if exists
    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    # Create engine and apply operations
    if explanation and explanation.operations:
        engine = ModelStateEngine.from_phenotype_and_operations(
            phenotype,
            {"operations": explanation.operations},
        )
        model_state = engine.current_state
    else:
        model_state = phenotype

    return genome, explanation, model_state


@router.post("/split-detection", response_model=SplitDetectionResponse)
async def detect_required_splits_endpoint(
    request: SplitDetectionRequest,
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Detect nodes that must be split before creating an annotation.

    Given a proposed coverage (set of nodes to annotate), this endpoint
    identifies any nodes that violate the entry/exit constraints and
    would need to be split first.

    A node requires splitting if it has BOTH:
    - External input (from outside proposed coverage)
    - External output (to outside proposed coverage)

    Returns violations and suggested split operations to resolve them.
    """
    genome, explanation, model_state = _get_current_model(genome_id, db)

    # Analyze proposed coverage
    result = analyze_coverage_for_splits(model_state, request.proposed_coverage)

    # Convert to response format
    violations = [
        ViolationDetail(
            node_id=v.node_id,
            reason=v.reason,
            external_inputs=list(v.external_inputs),
            external_outputs=list(v.external_outputs),
            internal_outputs=list(v.internal_outputs) if v.internal_outputs else None,
        )
        for v in result.violations
    ]

    suggested_operations = [
        OperationRequest(
            type=op["type"],
            params=SplitNodeParams(node_id=op["params"]["node_id"]),
        )
        for op in result.suggested_operations
    ]

    return SplitDetectionResponse(
        proposed_coverage=result.proposed_coverage,
        violations=violations,
        suggested_operations=suggested_operations,
        adjusted_coverage=result.adjusted_coverage,
    )


@router.post("/classify-nodes", response_model=ClassifyNodesResponse)
async def classify_nodes_endpoint(
    request: ClassifyNodesRequest,
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Classify nodes within a proposed coverage as entry, intermediate, or exit.

    - **Entry nodes**: Have external inputs, no external outputs
    - **Intermediate nodes**: No external inputs or outputs
    - **Exit nodes**: Have external outputs (may have internal inputs only)

    Also validates that the classification is valid (no entry nodes have
    external outputs, no intermediate nodes have external I/O).
    """
    genome, explanation, model_state = _get_current_model(genome_id, db)

    # Classify nodes
    result = classify_coverage(model_state, request.coverage)

    # Convert violations to response format
    violations = [
        ViolationDetail(
            node_id=v["node_id"],
            reason=v["reason"],
            external_inputs=v["external_inputs"],
            external_outputs=v["external_outputs"],
        )
        for v in result.violations
    ]

    classification = NodeClassification(
        entry=result.entry_nodes,
        intermediate=result.intermediate_nodes,
        exit=result.exit_nodes,
    )

    return ClassifyNodesResponse(
        coverage=result.coverage,
        classification=classification,
        valid=result.valid,
        violations=violations,
    )


@router.get("/coverage", response_model=CoverageResponse)
async def get_coverage_analysis(
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Get coverage analysis for the current explanation.

    Returns structural and compositional coverage metrics as defined
    in the explanation model paper:

    - **Structural coverage**: Fraction of nodes covered by annotations
    - **Compositional coverage**: Fraction of compositions explained

    Also returns lists of covered and uncovered nodes.
    """
    from ...analysis.coverage_report import coverage_report
    from ...core.model_state import ModelStateEngine

    genome, explanation, model_state = _get_current_model(genome_id, db)
    if explanation and explanation.operations:
        # Rebuild the engine to get the AnnotationData list (same as MCP get_coverage).
        engine = ModelStateEngine(_phenotype_for(genome, db))
        engine.load_operations({"operations": explanation.operations})
        model_state, annotations = engine.current_state, engine.annotations
    else:
        annotations = []
    rep = coverage_report(model_state, annotations)
    return CoverageResponse(
        structural_coverage=rep["structural_coverage"],
        compositional_coverage=rep["compositional_coverage"],
        covered_nodes=rep["node_coverage"]["covered"],
        uncovered_nodes=rep["node_coverage"]["uncovered"],
        annotations_count=rep["annotations_count"],
        leaf_count=rep["leaf_count"],
        composition_count=rep["composition_count"],
        uncovered_edges=rep["edge_coverage"]["uncovered"],
        by_annotation=rep["node_coverage"]["by_annotation"],
        hierarchy=rep.get("hierarchy"),
    )


@router.post("/auto-classify")
async def auto_classify_coverage(
    request: ClassifyNodesRequest,
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Automatically detect entry and exit nodes for a proposed coverage.

    Useful when you want to create an annotation but don't know which
    nodes should be entry vs exit. This endpoint analyzes the coverage
    and returns suggested entry/exit classification.
    """
    genome, explanation, model_state = _get_current_model(genome_id, db)

    entry_nodes, exit_nodes = auto_detect_entry_exit(model_state, request.coverage)

    # Also get full classification for context
    classification = classify_coverage(model_state, request.coverage)

    return {
        "coverage": request.coverage,
        "suggested_entry_nodes": entry_nodes,
        "suggested_exit_nodes": exit_nodes,
        "intermediate_nodes": classification.intermediate_nodes,
        "valid": classification.valid,
        "violations": classification.violations,
    }
