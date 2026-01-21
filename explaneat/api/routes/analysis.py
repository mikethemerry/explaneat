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
    """Load NEAT config from experiment's config text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(experiment.neat_config_text)
        config_path = f.name

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


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
    genome, explanation, model_state = _get_current_model(genome_id, db)

    if not explanation:
        # No explanation = no coverage
        all_nodes = list(model_state.get_node_ids())
        return CoverageResponse(
            structural_coverage=0.0,
            compositional_coverage=0.0,
            covered_nodes=[],
            uncovered_nodes=all_nodes,
            annotations_count=0,
        )

    # Get covered nodes from annotations in operations
    covered_nodes = set()
    annotations_count = 0

    for op in explanation.operations or []:
        if op.get("type") == "annotate":
            params = op.get("params", {})
            subgraph_nodes = params.get("subgraph_nodes", [])
            covered_nodes.update(subgraph_nodes)
            annotations_count += 1

    # Compute coverage metrics
    all_nodes = model_state.get_node_ids()
    uncovered_nodes = all_nodes - covered_nodes

    # Structural coverage = covered / total
    if len(all_nodes) > 0:
        structural_coverage = len(covered_nodes) / len(all_nodes)
    else:
        structural_coverage = 0.0

    # Compositional coverage - for now, use stored value or compute simple version
    # Full implementation would check composition explanations
    compositional_coverage = explanation.compositional_coverage or 0.0

    return CoverageResponse(
        structural_coverage=structural_coverage,
        compositional_coverage=compositional_coverage,
        covered_nodes=list(covered_nodes),
        uncovered_nodes=list(uncovered_nodes),
        annotations_count=annotations_count,
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
