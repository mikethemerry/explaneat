"""
Analysis-related API routes.

These routes handle split detection, node classification, and coverage analysis.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session

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
from ...db import Genome, Explanation


router = APIRouter()


@router.post("/split-detection", response_model=SplitDetectionResponse)
async def detect_required_splits(
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
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # TODO: Implement split detection algorithm
    # This will be implemented in Phase 3
    #
    # The algorithm:
    # 1. Get current model state (phenotype + operations applied)
    # 2. For each node in proposed_coverage:
    #    - Check if it has any external inputs (from outside coverage)
    #    - Check if it has any external outputs (to outside coverage)
    #    - If both, it's a violation
    # 3. Return violations and suggested splits

    proposed_coverage = request.proposed_coverage
    violations = []
    suggested_operations = []
    adjusted_coverage = list(proposed_coverage)

    # Placeholder response
    return SplitDetectionResponse(
        proposed_coverage=proposed_coverage,
        violations=violations,
        suggested_operations=suggested_operations,
        adjusted_coverage=adjusted_coverage,
    )


@router.post("/classify-nodes", response_model=ClassifyNodesResponse)
async def classify_nodes(
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
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # TODO: Implement node classification algorithm
    # This will be implemented in Phase 3
    #
    # The algorithm:
    # 1. Get current model state
    # 2. For each node in coverage:
    #    - Count external inputs and outputs
    #    - Classify as entry/intermediate/exit
    # 3. Validate classification (check for violations)

    coverage = request.coverage

    # Placeholder classification
    classification = NodeClassification(
        entry=[],
        intermediate=list(coverage),  # Default all to intermediate
        exit=[],
    )

    return ClassifyNodesResponse(
        coverage=coverage,
        classification=classification,
        valid=True,
        violations=[],
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
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    if not explanation:
        # No explanation = no coverage
        return CoverageResponse(
            structural_coverage=0.0,
            compositional_coverage=0.0,
            covered_nodes=[],
            uncovered_nodes=[],  # TODO: Get all phenotype nodes
            annotations_count=0,
        )

    # TODO: Implement coverage computation using existing coverage.py
    # This will connect to the existing CoverageComputer in Phase 3

    return CoverageResponse(
        structural_coverage=explanation.structural_coverage or 0.0,
        compositional_coverage=explanation.compositional_coverage or 0.0,
        covered_nodes=[],  # TODO: Compute from annotations
        uncovered_nodes=[],  # TODO: Compute from phenotype - covered
        annotations_count=len(explanation.annotations) if explanation.annotations else 0,
    )
