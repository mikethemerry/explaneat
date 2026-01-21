"""
Operation-related API routes.

These routes handle adding, removing, and validating operations
on the explanation event stream.
"""

from typing import List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..schemas import (
    OperationRequest,
    OperationResponse,
    OperationListResponse,
    OperationValidationResponse,
    OperationResult,
    ModelStateResponse,
)
from ...db import Genome, Explanation


router = APIRouter()


# =============================================================================
# Model State
# =============================================================================


@router.get("/model", response_model=ModelStateResponse)
async def get_current_model(
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Get the current model state after applying all operations.

    Returns the phenotype with all operations (splits, identity nodes, etc.)
    applied in order.
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    # TODO: Implement model state engine to apply operations
    # For now, return the phenotype without operations applied
    # This will be implemented in Phase 2

    raise HTTPException(
        status_code=501,
        detail="Model state engine not yet implemented. Use /phenotype for now.",
    )


# =============================================================================
# Operations CRUD
# =============================================================================


@router.get("/operations", response_model=OperationListResponse)
async def list_operations(
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Get all operations in the event stream.

    Operations are returned in sequence order.
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
        return OperationListResponse(operations=[], total=0)

    # TODO: Load operations from the operations JSON column
    # For now, return empty list
    operations = []

    return OperationListResponse(
        operations=operations,
        total=len(operations),
    )


@router.post("/operations", response_model=OperationResponse)
async def add_operation(
    operation: OperationRequest,
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Add a new operation to the event stream.

    The operation is validated against the current model state before being added.
    Operations are appended to the end of the event stream.

    Operation types:
    - **split_node**: Split a multi-output node into separate nodes
    - **consolidate_node**: Recombine previously split nodes
    - **remove_node**: Remove a pass-through node
    - **add_node**: Insert a node into a connection
    - **add_identity_node**: Intercept connections through an identity node
    - **annotate**: Create an annotation on a subgraph
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # Get or create explanation
    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    if not explanation:
        explanation = Explanation(
            genome_id=genome_id,
            is_well_formed=False,
        )
        db.add(explanation)
        db.commit()
        db.refresh(explanation)

    # TODO: Implement operation validation and application
    # For now, return a stub response
    # This will be implemented in Phase 2

    # Placeholder for next sequence number
    # In real implementation, we'd read from operations JSON and get len()
    next_seq = 0

    return OperationResponse(
        seq=next_seq,
        type=operation.type,
        params=operation.params.model_dump(),
        result=OperationResult(),
        created_at=datetime.utcnow(),
    )


@router.delete("/operations/{seq}")
async def remove_operation(
    seq: int = Path(..., description="Sequence number of operation to remove"),
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Remove an operation and all subsequent operations (undo).

    This removes the operation at the given sequence number and all
    operations that came after it. The model state is recomputed by
    replaying the remaining operations.
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
        raise HTTPException(status_code=404, detail="No explanation found")

    # TODO: Implement operation removal
    # This will be implemented in Phase 2

    return {
        "status": "removed",
        "removed_seq": seq,
        "remaining_operations": 0,
    }


@router.post("/operations/validate", response_model=OperationValidationResponse)
async def validate_operation(
    operation: OperationRequest,
    genome_id: UUID = Path(..., description="The genome ID"),
    db: Session = Depends(get_db),
):
    """
    Validate an operation without applying it.

    Returns validation results including any errors or warnings.
    This is useful for checking if an operation would succeed before
    actually adding it to the event stream.
    """
    genome = db.get(Genome, genome_id)
    if not genome:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")

    # TODO: Implement operation validation
    # For now, return a stub response indicating validation not implemented
    # This will be implemented in Phase 3

    return OperationValidationResponse(
        valid=True,
        errors=[],
        warnings=["Validation not yet fully implemented"],
    )
