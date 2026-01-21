"""
Operation-related API routes.

These routes handle adding, removing, and validating operations
on the explanation event stream.
"""

import tempfile
from typing import List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
import neat

from ..dependencies import get_db
from ..schemas import (
    OperationRequest,
    OperationResponse,
    OperationListResponse,
    OperationValidationResponse,
    OperationResult,
    ModelStateResponse,
    NodeSchema,
    ConnectionSchema,
    ModelMetadata,
)
from ...db import Genome, Population, Experiment, Explanation
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.model_state import ModelStateEngine, Operation
from ...core.operations import OperationError


router = APIRouter()


def _get_neat_config(experiment: Experiment) -> neat.Config:
    """Load NEAT config from experiment's config text and JSON."""
    from ...core.config_utils import load_neat_config

    return load_neat_config(
        experiment.neat_config_text or "",
        experiment.config_json,
    )


def _get_phenotype_and_engine(
    genome_id: UUID,
    db: Session,
) -> tuple[Genome, Explanation, ModelStateEngine]:
    """
    Get genome, explanation, and initialized ModelStateEngine.

    Returns:
        Tuple of (genome, explanation, engine)
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
            operations=[],
        )
        db.add(explanation)
        db.commit()
        db.refresh(explanation)

    # Create engine and load existing operations
    engine = ModelStateEngine.from_phenotype_and_operations(
        phenotype,
        {"operations": explanation.operations or []},
    )

    return genome, explanation, engine


def _network_to_response(network, is_original: bool = False) -> ModelStateResponse:
    """Convert NetworkStructure to API response."""
    nodes = [
        NodeSchema(
            id=node.id,
            type=node.type.value if hasattr(node.type, "value") else node.type,
            bias=node.bias,
            activation=node.activation,
            response=node.response,
            aggregation=node.aggregation,
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
        is_original=is_original or network.metadata.get("is_original", True),
    )

    return ModelStateResponse(
        nodes=nodes,
        connections=connections,
        metadata=metadata,
    )


def _operation_to_response(op: Operation) -> OperationResponse:
    """Convert Operation to API response."""
    return OperationResponse(
        seq=op.seq,
        type=op.type,
        params=op.params,
        result=OperationResult(**op.result) if op.result else None,
        created_at=op.created_at,
    )


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
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)

    current_state = engine.current_state
    is_original = len(engine.operations) == 0

    return _network_to_response(current_state, is_original=is_original)


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

    # Convert stored operations to response format
    operations = []
    for op_data in explanation.operations or []:
        created_at = op_data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not created_at:
            created_at = datetime.utcnow()

        operations.append(OperationResponse(
            seq=op_data["seq"],
            type=op_data["type"],
            params=op_data["params"],
            result=op_data.get("result"),
            created_at=created_at,
        ))

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
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)

    # Convert params to dict
    params = operation.params.model_dump()

    try:
        # Add operation to engine (validates and applies)
        new_op = engine.add_operation(operation.type, params, validate=True)

        # Save updated operations to database
        explanation.operations = engine.to_dict()["operations"]
        db.commit()

        return _operation_to_response(new_op)

    except OperationError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)

    if not explanation.operations:
        raise HTTPException(status_code=404, detail="No operations to remove")

    try:
        # Remove operations from seq onwards
        removed = engine.remove_operation(seq)

        # Save updated operations to database
        explanation.operations = engine.to_dict()["operations"]
        db.commit()

        return {
            "status": "removed",
            "removed_count": len(removed),
            "remaining_operations": len(engine.operations),
        }

    except OperationError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)

    # Convert params to dict
    params = operation.params.model_dump()

    # Validate operation
    errors = engine.validate_operation(operation.type, params)

    return OperationValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=[],
    )
