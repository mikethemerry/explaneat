"""
Operation-related API routes.

These routes handle adding, removing, and validating operations
on the explanation event stream.
"""

import tempfile
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query
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
    FunctionNodeMetadataSchema,
)
from ...db import Genome, Population, Experiment, Explanation
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.model_state import ModelStateEngine, Operation
from ...core.operations import OperationError
from ...core.collapse_transform import collapse_structure


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


def _network_to_response(
    network,
    is_original: bool = False,
    collapsed_annotations: Optional[List[str]] = None,
) -> ModelStateResponse:
    """Convert NetworkStructure to API response."""
    nodes = []
    for node in network.nodes:
        node_type = node.type.value if hasattr(node.type, "value") else node.type

        # Build function_metadata schema if present
        fn_meta = None
        if node.function_metadata is not None:
            fn_meta = FunctionNodeMetadataSchema(
                annotation_name=node.function_metadata.annotation_name,
                annotation_id=node.function_metadata.annotation_id,
                hypothesis=node.function_metadata.hypothesis,
                n_inputs=node.function_metadata.n_inputs,
                n_outputs=node.function_metadata.n_outputs,
                input_names=node.function_metadata.input_names,
                output_names=node.function_metadata.output_names,
                formula_latex=node.function_metadata.formula_latex,
                subgraph_nodes=node.function_metadata.subgraph_nodes,
                subgraph_connections=node.function_metadata.subgraph_connections,
            )

        nodes.append(
            NodeSchema(
                id=node.id,
                type=node_type,
                bias=node.bias,
                activation=node.activation,
                response=node.response,
                aggregation=node.aggregation,
                function_metadata=fn_meta,
                display_name=getattr(node, "display_name", None),
            )
        )

    connections = [
        ConnectionSchema(
            **{
                "from": conn.from_node,
                "to": conn.to_node,
                "weight": conn.weight,
                "enabled": conn.enabled,
                "output_index": conn.output_index,
            }
        )
        for conn in network.connections
    ]

    metadata = ModelMetadata(
        input_nodes=network.input_node_ids,
        output_nodes=network.output_node_ids,
        is_original=is_original or network.metadata.get("is_original", True),
        collapsed_annotations=collapsed_annotations or [],
        has_non_identity_ops=network.metadata.get("has_non_identity_ops", False),
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
        notes=op.notes,
    )


# =============================================================================
# Model State
# =============================================================================


@router.get("/model", response_model=ModelStateResponse)
async def get_current_model(
    genome_id: UUID = Path(..., description="The genome ID"),
    collapsed: Optional[str] = Query(
        None,
        description="Comma-separated annotation names to collapse into function nodes",
    ),
    db: Session = Depends(get_db),
):
    """
    Get the current model state after applying all operations.

    Returns the phenotype with all operations (splits, identity nodes, etc.)
    applied in order. Optionally collapse specified annotations into function nodes.

    Query params:
        collapsed: Comma-separated list of annotation names to collapse
                   (e.g. "ann1,ann2").
    """
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)

    current_state = engine.current_state
    is_original = len(engine.operations) == 0

    # Pass non-identity status through metadata
    current_state.metadata["has_non_identity_ops"] = engine.has_non_identity_ops

    collapsed_names: List[str] = []
    if collapsed:
        collapsed_names = [name.strip() for name in collapsed.split(",") if name.strip()]

    if collapsed_names:
        collapsed_ids = set(collapsed_names)
        current_state = collapse_structure(
            current_state, engine.annotations, collapsed_ids
        )

    return _network_to_response(
        current_state,
        is_original=is_original,
        collapsed_annotations=collapsed_names,
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
            notes=op_data.get("notes"),
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
    - **rename_node**: Set or clear a display name on a node
    - **rename_annotation**: Set or clear a display name on an annotation
    - **disable_connection**: Disable (turn off) a connection
    - **enable_connection**: Re-enable a previously disabled connection
    """
    genome, explanation, engine = _get_phenotype_and_engine(genome_id, db)

    # Convert params to dict
    params = operation.params.model_dump()

    try:
        # Add operation to engine (validates and applies)
        new_op = engine.add_operation(operation.type, params, validate=True, notes=operation.notes)

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
