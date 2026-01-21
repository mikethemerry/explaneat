"""
Experiment-related API routes.
"""

import tempfile
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
import neat

from ..dependencies import get_db
from ..schemas import (
    ExperimentListItem,
    ExperimentListResponse,
    GenomeDetail,
    ModelStateResponse,
    NodeSchema,
    ConnectionSchema,
    ModelMetadata,
)
from ...db import Experiment, Population, Genome
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.genome_network import NetworkStructure


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


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    db: Session = Depends(get_db),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    List experiments sorted by creation date (most recent first).

    This matches the CLI's experiment listing order.
    """
    # Get total count
    total = db.query(Experiment).count()

    # Get experiments sorted by created_at descending (most recent first)
    experiments = (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    result = []
    for exp in experiments:
        # Count generations (populations)
        generations = (
            db.query(Population)
            .filter(Population.experiment_id == exp.id)
            .count()
        )

        # Count total genomes
        total_genomes = (
            db.query(Genome)
            .join(Population)
            .filter(Population.experiment_id == exp.id)
            .count()
        )

        # Get best fitness
        best_genome = (
            db.query(Genome)
            .join(Population)
            .filter(
                Population.experiment_id == exp.id,
                Genome.fitness.isnot(None),
            )
            .order_by(Genome.fitness.desc())
            .first()
        )

        result.append(
            ExperimentListItem(
                id=str(exp.id),
                name=exp.name,
                status=exp.status,
                dataset_name=exp.dataset_name,
                generations=generations,
                total_genomes=total_genomes,
                best_fitness=best_genome.fitness if best_genome else None,
                created_at=exp.created_at,
            )
        )

    return ExperimentListResponse(
        experiments=result,
        total=total,
    )


@router.get("/{experiment_id}/best-genome")
async def get_best_genome(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the best genome from an experiment.

    This matches the CLI's behavior when selecting an experiment (s 0).
    Returns the genome with the highest fitness.
    """
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    # Get best genome by fitness
    best_genome = (
        db.query(Genome)
        .join(Population)
        .filter(
            Population.experiment_id == experiment_id,
            Genome.fitness.isnot(None),
        )
        .order_by(Genome.fitness.desc())
        .first()
    )

    if not best_genome:
        raise HTTPException(
            status_code=404,
            detail=f"No genomes with fitness found in experiment {experiment_id}",
        )

    return {
        "genome_id": str(best_genome.id),
        "neat_genome_id": best_genome.genome_id,
        "fitness": best_genome.fitness,
        "num_nodes": best_genome.num_nodes,
        "num_connections": best_genome.num_connections,
        "experiment_id": str(experiment_id),
        "experiment_name": experiment.name,
    }


@router.get("/{experiment_id}/best-genome/phenotype", response_model=ModelStateResponse)
async def get_best_genome_phenotype(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the phenotype of the best genome from an experiment.

    Combines experiment selection and phenotype loading in one call.
    """
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    # Get best genome by fitness
    best_genome = (
        db.query(Genome)
        .join(Population)
        .filter(
            Population.experiment_id == experiment_id,
            Genome.fitness.isnot(None),
        )
        .order_by(Genome.fitness.desc())
        .first()
    )

    if not best_genome:
        raise HTTPException(
            status_code=404,
            detail=f"No genomes with fitness found in experiment {experiment_id}",
        )

    # Load phenotype
    config = _get_neat_config(experiment)
    neat_genome = deserialize_genome(best_genome.genome_data, config)
    explaneat = ExplaNEAT(neat_genome, config)
    phenotype = explaneat.get_phenotype_network()
    phenotype.metadata["is_original"] = True

    return _network_to_response(phenotype)
