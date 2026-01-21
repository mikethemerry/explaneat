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
)
from ...db import Genome, Population, Experiment, Explanation
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.genome_network import NetworkStructure


router = APIRouter()


def _get_neat_config(experiment: Experiment) -> neat.Config:
    """Load NEAT config from experiment's config text."""
    # Write config to temporary file (neat-python requires file path)
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


@router.get("/{genome_id}/explanation", response_model=ExplanationResponse)
async def get_genome_explanation(
    genome_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the explanation for a genome.

    Creates a new explanation if one doesn't exist.
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

    # TODO: Load operations from the operations JSON column once we add it
    operations = []

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

    explanation = (
        db.query(Explanation)
        .filter(Explanation.genome_id == genome_id)
        .first()
    )

    if not explanation:
        raise HTTPException(
            status_code=404,
            detail="Explanation not found. Call GET first to create one.",
        )

    if update.name is not None:
        explanation.name = update.name
    if update.description is not None:
        explanation.description = update.description

    db.commit()
    db.refresh(explanation)

    return ExplanationResponse(
        id=str(explanation.id),
        genome_id=str(explanation.genome_id),
        name=explanation.name,
        description=explanation.description,
        operations=[],  # TODO: Load from operations column
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
