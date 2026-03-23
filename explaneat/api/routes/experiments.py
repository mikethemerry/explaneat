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
    ExperimentSplitResponse,
    LinkDatasetRequest,
    GenomeDetail,
    ModelStateResponse,
    NodeSchema,
    ConnectionSchema,
    ModelMetadata,
)
from ...db import Experiment, Population, Genome
from ...db.models import Dataset, DatasetSplit
from ...db.dataset_utils import create_or_get_split
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

        # Check if a split exists for this experiment
        has_split = (
            db.query(DatasetSplit)
            .filter(DatasetSplit.experiment_id == exp.id)
            .first()
            is not None
        )

        result.append(
            ExperimentListItem(
                id=str(exp.id),
                name=exp.name,
                status=exp.status,
                dataset_name=exp.dataset_name,
                dataset_id=str(exp.dataset_id) if exp.dataset_id else None,
                has_split=has_split,
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


@router.get("/{experiment_id}/split", response_model=ExperimentSplitResponse)
async def get_experiment_split(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the dataset split linked to an experiment.

    Returns split info + dataset metadata, or 404 if no split exists.
    """
    split = (
        db.query(DatasetSplit)
        .filter(DatasetSplit.experiment_id == experiment_id)
        .first()
    )
    if not split:
        raise HTTPException(
            status_code=404,
            detail=f"No dataset split found for experiment {experiment_id}",
        )

    dataset = db.get(Dataset, split.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {split.dataset_id} not found",
        )

    return ExperimentSplitResponse(
        split_id=str(split.id),
        dataset_id=str(split.dataset_id),
        dataset_name=dataset.name,
        num_samples=dataset.num_samples,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        split_type=split.split_type,
        test_size=split.test_size,
        random_state=split.random_state,
        train_size=split.train_size,
        test_size_actual=split.test_size_actual,
        feature_names=dataset.feature_names,
        feature_types=dataset.feature_types,
        feature_descriptions=dataset.feature_descriptions,
        target_name=dataset.target_name,
        target_description=dataset.target_description,
    )


@router.put("/{experiment_id}/dataset", response_model=ExperimentSplitResponse)
async def link_dataset_to_experiment(
    experiment_id: UUID,
    request: LinkDatasetRequest,
    db: Session = Depends(get_db),
):
    """
    Link a dataset to an experiment and create a train/test split.

    Sets experiment.dataset_id and creates a DatasetSplit in one call.
    """
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id} not found",
        )

    dataset = db.get(Dataset, UUID(request.dataset_id))
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {request.dataset_id} not found",
        )

    # Link dataset to experiment
    experiment.dataset_id = dataset.id
    db.flush()

    # Create or get split
    try:
        split = create_or_get_split(
            dataset_id=str(dataset.id),
            experiment_id=str(experiment_id),
            test_proportion=request.test_proportion,
            random_seed=request.random_seed,
            stratify=request.stratify,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db.commit()

    return ExperimentSplitResponse(
        split_id=str(split.id),
        dataset_id=str(split.dataset_id),
        dataset_name=dataset.name,
        num_samples=dataset.num_samples,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        split_type=split.split_type,
        test_size=split.test_size,
        random_state=split.random_state,
        train_size=split.train_size,
        test_size_actual=split.test_size_actual,
        feature_names=dataset.feature_names,
        feature_types=dataset.feature_types,
        feature_descriptions=dataset.feature_descriptions,
        target_name=dataset.target_name,
        target_description=dataset.target_description,
    )
