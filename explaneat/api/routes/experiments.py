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
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    ExperimentProgressResponse,
)
from ...db import Experiment, Population, Genome
from ...db.models import Dataset, DatasetSplit, ConfigTemplate
from ...db.dataset_utils import create_or_get_split
from ...db.serialization import deserialize_genome
from ...core.explaneat import ExplaNEAT
from ...core.genome_network import NetworkStructure
from ...core.config_resolution import resolve_config, config_to_neat_text


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

        has_split = exp.split_id is not None

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
    experiment = db.get(Experiment, experiment_id)
    if not experiment or not experiment.split_id:
        raise HTTPException(
            status_code=404,
            detail=f"No dataset split found for experiment {experiment_id}",
        )

    split = db.get(DatasetSplit, experiment.split_id)
    if not split:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset split {experiment.split_id} not found",
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
        validation_size=split.validation_size,
        feature_names=dataset.feature_names,
        feature_types=dataset.feature_types,
        feature_descriptions=dataset.feature_descriptions,
        target_name=dataset.target_name,
        target_description=dataset.target_description,
        class_names=dataset.class_names,
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

    # Create or get split
    try:
        split = create_or_get_split(
            dataset_id=str(dataset.id),
            test_proportion=request.test_proportion,
            random_seed=request.random_seed,
            stratify=request.stratify,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Link dataset and split to experiment
    experiment.dataset_id = dataset.id
    experiment.split_id = split.id
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
        validation_size=split.validation_size,
        feature_names=dataset.feature_names,
        feature_types=dataset.feature_types,
        feature_descriptions=dataset.feature_descriptions,
        target_name=dataset.target_name,
        target_description=dataset.target_description,
        class_names=dataset.class_names,
    )


# =============================================================================
# Experiment creation and monitoring
# =============================================================================


@router.post("/run", response_model=ExperimentCreateResponse)
async def create_and_run_experiment(
    request: ExperimentCreateRequest,
    db_session: Session = Depends(get_db),
):
    """Create and start a new NEAT experiment.

    Launches evolution in a background thread. Poll progress with
    GET /api/experiments/jobs/{job_id}.
    """
    from ..experiment_runner import experiment_runner

    dataset = db_session.get(Dataset, UUID(request.dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    split = db_session.get(DatasetSplit, UUID(request.dataset_split_id))
    if not split:
        raise HTTPException(status_code=404, detail="Dataset split not found")

    data = dataset.get_data()
    if data is None:
        raise HTTPException(status_code=400, detail="Dataset has no stored data")

    X_full, y_full = data

    # Auto-prepare dataset if it has categorical/ordinal features
    feature_names = dataset.feature_names or []
    ft_dict = dataset.feature_types or {}
    if (
        not dataset.source_dataset_id
        and any(ft in ("categorical", "ordinal") for ft in ft_dict.values())
    ):
        from ...db.encoding import build_encoding_config, prepare_dataset_arrays

        feature_types_list = [ft_dict.get(name, "numeric") for name in feature_names]
        enc_config = build_encoding_config(X_full, feature_names, feature_types_list)

        # Check for existing prepared version with same config
        prepared = (
            db_session.query(Dataset)
            .filter(
                Dataset.source_dataset_id == dataset.id,
                Dataset.encoding_config == enc_config,
            )
            .first()
        )

        if prepared is None:
            X_enc, new_names, new_types_dict = prepare_dataset_arrays(
                X_full, feature_names, feature_types_list, enc_config,
            )
            prepared = Dataset(
                name=f"{dataset.name} (prepared)",
                version=dataset.version,
                source=dataset.source,
                description=f"Auto-prepared from {dataset.name}",
                num_samples=X_enc.shape[0],
                num_features=X_enc.shape[1],
                num_classes=dataset.num_classes,
                feature_names=new_names,
                feature_types=new_types_dict,
                target_name=dataset.target_name,
                class_names=dataset.class_names,
                source_dataset_id=dataset.id,
                encoding_config=enc_config,
            )
            prepared.set_data(X_enc, y_full)
            db_session.add(prepared)
            db_session.commit()

        # Switch to prepared dataset
        dataset = prepared
        data = dataset.get_data()
        if data is None:
            raise HTTPException(status_code=500, detail="Failed to load prepared dataset")
        X_full, y_full = data
        feature_names = dataset.feature_names or []

    train_indices = split.train_indices or []
    if not train_indices:
        raise HTTPException(status_code=400, detail="Split has no training indices")

    X_train = X_full[train_indices]
    y_train = y_full[train_indices]

    # Standardize features so the network can learn effectively.
    # Save scaler params on the split so the evidence pipeline applies
    # the same transform.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    split.scaler_type = "StandardScaler"
    split.scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    db_session.commit()

    # Resolve config from template + overrides
    template_config = None
    if request.config_template_id:
        template_obj = db_session.get(ConfigTemplate, UUID(request.config_template_id))
        if not template_obj:
            raise HTTPException(status_code=404, detail="Config template not found")
        template_config = template_obj.config

    resolved_config = resolve_config(
        template=template_config,
        overrides=request.config_overrides,
    )

    num_inputs = X_train.shape[1]
    num_outputs = 1
    if dataset.num_classes and dataset.num_classes > 2:
        num_outputs = dataset.num_classes

    config_text = config_to_neat_text(resolved_config, num_inputs, num_outputs)
    config_json = {
        "pop_size": resolved_config["training"]["population_size"],
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "fitness_criterion": "max",
        "fitness_threshold": 999.0,
        "resolved_config": resolved_config,
        "config_template_id": request.config_template_id,
    }

    job_id = await experiment_runner.start(
        config_text=config_text,
        config_json=config_json,
        X_train=X_train,
        y_train=y_train,
        experiment_name=request.name,
        dataset_name=dataset.name,
        n_generations=resolved_config["training"]["n_generations"],
        n_epochs_backprop=resolved_config["training"]["n_epochs_backprop"],
        fitness_function=resolved_config["training"]["fitness_function"],
        description=request.description,
        dataset_id=str(dataset.id),
        split_id=str(split.id),
        config_template_id=request.config_template_id,
    )

    return ExperimentCreateResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=ExperimentProgressResponse)
async def get_experiment_progress(job_id: str):
    """Poll the progress of a running experiment."""
    from ..experiment_runner import experiment_runner

    progress = experiment_runner.get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Experiment job not found")

    return ExperimentProgressResponse(
        job_id=progress.job_id,
        experiment_id=progress.experiment_id,
        status=progress.status.value,
        current_generation=progress.current_generation,
        total_generations=progress.total_generations,
        best_fitness=progress.best_fitness,
        mean_fitness=progress.mean_fitness,
        pop_size=progress.pop_size,
        num_species=progress.num_species,
        error=progress.error,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_experiment(job_id: str):
    """Cancel a running experiment."""
    from ..experiment_runner import experiment_runner

    if experiment_runner.cancel(job_id):
        return {"status": "cancelled"}
    progress = experiment_runner.get_progress(job_id)
    status = progress.status.value if progress else "not_found"
    return {"status": status, "message": "Cannot cancel job in current state"}


def _default_neat_config_text(num_inputs: int, num_outputs: int, pop_size: int) -> str:
    """Generate a minimal NEAT config text for experiment creation."""
    return f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = relu
activation_mutate_rate  = 0.0
activation_options      = relu
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.3
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.15
node_delete_prob        = 0.05
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
