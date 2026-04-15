"""Retraining API routes for fine-tuning annotated models."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import APIRouter, HTTPException, Path

from ...db import db
from ...db.models import Dataset, DatasetSplit, Genome, Explanation
from ...db.dataset_utils import sample_dataset
from ...core.model_state import ModelStateEngine
from ..training_jobs import training_manager, JobStatus
from ..schemas import (
    RetrainStartRequest,
    RetrainStartResponse,
    RetrainStatusResponse,
    RetrainApplyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_engine(session, genome_id: str) -> ModelStateEngine:
    """Build a ModelStateEngine for a genome (same as evidence._build_engine)."""
    from ...core.explaneat import ExplaNEAT
    from ...core.config_utils import load_neat_config

    genome_db = session.query(Genome).filter_by(id=uuid.UUID(genome_id)).first()
    if not genome_db:
        raise HTTPException(status_code=404, detail="Genome not found")

    experiment = genome_db.population.experiment
    config = load_neat_config(
        experiment.neat_config_text or "",
        experiment.config_json,
    )
    neat_genome = genome_db.to_neat_genome(config)
    explainer = ExplaNEAT(neat_genome, config)
    phenotype = explainer.get_phenotype_network()

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == uuid.UUID(genome_id))
        .first()
    )

    engine = ModelStateEngine(phenotype)
    if explanation and explanation.operations:
        engine.load_operations({"operations": explanation.operations})
    return engine


def _load_split_data(session, split_id: str, split_choice: str, max_samples: int):
    """Load X, y data from a dataset split."""
    split = session.query(DatasetSplit).filter_by(id=uuid.UUID(split_id)).first()
    if not split:
        raise HTTPException(status_code=404, detail="Dataset split not found")

    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = dataset.get_data()
    if data is None:
        raise HTTPException(status_code=400, detail="Dataset has no stored data")

    X_full, y_full = data

    if split_choice == "train":
        indices = split.train_indices
    elif split_choice == "test":
        indices = split.test_indices
    else:
        indices = (split.train_indices or []) + (split.test_indices or [])

    if not indices:
        raise HTTPException(status_code=400, detail="No indices for requested split")

    X = X_full[indices]
    y = y_full[indices]

    X, y = sample_dataset(X, y, fraction=1.0, max_samples=max_samples)
    return X, y


@router.post("/retrain", response_model=RetrainStartResponse)
async def start_retrain(
    request: RetrainStartRequest,
    genome_id: str = Path(...),
):
    """Start a retraining job for the current model state.

    Builds a TrainableStructureNetwork from the current annotated model,
    trains it on the specified dataset, and returns a job_id for polling.
    """
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        structure = engine.current_state

        # Determine frozen nodes from annotations if requested
        frozen_nodes: Optional[Set[str]] = None
        if request.freeze_annotations:
            frozen_nodes = set()
            for ann in engine.annotations:
                for nid in ann.get("subgraph_nodes", []):
                    frozen_nodes.add(nid)

        X, y = _load_split_data(
            session, request.dataset_split_id,
            request.split, request.max_samples,
        )

    job_id = await training_manager.start_retrain(
        genome_id=genome_id,
        structure=structure,
        X=X,
        y=y,
        n_epochs=request.n_epochs,
        learning_rate=request.learning_rate,
        frozen_nodes=frozen_nodes,
    )

    return RetrainStartResponse(job_id=job_id)


@router.get("/retrain/{job_id}", response_model=RetrainStatusResponse)
async def get_retrain_status(
    job_id: str,
    genome_id: str = Path(...),
):
    """Poll the status of a retraining job."""
    job = training_manager.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.genome_id != genome_id:
        raise HTTPException(status_code=404, detail="Training job not found for this genome")

    return RetrainStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        metrics=job.metrics,
        error=job.error,
    )


@router.post("/retrain/{job_id}/apply", response_model=RetrainApplyResponse)
async def apply_retrain(
    job_id: str,
    genome_id: str = Path(...),
):
    """Apply a completed training job's weight updates as a retrain operation.

    This commits the trained weights as a new operation in the model's
    operation event stream.
    """
    job = training_manager.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.genome_id != genome_id:
        raise HTTPException(status_code=404, detail="Training job not found for this genome")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job.status.value}, not completed",
        )
    if not job.result:
        raise HTTPException(status_code=400, detail="Job has no results")

    with db.session_scope() as session:
        genome_db = session.query(Genome).filter_by(id=uuid.UUID(genome_id)).first()
        if not genome_db:
            raise HTTPException(status_code=404, detail="Genome not found")

        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == uuid.UUID(genome_id))
            .first()
        )
        if not explanation:
            raise HTTPException(status_code=404, detail="No explanation found")

        operations = list(explanation.operations or [])
        next_seq = max((op.get("seq", 0) for op in operations), default=0) + 1

        retrain_op = {
            "seq": next_seq,
            "type": "retrain",
            "params": {
                "weight_updates": job.result["weight_updates"],
                "bias_updates": job.result["bias_updates"],
                "metadata": {
                    "epochs": job.result["epochs_completed"],
                    "final_loss": job.result["final_loss"],
                    "final_val_loss": job.result["final_val_loss"],
                    "job_id": job_id,
                },
            },
            "result": {},
            "created_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }

        operations.append(retrain_op)
        explanation.operations = operations
        session.commit()

    return RetrainApplyResponse(
        operation_seq=next_seq,
        final_loss=job.result["final_loss"],
        final_val_loss=job.result["final_val_loss"],
        epochs_completed=job.result["epochs_completed"],
    )


@router.post("/retrain/{job_id}/cancel")
async def cancel_retrain(
    job_id: str,
    genome_id: str = Path(...),
):
    """Cancel a running training job."""
    job = training_manager.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.genome_id != genome_id:
        raise HTTPException(status_code=404, detail="Training job not found for this genome")

    if training_manager.cancel(job_id):
        return {"status": "cancelled"}
    else:
        return {"status": job.status.value, "message": "Cannot cancel job in current state"}
