"""Lightweight in-memory training job manager.

Launches retraining of annotated models using TrainableStructureNetwork
in a background thread via asyncio.to_thread().
"""

import asyncio
import logging
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import torch

from ..core.genome_network import NetworkStructure
from ..core.trainable_structure_network import TrainableStructureNetwork

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    job_id: str
    genome_id: str
    status: JobStatus = JobStatus.PENDING
    current_epoch: int = 0
    total_epochs: int = 0
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False


class TrainingJobManager:
    """Manages background retraining jobs."""

    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}

    def get_status(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    async def start_retrain(
        self,
        genome_id: str,
        structure: NetworkStructure,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 50,
        learning_rate: float = 0.01,
        frozen_nodes: Optional[Set[str]] = None,
    ) -> str:
        """Launch retraining in a background thread.

        Returns job_id for polling progress.
        """
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(
            job_id=job_id,
            genome_id=genome_id,
            total_epochs=n_epochs,
            metrics={"loss": [], "val_loss": []},
        )
        self._jobs[job_id] = job

        # Deep-copy the structure so the training thread owns its own copy
        structure_copy = deepcopy(structure)

        # Launch in thread pool
        asyncio.create_task(
            self._run_training(
                job, structure_copy, X, y,
                n_epochs, learning_rate, frozen_nodes,
            )
        )

        return job_id

    async def _run_training(
        self,
        job: TrainingJob,
        structure: NetworkStructure,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        frozen_nodes: Optional[Set[str]],
    ):
        """Run the actual training loop in a thread."""
        try:
            job.status = JobStatus.RUNNING
            result = await asyncio.to_thread(
                self._training_loop,
                job, structure, X, y, n_epochs, learning_rate, frozen_nodes,
            )
            if job.cancelled:
                job.status = JobStatus.CANCELLED
            else:
                job.status = JobStatus.COMPLETED
                job.result = result
        except Exception as e:
            logger.exception("Training job %s failed", job.job_id)
            job.status = JobStatus.FAILED
            job.error = str(e)

    @staticmethod
    def _training_loop(
        job: TrainingJob,
        structure: NetworkStructure,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        frozen_nodes: Optional[Set[str]],
    ) -> Dict[str, Any]:
        """Synchronous training loop (runs in thread pool)."""
        tn = TrainableStructureNetwork(structure, frozen_nodes=frozen_nodes)
        optimizer = torch.optim.Adam(tn.parameters(), lr=learning_rate)

        x_tensor = torch.as_tensor(X, dtype=torch.float64)
        y_tensor = torch.as_tensor(y, dtype=torch.float64)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        # Simple train/val split (80/20)
        n = len(X)
        n_train = max(1, int(n * 0.8))
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:] if n_train < n else indices[:1]

        x_train = x_tensor[train_idx]
        y_train = y_tensor[train_idx]
        x_val = x_tensor[val_idx]
        y_val = y_tensor[val_idx]

        for epoch in range(n_epochs):
            if job.cancelled:
                break

            # Training step
            optimizer.zero_grad()
            pred = tn.forward(x_train)
            loss = torch.nn.functional.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()

            # Validation
            with torch.no_grad():
                val_pred = tn.forward(x_val)
                val_loss = torch.nn.functional.mse_loss(val_pred, y_val)

            job.current_epoch = epoch + 1
            job.metrics["loss"].append(float(loss.item()))
            job.metrics["val_loss"].append(float(val_loss.item()))

        # Write trained weights back to the structure
        update_result = tn.update_structure_weights()

        return {
            "weight_updates": update_result["weight_updates"],
            "bias_updates": update_result["bias_updates"],
            "final_loss": job.metrics["loss"][-1] if job.metrics["loss"] else None,
            "final_val_loss": job.metrics["val_loss"][-1] if job.metrics["val_loss"] else None,
            "epochs_completed": job.current_epoch,
        }

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.cancelled = True
            return True
        return False


# Singleton instance
training_manager = TrainingJobManager()
