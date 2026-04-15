"""Tests for the TrainingJobManager."""

import asyncio
import pytest
import numpy as np

from explaneat.core.genome_network import (
    NetworkNode,
    NetworkConnection,
    NetworkStructure,
    NodeType,
)
from explaneat.api.training_jobs import TrainingJobManager, JobStatus


def _simple_network():
    """Simple: 2 inputs -> 1 hidden (sigmoid) -> 1 output (sigmoid)."""
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="sigmoid"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.1, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=0.8, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes, connections=connections,
        input_node_ids=["-1", "-2"], output_node_ids=["0"],
    )


def _make_data():
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1])))
    return X, y


async def _wait_for_completion(manager, job_id, timeout=10.0):
    """Poll until job finishes, with timeout."""
    elapsed = 0
    while elapsed < timeout:
        job = manager.get_status(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return job
        await asyncio.sleep(0.05)
        elapsed += 0.05
    return manager.get_status(job_id)


class TestTrainingJobManager:
    def test_start_retrain_returns_job_id(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=5,
            )
            assert isinstance(job_id, str)
            assert len(job_id) > 0

        asyncio.run(run())

    def test_get_status(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=5,
            )

            job = manager.get_status(job_id)
            assert job is not None
            assert job.genome_id == "test-genome"
            assert job.total_epochs == 5

        asyncio.run(run())

    def test_training_completes(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=10,
            )

            job = await _wait_for_completion(manager, job_id)
            assert job.status == JobStatus.COMPLETED
            assert job.current_epoch == 10
            assert len(job.metrics["loss"]) == 10
            assert len(job.metrics["val_loss"]) == 10

        asyncio.run(run())

    def test_training_produces_weight_updates(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=10,
            )

            job = await _wait_for_completion(manager, job_id)
            assert job.result is not None
            assert "weight_updates" in job.result
            assert "bias_updates" in job.result
            assert len(job.result["weight_updates"]) > 0

        asyncio.run(run())

    def test_loss_decreases(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=30,
                learning_rate=0.01,
            )

            job = await _wait_for_completion(manager, job_id)
            assert job.status == JobStatus.COMPLETED
            losses = job.metrics["loss"]
            assert losses[-1] < losses[0], \
                f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

        asyncio.run(run())

    def test_frozen_nodes(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=10,
                frozen_nodes={"5"},
            )

            job = await _wait_for_completion(manager, job_id)
            assert job.status == JobStatus.COMPLETED
            assert job.result is not None

        asyncio.run(run())

    def test_cancel_job(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=10000,  # Long enough to cancel
            )

            # Wait for it to start running
            await asyncio.sleep(0.1)

            cancelled = manager.cancel(job_id)
            assert cancelled

            job = await _wait_for_completion(manager, job_id)
            assert job.status == JobStatus.CANCELLED

        asyncio.run(run())

    def test_nonexistent_job(self):
        manager = TrainingJobManager()
        assert manager.get_status("nonexistent") is None

    def test_cancel_nonexistent_job(self):
        manager = TrainingJobManager()
        assert manager.cancel("nonexistent") is False

    def test_result_has_final_metrics(self):
        async def run():
            manager = TrainingJobManager()
            X, y = _make_data()
            struct = _simple_network()
            job_id = await manager.start_retrain(
                genome_id="test-genome",
                structure=struct,
                X=X, y=y,
                n_epochs=5,
            )

            job = await _wait_for_completion(manager, job_id)
            assert job.result["final_loss"] is not None
            assert job.result["final_val_loss"] is not None
            assert job.result["epochs_completed"] == 5

        asyncio.run(run())
