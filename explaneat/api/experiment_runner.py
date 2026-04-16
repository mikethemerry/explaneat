"""Experiment runner service for launching NEAT evolution from the UI.

Wraps DatabaseBackpropPopulation.run() in a background thread with
progress reporting for the API to poll.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class CancelledException(Exception):
    """Raised by ProgressReporter when the job is cancelled."""


@dataclass
class ExperimentProgress:
    job_id: str
    experiment_id: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    current_generation: int = 0
    total_generations: int = 0
    best_fitness: Optional[float] = None
    best_validation_fitness: Optional[float] = None
    mean_fitness: Optional[float] = None
    pop_size: int = 0
    num_species: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False


class ProgressReporter:
    """NEAT reporter that updates ExperimentProgress and supports cancellation.

    Must implement all methods that neat.reporting.ReporterSet dispatches to,
    otherwise AttributeError is raised when NEAT internals (e.g. reproduction)
    call reporter hooks like info().
    """

    def __init__(self, progress: ExperimentProgress):
        self.progress = progress

    # --- Required by neat.reporting.ReporterSet ---
    def info(self, msg):
        logger.info("[NEAT] %s", msg)

    def complete_extinction(self):
        logger.warning("Complete extinction")

    def found_solution(self, config, generation, best):
        logger.info("Solution found at generation %d", generation)

    def species_stagnant(self, sid, species):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def start_generation(self, generation):
        if self.progress.cancelled:
            raise CancelledException("Experiment cancelled by user")
        self.progress.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome and best_genome.fitness is not None:
            self.progress.best_fitness = float(best_genome.fitness)
        if species and hasattr(species, 'species'):
            self.progress.num_species = len(species.species)
        fitnesses = [
            g.fitness for g in population.values()
            if g.fitness is not None
        ]
        if fitnesses:
            self.progress.mean_fitness = float(np.mean(fitnesses))

        # Report best validation fitness if available
        val_fitnesses = [
            getattr(g, 'validation_fitness', None)
            for g in population.values()
        ]
        val_fitnesses = [f for f in val_fitnesses if f is not None]
        if val_fitnesses:
            self.progress.best_validation_fitness = float(max(val_fitnesses))


class ExperimentRunner:
    """Manages background experiment execution."""

    def __init__(self):
        self._jobs: Dict[str, ExperimentProgress] = {}

    def get_progress(self, job_id: str) -> Optional[ExperimentProgress]:
        return self._jobs.get(job_id)

    async def start(
        self,
        config_text: str,
        config_json: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        experiment_name: str,
        dataset_name: str,
        n_generations: int = 10,
        n_epochs_backprop: int = 5,
        fitness_function: str = "bce",
        description: str = "",
        dataset_id: str = None,
        split_id: str = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        patience: int = None,
        config_template_id: str = None,
    ) -> str:
        """Launch a NEAT experiment in a background thread.

        Returns job_id for polling progress.
        """
        job_id = str(uuid.uuid4())[:8]
        progress = ExperimentProgress(
            job_id=job_id,
            total_generations=n_generations,
        )
        self._jobs[job_id] = progress

        asyncio.create_task(
            self._run_experiment(
                progress, config_text, config_json,
                X_train, y_train,
                experiment_name, dataset_name, description,
                n_generations, n_epochs_backprop,
                fitness_function,
                dataset_id, split_id,
                X_val, y_val, patience,
                config_template_id,
            )
        )

        return job_id

    async def _run_experiment(
        self,
        progress: ExperimentProgress,
        config_text: str,
        config_json: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        experiment_name: str,
        dataset_name: str,
        description: str,
        n_generations: int,
        n_epochs_backprop: int,
        fitness_function: str,
        dataset_id: str,
        split_id: str,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        patience: int = None,
        config_template_id: str = None,
    ):
        """Run NEAT evolution in a thread."""
        try:
            progress.status = ExperimentStatus.RUNNING
            result = await asyncio.to_thread(
                self._evolution_loop,
                progress, config_text, config_json,
                X_train, y_train,
                experiment_name, dataset_name, description,
                n_generations, n_epochs_backprop,
                fitness_function,
                dataset_id, split_id,
                X_val, y_val, patience,
                config_template_id,
            )
            if progress.cancelled:
                progress.status = ExperimentStatus.CANCELLED
            else:
                progress.status = ExperimentStatus.COMPLETED
        except CancelledException:
            logger.info("Experiment job %s cancelled", progress.job_id)
            progress.status = ExperimentStatus.CANCELLED
        except Exception as e:
            logger.exception("Experiment job %s failed", progress.job_id)
            progress.status = ExperimentStatus.FAILED
            progress.error = str(e)

    @staticmethod
    def _evolution_loop(
        progress: ExperimentProgress,
        config_text: str,
        config_json: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        experiment_name: str,
        dataset_name: str,
        description: str,
        n_generations: int,
        n_epochs_backprop: int,
        fitness_function: str,
        dataset_id: str,
        split_id: str,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        patience: int = None,
        config_template_id: str = None,
    ) -> dict:
        """Synchronous evolution loop (runs in thread pool)."""
        from uuid import UUID as _UUID
        import neat
        from ..core.config_utils import load_neat_config
        from ..db.population import DatabaseBackpropPopulation
        from ..db.base import db
        from ..evaluators.evaluators import binary_cross_entropy, auc_fitness

        # Select fitness function
        fitness_fn_map: Dict[str, Callable] = {
            "bce": binary_cross_entropy,
            "auc": auc_fitness,
        }
        fitness_fn = fitness_fn_map.get(fitness_function, binary_cross_entropy)

        # Build NEAT config
        config = load_neat_config(config_text, config_json)

        # Create population with database tracking
        population = DatabaseBackpropPopulation(
            config,
            X_train,
            y_train,
            xs_val=X_val,
            ys_val=y_val,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            description=description,
            dataset_id=dataset_id,
            config_template_id=config_template_id,
        )

        progress.experiment_id = str(population.experiment_id)
        progress.pop_size = config.pop_size

        # Link the split to this experiment
        if split_id:
            with db.session_scope() as session:
                from ..db.models import Experiment as ExperimentModel
                exp = session.get(ExperimentModel, population.experiment_id)
                if exp:
                    exp.split_id = _UUID(split_id)

        # Add progress reporter for live updates and cancellation
        reporter = ProgressReporter(progress)
        population.reporters.add(reporter)

        # Run all generations in one call — DatabaseBackpropPopulation.run()
        # handles the generation loop internally, and ProgressReporter hooks
        # provide live updates via start_generation/post_evaluate
        population.run(fitness_fn, n=n_generations, nEpochs=n_epochs_backprop, patience=patience)

        return {
            "experiment_id": progress.experiment_id,
            "generations_completed": progress.current_generation,
            "best_fitness": progress.best_fitness,
        }

    def cancel(self, job_id: str) -> bool:
        progress = self._jobs.get(job_id)
        if progress and progress.status == ExperimentStatus.RUNNING:
            progress.cancelled = True
            return True
        return False


# Singleton instance
experiment_runner = ExperimentRunner()
