"""Run one seeded PropNEAT evolution for the 10-seed consistency experiment.

Seeds every RNG the evolution path touches (python `random` — used by
neat-python for initial weights, mutation, crossover and reproduction —
numpy, and torch) before the population is built, pins torch to a fixed
thread count, and runs the unchanged training path
(DatabaseBackpropPopulation.run with auc_fitness), exactly as
ExperimentRunner._evolution_loop does for UI-launched runs.

The paper's split row is only read: the StandardScaler is fit in memory on
the training indices (the API route would rewrite split.scaler_params) and is
asserted equal to the stored params.

Run each seed in its own process (global RNG state is per process), with
PYTHONHASHSEED=0:

    PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py heart 3
"""

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import neat
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit, Experiment  # noqa: E402
from explaneat.db.population import DatabaseBackpropPopulation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

# Per-dataset settings recovered from the paper runs (see PREREG.md).
DATASETS = {
    "heart": dict(
        dataset_id="eea247c2-bdb7-41af-8ec1-9e26dcb5050d",
        split_id="165cebbd-6609-46ca-867e-c09e17b2d6ec",
        neat_config=HERE / "configs/heart.ini",
        reference_experiment="c8f3805e-",
        n_generations=40,
    ),
    "adult": dict(
        dataset_id="46c63b17-42c5-4142-9fcf-3b5f369319c5",
        split_id="1df50169-8095-40d1-b61a-30bb541cd313",
        neat_config=HERE / "configs/adult.ini",
        reference_experiment="dbab685c-",
        n_generations=100,
    ),
    "monk2": dict(
        dataset_id="c2340904-9db2-4830-99be-e9ed637ed400",
        split_id="63257a45-49c9-42a6-8b28-8e346eab2bee",
        neat_config=HERE / "configs/monk2.ini",
        reference_experiment="45e998d7-",
        n_generations=100,
    ),
}
N_EPOCHS_BACKPROP = 5
TORCH_THREADS = 1
DESCRIPTION = "Thesis Ch7 consistency experiment, 2026-10"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)


def load_training_data(dataset_id: str, split_id: str):
    with db.session_scope() as session:
        dataset = session.get(Dataset, UUID(dataset_id))
        split = session.get(DatasetSplit, UUID(split_id))
        assert split.dataset_id == dataset.id, "split does not belong to dataset"
        X_full, y_full = dataset.get_data()
        train_idx = list(split.train_indices)
        stored = split.scaler_params
        dataset_name = dataset.name

    X_train = X_full[train_idx].astype(np.float64)
    y_train = y_full[train_idx]
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0.0] = 1.0  # sklearn StandardScaler convention
    np.testing.assert_allclose(mean, stored["mean"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(scale, stored["scale"], rtol=0, atol=1e-9)
    return (X_train - mean) / scale, y_train, dataset_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=None,
                    help="override (dry runs only)")
    ap.add_argument("--name", default=None, help="override experiment name")
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    spec = DATASETS[args.dataset]
    n_gen = args.generations or spec["n_generations"]
    name = args.name or f"10seed-{args.dataset}-s{args.seed}"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db.init_db()
    X_train, y_train, dataset_name = load_training_data(spec["dataset_id"], spec["split_id"])

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(spec["neat_config"]))

    seed_everything(args.seed)
    t0 = time.time()
    population = DatabaseBackpropPopulation(
        config, X_train, y_train,
        experiment_name=name,
        dataset_name=dataset_name,
        description=DESCRIPTION,
        dataset_id=spec["dataset_id"],
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    with db.session_scope() as session:
        exp = session.get(Experiment, population.experiment_id)
        exp.split_id = UUID(spec["split_id"])
        exp.random_seed = args.seed
        exp.hardware_info = {
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "processor": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "device": "cpu",
            "torch_threads": TORCH_THREADS,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "neat": "0.92",
            "run_params": {
                "dataset": args.dataset, "seed": args.seed, "n_generations": n_gen,
                "n_epochs_backprop": N_EPOCHS_BACKPROP, "fitness_function": "auc",
                "neat_config": str(spec["neat_config"].relative_to(REPO)),
                "reference_experiment": spec["reference_experiment"],
                "code_commit": commit,
            },
        }
    print(f"EXPERIMENT_ID {population.experiment_id}", flush=True)

    population.run(auc_fitness, n=n_gen, nEpochs=N_EPOCHS_BACKPROP)
    wall = time.time() - t0
    print(json.dumps({"experiment_id": str(population.experiment_id),
                      "dataset": args.dataset, "seed": args.seed,
                      "generations": n_gen, "wall_clock_s": round(wall, 1),
                      "best_fitness": population.best_genome.fitness}), flush=True)


if __name__ == "__main__":
    main()
