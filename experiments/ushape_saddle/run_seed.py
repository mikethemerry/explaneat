"""Run one seeded PropNEAT evolution for the U-shape and saddle study.

The 10-seed harness (experiments/10seed/run_seed.py: seeding, torch threads, the unchanged
DatabaseBackpropPopulation.run with auc_fitness, 5 backprop epochs) on one of this study's
datasets, read from datasets.json (written by make_datasets.py). Config: configs/syn10.ini, the
Heart thesis config with num_inputs 10 (pop 250, full_direct start, ReLU, sigmoid output).

    PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
        uv run python experiments/ushape_saddle/run_seed.py u 0 [--generations 100] [--name ...] \
            [--hidden-activation sigmoid]

``--hidden-activation sigmoid`` trains with sigmoid hidden units instead of ReLU
(``neuralneat.set_hidden_activation``; recorded as config_json["training"]["hidden_activation"]).
Omitted, the run is the original ReLU run, unchanged.
"""
import argparse
import json
import logging
import os
import platform
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
sys.path.insert(0, str(REPO / "experiments" / "10seed"))

from run_seed import N_EPOCHS_BACKPROP, TORCH_THREADS, load_training_data, seed_everything  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import Experiment  # noqa: E402
from explaneat.db.population import DatabaseBackpropPopulation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

N_GENERATIONS = 100
CONFIG = HERE / "configs" / "syn10.ini"
DESCRIPTION = "Synthetic U-shape and saddle study (experiments/ushape_saddle), exploratory"


def main():
    reg = json.loads((HERE / "datasets.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(reg))
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=N_GENERATIONS)
    ap.add_argument("--name", default=None)
    ap.add_argument("--hidden-activation", default=None, choices=["relu", "sigmoid"])
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    spec = reg[args.task]
    name = args.name or f"{spec['label']}-s{args.seed}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_db()
    X_train, y_train, dataset_name = load_training_data(spec["dataset_id"], spec["split_id"])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(CONFIG))
    if args.hidden_activation and args.hidden_activation != "relu":
        from explaneat.core.neuralneat import set_hidden_activation
        set_hidden_activation(config, args.hidden_activation)
    seed_everything(args.seed)
    t0 = time.time()
    population = DatabaseBackpropPopulation(config, X_train, y_train, experiment_name=name, dataset_name=dataset_name,
                                            description=DESCRIPTION, dataset_id=spec["dataset_id"], mask_phantom=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    with db.session_scope() as session:
        exp = session.get(Experiment, population.experiment_id)
        exp.split_id = UUID(spec["split_id"])
        exp.random_seed = args.seed
        exp.hardware_info = {
            "platform": os.name, "cpu_count": os.cpu_count(), "machine": platform.machine(),
            "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "device": "cpu", "torch_threads": TORCH_THREADS, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "neat": "0.92",
            "run_params": {"dataset": args.task, "label": spec["label"], "seed": args.seed, "n_generations": args.generations,
                           "n_epochs_backprop": N_EPOCHS_BACKPROP, "fitness_function": "auc",
                           "neat_config": str(CONFIG.relative_to(REPO)), "code_commit": commit, "mask_phantom": False,
                           **({"hidden_activation": args.hidden_activation}
                              if args.hidden_activation and args.hidden_activation != "relu" else {})},
        }
    print(f"EXPERIMENT_ID {population.experiment_id}", flush=True)
    population.run(auc_fitness, n=args.generations, nEpochs=N_EPOCHS_BACKPROP)
    print(json.dumps({"experiment_id": str(population.experiment_id), "task": args.task, "seed": args.seed,
                      "generations": args.generations, "wall_clock_s": round(time.time() - t0, 1),
                      "best_fitness": population.best_genome.fitness}), flush=True)


if __name__ == "__main__":
    main()
