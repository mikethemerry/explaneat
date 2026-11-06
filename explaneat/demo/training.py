"""Brief, seeded PropNEAT runs for the demo (in memory, or recorded in a DB)."""

from __future__ import annotations

import copy
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import neat
import numpy as np
import torch

from ..core.backproppop import BackpropPopulation
from ..core.neuralneat import NeuralNeat

# ReLU hidden units, sigmoid output (NeuralNeat applies it), backprop trains
# the weights so NEAT's own weight mutation is off. Two initial hidden nodes
# give the explanation something to compose from generation 0.
CLASSIFICATION_CONFIG = """
[NEAT]
fitness_criterion = max
fitness_threshold = 1e9
pop_size = {pop_size}
reset_on_extinction = True

[DefaultGenome]
activation_default = relu
activation_mutate_rate = 0.0
activation_options = relu
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.0
bias_mutate_rate = 0.0
bias_replace_rate = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.4
conn_delete_prob = 0.2
enabled_default = True
enabled_mutate_rate = 0.05
feed_forward = True
initial_connection = full_direct
node_add_prob = 0.3
node_delete_prob = 0.1
num_hidden = 2
num_inputs = {num_inputs}
num_outputs = 1
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.0
weight_mutate_rate = 0.0
weight_replace_rate = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
"""


def load_config(text: str, work_dir: Path) -> neat.Config:
    """neat-python only reads configs from a file: write one, read it, delete it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".ini", dir=work_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, name)
    finally:
        os.unlink(name)
    config.explaneat_config_text = text
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _HoldOutInvalid:
    """Mutation can delete every input->output path; NeuralNeat rejects such
    genomes, so they sit out backprop (and score as invalid) instead of
    crashing the run."""

    def backpropagate(self, xs, ys, nEpochs=5):
        held = {}
        for k, g in list(self.population.items()):
            try:
                NeuralNeat(g, self.config)
            except Exception:
                held[k] = self.population.pop(k)
        try:
            return super().backpropagate(xs, ys, nEpochs=nEpochs)
        finally:
            self.population.update(held)


class DemoPopulation(_HoldOutInvalid, BackpropPopulation):
    pass


def safe_auc_fitness(genomes, config, xs, ys, device):
    """Training-set ROC AUC; invalid genomes score 0.5 (chance)."""
    from sklearn.metrics import roc_auc_score

    y = ys.detach().cpu().numpy().ravel()
    for _, g in genomes.items():
        try:
            with torch.no_grad():
                p = NeuralNeat(g, config).forward(xs).detach().cpu().numpy().ravel()
            g.fitness = float(roc_auc_score(y, p)) if np.all(np.isfinite(p)) else 0.5
        except Exception:
            g.fitness = 0.5


def train_classifier(X, y, seed: int, generations: int, pop_size: int, epochs: int,
                     work_dir: Path, database_url: Optional[str] = None,
                     experiment_name: Optional[str] = None):
    """Return (best_genome, config, info). ``database_url`` records the run."""
    config = load_config(CLASSIFICATION_CONFIG.format(pop_size=pop_size, num_inputs=X.shape[1]),
                         work_dir)
    seed_everything(seed)
    info = {"seed": seed, "generations": generations, "pop_size": pop_size,
            "epochs_per_generation": epochs, "mode": "in-memory"}
    if database_url:
        from ..core.ancestry_reporter import AncestryReporter
        from ..db.population import DatabaseBackpropPopulation

        class DemoDbPopulation(_HoldOutInvalid, DatabaseBackpropPopulation):
            pass

        pop = DemoDbPopulation(config, X, y, experiment_name=experiment_name,
                               dataset_name="demo-hinge (synthetic)",
                               description="explaneat demo: synthetic hinge classification",
                               database_url=database_url, ancestry_reporter=AncestryReporter(),
                               mask_phantom=True)
        pop.run(safe_auc_fitness, n=generations, nEpochs=epochs)
        info.update(mode="database", experiment_id=str(pop.experiment_id))
    else:
        pop = DemoPopulation(config, X, y, mask_phantom=True)
        pop.run(safe_auc_fitness, n=generations, nEpochs=epochs)
    best = copy.deepcopy(pop.best_genome)
    info["train_fitness_auc"] = float(best.fitness) if best.fitness is not None else None
    return best, config, info


def predict(genome, config, X) -> np.ndarray:
    with torch.no_grad():
        out = NeuralNeat(genome, config).forward(torch.as_tensor(X, dtype=torch.float64))
    return out.numpy().reshape(len(X), -1)[:, 0]
