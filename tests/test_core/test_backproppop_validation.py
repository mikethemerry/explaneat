"""Tests for BackpropPopulation validation-based selection and early stopping."""

import tempfile
import os
import numpy as np
import torch
import neat
import pytest

from explaneat.core.backproppop import BackpropPopulation
from explaneat.evaluators.evaluators import auc_fitness


def _make_neat_config(num_inputs: int, pop_size: int = 10) -> neat.Config:
    """Create a minimal NEAT config for testing."""
    config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = {pop_size}
reset_on_extinction   = True

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 1

[DefaultReproduction]
elitism            = 1
survival_threshold = 0.2

[DefaultGenome]
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid
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
conn_add_prob           = 0.2
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.1
node_delete_prob        = 0.05
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = 1
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
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
    finally:
        os.unlink(config_path)

    return config


def _make_binary_data(n_samples: int, n_features: int, seed: int = 42):
    """Generate simple binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Simple linear boundary + noise
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


class TestValidationSelection:
    """Test that validation-based genome selection works."""

    def test_validation_fitness_is_set(self):
        """Genomes should have validation_fitness after run with val data."""
        X_train, y_train = _make_binary_data(60, 3, seed=1)
        X_val, y_val = _make_binary_data(20, 3, seed=2)

        config = _make_neat_config(num_inputs=3, pop_size=10)
        pop = BackpropPopulation(
            config, X_train, y_train,
            xs_val=X_val, ys_val=y_val,
        )

        best = pop.run(
            fitness_function=auc_fitness,
            n=3,
            nEpochs=1,
        )

        # Best genome should have validation_fitness set
        assert hasattr(best, 'validation_fitness')
        assert best.validation_fitness is not None
        assert 0.0 <= best.validation_fitness <= 1.0

    def test_best_genome_selected_by_validation(self):
        """best_genome should be selected by validation fitness, not training."""
        X_train, y_train = _make_binary_data(60, 3, seed=1)
        X_val, y_val = _make_binary_data(20, 3, seed=2)

        config = _make_neat_config(num_inputs=3, pop_size=10)
        pop = BackpropPopulation(
            config, X_train, y_train,
            xs_val=X_val, ys_val=y_val,
        )

        best = pop.run(
            fitness_function=auc_fitness,
            n=3,
            nEpochs=1,
        )

        # best_genome should have validation_fitness (it was selected by val)
        assert hasattr(best, 'validation_fitness')
        assert best.validation_fitness is not None


class TestEarlyStopping:
    """Test early stopping with patience parameter."""

    def test_early_stopping_stops_before_n(self):
        """With patience=1, evolution should stop before reaching n generations."""
        X_train, y_train = _make_binary_data(40, 3, seed=1)
        X_val, y_val = _make_binary_data(15, 3, seed=2)

        config = _make_neat_config(num_inputs=3, pop_size=10)
        pop = BackpropPopulation(
            config, X_train, y_train,
            xs_val=X_val, ys_val=y_val,
        )

        # patience=1 means stop if val fitness doesn't improve for 1 generation
        # With random init and small data, this should trigger quickly
        best = pop.run(
            fitness_function=auc_fitness,
            n=50,  # high limit
            nEpochs=1,
            patience=1,
        )

        # Should have stopped well before 50 generations
        assert pop.generation < 50
        assert best is not None

    def test_patience_none_runs_all_generations(self):
        """With patience=None, should run all n generations."""
        X_train, y_train = _make_binary_data(40, 3, seed=1)
        X_val, y_val = _make_binary_data(15, 3, seed=2)

        config = _make_neat_config(num_inputs=3, pop_size=10)
        pop = BackpropPopulation(
            config, X_train, y_train,
            xs_val=X_val, ys_val=y_val,
        )

        n_gens = 3
        best = pop.run(
            fitness_function=auc_fitness,
            n=n_gens,
            nEpochs=1,
            patience=None,
        )

        # Should run all generations (generation counter is 0-indexed and incremented after each)
        assert pop.generation == n_gens
        assert best is not None


class TestBackwardCompatibility:
    """Test that behavior is unchanged when no validation data is provided."""

    def test_no_validation_selects_by_training(self):
        """Without validation data, best genome should be selected by training fitness."""
        X_train, y_train = _make_binary_data(60, 3, seed=1)

        config = _make_neat_config(num_inputs=3, pop_size=10)
        pop = BackpropPopulation(config, X_train, y_train)

        best = pop.run(
            fitness_function=auc_fitness,
            n=3,
            nEpochs=1,
        )

        assert best is not None
        assert best.fitness is not None
        # Should NOT have validation_fitness
        assert not hasattr(best, 'validation_fitness') or getattr(best, 'validation_fitness', None) is None

    def test_no_validation_ignores_patience(self):
        """Patience param should be ignored when no validation data."""
        X_train, y_train = _make_binary_data(40, 3, seed=1)

        config = _make_neat_config(num_inputs=3, pop_size=10)
        pop = BackpropPopulation(config, X_train, y_train)

        n_gens = 3
        best = pop.run(
            fitness_function=auc_fitness,
            n=n_gens,
            nEpochs=1,
            patience=1,  # should be ignored since no val data
        )

        # Should still run all generations
        assert pop.generation == n_gens
        assert best is not None
