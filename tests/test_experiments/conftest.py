"""Fixtures for the experiment tooling tests (experiments/churn/*).

`experiments/` is not a package, so the churn directory is put on sys.path.
"""

import os
import sys
import tempfile
from pathlib import Path

import neat
import pytest
from churn_test_helpers import build_genome  # noqa: E402

CHURN_DIR = Path(__file__).resolve().parents[2] / "experiments" / "churn"
if str(CHURN_DIR) not in sys.path:
    sys.path.insert(0, str(CHURN_DIR))

CONFIG_TEXT = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1.0
pop_size              = 10
reset_on_extinction   = False

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2

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
bias_mutate_rate        = 0.0
bias_replace_rate       = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.0
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.5
node_delete_prob        = 0.5
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
weight_mutate_rate      = 0.0
weight_replace_rate     = 0.0
num_inputs              = 3
num_hidden              = 0
num_outputs             = 1
"""


@pytest.fixture
def small_config():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False)
    f.write(CONFIG_TEXT)
    f.close()
    try:
        yield neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, f.name)
    finally:
        os.unlink(f.name)


@pytest.fixture
def recipient_genome():
    # hidden 1 <- (-1, -2); output <- (1, -3)
    return build_genome(1, {0: -0.3, 1: 0.2},
                        {(-1, 1): 0.8, (-2, 1): -0.5, (1, 0): 1.3, (-3, 0): 0.7})


@pytest.fixture
def donor_genome():
    # hidden 5 <- (-1, -2); hidden 6 <- (5, -3); output <- (6, -3, -1)
    return build_genome(2, {0: 0.1, 5: 0.4, 6: -0.2},
                        {(-1, 5): 1.1, (-2, 5): 0.9, (5, 6): -1.4, (-3, 6): 0.6,
                         (6, 0): 2.0, (-3, 0): -0.4, (-1, 0): 0.3})
