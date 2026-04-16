"""Config resolution utilities for ExplaNEAT training configs.

Merges three layers to produce a resolved training config:

    defaults -> template -> per-experiment overrides

The resolved config is a grouped dict with keys ``training``, ``neat``, and
``backprop``. It can be rendered to NEAT-python config text via
``config_to_neat_text``.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "training": {
        "population_size": 150,
        "n_generations": 10,
        "n_epochs_backprop": 5,
        "fitness_function": "bce",
    },
    "neat": {
        "bias_mutate_rate": 0.7,
        "bias_mutate_power": 0.5,
        "bias_replace_rate": 0.1,
        "weight_mutate_rate": 0.8,
        "weight_mutate_power": 0.5,
        "weight_replace_rate": 0.1,
        "enabled_mutate_rate": 0.01,
        "node_add_prob": 0.15,
        "node_delete_prob": 0.05,
        "conn_add_prob": 0.3,
        "conn_delete_prob": 0.1,
        "compatibility_threshold": 3.0,
        "compatibility_disjoint_coefficient": 1.0,
        "compatibility_weight_coefficient": 0.5,
        "max_stagnation": 15,
        "species_elitism": 2,
        "elitism": 2,
        "survival_threshold": 0.2,
    },
    "backprop": {
        "learning_rate": 1.5,
        "optimizer": "adadelta",
    },
}


def _apply_layer(
    base: Dict[str, Dict[str, Any]],
    layer: Optional[Mapping[str, Any]],
) -> None:
    """Apply ``layer`` into ``base`` in place.

    Only groups that already exist in ``base`` (i.e. the known groups from
    ``DEFAULT_CONFIG``) are considered. Unknown groups are silently dropped.
    Within a known group, only dict values are merged; per-key updates
    preserve existing keys that are not overridden.
    """
    if not layer:
        return
    for group_name, group_values in layer.items():
        if group_name not in base:
            # Unknown group — ignore.
            continue
        if not isinstance(group_values, Mapping):
            # Ignore non-dict values for known groups.
            continue
        for key, value in group_values.items():
            base[group_name][key] = value


def resolve_config(
    template: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Resolve a training config by merging defaults, template, and overrides.

    Layers (highest precedence last):
        1. ``DEFAULT_CONFIG`` (deep-copied so the default is never mutated)
        2. ``template`` — partial dict keyed by group name
        3. ``overrides`` — partial dict keyed by group name

    Unknown groups are ignored. Partial group overrides preserve other keys
    in that group.
    """
    resolved = copy.deepcopy(DEFAULT_CONFIG)
    _apply_layer(resolved, template)
    _apply_layer(resolved, overrides)
    return resolved


def config_to_neat_text(
    config: Mapping[str, Mapping[str, Any]],
    num_inputs: int,
    num_outputs: int,
) -> str:
    """Render a resolved config to NEAT-python INI-style config text.

    ``num_inputs``/``num_outputs`` are runtime-computed from the dataset and
    are not sourced from ``config``. ``population_size`` comes from
    ``config["training"]["population_size"]``. Mutation rates, topology
    probabilities, species/stagnation/reproduction values are read from
    ``config["neat"]``. Fixed bookkeeping values (activation, aggregation,
    init stats, bounds) stay hardcoded here.
    """
    training = config["training"]
    neat = config["neat"]

    pop_size = training["population_size"]

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
bias_mutate_power       = {neat["bias_mutate_power"]}
bias_mutate_rate        = {neat["bias_mutate_rate"]}
bias_replace_rate       = {neat["bias_replace_rate"]}
compatibility_disjoint_coefficient = {neat["compatibility_disjoint_coefficient"]}
compatibility_weight_coefficient   = {neat["compatibility_weight_coefficient"]}
conn_add_prob           = {neat["conn_add_prob"]}
conn_delete_prob        = {neat["conn_delete_prob"]}
enabled_default         = True
enabled_mutate_rate     = {neat["enabled_mutate_rate"]}
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = {neat["node_add_prob"]}
node_delete_prob        = {neat["node_delete_prob"]}
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
weight_mutate_power     = {neat["weight_mutate_power"]}
weight_mutate_rate      = {neat["weight_mutate_rate"]}
weight_replace_rate     = {neat["weight_replace_rate"]}

[DefaultSpeciesSet]
compatibility_threshold = {neat["compatibility_threshold"]}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {neat["max_stagnation"]}
species_elitism      = {neat["species_elitism"]}

[DefaultReproduction]
elitism            = {neat["elitism"]}
survival_threshold = {neat["survival_threshold"]}
"""
