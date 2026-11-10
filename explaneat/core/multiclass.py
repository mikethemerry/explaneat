"""Multi-class (softmax) PropNEAT: criterion, fitness and an output-sink genome.

Everything here is opt-in; binary runs never touch it.

* ``SoftmaxCrossEntropyCriterion``: the backprop criterion for ``k`` output
  logits (identity output activation, ``set_output_activation(config,
  "identity")``).  ``BackpropPopulation`` stores ``ys`` as a float ``[N, 1]``
  tensor; the criterion flattens it to integer class labels, so the population
  needs no change.
* ``make_multiclass_fitness(metric)``: fitness = ``-cross_entropy`` (default)
  or accuracy, on the data the population passes (the training rows).
* ``OutputSinkGenome``: ``neat.DefaultGenome`` whose add-connection mutation
  never starts a connection at an output node.  neat-python allows
  output -> hidden connections; with several outputs that lets one output feed
  another through a hidden node, so an "output" would also be a hidden unit.
  With sink outputs every output is a pure readout and ``NeuralNeat`` can put
  all outputs in one final layer (see ``NodeMapping``).  Genomes are ordinary
  ``DefaultGenome`` data, so serialisation / loading are unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neat.genome import DefaultGenome
from neat.graphs import creates_cycle
from random import choice

from explaneat.core.errors import GenomeNotValidError
from explaneat.core.neuralneat import NeuralNeat

logger = logging.getLogger("experimenter.multiclass")

INVALID_CE = 50.0  # cross-entropy charged to a genome NeuralNeat rejects


def labels_of(target: torch.Tensor) -> torch.Tensor:
    """Integer class labels from a ``[N]`` / ``[N, 1]`` target tensor."""
    return target.reshape(-1).long()


class SoftmaxCrossEntropyCriterion(nn.Module):
    """``F.cross_entropy(logits [N, k], labels)`` with labels from a float ``[N, 1]`` target."""

    def forward(self, logits, target):
        return F.cross_entropy(logits, labels_of(target))

    @staticmethod
    def describe():
        return {"name": "softmax_cross_entropy"}


def make_multiclass_fitness(metric: str = "neg_ce"):
    """Fitness function ``f(genomes, config, xs, ys, device)`` for k-class genomes.

    ``metric`` = "neg_ce" (default): fitness = -mean cross-entropy (higher is
    better, smooth, the quantity backprop minimises); "accuracy": fraction
    correct.  A genome NeuralNeat rejects scores ``-INVALID_CE`` / 0.
    """
    if metric not in ("neg_ce", "accuracy"):
        raise ValueError(metric)

    def fitness(genomes, config, xs, ys, device):
        y = labels_of(ys)
        for _, genome in genomes.items():
            try:
                net = NeuralNeat(genome, config)
            except GenomeNotValidError:
                genome.fitness = -INVALID_CE if metric == "neg_ce" else 0.0
                continue
            with torch.no_grad():
                logits = net.forward(xs)
                if metric == "neg_ce":
                    ce = float(F.cross_entropy(logits, y))
                    genome.fitness = -ce if np.isfinite(ce) else -INVALID_CE
                else:
                    genome.fitness = float((logits.argmax(dim=1) == y).double().mean())

    fitness.metric = metric
    return fitness


class OutputSinkGenome(DefaultGenome):
    """DefaultGenome that never adds a connection *from* an output node.

    neat-python reads a genome's config section by class name, so load the
    config with ``neat.DefaultGenome`` (section ``[DefaultGenome]``) and then
    call ``use_output_sink_genome(config)``.
    """

    def mutate_add_connection(self, config):
        possible_outputs = list(self.nodes)
        out_node = choice(possible_outputs)
        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)
        if in_node in config.output_keys:
            return  # outputs are sinks
        key = (in_node, out_node)
        if key in self.connections:
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return
        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg


def use_output_sink_genome(config):
    """Make ``config`` create ``OutputSinkGenome`` genomes (same genome config)."""
    config.genome_type = OutputSinkGenome
    return config
