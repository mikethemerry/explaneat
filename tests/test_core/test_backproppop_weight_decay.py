"""``BackpropPopulation(weight_decay=...)`` passes an L2 penalty to Adadelta.

The default (0.0) must leave training bit-identical to the original loop
(also covered by ``test_neuralneat_mask_phantom.TestDefaultUnchanged``, which
runs with the default); a positive value must equal manual training with
``optim.Adadelta(..., weight_decay=wd)`` and shrink the weights.
"""

import copy
import random

import torch.optim as optim

from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.neuralneat import NeuralNeat
from tests.test_core.test_neuralneat_mask_phantom import _data, config  # noqa: F401


def _manual(genome, config, xs, ys, crit, n_epochs, wd):
    net = NeuralNeat(genome, config, mask_phantom=True)
    opt = optim.Adadelta(net.parameters(), lr=1.5, weight_decay=wd)
    opt.zero_grad()
    for _ in range(n_epochs):
        loss = crit(net.forward(xs), ys)
        loss.backward()
        opt.step()
        opt.zero_grad()
    net.update_genome_weights()
    return net.genome


def _weights(g):
    return {k: c.weight for k, c in g.connections.items()}


def test_default_is_zero_and_matches_plain_adadelta(config):
    random.seed(5)
    xs, ys = _data()
    pop = BackpropPopulation(config, xs, ys, mask_phantom=True)
    assert pop.weight_decay == 0.0
    before = copy.deepcopy(pop.population)
    pop.backpropagate(pop.xs, pop.ys, nEpochs=4)
    for k, g in before.items():
        ref = _manual(copy.deepcopy(g), config, pop.xs, pop.ys, pop.criterion, 4, 0)
        assert _weights(pop.population[k]) == _weights(ref)


def test_weight_decay_is_used_and_shrinks(config):
    random.seed(6)
    xs, ys = _data()
    wd = 0.5
    pop = BackpropPopulation(config, xs, ys, mask_phantom=True, weight_decay=wd)
    before = copy.deepcopy(pop.population)
    pop.backpropagate(pop.xs, pop.ys, nEpochs=4)
    n_diff = 0
    norm_wd = norm_plain = 0.0
    for k, g in before.items():
        ref_wd = _manual(copy.deepcopy(g), config, pop.xs, pop.ys, pop.criterion, 4, wd)
        ref_0 = _manual(copy.deepcopy(g), config, pop.xs, pop.ys, pop.criterion, 4, 0)
        assert _weights(pop.population[k]) == _weights(ref_wd)
        n_diff += _weights(ref_wd) != _weights(ref_0)
        norm_wd += sum(w * w for w in _weights(ref_wd).values())
        norm_plain += sum(w * w for w in _weights(ref_0).values())
    assert n_diff > 0
    assert norm_wd < norm_plain
