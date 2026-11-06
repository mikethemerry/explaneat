"""``BackpropPopulation.backpropagate`` honours ``self.criterion``.

Before 2026-11-06 the ``criterion`` argument was stored and ignored: the loop
hardcoded ``nn.BCELoss()``.  The default path must stay bit-identical (that is
covered by ``test_neuralneat_mask_phantom.TestDefaultUnchanged``, which compares
the population against the original loop body); here we check a custom
criterion is actually used, and the survival target path.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.neuralneat import NeuralNeat, set_output_activation
from tests.test_core.test_neuralneat_mask_phantom import _data, config  # noqa: F401


class SpyCriterion(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0
        self.target_shapes = set()

    def forward(self, pred, target):
        self.calls += 1
        self.target_shapes.add(tuple(target.shape))
        return self.inner(pred, target)


def _manual(genome, config, xs, ys, crit, n_epochs, mask):
    net = NeuralNeat(genome, config, mask_phantom=mask)
    opt = optim.Adadelta(net.parameters(), lr=1.5)
    opt.zero_grad()
    for _ in range(n_epochs):
        loss = crit(net.forward(xs), ys)
        loss.backward()
        opt.step()
        opt.zero_grad()
    net.update_genome_weights()
    return net.genome


def test_default_criterion_is_bce(config):
    xs, ys = _data()
    pop = BackpropPopulation(config, xs, ys)
    assert isinstance(pop.criterion, nn.BCELoss)


def test_custom_criterion_is_called(config):
    random.seed(1)
    xs, ys = _data()
    spy = SpyCriterion(nn.MSELoss())
    pop = BackpropPopulation(config, xs, ys, criterion=spy)
    pop.backpropagate(pop.xs, pop.ys, nEpochs=3)
    # per genome: one pre-loss, 3 epochs, one post-loss
    assert spy.calls == len(pop.population) * (1 + 3 + 1)
    assert spy.target_shapes == {tuple(pop.ys.shape)}


def test_custom_criterion_drives_the_weights(config):
    """The trained genome equals manual training with that criterion, not BCE."""
    random.seed(2)
    xs, ys = _data()
    pop = BackpropPopulation(config, xs, ys, criterion=nn.MSELoss(), mask_phantom=True)
    before = copy.deepcopy(pop.population)
    pop.backpropagate(pop.xs, pop.ys, nEpochs=4)
    n_diff_bce = 0
    for k, g in before.items():
        mse_g = _manual(copy.deepcopy(g), config, pop.xs, pop.ys, nn.MSELoss(), 4, True)
        bce_g = _manual(copy.deepcopy(g), config, pop.xs, pop.ys, nn.BCELoss(), 4, True)
        for ck, c in pop.population[k].connections.items():
            assert c.weight == mse_g.connections[ck].weight
            n_diff_bce += c.weight != bce_g.connections[ck].weight
    assert n_diff_bce > 0


def test_survival_target_with_cox_criterion(config):
    """[N, 2] (time, event) target + identity output + CoxCriterion lowers the Cox loss."""
    from explaneat.survival import CoxCriterion, CoxRiskSets, cox_ph_loss

    random.seed(3)
    cfg = copy.deepcopy(config)
    set_output_activation(cfg, "identity")
    xs, _ = _data()
    rng = np.random.default_rng(0)
    x = xs.numpy()
    g_true = 1.5 * x[:, 0] - x[:, 1]
    T = rng.exponential(1.0 / np.exp(g_true))
    C = rng.exponential(2.0, size=len(T))
    y = np.column_stack([np.minimum(T, C), (T <= C).astype(float)])
    pop = BackpropPopulation(cfg, xs, y, criterion=CoxCriterion("efron"))
    assert pop.ys.shape == (len(y), 2)  # not unsqueezed
    rs = CoxRiskSets(y[:, 0], y[:, 1])
    before = copy.deepcopy(pop.population)
    pop.backpropagate(pop.xs, pop.ys, nEpochs=10)
    improved = 0
    for k, g in before.items():
        with torch.no_grad():
            l0 = float(cox_ph_loss(NeuralNeat(g, cfg).forward(pop.xs), rs))
            l1 = float(cox_ph_loss(NeuralNeat(pop.population[k], cfg).forward(pop.xs), rs))
        improved += l1 < l0
        assert np.isfinite(l1)
    assert improved >= len(before) // 2


def test_validation_fitness_function_is_opt_in(config):
    """run(validation_fitness_function=f) scores the validation fold with f; default reuses fitness."""
    from explaneat.evaluators.evaluators import auc_fitness

    xs, ys = _data()
    xv, yv = _data(seed=1)
    calls = []

    def val_fit(genomes, cfg, x, y, device):
        calls.append(x.shape[0])
        for g in genomes.values():
            g.fitness = 0.123

    for vf in (None, val_fit):
        random.seed(5)
        pop = BackpropPopulation(config, xs, ys, xs_val=xv, ys_val=yv)
        pop.run(auc_fitness, n=1, nEpochs=1, validation_fitness_function=vf)
        best = pop.best_genome
        if vf is None:
            assert best.validation_fitness != 0.123 and not calls
        else:
            assert calls == [xv.shape[0]]
            assert best.validation_fitness == 0.123
            assert best.fitness != 0.123  # training fitness restored


def test_skip_invalid_genomes_is_opt_in(config):
    import pytest

    from explaneat.core.errors import GenomeNotValidError

    random.seed(4)
    xs, ys = _data()
    for skip in (False, True):
        pop = BackpropPopulation(config, xs, ys, skip_invalid_genomes=skip)
        k0 = next(iter(pop.population))
        for c in pop.population[k0].connections.values():
            c.enabled = False  # output unreachable: NeuralNeat rejects it
        before = copy.deepcopy(pop.population[k0])
        if not skip:
            assert pop.skip_invalid_genomes is False
            with pytest.raises(GenomeNotValidError):
                pop.backpropagate(pop.xs, pop.ys, nEpochs=1)
        else:
            pop.backpropagate(pop.xs, pop.ys, nEpochs=1)
            after = pop.population[k0]
            assert {k: c.weight for k, c in after.connections.items()} == \
                {k: c.weight for k, c in before.connections.items()}
