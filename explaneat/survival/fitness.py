"""NEAT fitness functions for Cox survival models.

All have ExplaNEAT's ``fitness_fn(genomes, config, xs, ys, device)`` shape and
set ``genome.fitness``; ``ys`` is the [N, 2] (time, event) target.

Fitness convention: **fitness = -(Cox loss)**, not ``1 / loss``.  The Cox loss
is a mean negative log partial likelihood on a log scale; ``1/loss`` is badly
behaved and makes ``fitness_threshold`` meaningless.  A constant-output genome
scores exactly ``-null_cox_loss(...)`` (about -5.7 on WHAS500, -5.98 on GBSG2),
so the fitness floor is data-dependent and known in advance: set
``fitness_threshold`` above any attainable value (e.g. 999) and stop on
generations, or relative to the null loss (``null_cox_loss`` - margin), never as
an absolute number carried over from a classification config.

Prefer ``cox_partial_likelihood_fitness``: selection and backprop then optimise
the same objective, and unlike a C-index fitness it does not trade calibration
for ranking.  ``cox_concordance_fitness`` is the ablation analogue of
``auc_fitness``.

Parsimony: ``make_cox_fitness(complexity_penalty=lam)`` subtracts
``lam * k / n_events`` where ``k`` counts enabled connections plus hidden
nodes.  With ``lam = 1`` this is the per-event AIC (``-2 pll + 2k`` halved and
divided by the event count), the natural events-per-parameter pressure.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from .cox import CoxRiskSets, cox_ph_loss
from .metrics import concordance

INVALID_FITNESS = -1e3


def _predict(net, xs):
    out = net.forward(xs)
    return out.reshape(-1) if isinstance(out, torch.Tensor) else torch.as_tensor(out).reshape(-1)


def genome_complexity(genome, config) -> int:
    """Enabled connections + hidden nodes (the output bias is unidentifiable in Cox)."""
    n_conn = sum(1 for c in genome.connections.values() if c.enabled)
    out_keys = set(config.genome_config.output_keys)
    n_hidden = sum(1 for k in genome.nodes if k not in out_keys)
    return n_conn + n_hidden


def make_cox_fitness(ties: str = "efron", complexity_penalty: float = 0.0,
                     nneat_cls=None) -> Callable:
    """Build a Cox partial-likelihood fitness function (fitness = -loss - penalty)."""

    def cox_fitness(genomes, config, xs, ys, device):
        from explaneat.core.errors import GenomeNotValidError
        from explaneat.core.neuralneat import NeuralNeat as _NN
        NN = nneat_cls or _NN
        rs = CoxRiskSets(ys[:, 0].detach().cpu().numpy(), ys[:, 1].detach().cpu().numpy(),
                         device=ys.device, dtype=ys.dtype)
        for _, genome in genomes.items():
            try:
                net = NN(genome, config)
            except GenomeNotValidError:
                genome.fitness = INVALID_FITNESS
                continue
            try:
                with torch.no_grad():
                    loss = float(cox_ph_loss(_predict(net, xs), rs, ties=ties))
            except Exception:
                genome.fitness = INVALID_FITNESS
                continue
            if not math.isfinite(loss):
                genome.fitness = INVALID_FITNESS
                continue
            fit = -loss
            if complexity_penalty:
                fit -= complexity_penalty * genome_complexity(genome, config) / rs.n_events
            genome.fitness = fit

    cox_fitness.__name__ = "cox_partial_likelihood_fitness"
    cox_fitness.ties = ties
    cox_fitness.complexity_penalty = complexity_penalty
    return cox_fitness


#: The primary survival fitness: -(Efron Cox loss), no parsimony penalty.
cox_partial_likelihood_fitness = make_cox_fitness("efron", 0.0)


def cox_concordance_fitness(genomes, config, xs, ys, device, nneat_cls=None):
    """Fitness = Harrell's C on g(x).  Ablation only: rank-only fitness degrades calibration."""
    from explaneat.core.errors import GenomeNotValidError
    from explaneat.core.neuralneat import NeuralNeat as _NN
    NN = nneat_cls or _NN
    t = ys[:, 0].detach().cpu().numpy()
    e = ys[:, 1].detach().cpu().numpy()
    for _, genome in genomes.items():
        try:
            net = NN(genome, config)
        except GenomeNotValidError:
            genome.fitness = 0.5
            continue
        try:
            with torch.no_grad():
                risk = _predict(net, xs).cpu().numpy()
            c = concordance(risk, t, e)
            genome.fitness = float(c) if np.isfinite(c) else 0.5
        except Exception:
            genome.fitness = 0.5
