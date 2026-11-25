"""Fitness for the self-play value network: minus the soft-label BCE on the buffer.

``neg_bce_fitness(genomes, config, xs, ys, device)`` follows the PropNEAT fitness-function
signature (``BackpropPopulation.run``). Each genome's NeuralNeat (``mask_phantom=True``, the
same network backprop trained) predicts P(side to move wins) for every buffer row under
``torch.no_grad()``; fitness = -mean BCE against the soft labels (1 / 0 / 0.5 draws, or the
mean label of a deduplicated position), predictions clamped to ``[EPS, 1 - EPS]``.

Invalid genomes (NeuralNeat rejects them: no input-output path) and genomes whose
predictions or loss are not finite (NaN/inf weights, e.g. after Adadelta diverges) are scored
by ``INVALID_RULE``: ``INVALID_MARGIN`` below the worst valid fitness of the same call, or, if
no genome in the call is valid, ``INVALID_MARGIN`` below ``CLAMP_FLOOR`` (the fitness of a
maximally wrong clamped prediction, log(EPS) ~ -16.12). A large constant such as -1e9 would
break neat-python's reproduction: ``DefaultReproduction.reproduce`` normalises each species'
mean fitness by (max - min) over the whole population, so one -1e9 genome makes every valid
species' adjusted fitness ~1 (uniform spawn) and crushes the mean of any species holding it.

Fitness is only comparable within one round: the buffer, and so the target, changes
between rounds.
"""
import math

import netfn  # noqa: F401  (forces EXPLANEAT_DEVICE=cpu before explaneat is imported)

import torch  # noqa: E402

from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

EPS = 1e-7
# fitness of a maximally wrong clamped prediction on a hard label: -(-log(EPS))
CLAMP_FLOOR = math.log(EPS)
INVALID_MARGIN = 1e-3
INVALID_RULE = "min(valid fitness in the call) - INVALID_MARGIN, else CLAMP_FLOOR - INVALID_MARGIN"


def bce(pred, ys) -> float:
    """Mean soft-label binary cross-entropy, predictions clamped to [EPS, 1 - EPS]."""
    p = torch.clamp(pred.reshape(-1), EPS, 1.0 - EPS)
    y = ys.reshape(-1).to(p.dtype)
    return float(-(y * torch.log(p) + (1.0 - y) * torch.log1p(-p)).mean())


def invalid_fitness(valid_fitnesses) -> float:
    """The ``INVALID_RULE`` score for an invalid or non-finite genome."""
    base = min(valid_fitnesses) if valid_fitnesses else CLAMP_FLOOR
    return base - INVALID_MARGIN


def neg_bce_fitness(genomes, config, xs, ys, device=None):
    valid, invalid = [], []
    for _, genome in genomes.items():
        try:
            net = NeuralNeat(genome, config, mask_phantom=True)
        except GenomeNotValidError:
            invalid.append(genome)
            continue
        net.eval()
        with torch.no_grad():
            pred = net.forward(xs)
        if not bool(torch.isfinite(pred).all()):
            invalid.append(genome)
            continue
        loss = bce(pred, ys)
        if not math.isfinite(loss):
            invalid.append(genome)
            continue
        genome.fitness = -loss
        valid.append(genome.fitness)
    if invalid:
        fit = invalid_fitness(valid)
        for genome in invalid:
            genome.fitness = fit
