"""Fitness for the self-play value network: minus the soft-label BCE on the buffer.

``neg_bce_fitness(genomes, config, xs, ys, device)`` follows the PropNEAT fitness-function
signature (``BackpropPopulation.run``). Each genome's NeuralNeat (``mask_phantom=True``, the
same network backprop trained) predicts P(side to move wins) for every buffer row under
``torch.no_grad()``; fitness = -mean BCE against the soft labels (1 / 0 / 0.5 draws, or the
mean label of a deduplicated position), predictions clamped to ``[EPS, 1 - EPS]``. A genome
NeuralNeat rejects (no input-output path) gets ``INVALID_FITNESS``.

Fitness is only comparable within one round: the buffer, and so the target, changes
between rounds.
"""
import netfn  # noqa: F401  (forces EXPLANEAT_DEVICE=cpu before explaneat is imported)

import torch  # noqa: E402

from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

EPS = 1e-7
INVALID_FITNESS = -1e9


def bce(pred, ys) -> float:
    """Mean soft-label binary cross-entropy, predictions clamped to [EPS, 1 - EPS]."""
    p = torch.clamp(pred.reshape(-1), EPS, 1.0 - EPS)
    y = ys.reshape(-1).to(p.dtype)
    return float(-(y * torch.log(p) + (1.0 - y) * torch.log1p(-p)).mean())


def neg_bce_fitness(genomes, config, xs, ys, device=None):
    for _, genome in genomes.items():
        try:
            net = NeuralNeat(genome, config, mask_phantom=True)
        except GenomeNotValidError:
            genome.fitness = INVALID_FITNESS
            continue
        net.eval()
        with torch.no_grad():
            pred = net.forward(xs)
        genome.fitness = -bce(pred, ys)
