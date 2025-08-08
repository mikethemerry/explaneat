import torch
import torch.nn as nn
import neat
from explaneat.core.neuralneat import NeuralNeat as nneat
from explaneat.core.errors import GenomeNotValidError


import logging

logger = logging.getLogger("experimenter.evaluators")


def binary_cross_entropy(genomes, config, xs, ys, device):

    logger.info("Xs dtype{}".format(xs.dtype))
    logger.info("ys dtype{}".format(ys.dtype))
    loss = nn.BCELoss()
    loss = loss.to(device)
    for genome_id, genome in genomes.items():
        try:
            net = nneat(genome, config)
        except GenomeNotValidError:
            genome.fitness = 0
            continue
        preds = net.forward(xs)
        # preds = []
        # for xi in xs:
        #     preds.append(net.activate(xi))
        # logger.info("Preds dtype is {}".format(preds.dtype))
        # logger.info("Preds dtype is {}".format(preds.dtype))
        # logger.info("Ys dtype is {}".format(ys.dtype))
        # logger.info("device is {}".format(device))
        genome.fitness = float(
            1.0 / loss(torch.tensor(preds).to(device), torch.tensor(ys))
        )
