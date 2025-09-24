import torch
import torch.nn as nn
import neat
from explaneat.core.neuralneat import NeuralNeat as nneat
from explaneat.core.errors import GenomeNotValidError
from sklearn.metrics import roc_auc_score
import numpy as np

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
        # Ensure ys is of shape [size, 1] for BCELoss
        if ys.dim() == 1:
            ys_reshaped = ys.view(-1, 1)
        else:
            ys_reshaped = ys
        # preds = []
        # for xi in xs:
        #     preds.append(net.activate(xi))
        # logger.info("Preds dtype is {}".format(preds.dtype))
        # logger.info("Preds dtype is {}".format(preds.dtype))
        # logger.info("Ys dtype is {}".format(ys.dtype))
        # logger.info("device is {}".format(device))
        genome.fitness = float(
            1.0 / loss(torch.tensor(preds).to(device), torch.tensor(ys_reshaped))
        )


def auc_fitness(genomes, config, xs, ys, device):
    """
    AUC-based fitness function for binary classification.
    Uses ROC AUC score as the fitness metric.
    """
    logger.info("Evaluating fitness using AUC metric")
    logger.info("Xs dtype: {}".format(xs.dtype))
    logger.info("ys dtype: {}".format(ys.dtype))

    for genome_id, genome in genomes.items():
        try:
            net = nneat(genome, config)
        except GenomeNotValidError:
            genome.fitness = 0.5  # Random performance for invalid genomes
            continue

        # Get predictions
        preds = net.forward(xs)

        # Convert to numpy for AUC calculation
        if isinstance(preds, torch.Tensor):
            preds_np = preds.detach().cpu().numpy()
        else:
            preds_np = np.array(preds)

        # Flatten if needed
        if preds_np.ndim > 1:
            preds_np = preds_np.flatten()

        # Convert ys to numpy if it's a tensor
        if isinstance(ys, torch.Tensor):
            ys_np = ys.detach().cpu().numpy()
        else:
            ys_np = np.array(ys)

        # Flatten if needed
        if ys_np.ndim > 1:
            ys_np = ys_np.flatten()

        try:
            # Calculate AUC
            auc_score = roc_auc_score(ys_np, preds_np)
            genome.fitness = float(auc_score)
            logger.debug(f"Genome {genome_id} AUC: {auc_score:.4f}")
        except ValueError as e:
            # Handle edge cases (e.g., all predictions same class)
            logger.warning(f"Could not calculate AUC for genome {genome_id}: {e}")
            genome.fitness = 0.5  # Random performance
