"""Shared helpers for AAAI-27 paper result runs.

Frozen paper models (verified in verify_ground_truth.py):

- Adult:  genome 4bc8fa07-bbf5-48e4-96a8-77be8b577ba3, experiment "No pressure"
          (dbab685c-e9c8-4118-bb4e-0fbdfab4c4c9). NOTE: the working-draft task
          doc attributed this model to experiment "Adult - OHE - 250-100-5"
          (5c059459); that attribution is wrong — see verify_ground_truth.py.
          Split 1df50169 (== task's 7c025383: identical indices and scaler).
- Heart:  genome 9022a9eb-ecf9-41ba-b138-28ed4d131c9d, experiment "Heart Binary"
          (c8f3805e-bbb7-4eb9-aa07-cff73176d98c), split 165cebbd.

All evaluation uses the stored splits and stored scalers. Never re-split.
"""
import copy
import uuid

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from explaneat.db import db
from explaneat.db.models import Experiment, Genome, Dataset, DatasetSplit, Explanation
from explaneat.core.config_utils import load_neat_config
from explaneat.core.neuralneat import NeuralNeat

SEED = 42
N_BOOTSTRAP = 10_000

ADULT = {
    "label": "adult",
    "genome_id": "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3",
    "experiment_id": "dbab685c-e9c8-4118-bb4e-0fbdfab4c4c9",
    "task_split_id": "7c025383-db44-4825-9ee2-a709e05c15ff",  # index-identical to exp split
}
HEART = {
    "label": "heart",
    "genome_id": "9022a9eb-ecf9-41ba-b138-28ed4d131c9d",
    "experiment_id": "c8f3805e-bbb7-4eb9-aa07-cff73176d98c",
    "task_split_id": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
}


def load_model(session, spec):
    """Load (neat_genome, config, X_scaled, y, split, explanation) for a paper model."""
    exp = session.query(Experiment).filter_by(id=uuid.UUID(spec["experiment_id"])).first()
    genome_row = session.query(Genome).filter_by(id=uuid.UUID(spec["genome_id"])).first()
    split = session.query(DatasetSplit).filter_by(id=exp.split_id).first()
    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first()
    explanation = session.query(Explanation).filter_by(genome_id=genome_row.id).first()

    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    genome = genome_row.to_neat_genome(config)

    X, y = dataset.get_data()
    mean = np.array(split.scaler_params["mean"])
    scale = np.array(split.scaler_params["scale"])
    X_scaled = (X - mean) / scale

    return genome, config, X_scaled, y, split, explanation, dataset


def predict(genome, config, X):
    """Forward pass through NeuralNeat (the training engine). Returns 1-D scores."""
    net = NeuralNeat(genome, config)
    xt = torch.tensor(np.asarray(X), dtype=torch.float64)
    return net.forward(xt).detach().numpy().ravel()


def ablated_genome(genome, edges):
    """Copy of genome with the given (from_key, to_key) connection genes disabled.

    Measurement-only: the copy is never persisted. Raises if an edge is absent.
    """
    g = copy.deepcopy(genome)
    for edge in edges:
        key = tuple(edge)
        if key not in g.connections:
            raise KeyError(f"edge {key} not in genome connections")
        g.connections[key].enabled = False
    return g


def paired_bootstrap_delta(y_true, scores_a, scores_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap of AUC(b) - AUC(a) deltas on the same resamples.

    Resamples rows once per iteration, computes AUC for BOTH score vectors on
    the same resample, takes the difference. Resamples with a single class are
    skipped. Returns (deltas_array, n_skipped).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            skipped += 1
            continue
        deltas.append(roc_auc_score(yt, scores_b[idx]) - roc_auc_score(yt, scores_a[idx]))
    return np.array(deltas), skipped


def bootstrap_auc_ci(y_true, scores, n_boot=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap CI for a single model's AUC. Returns (aucs, n_skipped)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            skipped += 1
            continue
        aucs.append(roc_auc_score(yt, scores[idx]))
    return np.array(aucs), skipped


def pct_ci(values, lo=2.5, hi=97.5):
    return np.percentile(values, lo), np.percentile(values, hi)
