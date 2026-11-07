"""Synthetic data with a known structure (no downloads, no real records)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def hinge_classification(n: int = 600, d: int = 4, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Binary task whose true logit has a hinge (V-shape) in x0 and a linear x1.

        logit(x) = 4 * (|x0| - 0.5) + 1.5 * x1,   x ~ U(-1, 1)^d

    ``|x0| = relu(x0) + relu(-x0)`` is the known non-linear feature: a linear
    model cannot use x0 at all (it has no linear main effect), while a network
    with two ReLU units can. x2.. are noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    logit = 4.0 * (np.abs(X[:, 0]) - 0.5) + 1.5 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    truth = {"feature": "hinge abs(x0) = relu(x0) + relu(-x0), plus linear x1",
             "logit": "4 * (abs(x0) - 0.5) + 1.5 * x1", "noise_features": list(range(2, d))}
    return X, y, truth


def train_test_split_indices(n: int, test_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed + 10_000)
    idx = rng.permutation(n)
    k = int(round(n * (1 - test_frac)))
    return np.sort(idx[:k]), np.sort(idx[k:])
