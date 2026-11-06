"""Simulated survival data with a known log-risk: the implementation sanity check.

If a Cox-loss network cannot beat linear Cox where the true log-risk is
non-linear, the network or the loss plumbing is broken.  Run this before any
real data (brief section 7): on real tabular data a correct and a broken
pipeline both look like "parity with Cox".

Design follows the DeepSurv simulation (Katzman et al. 2018):

    h(t|x) = h0 * exp(g_true(x))
    T ~ Exponential(rate = h0 * exp(g_true(x)))
    C ~ Uniform(0, c_max);  observed = min(T, C),  event = 1{T <= C}

Regimes:
    "linear"     g_true(x) = x . beta (5 active of d)  -> Cox correctly specified
    "nonlinear"  g_true(x) = log(5) * exp(-(x0^2 + x1^2) / (2 * 0.5^2))
                 a Gaussian bump with zero linear main effect -> Cox is blind to it

Identical to the reference ``simulate.py`` (same RNG consumption), so seeds
reproduce the brief's numbers.
"""

from __future__ import annotations

import numpy as np


def simulate(n: int = 4000, d: int = 10, mode: str = "nonlinear",
             censor_frac: float = 0.30, seed: int = 0):
    """Return ``X [n, d], time [n], event [n] (bool), g_true [n]``."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))

    if mode == "linear":
        beta = np.zeros(d)
        beta[:5] = rng.normal(0, 1, 5)
        g = X @ beta
    elif mode == "nonlinear":
        r, lam = 0.5, 5.0
        g = np.log(lam) * np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) / (2 * r ** 2))
    else:
        raise ValueError(f"mode must be 'linear' or 'nonlinear', got {mode!r}")

    g = g - g.mean()
    h0 = 0.1
    T = rng.exponential(1.0 / (h0 * np.exp(g)))

    c_max = np.quantile(T, 1 - censor_frac) * 2.0
    C = rng.uniform(0, c_max, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(bool)
    return X, time, event, g
