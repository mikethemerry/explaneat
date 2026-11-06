"""Breslow baseline hazard: turns a fitted log-risk g(x) into survival curves.

    dH0(T_i) = D_i / sum_{j in R(T_i)} exp(g(x_j))
    H0(t)    = sum_{T_i <= t} dH0(T_i)
    S(t|x)   = exp(-exp(g(x)) * H0(t)) = S0(t) ** exp(g(x))

This is the step that makes a Cox-loss network produce Cox-comparable survival
curves (and so Brier scores and calibration).  It is fitted, not learned:
freeze the network first, fit on the TRAINING rows of the same fold, and always
pair a fitted g with its own baseline (never mix a network's g with a Cox
model's baseline).

Verified against scikit-survival's ``predict_survival_function`` to 1e-16
(``tests/test_survival/test_cox.py``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class BreslowBaseline:
    """Baseline cumulative hazard estimated on training data given fitted g(x)."""

    def __init__(self):
        self.times_ = None       # ascending distinct event times
        self.cum_hazard_ = None  # H0 at those times

    def fit(self, log_risk, time, event):
        g = np.asarray(log_risk, dtype=np.float64).ravel()
        t = np.asarray(time, dtype=np.float64).ravel()
        e = np.asarray(event, dtype=np.float64).ravel()

        order = np.argsort(t, kind="stable")     # ascending
        t_s, e_s, g_s = t[order], e[order], g[order]

        shift = g_s.max()
        expg = np.exp(g_s - shift)
        rev_cum = np.cumsum(expg[::-1])[::-1]    # sum over j with t_j >= t_i

        uniq = np.unique(t_s[e_s == 1])
        first = np.searchsorted(t_s, uniq, side="left")
        ev_t = t_s[e_s == 1]
        d = np.searchsorted(ev_t, uniq, side="right") - np.searchsorted(ev_t, uniq, side="left")
        dh = d / (rev_cum[first] * np.exp(shift))

        self.times_ = uniq
        self.cum_hazard_ = np.cumsum(dh)
        return self

    def baseline_cumulative_hazard(self, times):
        idx = np.searchsorted(self.times_, np.asarray(times, dtype=np.float64), side="right") - 1
        return np.where(idx >= 0, self.cum_hazard_[np.clip(idx, 0, None)], 0.0)

    def baseline_survival(self, times):
        """S0(t) = exp(-H0(t)), the survival of a subject with g(x) = 0."""
        return np.exp(-self.baseline_cumulative_hazard(times))

    def cumulative_hazard(self, log_risk, times):
        """H(t|x) = exp(g(x)) * H0(t).  Returns [n_samples, n_times]."""
        g = np.asarray(log_risk, dtype=np.float64).ravel()
        h0 = self.baseline_cumulative_hazard(times)
        return np.exp(g)[:, None] * h0[None, :]

    def survival(self, log_risk, times):
        """S(t|x) = exp(-H(t|x)).  Returns [n_samples, n_times]."""
        return np.exp(-self.cumulative_hazard(log_risk, times))

    def risk_at(self, log_risk, horizon):
        """Predicted cumulative incidence 1 - S(horizon|x), e.g. 5-year CVD risk."""
        return 1.0 - self.survival(log_risk, [horizon])[:, 0]


class SurvivalPredictor:
    """A frozen log-risk function plus its Breslow baseline.

    ``risk_fn`` maps an [n, p] feature array to the [n] log-risk g(x).  Works
    for a NeuralNeat net, a StructureNetwork over an annotated model, or a
    linear Cox model's ``predict``: the downstream maths is identical.
    """

    def __init__(self, risk_fn: Callable, baseline: BreslowBaseline):
        self.risk_fn = risk_fn
        self.baseline = baseline
        self.event_times_ = baseline.times_

    def risk(self, X):
        return np.asarray(self.risk_fn(X), dtype=np.float64).ravel()

    def cumulative_hazard(self, X, times):
        return self.baseline.cumulative_hazard(self.risk(X), times)

    def survival(self, X, times):
        return self.baseline.survival(self.risk(X), times)

    def risk_at(self, X, horizon):
        return self.baseline.risk_at(self.risk(X), horizon)


def fit_survival_predictor(risk_fn: Callable, X_train, time_train, event_train) -> SurvivalPredictor:
    """Fit the Breslow baseline on the training rows for a frozen risk function."""
    g = np.asarray(risk_fn(X_train), dtype=np.float64).ravel()
    return SurvivalPredictor(risk_fn, BreslowBaseline().fit(g, time_train, event_train))
