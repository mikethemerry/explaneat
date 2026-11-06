"""
Non-linear simulation: the implementation sanity check.

If your Cox-loss network cannot beat a linear Cox model on data where the true
log-risk is non-linear, the network or the loss is broken. This is the test to
run FIRST, before touching real data - it separates "the method has no headroom
on this dataset" from "my plumbing is wrong".

Design follows the DeepSurv simulation study (Katzman et al. 2018): exponential
survival times with a known risk function, independent uniform censoring.

    h(t|x) = h0 * exp(g_true(x))
    T ~ Exponential(rate = h0 * exp(g_true(x)))
    C ~ Uniform(0, c_max);  observed = min(T, C),  event = 1{T <= C}

Two regimes:
    "linear"    g_true(x) = x . beta            -> Cox is correctly specified
    "nonlinear" g_true(x) = gaussian interaction -> Cox is mis-specified
"""
from __future__ import annotations

import numpy as np
import torch
import warnings
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from cox_torch import CoxRiskSets, cox_ph_loss, concordance
from benchmark import train_nn

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


def simulate(n=4000, d=10, mode="nonlinear", censor_frac=0.30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))

    if mode == "linear":
        beta = np.zeros(d); beta[:5] = rng.normal(0, 1, 5)
        g = X @ beta
    elif mode == "nonlinear":
        # Gaussian log-risk over the first two coordinates: strong interaction,
        # zero linear main effect by symmetry. A linear Cox model is blind to it.
        r, lam = 0.5, 5.0
        g = np.log(lam) * np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) / (2 * r ** 2))
    else:
        raise ValueError(mode)

    g = g - g.mean()
    h0 = 0.1
    T = rng.exponential(1.0 / (h0 * np.exp(g)))

    # tune censoring ceiling to hit the requested censored fraction
    c_max = np.quantile(T, 1 - censor_frac) * 2.0
    C = rng.uniform(0, c_max, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(bool)
    return X, time, event, g


def run(mode, n=4000, seed=0):
    X, t, e, g_true = simulate(n=n, mode=mode, seed=seed)
    n_tr = int(0.6 * n); n_va = int(0.2 * n)
    idx = np.random.default_rng(seed).permutation(n)
    itr, iva, ite = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    mu, sd = X[itr].mean(0), X[itr].std(0)
    Xtr, Xva, Xte = (X[itr]-mu)/sd, (X[iva]-mu)/sd, (X[ite]-mu)/sd

    cox = CoxPHSurvivalAnalysis(alpha=1e-2, ties="efron").fit(
        Xtr, Surv.from_arrays(event=e[itr], time=t[itr]))
    c_cox = concordance_index_censored(e[ite], t[ite], cox.predict(Xte))[0]

    model = train_nn(Xtr, t[itr], e[itr], Xva, t[iva], e[iva], seed=seed, epochs=1500)
    with torch.no_grad():
        r_nn = model(torch.tensor(Xte)).numpy()
    c_nn = concordance_index_censored(e[ite], t[ite], r_nn)[0]

    # ceiling: C of the TRUE risk function on the same test set
    c_oracle = concordance_index_censored(e[ite], t[ite], g_true[ite])[0]

    # how well does each recover the true risk ordering?
    from scipy.stats import spearmanr
    rho_cox = spearmanr(cox.predict(Xte), g_true[ite]).statistic
    rho_nn = spearmanr(r_nn, g_true[ite]).statistic

    print(f"{mode:<10} n={n}  censored={100*(1-e.mean()):.0f}%   "
          f"Cox C={c_cox:.3f}   NN C={c_nn:.3f}   oracle C={c_oracle:.3f}   "
          f"| spearman vs true g:  Cox={rho_cox:+.3f}  NN={rho_nn:+.3f}")
    return c_cox, c_nn, c_oracle


if __name__ == "__main__":
    print("Sanity check: NN must match Cox when the truth is linear, and beat it when it isn't.\n")
    for mode in ["linear", "nonlinear"]:
        for seed in [0, 1, 2]:
            run(mode, seed=seed)
        print()
