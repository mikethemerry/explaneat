"""Heart-100 signal matrix in the form of the churn and adult catalogues (read-only).

Basis: ``experiments/linear_betas/betas.py`` ``heart_basis`` (the one the Heart-100 linear betas use):
a linear block per variable (a number: its z-column; a category: offsets for its levels against the
most common level), bends per number (hinges at the training quartiles; vessels at 1 and 2),
residualised on every linear column, and the interactions of every pair of variables that a
switching unit of one of the ten networks reads together (drivers and modifiers in
``experiments/heart100_audit/features.csv``), residualised on every linear and bend column.

- a network's **component** of signal ``s``: ``Phi_s beta_s`` of the least-squares projection of
  its logit on ``[1, Phi]`` (242 training rows); ``sd_train`` = its standard deviation over the
  training rows (logit), ``sd_robust`` the same after clipping it to its 1st-99th percentiles.
- **used**: ``sd_train >= 0.10`` logit (the churn and adult rule).
- the data's own version: a ridge logistic fit of the labels on ``Phi`` (columns scaled to unit SD,
  C chosen by 5-fold CV log-loss from 0.01, 0.1, 1); ``data_component_sd`` its component's SD,
  ``data_cv_logloss_gain`` the CV log-loss lost when the block is left out (> 0: the data use it);
  ``corr_with_data`` the correlation of a network's component with the data's.

Writes ``matrix.csv`` (seed x signal), ``signals.csv`` (one row per signal) and ``projection.csv``.

    OMP_NUM_THREADS=1 uv run python experiments/heart100_signals/signals.py
(uses the cache of ``experiments/linear_betas/betas.py``; run that first, with DATABASE_URL set to
explaneat_v4, if ``experiments/linear_betas/cache/heart100.pkl`` is missing)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/linear_betas"))
import betas as LB  # noqa: E402

USED_SD = 0.10


def fit_logit(P, y, C):
    sd = P.std(axis=0)
    sd[sd == 0] = 1
    lr = LogisticRegression(C=C, max_iter=20000, tol=1e-8).fit(P / sd, y)
    return lr.intercept_[0], lr.coef_[0] / sd


def cv_score(P, y, C, folds):
    ll = []
    for tr, va in folds:
        a, b = fit_logit(P[tr], y[tr], C)
        ll.append(log_loss(y[va], 1 / (1 + np.exp(-(a + P[va] @ b)))))
    return float(np.mean(ll))


def main():
    D = LB.load_set("heart100")
    spec = LB.linear_spec("heart100", D)
    Ltr, Lte = LB.L_of(spec, D["X"]), LB.L_of(spec, D["Xt"])
    HB = LB.heart_basis(D, dict(train=Ltr, test=Lte), [f"lin:{vs[0]}" for _, vs, _ in spec])
    P, Pt, sig = HB["Phi_tr"], HB["Phi_te"], HB["signals"]
    y, yt = D["y"], D["yt"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(P, y))
    scores = {C: cv_score(P, y, C, folds) for C in (0.01, 0.1, 1.0)}
    C = min(scores, key=scores.get)
    a_d, b_d = fit_logit(P, y, C)
    print("data fit C", C, "cv log-loss", {k: round(v, 4) for k, v in scores.items()})

    def comp(beta, s, M=P):
        idx = sig[s]
        return M[:, idx] @ beta[idx]

    rows, proj = [], []
    for s in range(10):
        lab = f"heart100-s{s}"
        e, et = D["eta_tr"][s], D["eta_te"][s]
        c = LB.ols(P, e)
        a, beta = c[0], c[1:]
        proj.append(dict(network=lab, seed=s, r2_train=LB.r2(e, a + P @ beta), r2_test=LB.r2(et, a + Pt @ beta),
                         auc_train=roc_auc_score(y, e), auc_test=roc_auc_score(yt, et),
                         auc_projection_train=roc_auc_score(y, a + P @ beta),
                         auc_projection_test=roc_auc_score(yt, a + Pt @ beta)))
        for name in sig:
            x = comp(beta, name)
            sd = float(np.std(x))
            q = np.quantile(x, [0.01, 0.99])
            xd = comp(b_d, name)
            rows.append(dict(network=lab, seed=s, signal=name, kind=name.split(":")[0], sd_train=sd,
                             sd_robust=float(np.std(np.clip(x, *q))), used=sd >= USED_SD,
                             corr_with_data=(float(np.corrcoef(x, xd)[0, 1]) if sd > 1e-9 and np.std(xd) > 1e-9
                                             else float("nan")),
                             n_columns=len(sig[name])))
    M = pd.DataFrame(rows)
    srows = []
    for name, idx in sig.items():
        keep = [i for i in range(P.shape[1]) if i not in set(idx)]
        mb = M[M.signal == name]
        u = mb[mb.used]
        srows.append(dict(signal=name, kind=name.split(":")[0], variables=" + ".join(name.split(":")[1].split("*")),
                          n_columns=len(idx), data_component_sd=float(np.std(comp(b_d, name))),
                          data_cv_logloss_gain=cv_score(P[:, keep], y, C, folds) - scores[C],
                          n_seeds_used=int(len(u)), seeds_used=" ".join(str(x) for x in u.seed),
                          median_sd_all=float(mb.sd_train.median()),
                          median_sd_used=float(u.sd_train.median()) if len(u) else float("nan")))
    S = pd.DataFrame(srows)
    S["data_relevant"] = S.data_cv_logloss_gain > 0
    S = S.sort_values(["kind", "data_component_sd"], ascending=[True, False],
                      key=lambda c: c.map({"lin": 0, "shape": 1, "int": 2}) if c.name == "kind" else c)
    M.to_csv(HERE / "matrix.csv", index=False)
    S.to_csv(HERE / "signals.csv", index=False)
    pd.DataFrame(proj).to_csv(HERE / "projection.csv", index=False)
    print(S[["signal", "n_columns", "data_component_sd", "data_cv_logloss_gain", "n_seeds_used",
             "median_sd_used"]].round(3).to_string())
    print(pd.DataFrame(proj)[["network", "r2_train", "r2_test"]].round(4).to_string())


if __name__ == "__main__":
    main()
