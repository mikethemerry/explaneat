"""
Cox PH vs a Cox-loss neural network: the benchmark protocol, end to end.

The NN here is a plain MLP standing in for an evolved ExplaNEAT genome. The
point is the PROTOCOL and the plumbing, not the architecture: swap the MLP for
a NEAT phenotype and everything downstream is unchanged.

Protocol (pre-specified, per the C-hacking guidance in Sonabend et al. 2022):
  * identical repeated stratified splits for both models
  * identical preprocessing (standardise on train only)
  * risk score = g(x) for BOTH models, stated up front, no post-hoc transform
  * metrics: Harrell's C, Uno's IPCW C, integrated Brier score
  * IBS evaluated on a time grid inside the observed follow-up of both folds
  * paired differences reported with a mean +/- sd across repeats
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_whas500, load_gbsg2, load_veterans_lung_cancer, load_flchain
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            integrated_brier_score)
from sksurv.util import Surv
from sksurv.preprocessing import OneHotEncoder

from cox_torch import CoxRiskSets, cox_ph_loss, BreslowBaseline

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ data
def get_dataset(name):
    loaders = {
        "whas500": load_whas500,
        "gbsg2": load_gbsg2,
        "veterans": load_veterans_lung_cancer,
        "flchain": load_flchain,
    }
    X, y = loaders[name]()
    X = OneHotEncoder().fit_transform(X)
    X = X.astype(float)
    X = X.fillna(X.median(numeric_only=True))
    ev_field, t_field = y.dtype.names
    return X.to_numpy(float), y[ev_field].astype(bool), y[t_field].astype(float)


# ------------------------------------------------------------------ model
class MLP(nn.Module):
    """Stand-in for an evolved genome. Note the LINEAR output: this is a
    log-risk g(x), not a probability. No sigmoid."""
    def __init__(self, d_in, hidden=(32, 32), dropout=0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1, bias=False)]   # bias is unidentifiable in Cox
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).reshape(-1)


def train_nn(Xtr, ttr, etr, Xva, tva, eva, epochs=500, lr=1e-2, wd=1e-3,
             ties="efron", patience=50, seed=0):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rs_tr = CoxRiskSets(ttr, etr)
    rs_va = CoxRiskSets(tva, eva)
    Xtr_t, Xva_t = torch.tensor(Xtr), torch.tensor(Xva)

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = cox_ph_loss(model(Xtr_t), rs_tr, ties=ties)   # FULL BATCH -> exact
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = float(cox_ph_loss(model(Xva_t), rs_va, ties=ties))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ------------------------------------------------------------------ eval
def evaluate(risk_tr, risk_te, ttr, etr, tte, ete, surv_te_fn=None):
    y_tr = Surv.from_arrays(event=etr, time=ttr)
    y_te = Surv.from_arrays(event=ete, time=tte)
    out = {}
    out["harrell_c"] = concordance_index_censored(ete, tte, risk_te)[0]

    # Uno's IPCW C: needs a tau inside the training censoring support
    tau = min(tte[ete].max(), ttr[etr].max())
    try:
        out["uno_c"] = concordance_index_ipcw(y_tr, y_te, risk_te, tau=tau)[0]
    except Exception:
        out["uno_c"] = np.nan

    if surv_te_fn is not None:
        lo = max(ttr.min(), tte.min())
        hi = min(np.percentile(tte[ete], 80), ttr[etr].max() * 0.99)
        if hi > lo:
            grid = np.linspace(lo, hi, 25)
            grid = grid[(grid > tte.min()) & (grid < tte.max())]
            try:
                S = surv_te_fn(grid)
                out["ibs"] = integrated_brier_score(y_tr, y_te, S, grid)
            except Exception:
                out["ibs"] = np.nan
        else:
            out["ibs"] = np.nan
    return out


def run(name, n_repeats=5, seed0=0):
    X, e, t = get_dataset(name)
    print(f"\n{'='*74}\n{name}: n={len(t)}  features={X.shape[1]}  "
          f"events={int(e.sum())} ({100*e.mean():.0f}%)  censored={100*(1-e.mean()):.0f}%")

    rows = {"cox": [], "nn": []}
    for rep in range(n_repeats):
        idx = np.arange(len(t))
        itr, ite = train_test_split(idx, test_size=0.25, random_state=seed0 + rep, stratify=e)
        itr, iva = train_test_split(itr, test_size=0.20,
                                    random_state=seed0 + rep, stratify=e[itr])

        mu, sd = X[itr].mean(0), X[itr].std(0)
        sd[sd == 0] = 1.0
        Xtr, Xva, Xte = (X[itr] - mu) / sd, (X[iva] - mu) / sd, (X[ite] - mu) / sd
        ttr, tva, tte = t[itr], t[iva], t[ite]
        etr, eva, ete = e[itr], e[iva], e[ite]

        # ---- Cox PH (same features, same split, ridge tuned on the val fold)
        y_tr = Surv.from_arrays(event=etr, time=ttr)
        y_va = Surv.from_arrays(event=eva, time=tva)
        best_a, best_c = None, -np.inf
        for a in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                m = CoxPHSurvivalAnalysis(alpha=a, ties="efron").fit(Xtr, y_tr)
                c = concordance_index_censored(eva, tva, m.predict(Xva))[0]
                if c > best_c:
                    best_a, best_c = a, c
            except Exception:
                continue
        cox = CoxPHSurvivalAnalysis(alpha=best_a, ties="efron").fit(Xtr, y_tr)
        r_tr_cox, r_te_cox = cox.predict(Xtr), cox.predict(Xte)
        bl_cox = BreslowBaseline().fit(r_tr_cox, ttr, etr)
        rows["cox"].append(evaluate(r_tr_cox, r_te_cox, ttr, etr, tte, ete,
                                    lambda g: bl_cox.survival(r_te_cox, g)))

        # ---- Cox-loss NN, identical split, early-stopped on the same val fold
        model = train_nn(Xtr, ttr, etr, Xva, tva, eva, seed=seed0 + rep)
        with torch.no_grad():
            r_tr_nn = model(torch.tensor(Xtr)).numpy()
            r_te_nn = model(torch.tensor(Xte)).numpy()
        bl_nn = BreslowBaseline().fit(r_tr_nn, ttr, etr)
        rows["nn"].append(evaluate(r_tr_nn, r_te_nn, ttr, etr, tte, ete,
                                   lambda g: bl_nn.survival(r_te_nn, g)))

    def agg(k, m):
        v = np.array([r[m] for r in rows[k]], dtype=float)
        v = v[np.isfinite(v)]
        return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)

    print(f"{'model':<10} {'Harrell C':>16} {'Uno C':>16} {'IBS':>16}")
    for k, label in [("cox", "Cox PH"), ("nn", "NN (Cox loss)")]:
        hm, hs = agg(k, "harrell_c"); um, us = agg(k, "uno_c"); im, isd = agg(k, "ibs")
        print(f"{label:<10} {hm:>10.3f}+-{hs:.3f} {um:>10.3f}+-{us:.3f} {im:>10.3f}+-{isd:.3f}")

    d = np.array([rows["nn"][i]["harrell_c"] - rows["cox"][i]["harrell_c"]
                  for i in range(n_repeats)])
    print(f"paired delta Harrell C (NN - Cox): {d.mean():+.4f} +- {d.std():.4f}   "
          f"per-repeat: {np.round(d, 3)}")
    return rows


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["whas500", "gbsg2", "veterans"]
    for nm in names:
        run(nm, n_repeats=5)
