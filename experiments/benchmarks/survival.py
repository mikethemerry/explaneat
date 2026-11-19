"""The comparator suite on the survival sets (WHAS500; GBSG2 as a side note), exploratory.

    uv run python experiments/benchmarks/survival.py

No database access. Data and split exactly as the Cox case (experiments/survival/cox_case): the
scikit-survival bundled data, the case's split file (train / validation / test), features
standardised with the training rows' mean and SD (explaneat.survival.data.standardise). The
networks trained on the training rows and were selected on the validation rows; the comparators
are fitted and tuned on the training rows only (the validation rows are not used), by 5-fold CV
stratified on the event indicator (shuffle, seed 42), scoring Harrell's C, then refitted on all
training rows; the test rows are read once.

    Cox PH            lifelines CoxPHFitter, penalizer 0 (Efron ties); nothing to tune
    Cox, L2 (ridge)   scikit-survival CoxPHSurvivalAnalysis (Efron), alpha by CV
    Cox, elastic net  scikit-survival CoxnetSurvivalAnalysis, alpha and l1_ratio by CV
    Random surv. forest  RandomSurvivalForest (500 trees, max_features sqrt), depth and leaf size by CV
    Gradient-boosted  GradientBoostingSurvivalAnalysis (Cox loss), rate, stages and depth by CV

Test metrics are explaneat.survival.metrics, as the Cox case runs: Harrell's C (95% percentile
interval over 10,000 bootstrap resamples of the test rows, RandomState(42)), Uno's C,
time-dependent AUC at the case's horizons (1, 3, 5 years where valid; their mean), and the
integrated Brier score. The Cox models' curves use the Breslow baseline on their training
log-risks (as the networks and experiments/survival/cox_case/references.py); the forest and the
boosted model use their own survival functions. The selected PropNEAT network of each dataset
(cox_case/selected.json; its genome pickle) is scored by the same code, as a check against its
run record. Output: results/survival.json.
"""
from __future__ import annotations

import pickle
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from common import REPO, RESULTS, SEED, boot_indices, jdump, jload  # noqa: E402

CASE = REPO / "experiments" / "survival" / "cox_case"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parent))

from explaneat.survival import BreslowBaseline  # noqa: E402
from explaneat.survival.data import load_builtin, load_split_indices, standardise  # noqa: E402
from explaneat.survival.metrics import (harrell_c, integrated_brier, time_dependent_auc,  # noqa: E402
                                        uno_c)

HORIZONS = [365.0, 1095.0, 1825.0]
GRIDS = {
    "cox_l2": {"alpha": [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]},
    "cox_enet": {"l1_ratio": [0.1, 0.5, 0.9], "alphas": [[a] for a in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]},
    "rsf": {"max_depth": [3, 5, None], "min_samples_leaf": [5, 10, 20]},
    "gbs": {"learning_rate": [0.03, 0.1], "n_estimators": [100, 300], "max_depth": [1, 2, 3]},
}


def surv_struct(y):
    from sksurv.util import Surv
    return Surv.from_arrays(event=y[:, 1].astype(bool), time=y[:, 0])


def estimators():
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    return {"cox_l2": CoxPHSurvivalAnalysis(ties="efron"),
            "cox_enet": CoxnetSurvivalAnalysis(fit_baseline_model=True, max_iter=100000),
            "rsf": RandomSurvivalForest(n_estimators=500, max_features="sqrt", random_state=SEED, n_jobs=4),
            "gbs": GradientBoostingSurvivalAnalysis(loss="coxph", random_state=0)}


def metrics(risk_te, ytr, yte, surv_fn, idx):
    ttr, etr, tte, ete = ytr[:, 0], ytr[:, 1], yte[:, 0], yte[:, 1]
    c = harrell_c(risk_te, tte, ete)
    boots = []
    for i in idx:
        if ete[i].sum() == 0:
            continue
        boots.append(harrell_c(risk_te[i], tte[i], ete[i]))
    td = time_dependent_auc(risk_te, ttr, etr, tte, ete, HORIZONS)
    return {"harrell_c": float(c), "harrell_c_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_boot_used": len(boots), "uno_c": float(uno_c(risk_te, ttr, etr, tte, ete)),
            "td_auc": {"times": [float(t) for t in td["times"]], "auc": [float(a) for a in td["auc"]],
                       "mean_auc": float(td["mean_auc"])},
            "ibs": float(integrated_brier(surv_fn, ttr, etr, tte, ete))}


def breslow_fn(g_tr, g_te, ytr):
    bl = BreslowBaseline().fit(g_tr, ytr[:, 0], ytr[:, 1])
    return lambda grid: bl.survival(g_te, grid)


def own_fn(model, Xte):
    fns = model.predict_survival_function(Xte)

    def surv(grid):
        g = np.clip(np.asarray(grid, dtype=float), fns[0].x.min(), fns[0].x.max())
        return np.array([np.atleast_1d(f(g)) for f in fns])
    return surv


def propneat(ds, Xtr, Xte, ytr, yte, idx):
    import neat
    import torch
    from explaneat.core.neuralneat import NeuralNeat, set_output_activation
    tag = jload(CASE / "selected.json")[ds]["tag"]
    b = pickle.loads((CASE / "runs" / f"{tag}.genome.pkl").read_bytes())
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write(b["config_text"])
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, f.name)
    set_output_activation(cfg, "identity")
    net = NeuralNeat(b["genome"], cfg)
    g = lambda X: net.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel()  # noqa: E731
    g_tr, g_te = g(Xtr), g(Xte)
    m = metrics(g_te, ytr, yte, breslow_fn(g_tr, g_te, ytr), idx)
    m["train_harrell_c"] = float(harrell_c(g_tr, ytr[:, 0], ytr[:, 1]))
    run = jload(CASE / "runs" / f"{tag}.json")["results"]["neat"]
    m["run_record_harrell_c"] = run["harrell_c"]
    assert abs(m["harrell_c"] - run["harrell_c"]) < 1e-9, (m["harrell_c"], run["harrell_c"])
    m["tag"] = tag
    return m


def main():
    from lifelines import CoxPHFitter
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    out = {}
    for ds in ("whas500", "gbsg2"):
        X, y, names = load_builtin(ds)
        sp = load_split_indices(CASE / "splits" / f"{ds}.splits.json")
        tr, te = sp["train"], sp["test"]
        _, _, Xtr, Xte = standardise(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        idx = boot_indices(len(te))
        res = {"n_train": len(tr), "n_val_unused": len(sp["val"]), "n_test": len(te),
               "events_train": int(ytr[:, 1].sum()), "events_test": int(yte[:, 1].sum()), "horizons": HORIZONS}
        res["propneat"] = propneat(ds, Xtr, Xte, ytr, yte, idx)
        # Cox PH, unpenalised
        cols = [f"x{i}" for i in range(Xtr.shape[1])]
        df = pd.DataFrame(Xtr, columns=cols)
        df["T"], df["E"] = ytr[:, 0], ytr[:, 1]
        cph = CoxPHFitter(penalizer=0.0).fit(df, "T", "E")
        lp = lambda A: cph.predict_log_partial_hazard(pd.DataFrame(A, columns=cols)).to_numpy(float)  # noqa: E731
        res["cox_ph"] = {"params": {}, "cv_c": None, **metrics(lp(Xte), ytr, yte, breslow_fn(lp(Xtr), lp(Xte), ytr), idx),
                         "train_harrell_c": float(harrell_c(lp(Xtr), ytr[:, 0], ytr[:, 1]))}
        print(ds, "cox_ph", res["cox_ph"]["harrell_c"], flush=True)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr[:, 1].astype(int)))
        for name, est in estimators().items():
            t0 = time.time()
            gs = GridSearchCV(est, GRIDS[name], cv=folds, n_jobs=2, error_score=np.nan).fit(Xtr, surv_struct(ytr))
            m = gs.best_estimator_
            r_tr, r_te = m.predict(Xtr), m.predict(Xte)
            fn = breslow_fn(r_tr, r_te, ytr) if name in ("cox_l2", "cox_enet") else own_fn(m, Xte)
            p = {k: (v[0] if isinstance(v, list) else v) for k, v in gs.best_params_.items()}
            res[name] = {"params": p, "cv_c": float(gs.best_score_), **metrics(r_te, ytr, yte, fn, idx),
                         "train_harrell_c": float(harrell_c(r_tr, ytr[:, 0], ytr[:, 1])),
                         "grid": GRIDS[name], "seconds": round(time.time() - t0, 1)}
            print(ds, name, p, "cv", round(gs.best_score_, 4), "test C", round(res[name]["harrell_c"], 4), flush=True)
        out[ds] = res
        jdump(out, RESULTS / "survival.json")


if __name__ == "__main__":
    main()
