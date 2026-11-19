"""The comparator suite on every binary-classification set (exploratory; not pre-registered).

    OMP_NUM_THREADS=2 <venv with interpret>/python experiments/benchmarks/classification.py [name ...]

(the EBM needs `interpret`; the conv-neat worktree venv has it, with the same scikit-learn 1.7.2 as
the uv environment). Reads cache/<name>.npz (export_data.py); no database access. Per dataset and
model: hyperparameters chosen by 5-fold stratified CV on the training rows only (shuffle, seed 42,
scoring ROC AUC), refit on all training rows, the test rows read once. Test AUC with a 95%
percentile interval over 10,000 bootstrap resamples of the test rows (RandomState(42), the 10-seed
study's index matrix), and test accuracy at the default decision rule (predict). The grids are
those of experiments/references/baselines.py (L2 LR, gradient boosting) and
experiments/references/heart_benchmark/bench.py (the other five), unchanged, for every dataset.
Results are written after each model (results/classification/<name>.json, one file per dataset so
datasets can run in parallel processes); a rerun skips what is there.
XOR (4 rows, no split): an L2 logistic regression and a decision tree fitted on the truth table,
accuracy on the same four rows.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from common import CACHE, CLASSIFICATION, RESULTS, SEED, auc_ci, boot_indices, jdump, jload  # noqa: E402

NJ = int(os.environ.get("BENCH_JOBS", "4"))
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
HGB_GRID = {"learning_rate": [0.03, 0.1], "max_iter": [100, 300], "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [5, 20], "l2_regularization": [0.0, 1.0]}


def models():
    from interpret.glassbox import ExplainableBoostingClassifier
    return {
        "lr_l2": (LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED), {"C": C_GRID}),
        "lr_l1": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000), {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}),
        "tree": (DecisionTreeClassifier(random_state=SEED), {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [5, 10, 20]}),
        "rf": (RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=2),
               {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]}),
        "gb": (HistGradientBoostingClassifier(random_state=0, early_stopping=False), HGB_GRID),
        "ebm": (ExplainableBoostingClassifier(random_state=SEED, n_jobs=2), {"interactions": [0, 5, 10], "max_bins": [32, 256]}),
        "mlp": (MLPClassifier(max_iter=3000, random_state=SEED),
                {"hidden_layer_sizes": [(8,), (16,), (32,)], "alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
    }


def jsonable(p):
    return {k: (list(v) if isinstance(v, tuple) else v.item() if hasattr(v, "item") else v) for k, v in p.items()}


def xor():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    out = {}
    for name, m in (("lr_l2", LogisticRegression(max_iter=5000, random_state=SEED)),
                    ("tree", DecisionTreeClassifier(random_state=SEED))):
        m.fit(X, y)
        out[name] = {"params": "defaults (4 rows: no CV)", "train_accuracy": float(accuracy_score(y, m.predict(X))),
                     "predictions": m.predict(X).tolist()}
    return out


def main():
    (RESULTS / "classification").mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(CLASSIFICATION) + ["xor"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name in names:
        path = RESULTS / "classification" / f"{name}.json"
        out = jload(path) if path.exists() else {}
        if name == "xor":
            out["xor"] = xor()
            jdump(out, path)
            print("xor", out["xor"], flush=True)
            continue
        d = np.load(CACHE / f"{name}.npz")
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
        idx = boot_indices(len(yte))
        res = out.setdefault(name, {})
        for mname, (est, grid) in models().items():
            if mname in res:
                continue
            t0 = time.time()
            g = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=NJ).fit(Xtr, ytr)
            p = g.predict_proba(Xte)[:, 1]
            ci, nb = auc_ci(yte, p, idx)
            res[mname] = {"params": jsonable(g.best_params_), "cv_auc": float(g.best_score_),
                          "train_auc": float(roc_auc_score(ytr, g.predict_proba(Xtr)[:, 1])),
                          "test_auc": float(roc_auc_score(yte, p)), "test_auc_ci95": ci, "n_boot_used": nb,
                          "test_accuracy": float(accuracy_score(yte, g.predict(Xte))),
                          "n_train": int(len(ytr)), "n_test": int(len(yte)), "grid": jsonable(grid),
                          "seconds": round(time.time() - t0, 1)}
            jdump(out, path)
            print(f"{name:13s} {mname:6s} cv {g.best_score_:.4f} test {res[mname]['test_auc']:.4f} "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}] acc {res[mname]['test_accuracy']:.4f} {g.best_params_} "
                  f"{res[mname]['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
