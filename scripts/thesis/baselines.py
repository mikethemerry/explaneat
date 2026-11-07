"""Task 6 references (ii) and (iii): logistic regression and gradient boosting
fitted on the SAME train/test split each PropNEAT run used.

- LR: the paper's protocol (scripts/aaai27/run_a_logistic.py): L2
  LogisticRegression, C chosen by 5-fold CV on the train split only
  (roc_auc), lbfgs, max_iter 5000, random_state 42.
- GB: sklearn HistGradientBoostingClassifier, random_state 0, no early
  stopping; hyperparameters chosen by 5-fold stratified CV (shuffle, seed 42)
  on the train split only (roc_auc) over GRID below; refit on all of train.
- Test AUC with a 95% percentile bootstrap CI (10,000 resamples, seed 42).
- Same z-normalised features as the networks (the split's stored scaler).
  n_jobs=1 throughout (CPU budget).

Run: PYTHONHASHSEED=0 uv run python scripts/thesis/baselines.py
Writes results/thesis/baselines.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explaneat.db import db  # noqa: E402
from mcp_server.helpers import load_split_data  # noqa: E402

SEED = 42
SPLITS = {
    "heart": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
    "monk2": "63257a45-49c9-42a6-8b28-8e346eab2bee",
    "adult": "1df50169-8095-40d1-b61a-30bb541cd313",
    "churn": "45962ea6-3c78-49f9-a01e-3665f6809b1e",
}
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
GRID = {
    "learning_rate": [0.03, 0.1],
    "max_iter": [100, 300],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [5, 20],
    "l2_regularization": [0.0, 1.0],
}


def boot_ci(y, p, n=10_000, seed=SEED):
    rng = np.random.RandomState(seed)
    a = []
    for _ in range(n):
        i = rng.randint(0, len(y), size=len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    out = {"protocol": __doc__, "grid_gb": GRID, "grid_lr_C": C_GRID}
    only = sys.argv[1:] or list(SPLITS)
    path = ROOT / "results" / "thesis" / "baselines.json"
    if path.exists():
        out.update({k: v for k, v in json.loads(path.read_text()).items() if k in SPLITS})
    for name in only:
        sid = SPLITS[name]
        with db.session_scope() as s:
            Xtr, ytr, *_ = load_split_data(s, sid, "train", 1.0, 10 ** 9)
            Xte, yte, *_ = load_split_data(s, sid, "test", 1.0, 10 ** 9)
        ytr, yte = ytr.ravel().astype(int), yte.ravel().astype(int)
        r = {"split_id": sid, "n_train": int(len(ytr)), "n_test": int(len(yte)), "n_features": int(Xtr.shape[1])}

        t0 = time.time()
        lr = GridSearchCV(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED),
                          {"C": C_GRID}, scoring="roc_auc", cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=1)
        lr.fit(Xtr, ytr)
        p = lr.best_estimator_.predict_proba(Xte)[:, 1]
        r["logistic"] = {"C": lr.best_params_["C"], "cv_auc": float(lr.best_score_),
                         "test_auc": float(roc_auc_score(yte, p)), "test_auc_ci95": boot_ci(yte, p),
                         "seconds": time.time() - t0}
        print(name, "LR", r["logistic"], flush=True)

        t0 = time.time()
        gb = GridSearchCV(HistGradientBoostingClassifier(random_state=0, early_stopping=False),
                          GRID, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
        gb.fit(Xtr, ytr)
        p = gb.best_estimator_.predict_proba(Xte)[:, 1]
        r["gradient_boosting"] = {"params": gb.best_params_, "cv_auc": float(gb.best_score_),
                                  "train_auc": float(roc_auc_score(ytr, gb.best_estimator_.predict_proba(Xtr)[:, 1])),
                                  "test_auc": float(roc_auc_score(yte, p)), "test_auc_ci95": boot_ci(yte, p),
                                  "seconds": time.time() - t0}
        # default-hyperparameter HGB as a sensitivity check
        d = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        r["gradient_boosting_default"] = {"test_auc": float(roc_auc_score(yte, d.predict_proba(Xte)[:, 1]))}
        print(name, "GB", r["gradient_boosting"], r["gradient_boosting_default"], flush=True)
        out[name] = r
        path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
