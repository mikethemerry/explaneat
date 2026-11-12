"""Logistic-regression and gradient-boosting references for the churn case, refitted on the
case's split (the 10-seed churn split: stratified 80:20, seed 42, StandardScaler on train).

    uv run python experiments/churn_case/references.py

Same models and settings as experiments/churn/ingest_churn.py (L2 logistic regression,
max_iter 5000; HistGradientBoostingClassifier defaults, random_state 0; 10,000 bootstrap
resamples of the test set, seed 42). Read-only on the database; writes references.json and
prints the values next to experiments/churn/baselines.json for the check.
"""
import json
import sys
from pathlib import Path
from uuid import UUID

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explaneat.db import db  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit  # noqa: E402

DATASET = "802129c2-b11b-43a5-b6af-8bcf80e4c252"
SPLIT = "45962ea6-3c78-49f9-a01e-3665f6809b1e"


def bootstrap_ci(y, p, n=10_000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    db.init_db()
    with db.session_scope() as s:
        ds = s.get(Dataset, UUID(DATASET))
        sp = s.get(DatasetSplit, UUID(SPLIT))
        assert sp.dataset_id == ds.id
        X, y = ds.get_data()
        tr, te = np.asarray(sp.train_indices), np.asarray(sp.test_indices)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    res = {"dataset_id": DATASET, "split_id": SPLIT, "n_inputs": int(X.shape[1]),
           "n_train": int(len(tr)), "n_test": int(len(te)), "positive_rate_train": float(y[tr].mean()),
           "positive_rate_test": float(y[te].mean())}
    for label, model in [("l2_logistic", LogisticRegression(max_iter=5000)),
                         ("gradient_boosting", HistGradientBoostingClassifier(random_state=0))]:
        model.fit(Xtr, y[tr])
        p_tr, p_te = model.predict_proba(Xtr)[:, 1], model.predict_proba(Xte)[:, 1]
        lo, hi = bootstrap_ci(y[te], p_te)
        res[label] = {"train_auc": float(roc_auc_score(y[tr], p_tr)), "test_auc": float(roc_auc_score(y[te], p_te)),
                      "test_auc_ci": [lo, hi], "test_acc": float(((p_te > 0.5) == y[te]).mean())}
    (HERE / "references.json").write_text(json.dumps(res, indent=2) + "\n")
    base = json.loads((HERE.parent / "churn" / "baselines.json").read_text())
    for k in ("l2_logistic", "gradient_boosting"):
        print(f"{k}: test AUC {res[k]['test_auc']:.4f} (baselines.json {base[k]['test_auc']:.4f}), "
              f"train {res[k]['train_auc']:.4f}, acc {res[k]['test_acc']:.3f}")


if __name__ == "__main__":
    main()
