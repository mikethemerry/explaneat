"""Run A: logistic-regression baselines (adult-logistic, heart-logistic).

sklearn LogisticRegression, L2 penalty, C tuned by 5-fold CV on the TRAIN
split only (roc_auc scoring), on the same one-hot features and z-normalisation
as the paper models (the split's stored StandardScaler). Reports test AUC-ROC
(3dp) with a 95% percentile bootstrap CI (10,000 resamples, seed 42).

Run: uv run python scripts/aaai27/run_a_logistic.py
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold

sys.path.insert(0, "scripts")
from aaai27.common import ADULT, HEART, SEED, load_model, bootstrap_auc_ci, pct_ci

from explaneat.db import db

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def main():
    with db.session_scope() as s:
        for spec in (ADULT, HEART):
            _, _, X, y, split, _, _ = load_model(s, spec)
            Xtr, ytr = X[split.train_indices], y[split.train_indices]
            Xte, yte = X[split.test_indices], y[split.test_indices]

            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            search = GridSearchCV(
                LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000,
                                   random_state=SEED),
                {"C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=-1,
            )
            search.fit(Xtr, ytr)
            best_c = search.best_params_["C"]

            scores = search.best_estimator_.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, scores)
            aucs, skipped = bootstrap_auc_ci(yte, scores)
            lo, hi = pct_ci(aucs)

            print(f"{spec['label']}-logistic: test AUC = {auc:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]  "
                  f"(C={best_c}, cv_train_auc={search.best_score_:.4f}, "
                  f"n_test={len(yte)}, skipped_resamples={skipped})")


if __name__ == "__main__":
    main()
