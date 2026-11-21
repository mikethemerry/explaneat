"""Check that the staged iteration search (search.py) equals refitting at each iteration count.

    <venv>/python experiments/benchmarks/check_staged.py      # needs cache/churn.npz and scikit-survival

Gradient boosting (HistGradientBoosting, no early stopping) on Churn's first CV fold and gradient-boosted
survival (Cox loss) on WHAS500's first fold: the validation score after k staged iterations of one fit with
the largest count, against a separate fit with max_iter = k (n_estimators = k). Prints the largest absolute
difference; asserts it is below 1e-12. Output: results/check_staged.json.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import CACHE, REPO, RESULTS, SEED, jdump
from search import staged_auc, staged_cindex

VALUES = [10, 30, 100, 300]


def gb():
    d = np.load(CACHE / "churn.npz")
    X, y = d["Xtr"], d["ytr"]
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    kw = dict(learning_rate=0.1, max_leaf_nodes=15, random_state=0, early_stopping=False)
    m = HistGradientBoostingClassifier(max_iter=max(VALUES), **kw).fit(X[tr], y[tr])
    staged = staged_auc(m, X[va], y[va], VALUES)
    refit = [roc_auc_score(y[va], HistGradientBoostingClassifier(max_iter=k, **kw).fit(X[tr], y[tr])
                           .decision_function(X[va])) for k in VALUES]
    return staged, refit


def gbs():
    sys.path.insert(0, str(REPO))
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    from explaneat.survival.data import load_builtin, load_split_indices, standardise
    X, yy, _ = load_builtin("whas500")
    sp = load_split_indices(REPO / "experiments/survival/cox_case/splits/whas500.splits.json")
    _, _, Xtr, _ = standardise(X[sp["train"]], X[sp["test"]])
    ytr = yy[sp["train"]]
    y = Surv.from_arrays(event=ytr[:, 1].astype(bool), time=ytr[:, 0])
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr[:, 1].astype(int)))
    kw = dict(loss="coxph", learning_rate=0.1, max_depth=2, random_state=0)
    m = GradientBoostingSurvivalAnalysis(n_estimators=max(VALUES), **kw).fit(Xtr[tr], y[tr])
    staged = staged_cindex(m, Xtr[va], y[va], VALUES)
    refit = [concordance_index_censored(y[va]["event"], y[va]["time"],
                                        GradientBoostingSurvivalAnalysis(n_estimators=k, **kw).fit(Xtr[tr], y[tr])
                                        .predict(Xtr[va]))[0] for k in VALUES]
    return staged, refit


def main():
    out = {}
    for name, f in (("gb_churn_fold0", gb), ("gbs_whas500_fold0", gbs)):
        s, r = f()
        diff = float(np.max(np.abs(np.array(s) - np.array(r))))
        out[name] = {"iterations": VALUES, "staged": s, "refit": r, "max_abs_diff": diff}
        print(name, "max |staged - refit| =", diff)
        assert diff < 1e-12, (name, s, r)
    jdump(out, RESULTS / "check_staged.json")


if __name__ == "__main__":
    main()
