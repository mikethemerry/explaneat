"""The comparator suite on every binary-classification set (exploratory; not pre-registered).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BENCH_JOBS=4 \
        <venv with interpret>/python experiments/benchmarks/classification.py [name ...] [--only model,...]

(one thread per worker: with more, the boosting and MLP fits oversubscribe the cores and run many times
slower; the results do not depend on it)

(the EBM needs `interpret`; the conv-neat worktree venv has it, with the same scikit-learn 1.7.2 as
the uv environment). Reads cache/<name>.npz (export_data.py); no database access. Per dataset and
model: hyperparameters chosen by 5-fold stratified CV on the training rows only (shuffle, seed 42,
scoring ROC AUC), refit on all training rows, the test rows read once. Test AUC with a 95%
percentile interval over 10,000 bootstrap resamples of the test rows (RandomState(42), the 10-seed
study's index matrix), and test accuracy at the default decision rule (predict).

The grids (STAGES below) are design choices, the same for every dataset. They are wide enough that no
CV choice on any of the nine sets lies at either end of its grid, except at the end of a parameter's
range (common.DOMAIN_ENDS: max_depth None, min_samples_leaf 1, ...); make_csv.py asserts this
(results/grid_check.md). Where the product grid is too large to fit, the search runs in stages
(search.py: a coarse-then-fine, one-pass coordinate search on the same folds; the boosting iteration
count is scored from staged predictions of one fit per fold). The narrower grids of 19 Nov 2026
(those of experiments/references) and their results are in results/narrow_grids/.

Results are written after each model (results/classification/<name>.json, one file per dataset so
datasets can run in parallel processes); a rerun skips a model whose stored search equals the current
one and refits it otherwise, reusing any leading stage that is unchanged (search.search, `previous`). XOR (4 rows, no split): an L2 logistic regression and a decision tree
fitted on the truth table, accuracy on the same four rows.
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
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from common import CACHE, CLASSIFICATION, RESULTS, SEED, auc_ci, boot_indices, jdump, jload  # noqa: E402
from search import jsonable, search, staged_auc  # noqa: E402

NJ = int(os.environ.get("BENCH_JOBS", "4"))

# ------------------------------------------------------------------ grids (design choices)
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4]
ALPHA_MLP = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
STAGES = {
    "lr_l2": [{"grid": {"C": C_GRID}}],
    "lr_l1": [{"grid": {"C": C_GRID}}],
    "tree": [{"grid": {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None],
                       "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100, 200]}}],
    # stage 1 at max_features "sqrt" (the default); stage 2 the feature fraction at the stage-1 tree size
    "rf": [{"grid": {"max_depth": [3, 5, 8, 12, 20, None], "min_samples_leaf": [1, 2, 5, 10, 20]}},
           {"grid": {"max_features": [1, "sqrt", 0.5, None]}}],
    # stage 1 rate x tree size x iterations (staged) at min_samples_leaf 20, l2 0 (the defaults);
    # stage 2 leaf size x l2 at the stage-1 choice; stage 3 tree size again, over a wider range; stage 4 the
    # iteration count again (staged), up to 30,000 (the wider ranges only in the later, one-parameter
    # stages: in stage 1 they would multiply its cost)
    "gb": [{"grid": {"learning_rate": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], "max_leaf_nodes": [2, 3, 7, 15, 31, 63, 127],
                     "max_iter": [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]}, "staged": "max_iter"},
           {"grid": {"min_samples_leaf": [1, 5, 20, 50, 100], "l2_regularization": [0.0, 0.1, 1.0, 10.0, 100.0]}},
           {"grid": {"max_leaf_nodes": [2, 3, 7, 15, 31, 63, 127, 255, 511]}},
           {"grid": {"max_iter": [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]}, "staged": "max_iter"}],
    # stage 1 interactions at 256 bins; stage 2 bins at the stage-1 interactions
    "ebm": [{"grid": {"interactions": [0, 5, 10, 20, 50, 100, 200]}, "base": {"max_bins": 256}},
            {"grid": {"max_bins": [16, 32, 64, 256, 1024, 4096, 16384, 32768]}}],
    # stage 1 alpha at one layer of 32; stage 2 the architecture at that alpha; stage 3 alpha again
    "mlp": [{"grid": {"alpha": ALPHA_MLP}, "base": {"hidden_layer_sizes": (32,)}},
            {"grid": {"hidden_layer_sizes": [(2,), (4,), (8,), (16,), (32,), (64,), (128,), (256,), (512,), (1024,),
                                             (32, 16), (64, 32), (128, 64), (256, 128), (512, 256), (1024, 512), (2048, 1024)]}},
            {"grid": {"alpha": ALPHA_MLP}}],
}


def models():
    from interpret.glassbox import ExplainableBoostingClassifier
    return {
        "lr_l2": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED),
        "lr_l1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED),
        "tree": DecisionTreeClassifier(random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=2),
        "gb": HistGradientBoostingClassifier(random_state=0, early_stopping=False),
        "ebm": ExplainableBoostingClassifier(random_state=SEED, n_jobs=2),
        "mlp": MLPClassifier(max_iter=3000, random_state=SEED),
    }


def range_ends(X):
    """Data-dependent ends of a parameter's range (search.edge_hits): EBM interactions at or above the number
    of feature pairs (all pairs) and max_bins at or above the largest number of distinct values of a
    training feature (no feature is binned further) change nothing beyond them."""
    p = X.shape[1]
    nu = max(len(np.unique(X[:, j])) for j in range(p))
    ints = [v for v in STAGES["ebm"][0]["grid"]["interactions"] if v >= p * (p - 1) // 2]
    bins = [v for v in STAGES["ebm"][1]["grid"]["max_bins"] if v >= nu]
    return {"interactions": ints, "max_bins": bins, "n_features": p, "max_distinct_values": nu}


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
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    names = args or list(CLASSIFICATION) + ["xor"]
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
        folds = list(cv.split(Xtr, ytr))
        idx = boot_indices(len(yte))
        res = out.setdefault(name, {})
        for mname, est in models().items():
            if only and mname not in only:
                continue
            spec = jsonable(STAGES[mname])
            if mname in res and res[mname].get("search") == spec:
                continue
            t0 = time.time()
            prev = res.get(mname, {}).get("stages")
            m, params, cv_score, record = search(est, STAGES[mname], Xtr, ytr, folds, NJ, scoring="roc_auc",
                                                 staged_scores=staged_auc, previous=prev)
            p = m.predict_proba(Xte)[:, 1]
            ci, nb = auc_ci(yte, p, idx)
            res[mname] = {"params": jsonable(params), "cv_auc": cv_score,
                          "train_auc": float(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])),
                          "test_auc": float(roc_auc_score(yte, p)), "test_auc_ci95": ci, "n_boot_used": nb,
                          "test_accuracy": float(accuracy_score(yte, m.predict(Xte))),
                          "n_train": int(len(ytr)), "n_test": int(len(yte)), "search": spec, "stages": record,
                          "range_ends": range_ends(Xtr),
                          "seconds": round(time.time() - t0, 1)}
            jdump(out, path)
            print(f"{name:13s} {mname:6s} cv {cv_score:.4f} test {res[mname]['test_auc']:.4f} "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}] acc {res[mname]['test_accuracy']:.4f} {params} "
                  f"{res[mname]['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
