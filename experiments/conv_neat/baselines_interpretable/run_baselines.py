"""Interpretable comparators for the ExConNEAT MNIST 10-class network, on the run's exact split.

  OMP_NUM_THREADS=2 .venv/bin/python experiments/conv_neat/baselines_interpretable/run_baselines.py --part lr
  parts: lr, l1, tree, hgb, ebm (each appends to results.json); CPU only, at most 2 threads.

Split: explaneat.conv.data.mnist_splits(seed=0, classes=0..9, n_val=10000) as in run_long.py:
validation = 10,000 rows carved from the 60,000 standard training images (stratified, seed 0);
"10k" = the run's n_train=10000 subset of the remaining 50,000 (same seed); "50k" = all 50,000;
test = the standard 10,000. Pixels / 255. Every setting is chosen on validation; test is read
once per final model.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import numpy as np  # noqa: E402

from explaneat.conv.data import mnist_splits  # noqa: E402

warnings.filterwarnings("ignore")
RES = os.path.join(HERE, "results.json")
MODELS = os.path.join(HERE, "models")


def load(pool):
    d = mnist_splits(0, classes=tuple(range(10)), n_train=10000 if pool == "10k" else None, n_val=10000)
    f = lambda a: a.reshape(len(a), -1).astype(np.float64)  # noqa: E731
    return f(d["X_train"]), d["y_train"], f(d["X_val"]), d["y_val"], f(d["X_test"]), d["y_test"], d["rows"]


def acc(m, X, y):
    return float((m.predict(X) == y).mean())


def save(key, rec):
    import fcntl
    with open(RES + ".lock", "w") as lk:  # two parts may run at once
        fcntl.flock(lk, fcntl.LOCK_EX)
        r = json.load(open(RES)) if os.path.exists(RES) else {}
        r[key] = rec
        json.dump(r, open(RES, "w"), indent=1)
    print(key, json.dumps({k: v for k, v in rec.items() if k != "tuning"}), flush=True)


def keep(name, m):
    os.makedirs(MODELS, exist_ok=True)
    pickle.dump(m, open(os.path.join(MODELS, name + ".pkl"), "wb"))


def part_lr(pool, D):
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, Xva, yva, Xte, yte, _ = D
    tun = []
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        t = time.time()
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
        tun.append({"C": C, "val": acc(m, Xva, yva), "secs": round(time.time() - t, 1)})
        print(" L2", pool, tun[-1], flush=True)
        if C == 1.0:
            ref = m
    best = max(tun, key=lambda r: r["val"])
    m = LogisticRegression(C=best["C"], max_iter=2000).fit(Xtr, ytr)
    keep(f"lr_l2_{pool}", m)
    save(f"lr_l2_{pool}", {"model": "logistic regression, L2 (lbfgs, multinomial)", "pool": pool,
                           "n_train": len(ytr), "C": best["C"], "val": best["val"], "test": acc(m, Xte, yte),
                           "val_C1_default": acc(ref, Xva, yva), "params": int(m.coef_.size + m.intercept_.size),
                           "tuning": tun})


def part_l1(pool, D):
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, Xva, yva, Xte, yte, _ = D
    tun, models = [], {}
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        t = time.time()
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=300, tol=1e-3, random_state=0).fit(Xtr, ytr)
        nz = int((m.coef_ != 0).sum())
        tun.append({"C": C, "val": acc(m, Xva, yva), "nonzero_coef": nz,
                    "pixels_used": int((m.coef_ != 0).any(0).sum()), "n_iter": int(m.n_iter_.max()),
                    "secs": round(time.time() - t, 1)})
        models[C] = m
        print(" L1", pool, tun[-1], flush=True)
    best = max(tun, key=lambda r: r["val"])
    m = models[best["C"]]
    keep(f"lr_l1_{pool}", m)
    save(f"lr_l1_{pool}", {"model": "logistic regression, L1 (saga, multinomial)", "pool": pool,
                           "n_train": len(ytr), "C": best["C"], "val": best["val"], "test": acc(m, Xte, yte),
                           "nonzero_coef": best["nonzero_coef"], "nonzero_with_intercepts": best["nonzero_coef"] + 10,
                           "pixels_used": best["pixels_used"], "params_dense": 7850, "tuning": tun})


def part_tree(pool, D):
    from sklearn.tree import DecisionTreeClassifier
    Xtr, ytr, Xva, yva, Xte, yte, _ = D
    tun, models = [], {}
    for depth in list(range(4, 31, 2)) + [None]:
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
        tun.append({"max_depth": depth, "val": acc(m, Xva, yva), "leaves": int(m.get_n_leaves()),
                    "depth": int(m.get_depth())})
        models[depth] = m
        print(" tree", pool, tun[-1], flush=True)
    for depth in (4, 6, 8):
        m = models[depth]
        keep(f"tree_d{depth}_{pool}", m)
        r = next(t for t in tun if t["max_depth"] == depth)
        save(f"tree_d{depth}_{pool}", {"model": f"CART, depth {depth} (readable)", "pool": pool, "n_train": len(ytr),
                                       "max_depth": depth, "val": r["val"], "test": acc(m, Xte, yte),
                                       "leaves": r["leaves"]})
    best = max(tun, key=lambda r: r["val"])  # first max: the shallowest best
    m = models[best["max_depth"]]
    keep(f"tree_best_{pool}", m)
    save(f"tree_best_{pool}", {"model": "CART, best depth on validation (not readable)", "pool": pool,
                               "n_train": len(ytr), "max_depth": best["max_depth"], "depth": best["depth"],
                               "val": best["val"], "test": acc(m, Xte, yte), "leaves": best["leaves"], "tuning": tun})


def part_hgb(pool, D):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr, Xva, yva, Xte, yte, _ = D
    t = time.time()
    m = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    n_trees = int(m.n_iter_ * 10)
    leaves = int(sum(p.get_n_leaf_nodes() for it in m._predictors for p in it))
    save(f"hgb_{pool}", {"model": "HistGradientBoostingClassifier (sklearn defaults)", "pool": pool,
                         "n_train": len(ytr), "val": acc(m, Xva, yva), "test": acc(m, Xte, yte),
                         "n_iter": int(m.n_iter_), "early_stopping_used": bool(m.do_early_stopping_),
                         "trees": n_trees, "leaves": leaves, "secs": round(time.time() - t, 1)})


def part_ebm(pool, D, configs):
    from interpret.glassbox import ExplainableBoostingClassifier
    import interpret
    Xtr, ytr, Xva, yva, Xte, yte, _ = D
    tun, models = [], {}
    for name, kw in configs:
        t = time.time()
        m = ExplainableBoostingClassifier(interactions=0, n_jobs=2, random_state=0, **kw).fit(Xtr, ytr)
        terms = len(m.term_features_)
        live = int(sum(np.abs(s).max() > 0 for s in m.term_scores_))
        tun.append({"config": name, "kwargs": kw, "val": acc(m, Xva, yva), "terms": terms,
                    "terms_nonzero": live, "secs": round(time.time() - t, 1)})
        models[name] = m
        print(" ebm", pool, tun[-1], flush=True)
    best = max(tun, key=lambda r: r["val"])
    m = models[best["config"]]
    keep(f"ebm_{pool}", m)
    bins = int(sum(np.prod(s.shape[:-1]) for s in m.term_scores_))  # bins per term (incl. missing/unknown), x 10 classes
    save(f"ebm_{pool}", {"model": "EBM (interpret, multiclass, 784 pixels, interactions=0)", "pool": pool,
                         "n_train": len(ytr), "interpret_version": interpret.__version__, "config": best["config"],
                         "kwargs": best["kwargs"], "val": best["val"], "test": acc(m, Xte, yte),
                         "terms": best["terms"], "terms_nonzero": best["terms_nonzero"],
                         "bins_total": bins, "scores_total": bins * 10, "secs": best["secs"], "tuning": tun})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True, choices=["lr", "l1", "tree", "hgb", "ebm"])
    p.add_argument("--pool", default="10k", choices=["10k", "50k"])
    a = p.parse_args()
    D = load(a.pool)
    print(a.part, a.pool, "train", len(D[1]), "val", len(D[3]), "test", len(D[5]), flush=True)
    if a.part == "ebm":
        cfgs = [("default_bags8", {"outer_bags": 8}), ("lr0.03_bags8", {"outer_bags": 8, "learning_rate": 0.03})]
        if a.pool == "50k":
            cfgs = [c for c in cfgs if c[0] == json.load(open(RES))["ebm_10k"]["config"]]
        part_ebm(a.pool, D, cfgs)
    else:
        globals()["part_" + a.part](a.pool, D)
