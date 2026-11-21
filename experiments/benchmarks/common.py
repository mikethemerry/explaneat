"""Shared registry for the comparator suite (experiments/benchmarks/).

Every dataset of Chapter 7 Experiments 1 and 2, on the exact split its PropNEAT runs used. The
classification sets are read from the database split the runs were trained on (inputs scaled with
the split's stored StandardScaler, as experiments/10seed/compute_runs.load_split); the survival sets
from the Cox case's own split files (scaler fitted on the training rows, as the runs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / "cache"
RESULTS = Path(os.environ.get("BENCH_RESULTS", HERE / "results"))   # BENCH_RESULTS: a scratch dir (determinism checks)

SEED = 42
N_BOOT = 10_000

# name -> (database, split id, what the PropNEAT runs used it for)
CLASSIFICATION = {
    "heart": ("explaneat_v4", "165cebbd-6609-46ca-867e-c09e17b2d6ec",
              "Heart (paper split; the paper genome and Heart-100)"),
    "monk2": ("explaneat_v4", "63257a45-49c9-42a6-8b28-8e346eab2bee", "MONK-2 (paper split)"),
    "adult": ("explaneat_v4", "1df50169-8095-40d1-b61a-30bb541cd313", "Adult (paper split)"),
    "churn": ("explaneat_v4", "45962ea6-3c78-49f9-a01e-3665f6809b1e", "Churn (churn100 split)"),
    "c4_cells": ("explaneat_dev", "69849f7e-1ff7-4c06-9784-1a7bc79e358f",
                 "Connect-4, 42 raw cells (c4_capacity)"),
    "c4_lines": ("explaneat_dev", "1bdaf11a-cb76-4d2c-84e8-2436313a42c5",
                 "Connect-4, 16 line features (c4_line_features, base 'lines')"),
    "shape_u": ("explaneat_dev", "fc59adac-cf59-4480-a2d7-7ec2fce379eb", "Synthetic U (shape-A)"),
    "shape_saddle": ("explaneat_dev", "cc5a37ed-efb3-4619-9e67-52343ed233aa", "Synthetic saddle (shape-B)"),
    "shape_both": ("explaneat_dev", "a80f5dcd-fa1a-42eb-ad10-1f40a70115bb", "Synthetic U + saddle (shape-C)"),
}
SURVIVAL = ("whas500", "gbsg2")

# Ends of a parameter's range: a grid that holds one of these on that side cannot be widened there, so a
# choice of it is not a grid-boundary choice (search.edge_hits). max_depth None and max_features None are
# "no limit"; min_samples_leaf 1, l2_regularization 0, interactions 0, max_features 1 (one feature per
# split), max_depth 1 (stumps), one boosting iteration and l1_ratio 1 (the lasso) are the other end of their parameter's range.
DOMAIN_ENDS = {"max_depth": [None, 1], "min_samples_leaf": [1], "l2_regularization": [0.0], "interactions": [0],
               "max_features": [1, None], "l1_ratio": [1.0], "max_iter": [1], "n_estimators": [1],
               "max_leaf_nodes": [2], "alpha": [0.0]}


def boot_indices(n, n_boot=N_BOOT, seed=SEED):
    """The 10-seed study's index matrix: RandomState(42).randint(0, n, n), drawn n_boot times in sequence."""
    rng = np.random.RandomState(seed)
    return np.stack([rng.randint(0, n, size=n) for _ in range(n_boot)])


def auc_ci(y, p, idx):
    """95% percentile interval of the test AUC over the resamples (single-class resamples skipped)."""
    from sklearn.metrics import roc_auc_score
    v = [roc_auc_score(y[i], p[i]) for i in idx if y[i].min() != y[i].max()]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], len(v)


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())
