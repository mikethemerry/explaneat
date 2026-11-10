"""MNIST data preparation for the exploratory MNIST runs (not pre-registered).

Source: OpenML ``mnist_784`` (version 1) via ``sklearn.datasets.fetch_openml``,
cached outside the repo as ``~/.explaneat/data/mnist_784.npz`` (uint8 pixels,
int labels).  OpenML row order is the standard MNIST order: rows 0..59999 are
the standard training set, rows 60000..69999 the standard test set.

Two tasks:

``mnist38``  digits 3 (y=0) vs 8 (y=1), 28x28 -> 14x14 by 2x2 mean pooling
             (196 inputs), pixels scaled to [0, 1] (raw / 255, then pooled).
             Train: a seeded subset of the standard-train 3s and 8s
             (``n_train``; default all 11,982).  Test: all standard-test 3s and
             8s (1,984 rows).
``mnist10``  all ten digits, 784 inputs, scaled to [0, 1] (raw / 255).
             Train: a stratified seeded subset of standard train (``n_train``,
             default 10,000; equal share per class up to rounding).  Test: the
             full standard test set (10,000 rows).

No z-scoring: [0, 1] keeps every input a pixel intensity, which is easier to
read in explanations; constant (always-0) border pixels are left in.

The subset seed (``SUBSET_SEED`` = 12345) is independent of the evolution seed,
so every evolution seed sees the same rows.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("EXPLANEAT_DATA", Path.home() / ".explaneat" / "data"))
NPZ = CACHE / "mnist_784.npz"
SUBSET_SEED = 12345
N_STD_TRAIN = 60000


def load_raw():
    """(X uint8 [70000, 784], y int64 [70000]); downloads + caches on first use."""
    if not NPZ.exists():
        from sklearn.datasets import fetch_openml

        CACHE.mkdir(parents=True, exist_ok=True)
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                            data_home=str(CACHE), return_X_y=True)
        np.savez_compressed(NPZ, X=X.astype(np.uint8), y=y.astype(np.int64))
    d = np.load(NPZ)
    return d["X"], d["y"]


def pool2x2(X01: np.ndarray) -> np.ndarray:
    """[N, 784] in [0, 1] -> [N, 196] by 2x2 mean pooling (row-major 14x14)."""
    n = X01.shape[0]
    return X01.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(n, 196)


def feature_names(task: str):
    if task == "mnist38":
        return [f"p{r:02d}_{c:02d}" for r in range(14) for c in range(14)]
    return [f"px{r:02d}_{c:02d}" for r in range(28) for c in range(28)]


def prepare(task: str, n_train: int | None = None):
    """Return dict(X_train, y_train, X_test, y_test, train_rows, test_rows, meta).

    ``train_rows`` / ``test_rows`` are indices into the 70,000-row OpenML array,
    so the subset is fully recorded.
    """
    X, y = load_raw()
    rng = np.random.default_rng(SUBSET_SEED)
    is_train = np.arange(len(y)) < N_STD_TRAIN
    if task == "mnist38":
        keep = (y == 3) | (y == 8)
        tr_pool = np.flatnonzero(keep & is_train)
        te_rows = np.flatnonzero(keep & ~is_train)
        if n_train is None or n_train >= len(tr_pool):
            tr_rows = tr_pool
        else:
            tr_rows = np.sort(rng.choice(tr_pool, size=n_train, replace=False))
        f = lambda rows: pool2x2(X[rows].astype(np.float64) / 255.0)  # noqa: E731
        lab = lambda rows: (y[rows] == 8).astype(np.float64)  # noqa: E731
        meta = dict(task="mnist38", classes={"0": "digit 3", "1": "digit 8"},
                    preprocessing="raw/255 then 2x2 mean pool -> 14x14 (196)")
    elif task == "mnist10":
        n_train = 10000 if n_train is None else n_train
        tr_idx = np.flatnonzero(is_train)
        per = [n_train // 10 + (1 if c < n_train % 10 else 0) for c in range(10)]
        parts = [rng.choice(tr_idx[y[tr_idx] == c], size=per[c], replace=False)
                 for c in range(10)]
        tr_rows = np.sort(np.concatenate(parts))
        te_rows = np.flatnonzero(~is_train)
        f = lambda rows: X[rows].astype(np.float64) / 255.0  # noqa: E731
        lab = lambda rows: y[rows].astype(np.int64)  # noqa: E731
        meta = dict(task="mnist10", classes=[str(c) for c in range(10)],
                    preprocessing="raw/255 (784 inputs)")
    else:
        raise ValueError(task)
    meta.update(subset_seed=SUBSET_SEED, n_train=int(len(tr_rows)), n_test=int(len(te_rows)),
                source="OpenML mnist_784 v1 (rows <60000 = standard train)")
    return dict(X_train=f(tr_rows), y_train=lab(tr_rows), X_test=f(te_rows), y_test=lab(te_rows),
                train_rows=tr_rows, test_rows=te_rows, meta=meta)
