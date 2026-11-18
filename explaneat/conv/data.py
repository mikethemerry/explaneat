"""Datasets for the ConvNEAT prototype (all generated or bundled, no download).

Synthetic known-truth task (16x16, values in [0, 1]):
  concepts, each planted independently at a random position:
    hbar  - horizontal bar, length 6-9, 1 px thick        P = 0.707
    blob  - small Gaussian blob (sigma 0.8, ~3x3)         P = 0.707
    vbar  - vertical bar, length 6-9 (distractor)         P = 0.5
    diag  - diagonal line, length 5-7 (distractor)        P = 0.5
  label = hbar AND blob  (P ~= 0.5)
  noise: Gaussian pixel noise (sd 0.08) + sparse salt pixels, clipped to [0, 1].
Because the concepts are independent, a node's correlation with one concept is
not confounded by the others (except through the label, which is reported as
its own column, "label=hbar&blob").
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

CONCEPTS = ("hbar", "blob", "vbar", "diag")
P_CONCEPT = {"hbar": 0.7071, "blob": 0.7071, "vbar": 0.5, "diag": 0.5}

# Every generator parameter, in one place (the report's Data panel reads this).
# Lengths are inclusive ranges; intensities are uniform ranges; all draws use the
# image's own NumPy Generator in the order: hbar, blob, vbar, diag, then noise.
SYNTH = {
    "hw": (16, 16),
    "hbar": {"length": (6, 9), "thickness": 1, "intensity": (0.7, 1.0),
             "position": "row uniform over all rows; start column uniform so the bar fits"},
    "vbar": {"length": (6, 9), "thickness": 1, "intensity": (0.7, 1.0),
             "position": "column uniform over all columns; start row uniform so the bar fits"},
    "diag": {"length": (5, 7), "thickness": 1, "intensity": (0.7, 1.0),
             "position": "top-left corner uniform so the line fits; direction down-right or down-left with p = 0.5"},
    "blob": {"sigma": 0.8, "peak": (0.8, 1.0), "centre": (1.5, 2.5),
             "position": "centre (row, col) uniform in [1.5, H - 2.5] x [1.5, W - 2.5] (sub-pixel)"},
    "noise_sd": 0.08, "salt_p": 0.02, "salt_value": (0.3, 0.8),
    "combine": "shapes are combined by pixel-wise max; then Gaussian noise is added, then salt pixels "
               "are overwritten, then the image is clipped to [0, 1]",
    "sizes": {"train": 3000, "val": 1000, "test": 2000},
    "seeds": "split s of run seed k uses Generator(10 k + {train: 0, val: 1, test: 2})",
}


def _draw(img, rng, concept):
    H, W = img.shape
    if concept == "hbar":
        p = SYNTH["hbar"]
        L = int(rng.integers(p["length"][0], p["length"][1] + 1)); r = int(rng.integers(0, H)); c = int(rng.integers(0, W - L + 1))
        img[r, c:c + L] = np.maximum(img[r, c:c + L], rng.uniform(*p["intensity"]))
    elif concept == "vbar":
        p = SYNTH["vbar"]
        L = int(rng.integers(p["length"][0], p["length"][1] + 1)); c = int(rng.integers(0, W)); r = int(rng.integers(0, H - L + 1))
        img[r:r + L, c] = np.maximum(img[r:r + L, c], rng.uniform(*p["intensity"]))
    elif concept == "diag":
        p = SYNTH["diag"]
        L = int(rng.integers(p["length"][0], p["length"][1] + 1)); r = int(rng.integers(0, H - L + 1)); c = int(rng.integers(0, W - L + 1))
        flip = rng.random() < 0.5
        v = rng.uniform(*p["intensity"])
        for i in range(L):
            cc = c + (L - 1 - i if flip else i)
            img[r + i, cc] = max(img[r + i, cc], v)
    elif concept == "blob":
        p = SYNTH["blob"]
        lo, hi = p["centre"]
        cy, cx = rng.uniform(lo, H - hi), rng.uniform(lo, W - hi)
        yy, xx = np.mgrid[0:H, 0:W]
        b = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * p["sigma"] ** 2)) * rng.uniform(*p["peak"])
        np.maximum(img, b, out=img)


def make_synthetic(n: int, seed: int, hw=(16, 16)) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, *hw), dtype=np.float32)
    C = np.zeros((n, len(CONCEPTS)), dtype=np.int64)
    for i in range(n):
        img = np.zeros(hw, dtype=np.float32)
        for j, c in enumerate(CONCEPTS):
            if rng.random() < P_CONCEPT[c]:
                C[i, j] = 1
                _draw(img, rng, c)
        img += rng.normal(0, SYNTH["noise_sd"], hw).astype(np.float32)
        salt = rng.random(hw) < SYNTH["salt_p"]
        img[salt] = rng.uniform(*SYNTH["salt_value"], salt.sum())
        X[i, 0] = np.clip(img, 0, 1)
    y = (C[:, 0] & C[:, 1]).astype(np.int64)
    return {"X": X, "y": y, "concepts": C}


def synthetic_splits(seed: int, n_train=3000, n_val=1000, n_test=2000):
    out = {}
    for s, n, off in (("train", n_train, 0), ("val", n_val, 1), ("test", n_test, 2)):
        d = make_synthetic(n, seed * 10 + off)
        out[f"X_{s}"], out[f"y_{s}"], out[f"c_{s}"] = d["X"], d["y"], d["concepts"]
    out["concept_names"] = list(CONCEPTS)
    return out


def digits_splits(seed: int, classes=None):
    """sklearn load_digits (8x8, bundled), pixels / 16 -> [0, 1]. 60/20/20 stratified."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)[:, None]
    y = d.target.astype(np.int64)
    if classes is not None:
        m = np.isin(y, classes)
        X, y = X[m], y[m]
        y = np.searchsorted(np.array(sorted(classes)), y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xte, yte, test_size=0.5, stratify=yte, random_state=seed)
    return {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva, "X_test": Xte, "y_test": yte}


MNIST_NPZ = os.path.join(os.path.expanduser("~"), ".explaneat", "data", "mnist_784.npz")
MNIST_N_STD_TRAIN = 60000


def load_mnist_raw(path: str = MNIST_NPZ):
    """(X uint8 [70000, 784], y int64 [70000]) from OpenML ``mnist_784`` v1.

    Cached as ``~/.explaneat/data/mnist_784.npz`` (the same cache as the main repo's
    ``experiments/mnist/data.py``). On first use it is fetched once with
    ``sklearn.datasets.fetch_openml`` (no extra dependency). OpenML row order is the
    standard MNIST order: rows 0..59,999 are the standard training set, 60,000..69,999 the test set.
    """
    if not os.path.exists(path):
        from sklearn.datasets import fetch_openml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                            data_home=os.path.dirname(path), return_X_y=True)
        np.savez_compressed(path, X=X.astype(np.uint8), y=y.astype(np.int64))
    d = np.load(path)
    return d["X"], d["y"]


def mnist_splits(seed: int, classes=(3, 8), n_train=10000, n_val=2000, raw=None):
    """Real MNIST at 28x28, pixels / 255 -> [0, 1].

    Test = every standard-test image of ``classes`` (all 10,000 for 10 classes; 1,984 for 3 vs 8).
    Validation = ``n_val`` images carved from the standard training images of ``classes``
    (stratified, seeded by ``seed``); train = ``n_train`` of the remaining ones (stratified,
    same rng; all of them if fewer remain). Labels are ranks in sorted(classes), so for
    3 vs 8, y = 1 means "8". Row indices into the 70,000-row array are returned too.
    """
    from sklearn.model_selection import train_test_split
    X, y = raw if raw is not None else load_mnist_raw()
    classes = sorted(classes)
    keep = np.isin(y, classes)
    is_tr = np.arange(len(y)) < MNIST_N_STD_TRAIN
    pool, te = np.flatnonzero(keep & is_tr), np.flatnonzero(keep & ~is_tr)
    rest, va = train_test_split(pool, test_size=n_val, stratify=y[pool], random_state=seed)
    if n_train is not None and n_train < len(rest):
        rest, _ = train_test_split(rest, train_size=n_train, stratify=y[rest], random_state=seed)
    tr = np.sort(rest)
    va = np.sort(va)
    lab = np.array(classes)

    def f(rows):
        return (X[rows].reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)

    return {"X_train": f(tr), "y_train": np.searchsorted(lab, y[tr]).astype(np.int64),
            "X_val": f(va), "y_val": np.searchsorted(lab, y[va]).astype(np.int64),
            "X_test": f(te), "y_test": np.searchsorted(lab, y[te]).astype(np.int64),
            "rows": {"train": tr, "val": va, "test": te}, "classes": classes}
