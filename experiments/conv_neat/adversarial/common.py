"""Shared loading for the adversarial study: ConvNEAT best genomes and dense CNN baselines."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from explaneat.conv import adversarial as ADV  # noqa: E402
from explaneat.conv.baseline import SmallCNN  # noqa: E402
from explaneat.conv.data import digits_splits, synthetic_splits  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.population import genome_from_dict  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "conv_neat", "results")
BASE_DIR = os.path.join(HERE, "baselines")
CNN_EPOCHS = 150  # the "fair" long baseline budget of recheck.py

LINF = [0.01, 0.02, 0.05, 0.1, 0.2]
SIGMA = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def l2_grid(d: int):
    """L2 radii matching the L-inf grid: the L2 norm of a perturbation of size eps on a
    quarter of the pixels, eps * sqrt(d) / 2."""
    return [round(e * np.sqrt(d) / 2, 4) for e in LINF]


def load_data(dataset: str, seed: int):
    if dataset == "synthetic":
        d = synthetic_splits(seed)
        # concept pixel masks for the test split (same generator calls, checked equal)
        X, C, M = ADV.synthetic_with_masks(2000, seed * 10 + 2)
        assert np.array_equal(X, d["X_test"]) and np.array_equal(C, d["c_test"])
        d["m_test"] = M
        return d
    if dataset == "digits38":
        return digits_splits(seed, classes=[3, 8])
    if dataset == "mnist38":
        return mnist38_splits(seed)
    raise ValueError(dataset)


MNIST_NPZ = os.path.join(os.path.expanduser("~"), ".explaneat", "data", "mnist_784.npz")
MNIST_RUNS = os.path.join(HERE, "mnist_runs")  # copies of the other branch's MNIST 3-vs-8 best genomes


def mnist38_splits(seed: int, n_train=10000, n_val=2000):
    """Replica of ``mnist_splits`` on the conv-neat-prototype branch (MNIST 3 vs 8 at 28x28,
    pixels / 255; test = all 1,984 standard-test 3s and 8s, the same for every seed)."""
    from sklearn.model_selection import train_test_split
    d = np.load(MNIST_NPZ)
    X, y = d["X"], d["y"]
    keep = np.isin(y, [3, 8])
    is_tr = np.arange(len(y)) < 60000
    pool, te = np.flatnonzero(keep & is_tr), np.flatnonzero(keep & ~is_tr)
    rest, va = train_test_split(pool, test_size=n_val, stratify=y[pool], random_state=seed)
    if n_train is not None and n_train < len(rest):
        rest, _ = train_test_split(rest, train_size=n_train, stratify=y[rest], random_state=seed)
    tr, va = np.sort(rest), np.sort(va)
    f = lambda rows: (X[rows].reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)  # noqa: E731
    lab = lambda rows: (y[rows] == 8).astype(np.int64)  # noqa: E731
    return {"X_train": f(tr), "y_train": lab(tr), "X_val": f(va), "y_val": lab(va),
            "X_test": f(te), "y_test": lab(te)}


def _run_dir(run: str) -> str:
    return os.path.join(MNIST_RUNS if run.startswith("mnist") else RESULTS, run)


def run_dirs(datasets=("synthetic", "digits38")):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*", "best_genome.json"))
                    + glob.glob(os.path.join(MNIST_RUNS, "*", "best_genome.json"))):
        run = os.path.basename(os.path.dirname(p))
        ds = run.split("_p")[0]
        if ds in datasets:
            out.append(run)
    return out


def load_convneat(run: str):
    with open(os.path.join(_run_dir(run), "results.json")) as f:
        r = json.load(f)
    cfgd = dict(r["config_initial"])
    cfgd["image_hw"] = tuple(cfgd["image_hw"])
    cfgd["kernel_sizes"] = tuple(cfgd["kernel_sizes"])
    cfg = ConvNeatConfig(**cfgd)
    with open(os.path.join(_run_dir(run), "best_genome.json")) as f:
        g = genome_from_dict(json.load(f))
    return g, cfg, r


def load_or_train_cnn(dataset: str, seed: int, width: int, data, log=print):
    """The dense baseline of run.py / recheck.py (same code, seed and 150-epoch budget),
    retrained once and cached as a state dict."""
    os.makedirs(BASE_DIR, exist_ok=True)
    path = os.path.join(BASE_DIR, f"{dataset}_s{seed}_w{width}.pt")
    meta_path = path.replace(".pt", ".json")
    if os.path.exists(path):
        m = SmallCNN(data["X_train"].shape[1], width, 1)
        m.load_state_dict(torch.load(path))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        with open(meta_path) as f:
            return m, json.load(f)
    log(f"  training CNN {dataset} s{seed} w{width} ({CNN_EPOCHS} epochs)")
    m, meta = ADV.train_cnn(data, width, 1, CNN_EPOCHS, seed)
    torch.save(m.state_dict(), path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return m, meta
