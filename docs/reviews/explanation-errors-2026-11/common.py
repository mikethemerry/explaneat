"""Shared helpers for the reproduction scripts.

Read-only: the database is read only through the verify cache (``~/.explaneat/verify_cache``,
filled by ``explaneat verify`` from a READ ONLY transaction); the operations come from the
pre-errata snapshot ``experiments/errata/2026-11-11/snapshot/<label>/operations.json``.
Run from this directory: ``uv run python <script>.py``.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
SNAP = REPO / "experiments/errata/2026-11-11/snapshot"

from explaneat.verify import model as vmod            # noqa: E402
from explaneat.db.dataset_utils import sample_dataset  # noqa: E402


def snapshot(label):
    d = SNAP / label
    rec = json.load(open(d / "record.json"))
    ops = json.load(open(d / "operations.json"))
    ops = ops if isinstance(ops, list) else ops.get("operations", ops)
    return rec, ops


def load(label, upto=None, genome_id=None, ops=None):
    """VerifyModel on the pre-errata operations (optionally only ops with seq <= upto)."""
    if ops is None:
        rec, ops = snapshot(label)
        genome_id = genome_id or rec["genome_id"]
    if upto is not None:
        ops = [o for o in ops if o.get("seq", 0) <= upto]
    return vmod.load(genome_id, operations=ops)


def vals(vm, X):
    return vm.m.ev.forward(np.asarray(X, dtype=float))


def logit(vm, X):
    return vm.m.ev.logit(np.asarray(X, dtype=float))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def rows(vm, split):
    """(X z-scaled, y, raw X) of a split: 'train', 'test' or 'both' (train+test)."""
    b = vm.base
    if split == "train":
        return b.X_train, b.y_train, b.raw_train
    if split == "test":
        return b.X_test, b.y_test, b.raw_test
    return (np.vstack([b.X_train, b.X_test]), np.concatenate([b.y_train, b.y_test]),
            np.vstack([b.raw_train, b.raw_test]))


def mcp_default_sample(vm, split="test", max_samples=200, fraction=1.0):
    """The rows an MCP evidence tool sees with its defaults (split='test', max_samples=200,
    sample_fraction=1.0, seed 42): (X z-scaled, y, raw X)."""
    X, y, raw = rows(vm, split)
    n = len(X)
    k = max(min(int(n * fraction), max_samples), 1)
    if k >= n:
        return X, y, raw
    idx = np.random.RandomState(42).choice(n, size=k, replace=False)
    return X[idx], y[idx], raw[idx]


def col(vm, name):
    return vm.m.feature_names.index(name)
