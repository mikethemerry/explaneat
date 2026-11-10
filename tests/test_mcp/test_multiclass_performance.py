"""compute_performance's k-output branch (MNIST 10-class, identity logits)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mcp_server.tools.evidence import _multiclass_performance


def test_multiclass_metrics_match_torch():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 10))
    y = rng.integers(0, 10, 200)
    logits[np.arange(100), y[:100]] += 3.0  # half the rows clearly right
    r = _multiclass_performance(logits, y.astype(float), "identity")
    ce = float(F.cross_entropy(torch.as_tensor(logits), torch.as_tensor(y)))
    assert r["cross_entropy"] == pytest.approx(ce, abs=1e-12)
    assert r["accuracy"] == pytest.approx(float(np.mean(logits.argmax(1) == y)))
    assert r["n_outputs"] == 10 and r["n_samples"] == 200
    assert 0.5 < r["auc_roc_ovr_macro"] <= 1.0
    assert set(r["per_class_accuracy"]) <= {str(c) for c in range(10)}


def test_non_identity_outputs_report_accuracy_only():
    out = np.array([[0.9, 0.1], [0.2, 0.8]])
    r = _multiclass_performance(out, np.array([0, 0]), "sigmoid")
    assert r["accuracy"] == 0.5 and "cross_entropy" not in r
