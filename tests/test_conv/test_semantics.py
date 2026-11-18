"""MNIST loader, stroke concepts and plain-language readings (explaneat/conv)."""

import numpy as np
import pytest
import torch

from explaneat.conv.data import SYNTH, make_synthetic, mnist_splits
from explaneat.conv.explain import semantics as SM
from explaneat.conv.explain.pipeline import explain_network
from explaneat.conv.explain.strokes import STROKES, stroke_concepts, validate
from explaneat.conv.genome import ConvNeatConfig, InnovationTracker
from explaneat.conv.mutation import random_genome_by_mutation

torch.set_num_threads(1)


def _fake_raw(n_train=600, n_test=200, seed=0):
    """A stand-in for the 70,000-row OpenML array: same layout, first rows = 'standard train'."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.integers(0, 256, size=(n, 784), dtype=np.uint8)
    y = rng.integers(0, 10, size=n).astype(np.int64)
    return X, y, n_train


def test_mnist_splits_layout(monkeypatch):
    X, y, ntr = _fake_raw()
    monkeypatch.setattr("explaneat.conv.data.MNIST_N_STD_TRAIN", ntr)
    d = mnist_splits(3, classes=(8, 3), n_train=50, n_val=20, raw=(X, y))
    r = d["rows"]
    # test = every standard-test 3 and 8; train / val from standard train, disjoint
    assert set(r["test"]) == set(np.flatnonzero(np.isin(y, [3, 8]) & (np.arange(len(y)) >= ntr)))
    assert r["train"].max() < ntr and r["val"].max() < ntr and not set(r["train"]) & set(r["val"])
    assert len(r["val"]) == 20 and len(r["train"]) == 50
    # labels: rank in sorted(classes), so 8 -> 1; images 28x28 in [0, 1]
    assert np.array_equal(d["y_test"], (y[r["test"]] == 8).astype(int))
    assert d["X_train"].shape[1:] == (1, 28, 28) and 0 <= d["X_train"].min() and d["X_train"].max() <= 1
    assert np.allclose(d["X_val"][0, 0].ravel(), X[r["val"][0]] / 255.0)
    # seeded: same seed, same split; other seed, other split
    d2 = mnist_splits(3, classes=(3, 8), n_train=50, n_val=20, raw=(X, y))
    d3 = mnist_splits(4, classes=(3, 8), n_train=50, n_val=20, raw=(X, y))
    assert np.array_equal(d2["rows"]["val"], r["val"]) and not np.array_equal(d3["rows"]["val"], r["val"])


def _digit(kind, H=28):
    """Hand-drawn 8 / 3 / 0-like images on an HxH canvas (strokes 2 px thick)."""
    x = np.zeros((H, H))
    s = H // 28
    if kind == "8":
        yy, xx = np.mgrid[0:H, 0:H]
        for cy, r in ((7.5, 4.5), (20.0, 5.0)):  # two rings ...
            d = np.hypot(yy - cy * s, xx - 14 * s)
            x[(d >= (r - 0.9) * s) & (d <= (r + 0.7) * s)] = 1
        x[11 * s:16 * s, 13 * s:16 * s] = 1      # ... joined by a narrow waist
    elif kind == "3":
        x[4:6, 8:19] = 1           # top stroke
        x[4:24, 17:19] = 1         # right spine
        x[13:15, 11:19] = 1        # middle prong
        x[22:24, 8:19] = 1         # bottom stroke
    return x


def test_stroke_rules_on_drawn_digits():
    C = stroke_concepts(np.stack([_digit("8"), _digit("3")])[:, None])
    c8, c3 = dict(zip(STROKES, C[0])), dict(zip(STROKES, C[1]))
    assert c8["upper_loop"] == 1 and c8["lower_loop"] == 1 and c8["open_left"] == 0 and c8["waist"] == 1
    assert c3["upper_loop"] == 0 and c3["lower_loop"] == 0 and c3["open_left"] == 1 and c3["top_bar"] == 1
    assert c3["left_heavy"] == 0
    v = validate(C, np.array([1, 0]), ["3", "8"])
    assert v["upper_loop"] == {"3": 0.0, "8": 1.0}


def test_synthetic_spec_matches_generator():
    """The Data panel reads SYNTH; check the generated images obey it."""
    d = make_synthetic(400, 11)
    assert d["X"].shape[2:] == SYNTH["hw"] and d["X"].min() >= 0 and d["X"].max() <= 1
    assert np.array_equal(d["y"], d["concepts"][:, 0] & d["concepts"][:, 1])
    assert 0.35 < d["y"].mean() < 0.65


def test_reading_rules():
    rng = np.random.default_rng(0)
    n = 400
    Cp = rng.integers(0, 2, size=(n, 4))
    data = {"c_test": Cp}
    lib = SM.concept_library("synthetic", data, ("absent", "present"))
    y = Cp[:, 0] & Cp[:, 1]
    # a perfect blob detector
    ev = SM.node_evidence(Cp[:, 1] * 2.0 + rng.random(n) * 0.1, lib, y)
    assert ev["status"] == "concept" and ev["concepts"] == ["blob"]
    assert ev["rates"]["blob"]["with"] == 1.0 and ev["rates"]["blob"]["without"] == 0.0
    assert SM.short_name(ev, lib) == "blob detector" and SM.concept_phrase(ev, lib, ("absent", "present")) == "Detects blobs"
    # vertical bars AND diagonals (independent concepts): both named
    ev = SM.node_evidence(Cp[:, 2] + Cp[:, 3] + rng.random(n) * 0.1, lib, y)
    assert set(ev["concepts"]) == {"vbar", "diag"}
    # noise: unexplained; all zero: inactive
    assert SM.node_evidence(rng.random(n), lib, y)["status"] == "unexplained"
    ev = SM.node_evidence(np.zeros(n), lib, y)
    assert ev["status"] == "inactive" and ev["fire_rate"] == 0.0
    # the label concept is recognised, and hbar is not repeated next to it (confounded)
    ev = SM.node_evidence(y * 1.0 + rng.random(n) * 0.1, lib, y)
    assert ev["concepts"] == ["both"]


def test_pipeline_readings_on_a_random_genome():
    cfg = ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=1, max_level=2)
    g = random_genome_by_mutation(0, cfg, InnovationTracker(1), np.random.default_rng(5), n_steps=25)
    tr_d, va, te = make_synthetic(50, 1), make_synthetic(120, 2), make_synthetic(150, 3)
    data = {"X_train": tr_d["X"], "y_train": tr_d["y"], "X_val": va["X"], "y_val": va["y"],
            "X_test": te["X"], "y_test": te["y"], "c_test": te["concepts"]}
    r = explain_network(g, cfg, data, "synthetic", ["hbar", "blob", "vbar", "diag"], n_walk=6)
    sm = r["semantics"]
    assert sum(sm["counts"].values()) == sm["n"] == r["size"]["hidden_functional"]
    for k, n in r["nodes"].items():
        if n["kind"] == "hidden":
            assert n["sem"]["phrase"] and n["sem"]["name"]
    for p in r["parts"]:
        assert p["sem"]["name"] and "P" not in p["sem"]["name"].split()[0]
    assert all(w["narrative"].startswith("This image") for w in r["walk"])
    assert len(sm["what"]) >= 2
