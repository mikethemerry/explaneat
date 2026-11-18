"""Multiclass (k-class) explanations: faithfulness, exact logit terms, readings, stroke rules."""

import numpy as np
import pytest
import torch

from explaneat.conv.explain.multiclass import (explain_network_multi, node_evidence10, profile_digits, reading10,
                                               library10)
from explaneat.conv.explain.strokes import STROKES10, stroke_concepts
from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, minimal_genome
from explaneat.conv.mutation import mutate_add_conn, mutate_add_node

torch.set_num_threads(1)


def _digit(kind, H=16):
    x = np.zeros((H, H), np.float32)
    if kind == "1":
        x[2:14, 7] = 1
    elif kind == "0":
        x[3, 4:12] = 1; x[12, 4:12] = 1; x[3:13, 4] = 1; x[3:13, 11] = 1
    elif kind == "7":
        x[2, 3:13] = 1
        for i in range(10):
            x[3 + i, 12 - i] = 1
    return x


def test_stroke_rules_10():
    X = np.stack([_digit("1"), _digit("0"), _digit("7")])[:, None]
    C = stroke_concepts(X, STROKES10)
    j = {s: i for i, s in enumerate(STROKES10)}
    assert C[0, j["centre_vstroke"]] == 1 and C[0, j["any_loop"]] == 0
    assert C[1, j["any_loop"]] == 1 and C[1, j["centre_vstroke"]] == 0
    assert C[2, j["top_bar"]] == 1
    assert C[1, j["heavy_ink"]] == 0  # 36 ink pixels < 101


def test_profile_digits_modes():
    assert profile_digits([0, 1, 1, 1, 1, 1, 1, 1, 1, 0.9])["mode"] == "except"
    pd = profile_digits([1, 0, 0, 0, 0, 0, 0, 0, 0.8, 0])
    assert pd["mode"] == "on" and pd["digits"] == [0, 8]


def test_reading_names_class_detector():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 30)
    X = rng.random((300, 1, 16, 16)).astype(np.float32) * 0.4
    lib = library10(X, y)
    pooled = (y == 3).astype(float) + rng.random(300) * 0.1
    ev = node_evidence10(pooled, lib, y)
    r = reading10(ev, lib)
    assert ev["status"] == "concept" and r["name"] == "'3' detector" and r["digits"]["digits"] == [3]


@pytest.mark.parametrize("seed", [0])
def test_explain_multi_is_faithful(seed):
    rng = np.random.default_rng(seed)
    cfg = ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=10, max_level=2)
    tr = InnovationTracker(cfg.n_out_maps)
    g = minimal_genome(0, cfg, tr, rng)
    for _ in range(4):
        mutate_add_node(g, cfg, tr, rng)
        mutate_add_conn(g, cfg, tr, rng)
    for c in g.conns.values():
        c.kernel = (rng.standard_normal(c.kernel.shape) * 0.4).astype(np.float32)
    g.head_w = (np.eye(10) + rng.standard_normal((10, 10)) * 0.1).astype(np.float32)
    y = np.repeat(np.arange(10), 12)
    X = np.stack([_digit("1" if d == 1 else "0" if d in (0, 6, 8) else "7") for d in y])[:, None]
    X = np.clip(X + rng.random(X.shape).astype(np.float32) * 0.3, 0, 1)
    data = {"X_test": X, "y_test": y, "X_val": X, "y_val": y}
    E = explain_network_multi(g, cfg, data, n_explain=120, n_imp=120, n_walk=4)
    assert E["check"]["max_abs_logit_diff"] < 1e-9
    assert E["check"]["max_abs_term_sum_err"] < 1e-9
    assert E["check"]["coverage"] == 1.0
    for w in E["walk"]:
        tot = np.sum([t["values"] for t in w["terms"]], axis=0)
        assert np.allclose(tot, w["logits"], atol=1e-9)
        assert w["pred"] == int(np.argmax(w["logits"]))
    hid = [n for n, r in E["nodes"].items() if r["kind"] == "hidden" and r["functional"]]
    assert all("profile" in E["nodes"][n]["sem"] and len(E["nodes"][n]["sem"]["profile"]["fire"]) == 10 for n in hid)
