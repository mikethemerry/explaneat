"""v5f: partial-feature selection and reproduction, the deep-dive boundary claim (2 and 3 inputs),
and readable category charts."""

import numpy as np
import pytest

from explaneat.analysis.plain_language import PARTIAL_MAX, build_layer
from explaneat.analysis.plain_language_store import load_layer
from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import META, _node


def _layer(label):
    lay = load_layer(label)
    if lay is None or "partials" not in (lay.get("graph") or {}):
        pytest.skip(f"{label} layer without partial features")
    return lay


@pytest.mark.parametrize("label", ["heart-paper", "churn100-s4"])
def test_partial_selection(label):
    lay = _layer(label)
    parts = {p["id"]: p for p in lay["parts"]}
    sel = lay["graph"]["partials"]
    assert 0 < len(sel) <= PARTIAL_MAX
    for r in sel:
        p = parts[r["region"]]
        assert p["nonlinear"] and p["role"] == "part"        # never a linear or additive part
        assert r["why"].startswith(("output of the non-linear part", "combines"))
    imp = [r["importance"] for r in sel]
    assert imp == sorted(imp, reverse=True)                   # ranked by importance


@pytest.mark.parametrize("label", ["heart-paper", "churn100-s4"])
def test_partial_reproduces_the_activation(label):
    lay = _layer(label)
    for r in lay["graph"]["partials"]:
        assert r.get("reproduction_max_abs_err") is not None, r["node"]
        assert r["reproduction_max_abs_err"] < 1e-6           # StructureNetwork vs the evaluator, training rows


def three_input_model(n=400):
    """score = -2 + relu(x1 + 0.5 x2 - 0.25 x3 - 0.2): one hinge unit over three numeric inputs."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    rng = np.random.RandomState(3)
    raw = rng.normal(size=(n, 3)).round(3)
    raw_t = rng.normal(size=(150, 3)).round(3)
    nodes = [_node("-1", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("-3", NodeType.INPUT, "identity"), _node("10", NodeType.HIDDEN, "relu", -0.2),
             _node("0", NodeType.OUTPUT, "sigmoid", -2.0)]
    conns = [NetworkConnection(from_node="-1", to_node="10", weight=1.0, enabled=True),
             NetworkConnection(from_node="-2", to_node="10", weight=0.5, enabled=True),
             NetworkConnection(from_node="-3", to_node="10", weight=-0.25, enabled=True),
             NetworkConnection(from_node="10", to_node="0", weight=2.0, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1", "-2", "-3"], output_node_ids=["0"])
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-09T10:00:00",
            "params": {"name": "plane", "hypothesis": "A plane.", "entry_nodes": ["-1", "-2", "-3"],
                       "exit_nodes": ["10"], "subgraph_nodes": ["-1", "-2", "-3", "10"],
                       "subgraph_connections": [["-1", "10"], ["-2", "10"], ["-3", "10"]]}},
           {"seq": 2, "type": "annotate", "result": None, "created_at": "2026-11-09T10:01:00",
            "params": {"name": "root", "hypothesis": "Root.", "entry_nodes": ["10"], "exit_nodes": ["0"],
                       "subgraph_nodes": ["10", "0"], "subgraph_connections": [["10", "0"]],
                       "child_annotation_ids": ["plane"]}}]
    ev = StateEvaluator(s)
    r = np.random.RandomState(9)
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(raw)))).astype(float)
    yt = (r.uniform(size=150) < 1 / (1 + np.exp(-ev.logit(raw_t)))).astype(float)
    cb = {"id": "plane-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "the event", "risk": "the risk of the event", "predicted": "the predicted risk of the event"},
          "patient": "patients", "patient_one": "patient",
          "variables": {k: {"label": k, "unit": "units", "decimals": 2, "kind": "continuous"} for k in ("a", "b", "c")}}
    return PlainModel(s, ops, raw, y, raw_t, yt, ["a", "b", "c"], np.zeros(3), np.ones(3), cb,
                      raw_train=raw, raw_test=raw_t)


def test_three_input_plane_rule():
    lay = build_layer(three_input_model(), dict(META, label="plane"))
    bc = next(c for c in lay["claims"] if c["type"] == "boundary")
    st = bc["statement"]
    assert bc["verified"] and st["kind"] == "plane" and st["agreement"] == pytest.approx(1.0)
    assert set(st["variables"]) == {"a", "b", "c"} and len(bc["evidence"]["maps"]) == 3     # a triple of maps
    co = st["rule"]["coefficients"]
    assert co["b"] / co["a"] == pytest.approx(0.5) and co["c"] / co["a"] == pytest.approx(-0.25)
    assert st["rule"]["text"].startswith("a plus about 0.50 × b minus about 0.25 × c exceed about 0.20")
    for mp in bc["evidence"]["maps"]:                         # the third input held at its median
        assert len(mp["held"]) == 1
    # every map's on/off grid agrees with the rule
    for mp in bc["evidence"]["maps"]:
        xs, ys = np.array(mp["x"]["grid"]), np.array(mp["y"]["grid"])
        held = {mp["x"]["variable"]: None, mp["y"]["variable"]: None}
        third = next(iter(mp["held"]))
        tv = mp["held"][third]["value"]
        off = np.array(mp["off"])
        for i, yv in enumerate(ys[::8]):
            for j, xv in enumerate(xs[::8]):
                v = {mp["x"]["variable"]: xv, mp["y"]["variable"]: yv, third: tv}
                on = v["a"] + 0.5 * v["b"] - 0.25 * v["c"] - 0.2 > 0
                assert on == (not off[i * 8, j * 8])


def test_churn_heavy_daytime_boundary():
    lay = _layer("churn100-s4")
    bc = next(c for c in lay["claims"] if c["type"] == "boundary" and c["annotation"]["id"] == "ann_82")
    assert bc["verified"] and bc["statement"]["kind"] in ("kinked", "line")
    assert bc["statement"]["agreement"] >= 0.99


def test_category_charts_sized_and_readable():
    from explaneat.export.plain_plots import PER_LEVEL, _cat_rows, category_figure, labels_overlap
    lay = _layer("churn100-s4")
    rows = _cat_rows([c for c in lay["claims"] if c["annotation"]["id"] == "ann_80" and c.get("verified")])
    big = [c for c in rows if len(c["shape"]["levels"]) > 6]
    small = [c for c in rows if len(c["shape"]["levels"]) <= 6]
    assert big and small
    fig, _ = category_figure(lay, big[:1])
    n = len(big[0]["shape"]["levels"])
    assert fig.get_size_inches()[1] >= PER_LEVEL * n        # height grows with the levels
    assert len(fig.axes) == 1                                # a many-level variable alone, full width
    assert not labels_overlap(fig)
    fig2, _ = category_figure(lay, small)
    assert not labels_overlap(fig2)
    assert len([a for a in fig2.axes if a.get_visible()]) == len(small)
