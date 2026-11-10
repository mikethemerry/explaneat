"""Linear factors: classification of inputs (exact / support / mixed / nonlinear), the always-active
check, the linearised-model AUC, and the templated sentence."""

import numpy as np
import pytest

from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import META, _node


def _model():
    """score = -8 + 1.0 a                      (direct: exact)
                 + 0.5 * id(0.8 e)             (identity unit: exact)
                 + 0.7 * relu(b + 10)          (always on: support)
                 + 0.4 c + 1.5 relu(c - 0.2)   (direct + switching: mixed)
                 + 2.0 relu(d)                 (switching only: nonlinear)"""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    rng = np.random.RandomState(4)
    n = 500
    raw = rng.normal(size=(n, 5)).round(3)
    raw_t = rng.normal(size=(200, 5)).round(3)
    ids = ["-1", "-2", "-3", "-4", "-5"]          # a b c d e
    nodes = [_node(i, NodeType.INPUT, "identity") for i in ids] + [
        _node("10", NodeType.HIDDEN, "relu", 10.0), _node("11", NodeType.HIDDEN, "relu", -0.2),
        _node("12", NodeType.HIDDEN, "relu", 0.0), _node("13", NodeType.HIDDEN, "identity", 0.0),
        _node("0", NodeType.OUTPUT, "sigmoid", -8.0)]
    E = [("-1", "0", 1.0), ("-2", "10", 1.0), ("10", "0", 0.7), ("-3", "0", 0.4), ("-3", "11", 1.0),
         ("11", "0", 1.5), ("-4", "12", 1.0), ("12", "0", 2.0), ("-5", "13", 0.8), ("13", "0", 0.5)]
    conns = [NetworkConnection(from_node=a, to_node=b, weight=w, enabled=True) for a, b, w in E]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=ids, output_node_ids=["0"])
    hid = ["10", "11", "12", "13"]
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-10T10:00:00",
            "params": {"name": "gate", "hypothesis": "A gate.", "entry_nodes": ["-3", "-4"], "exit_nodes": ["11", "12"],
                       "subgraph_nodes": ["-3", "-4", "11", "12"],
                       "subgraph_connections": [["-3", "11"], ["-4", "12"]]}},
           {"seq": 2, "type": "annotate", "result": None, "created_at": "2026-11-10T10:01:00",
            "params": {"name": "root", "hypothesis": "Root.", "entry_nodes": ids, "exit_nodes": ["0"],
                       "subgraph_nodes": ids + hid + ["0"],
                       "subgraph_connections": [[a, b] for a, b, _ in E],
                       "child_annotation_ids": ["gate"]}}]
    ev = StateEvaluator(s)
    r = np.random.RandomState(9)
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(raw)))).astype(float)
    yt = (r.uniform(size=200) < 1 / (1 + np.exp(-ev.logit(raw_t)))).astype(float)
    cb = {"id": "lin-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "the event", "risk": "the risk of the event", "predicted": "the predicted risk of the event"},
          "patient": "patients", "patient_one": "patient",
          "variables": {k: {"label": k, "unit": "units", "decimals": 2, "kind": "continuous"} for k in "abcde"}}
    return PlainModel(s, ops, raw, y, raw_t, yt, list("abcde"), np.zeros(5), np.ones(5), cb,
                      raw_train=raw, raw_test=raw_t)


def test_classification_of_inputs():
    from explaneat.analysis.plain_language_linearity import classify_variables, unit_kinds
    m = _model()
    k = unit_kinds(m)
    assert k["10"] == "always_on" and k["13"] == "linear"
    assert k["11"] == "switching" and k["12"] == "switching"
    info = classify_variables(m)
    assert {b: i["class"] for b, i in info.items()} == {
        "a": "exact_linear", "b": "support_linear", "c": "mixed", "d": "nonlinear", "e": "exact_linear"}
    c = lambda b: next(iter(info[b]["coef_raw"].values()))
    assert c("a") == pytest.approx(1.0) and c("e") == pytest.approx(0.4)
    assert c("b") == pytest.approx(0.7) and c("c") == pytest.approx(0.4)     # mixed: the direct path only
    assert info["a"]["fd_match_share"] == 1.0 and info["b"]["fd_match_share"] == 1.0
    assert info["c"]["fd_varies"] and info["d"]["fd_varies"]


def test_always_active_check_is_numeric():
    """A ReLU is support-linear only while it is active on every training row: push one row below
    zero and it becomes a switching unit (and its input mixed/nonlinear)."""
    from explaneat.analysis.plain_language_linearity import classify_variables, unit_kinds
    m = _model()
    m.X[0, 1] = -20.0                       # b = -20 on one row: relu(b + 10) is off there
    m.base_logit = m.logit(m.X)
    m._vals_cache = None
    assert unit_kinds(m)["10"] == "switching"
    assert classify_variables(m)["b"]["class"] == "nonlinear"


def test_linearised_auc_reproduces_on_a_known_case():
    """Replacing the switching units by their affine fits gives the AUC of the explicitly fitted
    straight-line model."""
    from explaneat.analysis.explanation_tools import default_auc
    from explaneat.analysis.plain_language_linearity import linearised_auc, unit_kinds
    m = _model()
    res = linearised_auc(m, unit_kinds(m))
    assert res["switching_units"] == 2
    assert res["auc_full_train"] == pytest.approx(default_auc(m.y, m.ev.logit(m.X)))
    # by hand: the fits of relu(c - 0.2) on c and relu(d) on d
    X = m.X
    def fit(x, yv):
        A = np.column_stack([np.ones_like(x), x])
        return A @ np.linalg.lstsq(A, yv, rcond=None)[0]
    a, b, c, d, e = X.T
    lin = (-8 + a + 0.5 * 0.8 * e + 0.7 * np.maximum(b + 10, 0) + 0.4 * c
           + 1.5 * fit(c, np.maximum(c - 0.2, 0)) + 2.0 * fit(d, np.maximum(d, 0)))
    assert res["auc_linearised_train"] == pytest.approx(default_auc(m.y, lin))
    assert "auc_linearised_test" in res


def test_layer_claims_and_sentence_lint_clean():
    from explaneat.analysis.plain_language import build_layer
    from explaneat.analysis.plain_language_store import lint_text
    from explaneat.export.render_plain import linear_sentence
    lay = build_layer(_model(), dict(META, label="lin"))
    lc = [c for c in lay["claims"] if c["type"] == "linearity"]
    model = next(c for c in lc if c["statement"]["kind"] == "model")
    st = model["statement"]
    assert (st["n_inputs"], st["n_exact_linear"], st["n_support_linear"], st["n_mixed"], st["n_nonlinear"]) == (5, 2, 1, 1, 1)
    assert 0 < st["variance_share_linear"] < 1
    assert all(c["verified"] for c in lc)
    assert lay["graph"]["linearity"]["variables"]["b"] == "support_linear"
    s = linear_sentence(lay)
    assert s and s["claims"] == [model["id"]]
    assert not lint_text(s["text"]), lint_text(s["text"])
    assert "3 of the model's 5 inputs" in s["text"]


@pytest.mark.parametrize("label,meaning,inverted", [
    ("adult-s0", "an income of $50,000 a year or less", True),
    ("churn100-s4", "the customer left (churned)", False),
    ("heart-paper", "heart disease present", False),
    ("monk2-s2", "exactly two of a1..a6 equal 1", False)])
def test_outcome_callout_is_data_verified(label, meaning, inverted):
    from explaneat.analysis.plain_language_store import load_layer
    from explaneat.export.render_plain import outcome_callout, outcome_text
    lay = load_layer(label)
    if lay is None or "outcome_check" not in lay:
        pytest.skip(f"{label} not re-extracted with the outcome check")
    oc = lay["outcome_check"]
    assert oc["verified"] and all(c["passed"] for c in oc["checks"]) and oc["checks"]
    assert meaning in oc["meaning"] and oc["inverted"] == inverted
    t = outcome_text(lay)
    assert f"{100 * oc['base_rate_train']:.0f}%" in t
    if inverted:
        assert t.startswith("Note: this model predicts the chance of an income of $50,000 or less")
        assert "reverse of the usual framing" in t
    assert outcome_callout(lay)


def test_grammar_lint_rejects_broken_outcome_phrases():
    from explaneat.analysis.plain_language_store import lint_text
    bad = ["It raises the predicted an income of $50,000 or less.",
           "It lowers a typical person's the predicted chance of a low income.",
           "It raises the the predicted risk.",
           "It raises the predicted heart disease."]
    for s in bad:
        assert lint_text(s), s
    ok = ["It lowers a typical person's predicted chance of a low income ($50,000 or less) from about 61% to about 11%.",
          "It raises the predicted risk of heart disease steadily."]
    for s in ok:
        assert not lint_text(s), (s, lint_text(s))
