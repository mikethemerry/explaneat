"""Binary one-step units: a ReLU that switches on the data but reads exactly one input node whose
column takes two values on the training rows is exactly affine in it there, so it counts as linear on
the data (class support_linear, evidence binary_step_units). Continuous inputs, one-group multi-input
units and units downstream of a binary-step unit are not reclassified."""

import numpy as np
import pytest

from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import META, _node

FEATS = ["a", "b", "g:0", "g:1", "g:2", "e"]


def _model():
    """score = -0.5 + 0.5 a + 0.8 relu(3 a - 0.5)              (a binary: unit 20 is binary one-step)
                 + 0.4 b + 1.2 relu(b)                         (b continuous: unit 21 switching)
                 + 0.6 relu(g:1 + g:2 - 0.2)                   (one group, two inputs: unit 22)
                 + 0.7 relu(1.5 relu(2 e - 0.3) + 0.5 b - 0.4) (chain: 23 binary one-step, 24 switching)
    on z-scored columns."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    rng = np.random.RandomState(7)

    def raw(n):
        a = (rng.uniform(size=n) < 0.4).astype(float)
        b = rng.normal(size=n).round(3)
        lev = rng.choice(3, size=n, p=[0.4, 0.3, 0.3])
        g = np.eye(3)[lev]
        e = (rng.uniform(size=n) < 0.5).astype(float)
        return np.column_stack([a, b, g, e])
    R, Rt = raw(600), raw(200)
    mean, scale = R.mean(0), R.std(0)
    X, Xt = (R - mean) / scale, (Rt - mean) / scale
    ids = ["-1", "-2", "-3", "-4", "-5", "-6"]
    nodes = [_node(i, NodeType.INPUT, "identity") for i in ids] + [
        _node("20", NodeType.HIDDEN, "relu", -0.5), _node("21", NodeType.HIDDEN, "relu", 0.0),
        _node("22", NodeType.HIDDEN, "relu", -0.2), _node("23", NodeType.HIDDEN, "relu", -0.3),
        _node("24", NodeType.HIDDEN, "relu", -0.4), _node("0", NodeType.OUTPUT, "sigmoid", -0.5)]
    E = [("-1", "0", 0.5), ("-1", "20", 3.0), ("20", "0", 0.8),
         ("-2", "0", 0.4), ("-2", "21", 1.0), ("21", "0", 1.2),
         ("-4", "22", 1.0), ("-5", "22", 1.0), ("22", "0", 0.6),
         ("-6", "23", 2.0), ("23", "24", 1.5), ("-2", "24", 0.5), ("24", "0", 0.7)]
    conns = [NetworkConnection(from_node=a, to_node=b, weight=w, enabled=True) for a, b, w in E]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=ids, output_node_ids=["0"])
    hid = ["20", "21", "22", "23", "24"]
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-10T10:00:00",
            "params": {"name": "a_step", "hypothesis": "A step on a.", "entry_nodes": ["-1"], "exit_nodes": ["20"],
                       "subgraph_nodes": ["-1", "20"], "subgraph_connections": [["-1", "20"]]}},
           {"seq": 2, "type": "annotate", "result": None, "created_at": "2026-11-10T10:01:00",
            "params": {"name": "chain", "hypothesis": "A chain.", "entry_nodes": ["-6", "-2"], "exit_nodes": ["24"],
                       "subgraph_nodes": ["-6", "-2", "23", "24"],
                       "subgraph_connections": [["-6", "23"], ["23", "24"], ["-2", "24"]]}},
           {"seq": 3, "type": "annotate", "result": None, "created_at": "2026-11-10T10:02:00",
            "params": {"name": "root", "hypothesis": "Root.", "entry_nodes": ids, "exit_nodes": ["0"],
                       "subgraph_nodes": ids + hid + ["0"],
                       "subgraph_connections": [[a, b] for a, b, _ in E],
                       "child_annotation_ids": ["a_step", "chain"]}}]
    ev = StateEvaluator(s)
    r = np.random.RandomState(9)
    y = (r.uniform(size=len(X)) < 1 / (1 + np.exp(-ev.logit(X)))).astype(float)
    yt = (r.uniform(size=len(Xt)) < 1 / (1 + np.exp(-ev.logit(Xt)))).astype(float)
    cb = {"id": "bstep-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "the event", "risk": "the risk of the event", "predicted": "the predicted risk of the event"},
          "patient": "patients", "patient_one": "patient",
          "variables": {"a": {"label": "a", "kind": "binary", "levels": {
                              "0": {"label": "not a", "group": "patients without a", "cond": "the patient does not have a"},
                              "1": {"label": "a", "group": "patients with a", "cond": "the patient has a"}}},
                        "b": {"label": "b", "unit": "units", "decimals": 2, "kind": "continuous"},
                        "g": {"label": "g", "kind": "category", "levels": {
                              k: {"label": f"g{k}", "group": f"patients in g{k}", "cond": f"the patient is in g{k}"}
                              for k in "012"}},
                        "e": {"label": "e", "kind": "binary", "levels": {
                              "0": {"label": "not e", "group": "patients without e", "cond": "the patient does not have e"},
                              "1": {"label": "e", "group": "patients with e", "cond": "the patient has e"}}}}}
    return PlainModel(s, ops, X, y, Xt, yt, FEATS, mean, scale, cb, raw_train=R, raw_test=Rt)


def test_unit_kinds_binary_step_only_for_single_binary_input():
    from explaneat.analysis.plain_language_linearity import unit_kinds
    m = _model()
    det = {}
    k = unit_kinds(m, details=det)
    assert k["20"] == "binary_step"          # one binary input, switches on the data
    assert k["21"] == "switching"            # continuous input
    assert k["22"] == "switching"            # two inputs (one one-hot group): not reclassified
    assert k["23"] == "binary_step"
    assert k["24"] == "switching"            # downstream of a binary-step unit: not reclassified
    assert det["20"]["passed"] and det["20"]["max_affine_error"] < 1e-9
    assert set(det) == {"20", "23"}
    # without the new kind: the classification before it
    k0 = unit_kinds(m, binary_step=False)
    assert k0["20"] == "switching" and k0["23"] == "switching"


def test_binary_step_variable_is_support_linear_with_exact_level_effect():
    from explaneat.analysis.plain_language_linearity import classify_variables, unit_kinds
    m = _model()
    info = classify_variables(m)
    assert info["a"]["class"] == "support_linear"
    assert info["a"]["binary_step_units"] == ["20"]
    assert info["a"]["fd_match_share"] == 1.0
    # the per-level effect (raw 0 -> 1): direct edge plus the unit's jump
    a0, a1 = (0.0 - m.mean[0]) / m.scale[0], (1.0 - m.mean[0]) / m.scale[0]
    want = 0.5 * (a1 - a0) + 0.8 * (max(3 * a1 - 0.5, 0) - max(3 * a0 - 0.5, 0))
    assert info["a"]["coef_raw"][0] == pytest.approx(want, rel=1e-9)
    # and it is what the model does on every row
    eff = m.logit(m.X, {0: a1}) - m.logit(m.X, {0: a0})
    assert np.allclose(eff, want, atol=1e-12)
    # the others are not reclassified
    assert info["b"]["class"] == "mixed"
    assert info["g"]["class"] == "nonlinear"
    assert info["e"]["class"] == "nonlinear"
    # before: a was mixed (direct edge + a switching unit)
    k0 = unit_kinds(m, binary_step=False)
    assert classify_variables(m, k0, {})["a"]["class"] == "mixed"


def test_one_group_multi_input_unit_reported_not_reclassified():
    from explaneat.analysis.plain_language_linearity import one_group_exact, unit_kinds
    m = _model()
    og = one_group_exact(m, unit_kinds(m))
    assert list(og) == ["22"] and og["22"]["variable"] == "g"
    assert og["22"]["max_affine_error"] < 1e-9


def test_linearised_model_unchanged_and_layer_claims():
    from explaneat.analysis.plain_language import build_layer
    from explaneat.analysis.plain_language_linearity import linearised_auc, unit_kinds
    m = _model()
    res = linearised_auc(m, unit_kinds(m))
    res0 = linearised_auc(m, unit_kinds(m, binary_step=False))
    assert res["switching_units"] == 3 and res0["switching_units"] == 5
    assert res["binary_step_linearisation_max_diff"] < 1e-9
    assert res["auc_linearised_train"] == pytest.approx(res0["auc_linearised_train"], abs=1e-12)
    lay = build_layer(m, dict(META, label="bstep"))
    lc = [c for c in lay["claims"] if c["type"] == "linearity"]
    assert all(c["verified"] for c in lc), [(c["key"], c["reasons"]) for c in lc if not c["verified"]]
    va = next(c for c in lc if c["statement"].get("variable") == "a")
    assert va["statement"]["class"] == "support_linear" and va["evidence"]["binary_step_units"] == 1
    parts = lay["graph"]["linearity"]["parts"]
    step_part = next(p["id"] for p in lay["parts"] if p["name"] == "a_step")
    chain_part = next(p["id"] for p in lay["parts"] if p["name"] == "chain")
    assert parts[step_part] == "support_linear" and parts[chain_part] == "nonlinear"
    model = next(c for c in lc if c["statement"]["kind"] == "model")
    assert model["statement"]["binary_step_units"] == 2
    assert model["evidence"]["one_group_exact_not_reclassified"] == ["22"]
    assert lay["graph"]["linearity"]["binary_step"]["20"]["passed"]
