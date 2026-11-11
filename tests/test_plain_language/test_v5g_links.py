"""v5g: linked inputs move together, off-data cases are marked, and the proof rule for
universal words (authoring-v4 evaluation, draft errors K1 and K2)."""

import numpy as np
import pytest

from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType

# education category (one-hot) that determines a numeric education level, plus an unrelated dose
FEATURES = ["edu:0", "edu:1", "edu:2", "edunum", "dose"]
EDUNUM = {0: 5.0, 1: 9.0, 2: 13.0}
CODEBOOK = {
    "id": "links-test", "source": "synthetic", "gaps": [],
    "outcome": {"positive": "the event", "short": "the event", "risk": "the risk of the event",
                "predicted": "the predicted risk of the event"},
    "patient": "people", "patient_one": "person",
    "variables": {
        "edu": {"label": "education", "kind": "category", "levels": {
            "0": {"label": "school", "group": "people whose education is school", "cond": "the education is school"},
            "1": {"label": "high school", "group": "people whose education is high school",
                  "cond": "the education is high school"},
            "2": {"label": "degree", "group": "people whose education is a degree", "cond": "the education is a degree"}}},
        "edunum": {"label": "education level", "unit": "years", "decimals": 0, "kind": "continuous"},
        "dose": {"label": "dose", "unit": "mg", "decimals": 1, "kind": "continuous"},
    },
}


def _node(i, t, act, bias=0.0):
    return NetworkNode(id=i, type=t, bias=bias, activation=act, response=1.0, aggregation="sum")


def _raw(n, seed):
    rng = np.random.RandomState(seed)
    e = rng.choice(3, size=n, p=[0.2, 0.5, 0.3])
    oh = np.eye(3)[e]
    edunum = np.array([EDUNUM[k] for k in e])
    dose = np.round(rng.uniform(0, 10, n), 1)
    return np.column_stack([oh, edunum, dose])


def linked_model(n=600):
    """logit = -0.2 + 1.5 z(edunum) + 0.4 z(edu:2) + 0.6 z(dose): a one-input change of the
    category moves only the small direct term; the linked move adds the education level."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    ids = ["-1", "-2", "-3", "-4", "-5"]
    nodes = [_node(i, NodeType.INPUT, "identity") for i in ids] + [_node("0", NodeType.OUTPUT, "sigmoid", -0.2)]
    conns = [NetworkConnection(from_node="-4", to_node="0", weight=1.5, enabled=True),
             NetworkConnection(from_node="-5", to_node="0", weight=0.6, enabled=True),
             NetworkConnection(from_node="-3", to_node="0", weight=0.4, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=ids, output_node_ids=["0"])
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-11T10:00:00",
            "params": {"name": "root", "hypothesis": "Logistic.", "entry_nodes": ids, "exit_nodes": ["0"],
                       "subgraph_nodes": ids + ["0"], "subgraph_connections": [["-3", "0"], ["-4", "0"], ["-5", "0"]]}}]
    raw, raw_t = _raw(n, 0), _raw(200, 1)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X, Xt = (raw - mean) / scale, (raw_t - mean) / scale
    ev = StateEvaluator(s)
    rng = np.random.RandomState(3)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(X)))).astype(float)
    yt = (rng.uniform(size=len(Xt)) < 1 / (1 + np.exp(-ev.logit(Xt)))).astype(float)
    return PlainModel(s, ops, X, y, Xt, yt, FEATURES, mean, scale, CODEBOOK, raw_train=raw, raw_test=raw_t)


@pytest.fixture(scope="module")
def m():
    return linked_model()


def test_links_detected_from_the_data(m):
    rec = m.links.record()["links"]
    pairs = {(r["changed"], r["moves"]) for r in rec}
    assert ("edu", "edunum") in pairs and ("edunum", "edu") in pairs
    assert not any("dose" in p for p in pairs)


def test_override_removes_a_link():
    from explaneat.analysis.plain_language_links import Links
    mm = linked_model(300)
    L = Links(mm, {"remove": [["edu", "edunum"]]})
    assert "edunum" not in L.linked("edu") and "edu" in L.linked("edunum")


def test_typical_person_is_on_the_data(m):
    from explaneat.analysis.plain_language import typical_raw
    raw = typical_raw(m)
    # most common category is high school; its education level is 9 (the overall median is 9 too,
    # so check a subset whose median level differs from its most common category)
    assert raw[1] == 1.0 and raw[3] == 9.0
    mask = np.asarray(m.var_by_base["edu"].raw_train != "1")      # school and degree only
    sub = typical_raw(m, mask)
    code = "0" if sub[0] > 0.5 else "2"
    assert sub[3] == EDUNUM[int(code)]


def test_risk_pair_moves_linked_inputs(m):
    from explaneat.analysis.plain_language import risk_pair, typical_raw, _z
    v = m.var_by_base["edu"]
    raw = typical_raw(m)
    info = {}
    pa, pb = risk_pair(m, raw, v, "1", "2", None, info=info)
    # the same as a real degree holder's row with the typical dose
    row = raw.copy()
    row[[0, 1, 2]] = [0, 0, 1]
    row[3] = 13.0
    want = float(m.risk(m.logit(_z(m, row)))[0])
    assert abs(pb - want) < 1e-12 and pb > pa
    assert info["moved"]["to"]["edunum"]["to"].startswith("13")
    assert "off_data" not in info
    # one input alone (the old behaviour) moves only the direct term
    z = _z(m, raw)
    lo = float(m.logit(z)[0])
    one = float(m.set_logit(z, v, "2", linked=False)[0])
    assert abs(one - lo - 0.4 / m.scale[2]) < 1e-9


def test_setting_two_linked_inputs_apart_is_off_data(m):
    from explaneat.analysis.plain_language import risk_pair, typical_raw
    v = m.var_by_base["edu"]
    nv = m.var_by_base["edunum"]
    info = {}
    risk_pair(m, typical_raw(m), v, "1", "2", None, extra_raw={nv.columns[0]: 5.0}, info=info)
    assert info.get("off_data") and "education level" in info["off_data"]["reasons"][0]


def test_per_row_effect_moves_linked_inputs(m):
    v = m.var_by_base["edu"]
    e = m.effect(m.X, v, "1", "2")
    # every row: edunum 9 -> 13 through weight 1.5, plus the category's own 0.4
    want = 1.5 * (13.0 - 9.0) / m.scale[3] + 0.4 / m.scale[2]
    assert np.allclose(e, want)


def test_claims_record_moves_and_templates_skip_off_data(m):
    from explaneat.analysis.plain_language import extract_claims
    from explaneat.analysis.plain_language_templates import _verified
    claims = extract_claims(m)["claims"]
    d = [c for c in claims if c["type"] == "direction" and c["statement"]["variable"] == "edu"]
    assert d and all(c.get("linked", {}).get("moved") for c in d)
    assert all(not c.get("off_data") for c in d)
    fake = dict(d[0], off_data={"reasons": ["x"]})
    assert _verified([fake, d[0]]) == [d[0]]


# ---------------------------------------------------------------- lint

def _claim(cid="C01", **kw):
    c = {"id": cid, "type": "direction", "verified": True, "statement": {}, "confidence": {}}
    c.update(kw)
    return c


def test_lint_off_data_refuses_real_people():
    from explaneat.analysis.plain_language_store import lint_off_data
    c = _claim(off_data={"reasons": ["no training person has education school with education level 13"]})
    assert lint_off_data("People whose education is school go from about 40% to about 6%.", [c])
    assert lint_off_data("For a typical person, school gives about 6%.", [c])
    assert not lint_off_data("In a constructed case for the typical person (no training person has this), "
                             "school gives about 6%.", [c])
    assert not lint_off_data("People whose education is school go from 40% to 6%.", [_claim()])


def test_lint_universal_proof_rule():
    from explaneat.analysis.plain_language_store import lint_universal
    plain = _claim()
    ok = _claim("C02", statement={"support_check": {"exceptions": 0, "n": 242}})
    bad = _claim("C03", statement={"support_check": {"exceptions": 18, "n": 242, "check": "x"}})
    s = "It adds to the risk only once ST depression rises above about 0.9 mm."
    assert lint_universal(s, [plain])
    assert "18 of 242" in lint_universal(s, [bad])[0]
    assert not lint_universal(s, [ok])
    assert lint_universal("An international plan raises it more than anything else.", [ok])
    assert lint_universal("It applies to all 38 customers without a plan.", [plain])
    for t in ("It rises (on the training data only).", "It changes only slightly.",
              "It rises with nothing until about 1 mm, then steadily."):
        assert not lint_universal(t, [plain]), t


def test_template_threshold_only_needs_the_check():
    from explaneat.analysis.plain_language_templates import threshold_sentence
    layer = {"population": {"plural": "patients", "singular": "patient"},
             "outcome": {"predicted": "the predicted risk"}}
    st = {"variable": "st", "kind": "above", "threshold": {"text": "0.9 mm"}, "direction": "raises",
          "conditions": [], "change": {"from": {"text": "0 mm"}, "to": {"text": "3 mm"}},
          "support_check": {"exceptions": 18, "n": 242}}
    c = {"id": "C1", "type": "threshold", "statement": st, "variables": [{"label": "ST depression"}],
         "risk": {"from": 0.2, "to": 0.6}, "confidence": {"test": "agrees", "word": "clearly"}, "shape": {}}
    t = threshold_sentence(layer, c)["text"]
    assert " only " not in t and "18 of the 242" in t
    st["support_check"] = {"exceptions": 0, "n": 242}
    assert " only " in threshold_sentence(layer, c)["text"]


def test_lint_universal_typical_case_scope():
    from explaneat.analysis.plain_language_store import lint_universal
    c = _claim()
    assert not lint_universal("For a typical patient with a reversible defect this part is off at every age, "
                              "so age changes nothing through it.", [c])
    assert lint_universal("For a typical patient it rises; it acts only above about 46 years.", [c])
