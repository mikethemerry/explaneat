"""Plain-language layer for a Cox survival genome: hazard ratios on the claims, the risk by the
horizon for the typical patient, a Harrell's-C test check, survival wording in the templates
and the proportional-hazards caveat, all passing the lint."""

import math

import numpy as np
import pytest

from explaneat.analysis.plain_language import (PlainModel, build_layer, fit_survival_scale, horizon_label,
                                               survival_outcome)
from explaneat.analysis.plain_language_store import lint_draft
from explaneat.analysis.plain_language_templates import sentences_text
from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import CODEBOOK, FEATURES, META, _node, raw_data

W_DOSE, W_GRP = 0.8, -0.6          # log-risk = 0.8 z(dose) - 0.6 z(grp:1); identity output


def cox_structure() -> NetworkStructure:
    nodes = [_node("-1", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("-3", NodeType.INPUT, "identity"), _node("-4", NodeType.INPUT, "identity"),
             _node("0", NodeType.OUTPUT, "identity", 0.0)]
    conns = [NetworkConnection(from_node="-1", to_node="0", weight=W_DOSE, enabled=True),
             NetworkConnection(from_node="-3", to_node="0", weight=W_GRP, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1", "-2", "-3", "-4"],
                         output_node_ids=["0"])
    s.metadata["output_activation"] = "identity"
    return s


def simulate(X, seed):
    """Exponential event times from the model's own log-risk, censored uniformly (years)."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    rng = np.random.RandomState(seed)
    g = StateEvaluator(cox_structure()).logit(X)
    t = rng.exponential(1.0 / (0.08 * np.exp(g)))
    c = rng.uniform(2, 12, size=len(t))
    e = (t <= c).astype(float)
    return np.minimum(t, c), e


def survival_model(n=400):
    s = cox_structure()
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-13T10:00:00",
            "params": {"name": "root", "hypothesis": "A linear Cox model.", "entry_nodes": ["-1", "-2", "-3", "-4"],
                       "exit_nodes": ["0"], "subgraph_nodes": ["-1", "-2", "-3", "-4", "0"],
                       "subgraph_connections": [["-1", "0"], ["-3", "0"]]}}]
    raw, raw_t = raw_data(n, 0), raw_data(150, 1)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X, Xt = (raw - mean) / scale, (raw_t - mean) / scale
    t, e = simulate(X, 3)
    tt, et = simulate(Xt, 4)
    scale_ = fit_survival_scale(s, X, t, e, horizon=5.0, time_unit="years")
    cb = dict(CODEBOOK, outcome=survival_outcome("death", scale_.horizon_label))
    return PlainModel(s, ops, X, e, Xt, et, FEATURES, mean, scale, cb, raw_train=raw, raw_test=raw_t,
                      risk_scale=scale_, time_train=t, time_test=tt)


@pytest.fixture(scope="module")
def lay():
    return build_layer(survival_model(), dict(META, label="cox-synthetic"))


def _dose(lay):
    return next(c for c in lay["claims"] if c["type"] == "direction" and c["statement"]["variable"] == "dose")


def test_horizon_labels_and_outcome_wording():
    assert horizon_label(5, "years") == "5-year"
    assert horizon_label(1826.25, "days") == "5-year"
    assert horizon_label(60, "months") == "5-year"
    assert horizon_label(180, "days") == "180-day"
    assert horizon_label(3, None) == "3"
    o = survival_outcome("recurrence", "5-year")
    assert o["predicted"] == "the predicted 5-year risk of recurrence" and o["short"] == "recurrence"


def test_survival_scale_and_hazard_ratios_on_the_claims(lay):
    rs = lay["risk_scale"]
    assert rs["kind"] == "survival" and rs["horizon"] == 5.0 and rs["horizon_label"] == "5-year"
    assert 0 < rs["baseline_survival_at_horizon"] < 1 and "ph_caveat" in rs
    assert lay["ph_caveat"] and "proportional-hazards" in lay["ph_caveat"]
    assert lay["outcome"]["predicted"] == "the predicted 5-year risk of death"
    d = _dose(lay)
    assert d["verified"]
    e = d["estimate"]
    assert e["hazard_ratio"] == pytest.approx(math.exp(e["value"]), rel=1e-4)   # both rounded to 6 figures
    assert e["hazard_ratio"] == e["odds_ratio"] and "log hazard ratio" in e["unit"]
    # a positive weight on dose: higher dose, higher hazard, higher risk by 5 years for the typical patient
    assert e["value"] > 0 and d["risk"]["to"] > d["risk"]["from"] and 0 < d["risk"]["from"] < d["risk"]["to"] < 1
    # the typical patient's risk is the horizon risk, not a sigmoid of the log-risk
    assert 0 < lay["reference_patient"]["predicted_risk"] < 1
    assert "hazard ratio" in lay["settings"]["effect_scale"]


def test_test_check_uses_harrells_c_over_the_follow_up_times(lay):
    d = _dose(lay)
    tc = d["test_check"]
    assert tc["status"] in ("agrees", "unclear", "disagrees"), tc
    assert "Harrell" in tc["measure"] and tc["n"] == 150 and tc["benefit_mean"] is not None
    # the simulated effect is large: the mechanism helps rank test patients
    assert tc["status"] == "agrees" and d["confidence"]["word"] == "clearly"


def test_template_states_hazard_ratios_and_closes_with_the_ph_caveat(lay):
    t = lay["template"]
    summ = t["summary"]
    text = sentences_text(summ, cite=False)
    assert "the predicted 5-year risk of death" in text
    assert "a hazard ratio of about" in text
    assert summ[-1]["text"] == lay["ph_caveat"][0].upper() + lay["ph_caveat"][1:] and summ[-1]["claims"]
    assert "odds" not in text.lower()
    root = next(p for p in lay["parts"] if p["role"] == "root")
    para = sentences_text(t["annotations"][root["id"]], cite=False)
    assert "hazard ratio of about" in para and "ignores" in para     # noise has no edge


def test_template_text_passes_the_lint(lay):
    root = next(p for p in lay["parts"] if p["role"] == "root")
    t = lay["template"]
    draft = {"summary": t["summary"], "annotations": {root["id"]: t["annotations"][root["id"]]},
             "worth_checking": [], "hypothesis_revisions": {}}
    assert lint_draft(lay, draft) == []


def test_classification_layers_are_unchanged():
    from .helpers import model
    L = build_layer(model("interaction"), META)
    assert L["risk_scale"]["kind"] == "binary" and "ph_caveat" not in L
    assert all("hazard_ratio" not in (c.get("estimate") or {}) for c in L["claims"])
    assert "hazard ratio" not in sentences_text(L["template"]["summary"], cite=False)
