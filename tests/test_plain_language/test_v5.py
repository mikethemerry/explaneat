"""v5: structure claims (what a part is made of, a copy, a formula), direct contributions, and
the read-down lint."""

import numpy as np
import pytest

from explaneat.analysis.plain_language import build_layer
from explaneat.analysis.plain_language_store import (ADDITIVE, block_claims, lint_draft, readdown_order,
                                                     summary_blocks)
from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import META, _node


def copy_model(n=400):
    """score = -3 + 0.2*dose + 1.0*relu(dose - 4) + 0.5*relu(dose - 4)': the hinge is computed twice
    (two identical hidden nodes, two annotations), and dose also feeds the score directly."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    rng = np.random.RandomState(5)
    raw = np.column_stack([np.round(rng.uniform(0, 10, n), 2), rng.normal(size=n)])
    raw_t = np.column_stack([np.round(rng.uniform(0, 10, 150), 2), rng.normal(size=150)])
    nodes = [_node("-1_a", NodeType.INPUT, "identity"), _node("-1_b", NodeType.INPUT, "identity"),
             _node("-1_c", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("10", NodeType.HIDDEN, "relu", -4.0), _node("11", NodeType.HIDDEN, "relu", -4.0),
             _node("0", NodeType.OUTPUT, "sigmoid", -3.0)]
    conns = [NetworkConnection(from_node="-1_a", to_node="10", weight=1.0, enabled=True),
             NetworkConnection(from_node="-1_c", to_node="11", weight=1.0, enabled=True),
             NetworkConnection(from_node="10", to_node="0", weight=1.0, enabled=True),
             NetworkConnection(from_node="11", to_node="0", weight=0.5, enabled=True),
             NetworkConnection(from_node="-1_b", to_node="0", weight=0.2, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1_a", "-1_b", "-1_c", "-2"],
                         output_node_ids=["0"])
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-09T10:00:00",
            "params": {"name": "fold_part", "hypothesis": "A fold.", "entry_nodes": ["-1_a"], "exit_nodes": ["10"],
                       "subgraph_nodes": ["-1_a", "10"], "subgraph_connections": [["-1_a", "10"]]}},
           {"seq": 2, "type": "annotate", "result": None, "created_at": "2026-11-09T10:00:30",
            "params": {"name": "fold_copy", "hypothesis": "The same fold.", "entry_nodes": ["-1_c"],
                       "exit_nodes": ["11"], "subgraph_nodes": ["-1_c", "11"], "subgraph_connections": [["-1_c", "11"]]}},
           {"seq": 3, "type": "annotate", "result": None, "created_at": "2026-11-09T10:01:00",
            "params": {"name": "root", "hypothesis": "Root.", "entry_nodes": ["10", "11", "-1_b", "-2"],
                       "exit_nodes": ["0"], "subgraph_nodes": ["10", "11", "-1_b", "-2", "0"],
                       "subgraph_connections": [["10", "0"], ["11", "0"], ["-1_b", "0"]],
                       "child_annotation_ids": ["fold_part", "fold_copy"]}}]
    ev = StateEvaluator(s)
    r = np.random.RandomState(9)
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(raw)))).astype(float)
    yt = (r.uniform(size=150) < 1 / (1 + np.exp(-ev.logit(raw_t)))).astype(float)
    cb = {"id": "copy-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "the event", "risk": "the risk of the event", "predicted": "the predicted risk of the event"},
          "patient": "patients", "patient_one": "patient",
          "variables": {"dose": {"label": "dose", "unit": "mg", "decimals": 1, "kind": "continuous"},
                        "noise": {"label": "noise", "unit": "units", "decimals": 2, "kind": "continuous"}}}
    return PlainModel(s, ops, raw, y, raw_t, yt, ["dose", "noise"], np.zeros(2), np.ones(2), cb,
                      raw_train=raw, raw_test=raw_t)


@pytest.fixture(scope="module")
def lay():
    return build_layer(copy_model(), dict(META, label="copy"))


def _parts(lay):
    by = {p["name"]: p for p in lay["parts"]}
    return by["root"], by["fold_part"], by["fold_copy"]


def _structure(lay, pid, kind):
    return next(c for c in lay["claims"] if c["type"] == "structure" and c["annotation"]["id"] == pid
                and c["statement"]["kind"] == kind)


def test_structure_claims_from_the_graph(lay):
    root, part, cp = _parts(lay)
    assert cp["role"] == "duplicate" and cp["duplicate_of"] == part["id"]
    comp = _structure(lay, root["id"], "composition")
    assert comp["verified"]
    assert set(comp["statement"]["fed_by_parts"]) == {part["id"], cp["id"]}
    assert comp["statement"]["combination"] == "added" and comp["statement"]["direct_inputs"] == ["dose"]
    copy = _structure(lay, cp["id"], "copy")
    assert copy["verified"] and copy["statement"]["copy_of"] == part["id"]
    formula = _structure(lay, part["id"], "formula")
    assert formula["verified"] and formula["statement"]["max_abs_err"] < 1e-9
    assert "relu" in (part["closed_form"]["latex"] or "")


def test_direct_contribution_claim(lay):
    root, part, _ = _parts(lay)
    d = [c for c in lay["claims"] if c.get("scope") == "direct"]
    assert len(d) == 1 and d[0]["statement"]["variable"] == "dose" and d[0]["verified"]
    ch = d[0]["statement"]["change"]
    span = ch["to"]["value"] - ch["from"]["value"]
    assert d[0]["estimate"]["value"] == pytest.approx(0.2 * span, rel=1e-6)    # the direct edge only
    whole = next(c for c in lay["claims"] if c.get("scope") == "model" and c["type"] == "direction"
                 and c["statement"]["variable"] == "dose")
    assert whole["estimate"]["value"] > d[0]["estimate"]["value"]               # includes the hinge
    add = next(b for b in summary_blocks(lay) if b["block"] == ADDITIVE)
    ok = block_claims(lay, add)
    assert d[0]["id"] in ok and whole["id"] not in ok                           # no double counting


def _s(t, *c):
    return {"text": t, "claims": list(c)}


def _draft(lay):
    root, part, cp = _parts(lay)
    comp = _structure(lay, root["id"], "composition")
    copy = _structure(lay, cp["id"], "copy")
    formula = _structure(lay, part["id"], "formula")
    direct = next(c for c in lay["claims"] if c.get("scope") == "direct")
    pdir = next(c for c in lay["claims"] if c["annotation"]["id"] == part["id"] and c["type"] == "direction")
    sub = next(c for c in lay["claims"] if c["annotation"]["id"] == part["id"] and c["type"] == "subgroup")
    return {
        "lead": [_s("The model adds dose up directly and through a fold that is computed twice.", comp["id"])],
        "blocks": [
            {"block": ADDITIVE, "sentences": [_s("These add up independently: through this sum alone a higher dose "
                                                 "raises the predicted risk steadily.", direct["id"])]},
            {"block": part["id"], "sentences": [_s("This part adds nothing until about 4.0 mg, then raises the "
                                                   "predicted risk steadily; a copy of it is added too.",
                                                   pdir["id"], copy["id"])]},
        ],
        "titles": {ADDITIVE: "Inputs that add up", root["id"]: "The whole model", part["id"]: "Dose fold"},
        "readdown": {
            root["id"]: {"hypothesis": [_s("The score is a sum of dose and the fold.", comp["id"])],
                         "evidence": [_s("It is made up of dose and two copies of the fold (each explained "
                                         "below), added together.", comp["id"], copy["id"])]},
            part["id"]: {"hypothesis": [_s("This part adds nothing until about 4.0 mg, then raises the predicted "
                                           "risk steadily.", pdir["id"])],
                         "evidence": [_s("It acts on some patients.", sub["id"]),
                                      _s("Its formula reproduces the part exactly.", formula["id"])]},
        },
    }


def test_readdown_order_and_valid_draft(lay):
    root, part, cp = _parts(lay)
    assert readdown_order(lay) == [root["id"], part["id"]]       # the copy is described by its parent
    assert lint_draft(lay, _draft(lay)) == []


def test_readdown_lint(lay):
    root, part, cp = _parts(lay)
    d = _draft(lay)
    del d["readdown"][part["id"]]
    assert any("needs a section" in e for e in lint_draft(lay, d))
    d = _draft(lay)                         # a composition's evidence must say what it is made of
    d["readdown"][root["id"]]["evidence"] = [_s("It adds up.", _structure(lay, cp["id"], "copy")["id"])]
    assert any("made of" in e for e in lint_draft(lay, d))
    d = _draft(lay)                         # ... and name the copy it contains
    d["readdown"][root["id"]]["evidence"][0]["claims"] = [_structure(lay, root["id"], "composition")["id"]]
    assert any("name the copy" in e for e in lint_draft(lay, d))
    d = _draft(lay)                         # a leaf may not cite the root's claims
    d["readdown"][part["id"]]["evidence"].append(_s("Fine.", _structure(lay, root["id"], "composition")["id"]))
    assert any("belongs to another part" in e for e in lint_draft(lay, d))
    d = _draft(lay)                         # no confidence words, here either
    d["readdown"][part["id"]]["hypothesis"][0]["text"] = d["readdown"][part["id"]]["hypothesis"][0]["text"].replace(
        "raises", "clearly raises")
    assert any("confidence word" in e for e in lint_draft(lay, d))
    d = _draft(lay)                         # every part read down has a plain heading
    del d["titles"][root["id"]]
    assert any("plain name" in e for e in lint_draft(lay, d))
    d = _draft(lay)                         # the additive paragraph quotes the direct contribution only
    whole = next(c for c in lay["claims"] if c.get("scope") == "model" and c["type"] == "direction")
    d["blocks"][0]["sentences"][0]["claims"] = [whole["id"]]
    assert any("another block" in e for e in lint_draft(lay, d))
