"""v3.1: one named point per fold, no mechanical shape clause, paragraph caps."""

import copy

import numpy as np
import pytest

from explaneat.analysis.plain_language import build_layer
from explaneat.analysis.plain_language_store import lint_draft, lint_points
from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import META, _node


def fold_model(fold=4.0, n=400):
    """dose has a direct effect and, through a separate part, a hinge at ``fold`` mg for every
    patient: score = -3 + 0.2*dose + 1.0*relu(dose - fold)."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    rng = np.random.RandomState(5)
    raw = np.column_stack([np.round(rng.uniform(0, 10, n), 2), rng.normal(size=n)])
    raw_t = np.column_stack([np.round(rng.uniform(0, 10, 150), 2), rng.normal(size=150)])
    nodes = [_node("-1_a", NodeType.INPUT, "identity"), _node("-1_b", NodeType.INPUT, "identity"),
             _node("-2", NodeType.INPUT, "identity"), _node("10", NodeType.HIDDEN, "relu", -fold),
             _node("0", NodeType.OUTPUT, "sigmoid", -3.0)]
    conns = [NetworkConnection(from_node="-1_a", to_node="10", weight=1.0, enabled=True),
             NetworkConnection(from_node="10", to_node="0", weight=1.0, enabled=True),
             NetworkConnection(from_node="-1_b", to_node="0", weight=0.2, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1_a", "-1_b", "-2"], output_node_ids=["0"])
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-09T10:00:00",
            "params": {"name": "fold_part", "hypothesis": "A fold.", "entry_nodes": ["-1_a"], "exit_nodes": ["10"],
                       "subgraph_nodes": ["-1_a", "10"], "subgraph_connections": [["-1_a", "10"]]}},
           {"seq": 2, "type": "annotate", "result": None, "created_at": "2026-11-09T10:01:00",
            "params": {"name": "root", "hypothesis": "Root.", "entry_nodes": ["10", "-1_b", "-2"], "exit_nodes": ["0"],
                       "subgraph_nodes": ["10", "-1_b", "-2", "0"], "subgraph_connections": [["10", "0"], ["-1_b", "0"]],
                       "child_annotation_ids": ["fold_part"]}}]
    ev = StateEvaluator(s)
    r = np.random.RandomState(9)
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(raw)))).astype(float)
    yt = (r.uniform(size=150) < 1 / (1 + np.exp(-ev.logit(raw_t)))).astype(float)
    cb = {"id": "fold-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "the event", "risk": "the risk of the event", "predicted": "the predicted risk of the event"},
          "patient": "patients", "patient_one": "patient",
          "variables": {"dose": {"label": "dose", "unit": "mg", "decimals": 1, "kind": "continuous"},
                        "noise": {"label": "noise", "unit": "units", "decimals": 2, "kind": "continuous"}}}
    return PlainModel(s, ops, raw, y, raw_t, yt, ["dose", "noise"], np.zeros(2), np.ones(2), cb,
                      raw_train=raw, raw_test=raw_t)


@pytest.fixture(scope="module")
def fold_layer():
    return build_layer(fold_model(), dict(META, label="fold"))


def _part_claims(lay, ctype):
    part = next(p for p in lay["parts"] if p["name"] == "fold_part")
    return [c for c in lay["claims"] if c["annotation"]["id"] == part["id"] and c["type"] == ctype
            and c["statement"].get("variable") == "dose"]


def test_threshold_and_hinge_reconciled_to_one_point(fold_layer):
    th = _part_claims(fold_layer, "threshold")[0]
    d = _part_claims(fold_layer, "direction")[0]
    assert th["verified"]
    for c in (th, d):
        sh = c["shape"]
        assert sh["class"] == "hinge" and sh["point_reconciled"] is True, sh.get("point_reason")
        pt = sh["point"]
        assert pt["value"] == pytest.approx(4.0, abs=0.15)               # the median fold
        assert pt["typical_hinge"]["value"] == pytest.approx(4.0, abs=0.3)
        lo, hi = pt["spread_p10_p90"]
        assert lo == pytest.approx(4.0, abs=0.15) and hi == pytest.approx(4.0, abs=0.15)
        assert pt["wide"] is False and pt["source_claim"] == th["id"]
        assert sh["points_text"]["at"] == pt["text"]


def test_whole_model_point_kept_separate(fold_layer):
    root = next(p for p in fold_layer["parts"] if p["role"] == "root")
    d = next(c for c in fold_layer["claims"] if c["annotation"]["id"] == root["id"] and c["type"] == "direction"
             and c["statement"]["variable"] == "dose")
    sh = d["shape"]
    assert sh["point_reconciled"] is False      # the whole-model curve keeps its own description
    assert sh["class"] == "accelerating"        # 0.2 per mg, then 1.2 per mg above the fold


def test_template_quotes_one_point_and_passes_lint(fold_layer):
    t = fold_layer["template"]
    draft = {"summary": (t["summary"] * 3)[:3], "annotations": {k: v for k, v in t["annotations"].items() if v}}
    assert lint_draft(fold_layer, draft) == []


def test_lint_rejects_two_points_for_one_fold(fold_layer):
    c = copy.deepcopy(_part_claims(fold_layer, "direction")[0])
    c["shape"]["point"]["text"] = "4.0 mg"
    c["shape"]["point"]["typical_hinge"]["text"] = "4.6 mg"
    c["shape"]["point"]["spread_text"] = ["3.9 mg", "4.1 mg"]
    both = "It adds nothing until about 4.0 mg (4.6 mg for a typical patient), then steadily."
    assert lint_points(both, [c])
    assert lint_points("It adds nothing until about 4.6 mg, then steadily.", [c])
    assert lint_points("It adds nothing until about 4.0 mg, then steadily.", [c]) == []
    c["shape"]["point"].update(value=4.0, typical_hinge={"value": 4.62, "text": "4.62 mg"},
                               spread_text=["3.9 mg", "4.1 mg"])
    assert lint_points("It adds nothing until about 4.6, then steadily.", [c])          # other precision
    assert lint_points("It adds nothing from about 3.5 to 4.0 mg, then steadily.", [c])  # a range for one fold
    assert lint_points("It adds nothing until about 4.0 mg (between 3.9 and 4.1 mg), then steadily.", [c]) == []


def test_lint_rejects_mechanical_clause_and_long_paragraphs(fold_layer):
    lay = copy.deepcopy(fold_layer)
    ok = next(c for c in lay["claims"] if c["verified"] and c["type"] == "subgroup")
    sent = {"text": "This part acts on some patients.", "claims": [ok["id"]]}
    mech = {"text": "This part acts on some patients; by shape: dose steadily.", "claims": [ok["id"]]}
    part = next(p for p in lay["parts"] if p["role"] == "part")["id"]
    root = next(p for p in lay["parts"] if p["role"] == "root")["id"]
    errs = "\n".join(lint_draft(lay, {"summary": [sent] * 3, "annotations": {part: [mech]}}))
    assert "mechanical shape clause" in errs
    errs = "\n".join(lint_draft(lay, {"summary": [sent] * 3, "annotations": {part: [sent] * 7}}))
    assert "at most 6" in errs
    assert lint_draft(lay, {"summary": [sent] * 3, "annotations": {part: [sent] * 6}}) == []
    errs = "\n".join(lint_draft(lay, {"summary": [sent] * 3, "annotations": {root: [sent] * 9}}))
    assert "at most 8" in errs
    assert lint_draft(lay, {"summary": [sent] * 3, "annotations": {root: [sent] * 8}}) == []


def test_template_paragraphs_respect_caps(fold_layer):
    from explaneat.analysis.plain_language_templates import cap_paragraph
    lay = copy.deepcopy(fold_layer)
    part = next(p for p in lay["parts"] if p["role"] == "part")
    d = [c for c in lay["claims"] if c["verified"] and c["type"] in ("direction", "threshold")]
    sents = [{"text": f"Sentence {i}.", "claims": [d[i % len(d)]["id"]]} for i in range(10)]
    capped = cap_paragraph(lay, part, sents, 6)
    assert len(capped) <= 6
    assert "claude" not in str(capped).lower()


def test_typical_fold_outside_the_spread_kept_separate():
    from explaneat.analysis.plain_language_shapes import reconcile_point
    fold = {"part": "ann_1", "claim": "C9", "value": 4.0, "text": "4.0 mg", "p10_p90": [3.8, 4.4],
            "p10_p90_text": ["3.8 mg", "4.4 mg"], "count_step": None}
    claim = {"annotation": {"id": "ann_1"}, "type": "direction", "scope": "part"}

    def shape(at):
        return {"class": "hinge", "points": {"at": at}, "points_text": {"at": f"{at} mg"}, "grid": [0.0, 10.0]}
    near = shape(4.9)        # within 10% of the range of the spread: the same fold, read off the grid
    reconcile_point(near, claim, None, [fold])
    assert near["point_reconciled"] is True and near["point"]["text"] == "4.0 mg"
    far = shape(7.5)         # this patient folds somewhere else
    reconcile_point(far, claim, None, [fold])
    assert far["point_reconciled"] is False and "outside" in far["point_reason"]
    assert "point" not in far
