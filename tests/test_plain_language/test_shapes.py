"""Shapes of effects (layer v3): classification, named points, consistency, lint, wording."""

import copy
import json
import re

import numpy as np
import pytest

from explaneat.analysis.plain_language import build_layer
from explaneat.analysis.plain_language_shapes import SENSITIVITY, TOL, classify_curve, classify_robust
from explaneat.analysis.plain_language_store import lint_draft, lint_shape

from .helpers import META, model
from .shape_helpers import SHAPES, curve_model


def _dose_claim(lay):
    return next(c for c in lay["claims"] if c["type"] == "direction" and c["statement"]["variable"] == "dose")


@pytest.mark.parametrize("name", list(SHAPES))
def test_known_shapes_are_classified(name):
    lay = build_layer(curve_model(**SHAPES[name]), dict(META, label=f"shape-{name}"))
    sh = _dose_claim(lay)["shape"]
    assert sh["class"] == name, (name, sh["class"], sh.get("pieces"))
    assert sh["status"] == "verified" and sh["consistent"] and sh["stable"]
    assert sh["share_same_shape"] == 1.0
    pts = sh["points"]
    if name == "hinge":
        assert pts["at"] == pytest.approx(4.0, abs=0.3)
        assert sh["hinge_kind"] == "flat_then_moving"
    if name == "step":
        assert 4.6 <= pts["at"] <= 5.8
    if name == "peak":
        assert pts["turning"] == pytest.approx(6.0, abs=0.3)


def test_named_points_are_in_real_units_and_tolerances_documented():
    lay = build_layer(curve_model(**SHAPES["hinge"]), dict(META))
    sh = _dose_claim(lay)["shape"]
    assert sh["points_text"]["at"].endswith("mg")
    assert set(TOL) <= set(sh["tolerances"]) and sh["consistent_share"] == 0.9
    assert sh["scale"].startswith("model score")


def test_classifier_robust_to_tolerances():
    x = np.linspace(0, 10, 41)
    for L, want in ((np.maximum(x - 4, 0), "hinge"), (np.minimum(3 * x, x + 4), "levelling_off"),
                    (np.maximum(x, 3 * x - 8), "accelerating"), (-(x - 6) ** 2 + 40, "peak"),
                    (np.clip((x - 4.8) * 10, 0, 4), "step"), (2 * x, "steady"), (0.001 * x, "flat")):
        r = classify_robust(x, L)
        assert r["class"] == want and r["stable"], (want, r["class"], r["sensitivity"])
        for s in SENSITIVITY:
            assert classify_curve(x, L, scale=s)["class"] == want


def test_ordinal_patterns():
    xo = np.array([0, 1, 2, 3.0])
    eq = classify_robust(xo, np.array([0, 1, 2, 3.0]), ordinal=True)
    assert eq["class"] == "steady" and eq["ordinal_pattern"] == "equal_steps"
    first = classify_robust(xo, np.array([0, 2, 2.3, 2.5]), ordinal=True)
    assert first["class"] == "step" and first["ordinal_pattern"] == "first_step"
    assert first["increments"] == pytest.approx([2, 0.3, 0.2])


def test_consistency_catches_mixed_shapes():
    """In group A the gate opens at a dose inside the range (a hinge); in group B it is open over
    the whole range (steady): individual curves differ, so the shape is not 'verified' as one."""
    lay = build_layer(model("mixed"), dict(META))
    part = next(p for p in lay["parts"] if p["name"] == "gate")
    c = next(c for c in lay["claims"] if c["annotation"]["id"] == part["id"] and c["type"] == "direction"
             and c["statement"]["variable"] == "dose")
    sh = c["shape"]
    assert len(sh["classes_across_patients"]) >= 2
    assert not sh["consistent"] and sh["share_same_shape"] < 0.9
    assert sh["status"] in ("mixed", "fallback")
    if sh["status"] == "mixed":
        assert sh["exceptions"] and sh["exceptions"][0]["who"]


def test_saturation_recorded_separately():
    big = dict(hidden=[], direct=2.0, bias=-10.0)      # risk runs from near 0% to near 100%
    lay = build_layer(curve_model(**big), dict(META))
    sh = _dose_claim(lay)["shape"]
    assert sh["class"] == "steady"                     # the mechanism (score scale) is steady
    assert sh["risk_saturation"] in ("near 0%", "near 100%")   # the risk curve saturates
    text = next(s["text"] for s in lay["template"]["annotations"][lay["parts"][0]["id"]]
                if _dose_claim(lay)["id"] in s["claims"])
    assert "steadily" in text and "flattening out" in text


def test_lint_rejects_missing_or_mismatched_shape_phrases():
    lay = build_layer(curve_model(**SHAPES["hinge"]), dict(META))
    c = _dose_claim(lay)
    cid = c["id"]
    tmpl = next(s for s in lay["template"]["annotations"][lay["parts"][0]["id"]] if cid in s["claims"])
    assert lint_shape(tmpl["text"], [c]) == []
    bare = "For a typical patient, a higher dose raises the predicted risk from about 12% to about 40%."
    assert lint_shape(bare, [c])                       # missing
    wrong = "A higher dose raises the predicted risk steadily, from about 12% to about 40%."
    assert lint_shape(wrong, [c])                      # mismatched (steady for a hinge)
    right = "A higher dose raises the predicted risk, with nothing until about 4 mg, then steadily."
    assert lint_shape(right, [c]) == []
    fb = copy.deepcopy(c)
    fb["shape"]["status"] = "fallback"
    assert lint_shape(right, [fb])
    assert lint_shape("A higher dose raises the predicted risk, in an uneven way.", [fb]) == []
    draft = {"summary": [{"text": "The model's score is linear in dose and raises the predicted risk "
                                  "with nothing until about 4 mg, then steadily.", "claims": [cid]}] * 3}
    assert any("shape jargon" in e for e in lint_draft(lay, draft))


def test_template_uses_shape_words_not_jargon():
    for name, kw in SHAPES.items():
        lay = build_layer(curve_model(**kw), dict(META))
        text = " ".join(s["text"] for ss in lay["template"]["annotations"].values() for s in ss)
        assert not re.search(r"\blinear|convex|concave|logit", text, re.I)
        t = lay["template"]
        draft = {"summary": (t["summary"] * 3)[:3], "annotations": {k: v for k, v in t["annotations"].items() if v}}
        assert lint_draft(lay, draft) == [], (name, lint_draft(lay, draft))


def test_no_claude_in_shape_output():
    lay = build_layer(curve_model(**SHAPES["peak"]), dict(META))
    assert "claude" not in json.dumps(lay["template"]).lower()
    assert "claude" not in json.dumps([c.get("shape") for c in lay["claims"]]).lower()
