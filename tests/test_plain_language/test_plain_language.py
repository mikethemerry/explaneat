"""Plain-language layer: claims, units, interaction checks, rendering, drafts (no database)."""

import copy
import json
import re

import numpy as np
import pytest

from explaneat.analysis.plain_language import (Variable, build_layer, claims_hash,
                                               extract_claims)
from explaneat.analysis.plain_language_codebooks import UCI_HEART, find_codebook
from explaneat.analysis.plain_language_store import (export_section, layer_path, lint_draft,
                                                     load_layer, record_verdict, save_layer,
                                                     set_draft)
from explaneat.analysis.plain_language_templates import render_template

from .helpers import META, model

CLAUDE = re.compile(r"claude", re.I)


@pytest.fixture(scope="module")
def layer():
    return build_layer(model("interaction"), dict(META))


def _cited(layer):
    t = layer["template"]
    sents = list(t["summary"]) + [s for ss in t["annotations"].values() for s in ss]
    return sents, {cid for s in sents for cid in s["claims"]}


# ------------------------------------------------------------------ determinism
def test_claims_recomputed_identically():
    a = build_layer(model("interaction"), dict(META))
    b = build_layer(model("interaction"), dict(META))
    assert json.dumps(a["claims"], sort_keys=True) == json.dumps(b["claims"], sort_keys=True)
    assert a["claims_hash"] == b["claims_hash"] == claims_hash(b["claims"])
    assert [c["id"] for c in a["claims"]] == [c["id"] for c in b["claims"]]


def test_claim_order_does_not_change_bootstrap():
    """Every claim's bootstrap starts from seed 42 afresh: recomputing one part alone gives
    the same intervals as inside the full extraction."""
    from explaneat.analysis.plain_language import build_parts, direction_claims
    m = model("interaction")
    full = {c["key"]: c for c in extract_claims(m)["claims"]}
    part = next(p for p in build_parts(m) if p.name == "gate")
    for c in direction_claims(m, part, part.input_nodes):
        assert c["estimate"]["ci95"] == pytest.approx(full[c["key"]]["estimate"]["ci95"])


def test_template_renderer_deterministic(layer):
    again = render_template(copy.deepcopy(layer))
    assert json.dumps(again, sort_keys=True) == json.dumps(layer["template"], sort_keys=True)


# ------------------------------------------------------------------ units
def test_unit_conversion_round_trips():
    v = Variable("chol", "continuous", [0], [None], {"label": "serum cholesterol", "unit": "mg/dL", "decimals": 0},
                 np.array([245.6]), np.array([51.8]), np.array([200.0, 300.0]), None)
    for raw in (126.0, 245.6, 417.0):
        z = v.encode(raw)[0]
        assert v.decode(z) == pytest.approx(raw, abs=1e-9)
        assert v.encode(v.decode(z))[0] == pytest.approx(z, abs=1e-12)
    assert v.fmt(240.0) == "240 mg/dL"


def test_model_round_trip_recorded():
    m = model("interaction")
    assert m.round_trip_err is not None and m.round_trip_err < 1e-9


def test_category_encoding_and_decoding():
    m = model("interaction")
    g = m.var_by_base["grp"]
    assert g.is_category() and g.codes == ["0", "1"]
    enc = g.encode("1")
    on, off = enc[2], enc[1]
    assert on == pytest.approx((1 - m.mean[2]) / m.scale[2])
    assert off == pytest.approx((0 - m.mean[1]) / m.scale[1])
    assert g.level("1")["label"] == "group B"
    assert g.level("9")["decoded"] is False   # unknown codes are never guessed


def test_heart_codebook_decodes_one_hots_and_reports_gaps():
    names = ["age", "sex:0", "sex:1", "cp:1", "cp:2", "cp:3", "cp:4", "trestbps", "chol", "fbs:0",
             "fbs:1", "restecg:0", "restecg:1", "restecg:2", "thalach", "exang:0", "exang:1",
             "oldpeak", "slope:1", "slope:2", "slope:3", "ca", "thal:3", "thal:4", "thal:6", "thal:7"]
    cb = find_codebook("Heart Disease (binary, prepared)", names)
    assert cb is UCI_HEART
    assert cb["variables"]["cp"]["levels"]["4"]["label"] == "no chest pain (asymptomatic)"
    assert "4" not in cb["variables"]["thal"]["levels"]
    assert any("thal:4" in g for g in cb["gaps"])
    assert find_codebook("Adult", names) is None   # same short names, different dataset


# ------------------------------------------------------------------ interaction checks
def _interactions(layer):
    return [c for c in layer["claims"] if c["type"] == "interaction"]


def test_interaction_found_on_model_with_interaction(layer):
    ints = _interactions(layer)
    assert len(ints) == 1
    c = ints[0]
    assert c["verified"], c["reasons"]
    assert c["statement"]["variable"] == "dose" and c["statement"]["modifier"] == "grp"
    assert c["statement"]["direction"] == "raises"
    assert c["statement"]["stronger_group"]["label"] == "group B"
    sg, wg = c["statement"]["stronger_group"], c["statement"]["weaker_group"]
    assert sg["estimate"] > wg["estimate"] > 0
    # the estimate is (first group - second group); the part's own counterfactual
    # difference-in-differences, in the same orientation, has the same sign
    d = c["estimate"]["value"]
    assert np.sign(c["evidence"]["counterfactual_did_through_part"]) == np.sign(d)
    lo, hi = c["estimate"]["ci95"]
    assert (lo > 0 or hi < 0) and abs(d) >= 0.1
    first = c["groups"][0]
    assert d == pytest.approx(first["estimate"] - c["groups"][1]["estimate"], rel=1e-4)
    assert {g["n"] for g in c["groups"]} and sum(g["n"] for g in c["groups"]) == 300
    text = " ".join(s["text"] for s in layer["template"]["annotations"][c["annotation"]["id"]])
    assert "even more so if the patient is in group B" in text or "only if the patient is in group B" in text


def test_no_interaction_claim_on_additive_model():
    lay = build_layer(model("additive"), dict(META))
    assert not [c for c in _interactions(lay) if c["verified"]]
    # the gate reads dose only, so there is no candidate at all
    assert not _interactions(lay)


def test_negligible_interaction_fails_and_is_not_rendered():
    lay = build_layer(model("tiny"), dict(META))
    ints = _interactions(lay)
    assert ints and not any(c["verified"] for c in ints)
    assert any("below" in r for c in ints for r in c["reasons"])
    _, cited = _cited(lay)
    assert not cited & {c["id"] for c in ints}
    assert {c["id"] for c in ints} <= {x["claim"] for x in lay["template"]["not_supported"]}


# ------------------------------------------------------------------ other claim types
def test_null_threshold_subgroup_claims(layer):
    by_type = {}
    for c in layer["claims"]:
        by_type.setdefault(c["type"], []).append(c)
    null = [c for c in by_type["null"] if c["statement"].get("variable") == "noise"]
    assert null and null[0]["verified"]
    th = [c for c in by_type["threshold"] if c["statement"]["variable"] == "dose"]
    assert th and th[0]["verified"], th[0]["reasons"] if th else None
    # group A: the gate opens at 1.5 z_dose = 1 + 2 * |off| ... check the fold is inside the range
    assert 0.0 < th[0]["statement"]["threshold"]["value"] < 10.0
    sub = by_type["subgroup"][0]
    assert sub["verified"] and 0 < sub["statement"]["n_active"] < 300
    dirs = [c for c in by_type["direction"] if c["scope"] == "model" and c["statement"]["variable"] == "dose"]
    assert dirs[0]["verified"] and dirs[0]["statement"]["direction"] == "raises"
    assert dirs[0]["test_check"]["consistent"] is True


def test_every_claim_has_evidence_fields(layer):
    for c in layer["claims"]:
        assert c["split"] == "train"
        assert c["annotation"]["id"]
        assert "estimate" in c and "verified" in c and c["reasons"]
        assert isinstance(c["small_n"], bool)
        est = c["estimate"]
        if c["type"] in ("direction", "magnitude", "interaction"):
            assert est["bootstrap"]["seed"] == 42 and len(est["ci95"]) == 2


# ------------------------------------------------------------------ rendering
def test_failing_claims_never_rendered(layer):
    lay = copy.deepcopy(layer)
    failed = {c["id"] for c in lay["claims"][::2]}
    for c in lay["claims"]:
        if c["id"] in failed:
            c["verified"] = False
            c["reasons"] = ["forced failure for the test"]
    t = render_template(lay)
    sents = list(t["summary"]) + [s for ss in t["annotations"].values() for s in ss]
    cited = {cid for s in sents for cid in s["claims"]}
    assert not cited & failed
    assert {x["claim"] for x in t["not_supported"]} == failed
    for x in t["not_supported"]:
        assert "not supported by the model" in x["text"]


def test_all_failed_gives_no_statements(layer):
    lay = copy.deepcopy(layer)
    for c in lay["claims"]:
        c["verified"] = False
    t = render_template(lay)
    assert all(not s["claims"] for s in t["summary"])
    assert all(not ss for ss in t["annotations"].values())


def test_template_text_is_plain(layer):
    sents, _ = _cited(layer)
    text = " ".join(s["text"] for s in sents)
    assert text
    for bad in ("relu", "sigmoid", "logit", "z-score", "-1_a", "grp:1", "node 5"):
        assert bad not in text.lower()
    assert "on the model's odds scale" in text


def test_no_claude_in_rendered_output(layer, tmp_path):
    from explaneat.export.render_plain import appendix_section, render_review
    lay = copy.deepcopy(layer)
    assert CLAUDE.search(json.dumps([p["original_hypothesis"] for p in lay["parts"]]))
    save_layer(lay, tmp_path)
    draft = {"summary": [{"text": s["text"], "claims": s["claims"]} for s in lay["template"]["summary"] if s["claims"]][:3]}
    while len(draft["summary"]) < 3:
        draft["summary"].append(draft["summary"][0])
    lay = set_draft(lay["label"], draft, tmp_path, now="2026-11-08T00:00:00+00:00")
    tex = render_review([lay], "Review")
    assert not CLAUDE.search(tex)
    assert not CLAUDE.search("\n".join(appendix_section(export_section(lay["label"], tmp_path), "app:x:")))
    assert not CLAUDE.search(json.dumps(lay["template"]))
    assert "AI-assisted draft" in tex


# ------------------------------------------------------------------ drafts and storage
def test_lint_rejects_unverified_jargon_and_uncited(layer):
    lay = copy.deepcopy(layer)
    ok_id = next(c["id"] for c in lay["claims"] if c["verified"])
    lay["claims"][-1]["verified"] = False
    bad_id = lay["claims"][-1]["id"]
    draft = {"summary": [
        {"text": "The model treats a higher dose as raising the risk.", "claims": [ok_id]},
        {"text": "Node 5 is a ReLU on z-scored dose.", "claims": [ok_id]},
        {"text": "Claude says dose causes the event.", "claims": [ok_id]},
        {"text": "Group B raises the risk.", "claims": [bad_id]},
        {"text": "No citation here.", "claims": []},
        {"text": "Odds about ×3 for group B.", "claims": [ok_id]},
    ]}
    errs = "\n".join(lint_draft(lay, draft))
    assert "not verified" in errs
    assert "activation-function jargon" in errs and "z-score jargon" in errs
    assert "node or annotation id" in errs
    assert "names the drafting assistant" in errs and "causal wording" in errs
    assert "cites no claim" in errs
    assert "model's odds scale" in errs
    good = {"summary": draft["summary"][:1] * 3}
    assert lint_draft(lay, good) == []


def test_store_draft_and_verdict(layer, tmp_path):
    lay = copy.deepcopy(layer)
    save_layer(lay, tmp_path)
    ok = [c["id"] for c in lay["claims"] if c["verified"]][:1]
    pid = next(p["id"] for p in lay["parts"] if p["role"] == "part")
    draft = {"summary": [{"text": "The model treats a higher dose as raising the risk.", "claims": ok}] * 3,
             "hypothesis_revisions": {pid: {"proposed": "A gate on dose.", "claims": ok}}}
    with pytest.raises(ValueError):
        set_draft(lay["label"], {"summary": []}, tmp_path)
    stored = set_draft(lay["label"], draft, tmp_path, now="2026-11-08T00:00:00+00:00")
    d = stored["drafted"]
    assert d["drafted_by"] == "ai-assisted" and d["verified_by"] is None
    rev = d["hypothesis_revisions"][pid]
    assert rev["original"] == next(p["original_hypothesis"] for p in lay["parts"] if p["id"] == pid)
    assert rev["proposed"] == "A gate on dose."
    v = record_verdict(lay["label"], True, minutes=5, notes="fine", directory=tmp_path)
    assert v["verification"]["verified"] and v["verification"]["verified_by"] == "M. Merry"
    assert load_layer(lay["label"], tmp_path)["drafted"]["verified_by"] == "M. Merry"
    assert layer_path(lay["label"], tmp_path).read_text().endswith("\n")


def test_reextract_marks_changed_draft_stale(layer):
    from explaneat.analysis.plain_language_store import merge_existing
    old = copy.deepcopy(layer)
    old["drafted"] = {"summary": [], "claims_hash": old["claims_hash"], "verified_by": "M. Merry"}
    old["verification"] = {"verified": True}
    same = merge_existing(copy.deepcopy(layer), old)
    assert not same["drafted"].get("stale") and same["verification"]["verified"]
    new = copy.deepcopy(layer)
    new["claims_hash"] = "different"
    changed = merge_existing(new, old)
    assert changed["drafted"]["stale"] and changed["drafted"]["verified_by"] is None
    assert changed["verification"]["verified"] is None


def test_export_section_validates_in_export_schema(layer, tmp_path):
    from explaneat.export.pack import validate
    from tests.test_export.helpers import toy_doc
    lay = copy.deepcopy(layer)
    lay["label"] = "toy-s0"
    save_layer(lay, tmp_path)
    doc = toy_doc()
    assert doc["schema_version"] == "1.1.0" and doc["plain_language"] is None
    doc["plain_language"] = export_section("toy-s0", tmp_path)
    validate(doc)
    from explaneat.export.render_tex import render_appendix
    tex = render_appendix(doc)
    printed_lines = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    assert "Plain-language summary" in tex and not CLAUDE.search(printed_lines)
