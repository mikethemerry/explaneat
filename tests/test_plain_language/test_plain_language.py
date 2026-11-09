"""Plain-language layer (v2): claims, units, confidence, risk-scale magnitudes, rendering, drafts."""

import copy
import json
import re

import numpy as np
import pytest

from explaneat.analysis.plain_language import (CONFIDENCE_RULES, PROB_SLIGHT, SMALL_N, BinaryRisk,
                                               SurvivalRisk, Variable, build_layer, claims_hash,
                                               confidence, extract_claims, typical_raw)
from explaneat.analysis.plain_language_codebooks import (ADULT, MONK2, UCI_HEART, codebook_checks,
                                                         find_codebook)
from explaneat.analysis.plain_language_store import (export_section, layer_path, lint_confidence,
                                                     lint_draft, load_layer, record_verdict, save_layer,
                                                     set_draft)
from explaneat.analysis.plain_language_templates import render_template

from .helpers import LIN_B, LIN_W, META, linear_model, model, rule_model

CLAUDE = re.compile(r"claude", re.I)


@pytest.fixture(scope="module")
def layer():
    return build_layer(model("interaction"), dict(META))


def _sents(layer):
    t = layer["template"]
    return list(t["summary"]) + [s for ss in t["annotations"].values() for s in ss]


def _cited(layer):
    return {cid for s in _sents(layer) for cid in s["claims"]}


# ------------------------------------------------------------------ determinism
def test_claims_recomputed_identically():
    a = build_layer(model("interaction"), dict(META))
    b = build_layer(model("interaction"), dict(META))
    assert json.dumps(a["claims"], sort_keys=True) == json.dumps(b["claims"], sort_keys=True)
    assert a["claims_hash"] == b["claims_hash"] == claims_hash(b["claims"])


def test_test_bootstrap_independent_of_order():
    from explaneat.analysis.plain_language import build_parts, direction_claims
    m = model("interaction")
    full = {c["key"]: c for c in extract_claims(m)["claims"]}
    part = next(p for p in build_parts(m) if p.name == "gate")
    for c in direction_claims(m, part, part.input_nodes):
        assert c["test_check"]["benefit_ci95"] == full[c["key"]]["test_check"]["benefit_ci95"]


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
    assert enc[2] == pytest.approx((1 - m.mean[2]) / m.scale[2])
    assert enc[1] == pytest.approx((0 - m.mean[1]) / m.scale[1])
    assert g.level("1")["label"] == "group B"
    assert g.level("9")["decoded"] is False   # unknown codes are never guessed


HEART_NAMES = ["age", "sex:0", "sex:1", "cp:1", "cp:2", "cp:3", "cp:4", "trestbps", "chol", "fbs:0",
               "fbs:1", "restecg:0", "restecg:1", "restecg:2", "thalach", "exang:0", "exang:1",
               "oldpeak", "slope:1", "slope:2", "slope:3", "ca", "thal:3", "thal:4", "thal:6", "thal:7"]


def test_codebooks_match_by_dataset_and_report_gaps():
    cb = find_codebook("Heart Disease (binary, prepared)", HEART_NAMES)
    assert cb is UCI_HEART and any("thal:4" in g for g in cb["gaps"])
    assert find_codebook("Adult", HEART_NAMES) is None
    monk = [f"a{i}:{v}" for i, n in ((1, 3), (2, 3), (3, 2), (4, 3), (5, 4), (6, 2)) for v in range(1, n + 1)]
    assert find_codebook("monk2_onehot", monk) is MONK2
    assert MONK2["variables"]["a5"]["levels"]["4"]["label"] == "a5 = 4"


def test_adult_codebook_checks_catch_a_wrong_mapping():
    """The Adult decoding is checked against the data: education code -> education-num."""
    names = ["education:0", "education:13", "education-num", "relationship:0", "relationship:5", "sex",
             "workclass:0", "occupation:0", "native-country:39"]
    # code 0 = 10th grade (education-num 6), code 13 = preschool (1); husbands male, wives female
    raw = np.array([[1, 0, 6, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1]] * 10, dtype=float)
    y = np.array([1, 0] * 10, dtype=float) * 0 + np.r_[np.ones(15), np.zeros(5)]
    ok = {c["check"]: c["passed"] for c in codebook_checks(ADULT, names, raw, y)}
    assert ok["education code -> education-num (UCI)"] and ok["husbands are men (sex = 1)"]
    bad = raw.copy()
    bad[:, 2] = 9                               # education-num no longer matches the codes
    ok2 = {c["check"]: c["passed"] for c in codebook_checks(ADULT, names, bad, y)}
    assert not ok2["education code -> education-num (UCI)"]


# ------------------------------------------------------------------ probability-scale magnitudes
def test_typical_patient_risk_matches_closed_form():
    """On a known logistic model, the typical-patient risks are sigmoid of the linear predictor."""
    m = linear_model()
    lay = build_layer(m, dict(META))
    d = next(c for c in lay["claims"] if c["type"] == "direction" and c["statement"]["variable"] == "dose")
    ref = typical_raw(m)
    z_grp = (ref[2] - m.mean[2]) / m.scale[2]

    def p(dose):
        z = (dose - m.mean[0]) / m.scale[0]
        return 1 / (1 + np.exp(-(LIN_B + LIN_W * z - 0.8 * z_grp)))
    lo, hi = np.round(np.percentile(m.raw_train[:, 0], [10, 90]), 1)
    assert d["risk"]["reference"] == "typical patient"
    assert d["risk"]["from"] == pytest.approx(p(lo), rel=1e-5)
    assert d["risk"]["to"] == pytest.approx(p(hi), rel=1e-5)
    assert d["risk"]["delta"] == pytest.approx(p(hi) - p(lo), rel=1e-4)
    # odds ratio stays in the evidence: exp(w * (z_hi - z_lo))
    assert d["estimate"]["odds_ratio"] == pytest.approx(np.exp(LIN_W * (hi - lo) / m.scale[0]), rel=1e-4)
    # a linear term has the same effect for every patient: a zero-width range across patients
    r = d["estimate"]["range_across_patients"]
    assert r[0] == pytest.approx(r[1])
    # ... and the prose leads with the risk, never with odds
    text = " ".join(s["text"] for s in _sents(lay) if d["id"] in s["claims"])
    assert f"{100 * p(lo):.0f}%" in text and f"{100 * p(hi):.0f}%" in text
    assert "odds" not in text.lower() and "×" not in text


def test_interaction_risk_per_group(layer):
    c = next(c for c in layer["claims"] if c["type"] == "interaction")
    assert c["verified"], c["reasons"]
    groups = {g["label"]: g for g in c["risk"]["groups"]}
    assert set(groups) == {"group A", "group B"}
    dB = groups["group B"]["to"] - groups["group B"]["from"]
    dA = groups["group A"]["to"] - groups["group A"]["from"]
    assert dB > dA > -1e-9                      # dose raises the risk more in group B
    assert c["statement"]["stronger_group"]["label"] == "group B"
    assert np.sign(c["evidence"]["counterfactual_did_through_part"]) == np.sign(c["estimate"]["value"])
    text = next(s["text"] for s in _sents(layer) if c["id"] in s["claims"])
    assert "even more so if the patient is in group B" in text or "only if the patient is in group B" in text
    assert "if the patient is in group A" in text and "%" in text


def test_survival_risk_scale():
    rs = SurvivalRisk(0.9, 5.0, "5-year")
    assert rs.risk(np.array([0.0]))[0] == pytest.approx(0.1)
    assert rs.risk(np.array([np.log(2.0)]))[0] == pytest.approx(1 - 0.81)
    assert rs.describe()["label"] == "predicted 5-year risk"
    assert BinaryRisk().risk(np.array([0.0]))[0] == pytest.approx(0.5)


# ------------------------------------------------------------------ confidence words
def _c(delta, test="agrees", small=False, ctype="direction"):
    return {"type": ctype, "risk": {"delta": delta}, "test_check": {"status": test}, "small_n": small}


def test_confidence_thresholds():
    assert confidence(_c(0.30))["word"] == "clearly"
    assert confidence(_c(PROB_SLIGHT))["word"] == "clearly"            # meaningful from 5 points
    c = confidence(_c(PROB_SLIGHT - 0.001))
    assert c["word"] is None and c["size"] == "slight" and "slightly" in c["qualifiers"]
    for st in ("unclear", "disagrees", "not available"):
        c = confidence(_c(0.30, test=st))
        assert c["word"] is None and "on the training data only" in c["qualifiers"]
    c = confidence(_c(0.30, small=True))
    assert c["word"] is None and "the model has little data on this" in c["qualifiers"]
    assert set(CONFIDENCE_RULES) >= {"clearly", "slightly", "the model has little data on this",
                                     "on the training data only"}
    assert str(SMALL_N) in CONFIDENCE_RULES["the model has little data on this"]


def test_linear_term_is_not_automatically_clear():
    """A linear term's range across patients has zero width, but that no longer makes it 'clear':
    a tiny effect is 'slightly' and an effect the test split cannot confirm is 'training only'."""
    m = linear_model()
    lay = build_layer(m, dict(META))
    for c in lay["claims"]:
        cf = c.get("confidence") or {}
        if cf.get("word") == "clearly":
            assert cf["size"] != "slight" and cf["test"] == "agrees" and cf["n_ok"]
            assert c["test_check"]["benefit_ci95"][0] > 0


def test_lint_enforces_confidence_words():
    clear = {"id": "C1", "type": "direction", "verified": True,
             "confidence": {"word": "clearly", "size": "large", "test": "agrees", "n_ok": True}}
    slight = {"id": "C2", "type": "direction", "verified": True,
              "confidence": {"word": None, "size": "slight", "test": "agrees", "n_ok": True}}
    trainonly = {"id": "C3", "type": "direction", "verified": True,
                 "confidence": {"word": None, "size": "large", "test": "unclear", "n_ok": True}}
    small = {"id": "C4", "type": "direction", "verified": True,
             "confidence": {"word": None, "size": "large", "test": "agrees", "n_ok": False}}
    s = "For a typical patient, it clearly raises the predicted risk from about 10% to about 40%."
    assert lint_confidence(s, [clear]) == []
    assert lint_confidence(s, [trainonly])                       # not clear, and not training-only
    assert lint_confidence("It raises the predicted risk (on the training data only).", [trainonly]) == []
    assert lint_confidence("It raises the predicted risk.", [slight])
    assert lint_confidence("It slightly raises the predicted risk.", [slight]) == []
    assert lint_confidence("It slightly raises the predicted risk.", [clear])   # 'slightly' needs a small claim
    assert lint_confidence("It raises the predicted risk.", [small])
    assert lint_confidence("It raises the predicted risk; the model has little data on this.", [small]) == []
    assert lint_confidence("This part acts on 8 of 242 patients.", [trainonly, small]) == []   # no effect stated


def test_template_passes_its_own_lint(layer):
    lay = copy.deepcopy(layer)
    t = lay["template"]
    draft = {"summary": (t["summary"] * 3)[:max(3, min(6, len(t["summary"])))],
             "annotations": {k: v for k, v in t["annotations"].items() if v}}
    assert lint_draft(lay, draft) == []
    lin = build_layer(linear_model(), dict(META))
    t2 = lin["template"]
    assert lint_draft(lin, {"summary": (t2["summary"] * 3)[:3],
                            "annotations": {k: v for k, v in t2["annotations"].items() if v}}) == []


# ------------------------------------------------------------------ other claim types
def test_interaction_checks_on_models_without_one():
    assert not [c for c in build_layer(model("additive"), dict(META))["claims"] if c["type"] == "interaction"]
    tiny = build_layer(model("tiny"), dict(META))
    ints = [c for c in tiny["claims"] if c["type"] == "interaction"]
    assert ints and not any(c["verified"] for c in ints)
    assert not _cited(tiny) & {c["id"] for c in ints}
    assert {c["id"] for c in ints} <= {x["claim"] for x in tiny["template"]["not_supported"]}


def test_null_threshold_subgroup_claims(layer):
    by_type = {}
    for c in layer["claims"]:
        by_type.setdefault(c["type"], []).append(c)
    assert any(c["verified"] and c["statement"].get("variable") == "noise" for c in by_type["null"])
    th = [c for c in by_type["threshold"] if c["statement"]["variable"] == "dose"]
    assert th and th[0]["verified"] and 0.0 < th[0]["statement"]["threshold"]["value"] < 10.0
    sub = by_type["subgroup"][0]
    assert sub["verified"] and 0 < sub["statement"]["n_active"] < 300
    assert sub["risk"]["reference"].startswith("typical patient this part acts on")


def test_count_rule_claim():
    lay = build_layer(rule_model(), dict(META, label="rule"))
    r = next(c for c in lay["claims"] if c["type"] == "rule")
    assert r["verified"] and r["statement"]["counts_predicted_positive"] == [1]
    text = next(s["text"] for s in _sents(lay) if r["id"] in s["claims"])
    assert "exactly one of the attributes take their first value" in text


def test_every_claim_has_evidence_fields(layer):
    for c in layer["claims"]:
        assert c["split"] == "train" and c["annotation"]["id"] and c["reasons"]
        assert isinstance(c["small_n"], bool)
        if c["type"] in ("direction", "magnitude", "interaction", "threshold", "subgroup"):
            assert "confidence" in c and c["risk"]["delta"] is not None
            assert c["test_check"]["status"] in ("agrees", "unclear", "disagrees", "not available")
            if c["test_check"].get("benefit_ci95"):
                assert c["test_check"]["split"] == "test"


# ------------------------------------------------------------------ rendering
def test_failing_claims_never_rendered(layer):
    lay = copy.deepcopy(layer)
    failed = {c["id"] for c in lay["claims"][::2]}
    for c in lay["claims"]:
        if c["id"] in failed:
            c["verified"] = False
            c["reasons"] = ["forced failure for the test"]
    t = render_template(lay)
    cited = {cid for s in list(t["summary"]) + [s for ss in t["annotations"].values() for s in ss]
             for cid in s["claims"]}
    assert not cited & failed
    assert {x["claim"] for x in t["not_supported"]} == failed


def test_all_failed_gives_no_statements(layer):
    lay = copy.deepcopy(layer)
    for c in lay["claims"]:
        c["verified"] = False
    t = render_template(lay)
    assert all(not s["claims"] for s in t["summary"])
    assert all(not ss for ss in t["annotations"].values())


def test_template_text_is_plain(layer):
    text = " ".join(s["text"] for s in _sents(layer))
    for bad in ("relu", "sigmoid", "logit", "z-score", "-1_a", "grp:1", "node 5", "odds", "×"):
        assert bad not in text.lower()
    assert "predicted risk of the event" in text and "%" in text


def test_no_claude_in_rendered_output(layer, tmp_path):
    from explaneat.export.render_plain import appendix_section, render_review
    lay = copy.deepcopy(layer)
    assert CLAUDE.search(json.dumps([p["original_hypothesis"] for p in lay["parts"]]))
    save_layer(lay, tmp_path)
    t = lay["template"]
    draft = {"summary": (t["summary"] * 3)[:3]}
    lay = set_draft(lay["label"], draft, tmp_path, now="2026-11-08T00:00:00+00:00")
    tex = render_review([lay], "Review")
    assert not CLAUDE.search(tex)
    assert not CLAUDE.search("\n".join(appendix_section(export_section(lay["label"], tmp_path), "app:x:")))
    assert not CLAUDE.search(json.dumps(lay["template"]))
    assert "AI-assisted draft" in tex and "What changed in v3" in tex


# ------------------------------------------------------------------ drafts and storage
def test_lint_rejects_unverified_jargon_odds_and_uncited(layer):
    lay = copy.deepcopy(layer)
    ok = next(c for c in lay["claims"] if c["verified"] and c["type"] == "direction")
    lay["claims"][-1]["verified"] = False
    bad_id = lay["claims"][-1]["id"]
    draft = {"summary": [
        {"text": "Node 5 is a ReLU on z-scored dose.", "claims": [ok["id"]]},
        {"text": "Claude says dose causes the event.", "claims": [ok["id"]]},
        {"text": "Group B raises the risk.", "claims": [bad_id]},
        {"text": "No citation here.", "claims": []},
        {"text": "Odds about ×3 for group B.", "claims": [ok["id"]]},
    ], "worth_checking": [{"text": "Worth checking: dose.", "claims": [ok["id"]]}]}
    errs = "\n".join(lint_draft(lay, draft))
    for msg in ("not verified", "activation-function jargon", "z-score jargon", "node or annotation id",
                "names the drafting assistant", "causal wording", "cites no claim", "odds or odds ratios",
                "needs a 'flag'"):
        assert msg in errs, msg


def test_store_draft_and_verdict(layer, tmp_path):
    lay = copy.deepcopy(layer)
    save_layer(lay, tmp_path)
    t = lay["template"]
    pid = next(p["id"] for p in lay["parts"] if p["role"] == "part")
    sub = next(c for c in lay["claims"] if c["type"] == "subgroup")
    draft = {"summary": (t["summary"] * 3)[:3],
             "worth_checking": [{"text": "Worth checking: this part acts on few patients.", "claims": [sub["id"]],
                                 "flag": "gate acts on few patients"}],
             "hypothesis_revisions": {pid: {"proposed": "A gate on dose.", "claims": [sub["id"]]}}}
    with pytest.raises(ValueError):
        set_draft(lay["label"], {"summary": []}, tmp_path)
    stored = set_draft(lay["label"], draft, tmp_path, now="2026-11-08T00:00:00+00:00")
    d = stored["drafted"]
    assert d["drafted_by"] == "ai-assisted" and d["verified_by"] is None
    assert d["hypothesis_revisions"][pid]["original"] == next(p["original_hypothesis"] for p in lay["parts"] if p["id"] == pid)
    v = record_verdict(lay["label"], True, minutes=5, notes="fine", directory=tmp_path)
    assert v["verification"]["verified_by"] == "M. Merry"
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


def test_reextract_keeps_a_draft_that_still_passes_the_lint(layer):
    from explaneat.analysis.plain_language_store import merge_existing
    old = copy.deepcopy(layer)
    t = old["template"]
    old["drafted"] = {"summary": (t["summary"] * 3)[:3], "annotations": {}, "worth_checking": [],
                      "hypothesis_revisions": {}, "claims_hash": old["claims_hash"], "verified_by": None}
    new = copy.deepcopy(layer)
    new["claims_hash"] = "different"
    kept = merge_existing(new, old)
    assert not kept["drafted"].get("stale") and kept["drafted"]["claims_hash"] == "different"
    assert kept["drafted"]["revalidated"]["from_claims_hash"] == layer["claims_hash"]


def test_interaction_with_a_rare_category_is_small_n():
    """An interaction involving a category with fewer than SMALL_N training patients is small-n."""
    from explaneat.analysis.plain_language import interaction_claims, build_parts
    m = model("interaction")
    g = m.var_by_base["grp"]
    keep = np.r_[np.where(g.raw_train == "1")[0][:10], np.where(g.raw_train == "0")[0]]
    from .helpers import CODEBOOK, FEATURES, operations, structure
    from explaneat.analysis.plain_language import PlainModel
    m2 = PlainModel(structure("interaction"), operations("interaction"), m.X[keep], m.y[keep], m.Xt, m.yt,
                    FEATURES, m.mean, m.scale, CODEBOOK, raw_train=m.raw_train[keep], raw_test=m.raw_test)
    part = next(p for p in build_parts(m2) if p.name == "gate")
    ints = interaction_claims(m2, part, {})
    assert ints and all(c["small_n"] for c in ints)
    assert all(c["confidence"]["word"] is None for c in ints)


def test_export_section_validates_in_export_schema(layer, tmp_path):
    from explaneat.export.pack import validate
    from explaneat.export.render_tex import render_appendix
    from tests.test_export.helpers import toy_doc
    lay = copy.deepcopy(layer)
    lay["label"] = "toy-s0"
    save_layer(lay, tmp_path)
    doc = toy_doc()
    assert doc["schema_version"] == "1.4.0" and doc["plain_language"] is None
    doc["plain_language"] = export_section("toy-s0", tmp_path)
    validate(doc)
    tex = render_appendix(doc)
    printed_lines = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    assert "Plain-language summary" in tex and not CLAUDE.search(printed_lines)
