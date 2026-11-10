"""Human cross-verification kit (explaneat.review): sentence ids, verdicts, badges, the risk queue on
the 2026-11 errata, and the signoff-v1 logic. No database: the errata test reads the pre-errata
snapshot and the verifier-benchmark run-1 verdicts from the repository."""

import copy
import json
from pathlib import Path

import pytest

from explaneat.review import diff as rdiff
from explaneat.review import signoff as so
from explaneat.review import verdicts as vd
from explaneat.review.overlay import overlay
from explaneat.review.provenance import DictProvenance
from explaneat.review.queue import Item, build_queue, score_features
from explaneat.review.sentences import Sentence, extract, make_sid, node_ids_of, report_hash, split_sentences
from explaneat.review.status import sentence_status, statuses

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "experiments" / "errata" / "2026-11-11" / "snapshot"


def layer(sentences_by_loc, claims=None, drafted_at="2026-11-01T00:00:00"):
    return {"claims": claims or [{"id": "C01", "risk": {"from": 0.2, "to": 0.4}, "verified": True}],
            "claims_hash": "h1",
            "drafted": {"drafted_at": drafted_at, "lead": sentences_by_loc.get("lead", []),
                        "readdown": sentences_by_loc.get("readdown", {})}}


def ops(hyp="Gate 290 = ReLU(1.2*age - 0.5): on for 35 patients, mean risk 0.13. It only acts on men.",
        seq=12):
    return [{"seq": seq, "type": "annotate", "result": {},
             "params": {"name": "gate", "hypothesis": hyp, "subgraph_nodes": ["290"], "entry_nodes": ["-1"],
                        "exit_nodes": ["290"], "evidence": {"records": []}}}]


# ----------------------------------------------------------------------------- sentence ids
def test_sid_stable_when_a_number_changes():
    a = extract(None, None, ops("Mean predicted risk of the 35 patients is 0.13."))[0]
    b = extract(None, None, ops("Mean predicted risk of the 35 patients is 0.18."))[0]
    assert a.sid == b.sid and a.content_hash != b.content_hash


def test_sid_changes_with_wording_and_part_but_not_with_position():
    a = extract(None, None, ops("First. Mean risk 0.13 for 35 patients."))[1]
    b = extract(None, None, ops("Mean risk 0.13 for 35 patients. Then more text."))[0]
    assert a.sid == b.sid and a.index != b.index
    c = extract(None, None, ops("Mean risk 0.13 for 35 women."))[0]
    d = extract(None, None, ops("Mean risk 0.13 for 35 patients.", seq=13))[0]
    assert c.sid != b.sid and d.sid != b.sid                 # wording, part (ann_13 vs ann_12)
    assert b.part == "ann_12"


def test_drafted_sid_survives_reordering_of_a_paragraph():
    s1 = {"text": "It acts on 117 of 242 patients.", "claims": ["C01"]}
    s2 = {"text": "Its formula reproduces the part exactly.", "claims": ["C01"]}
    a = extract(None, layer({"readdown": {"ann_36": {"evidence": [s1, s2]}}}), [])
    b = extract(None, layer({"readdown": {"ann_36": {"evidence": [s2, s1]}}}), [])
    assert {s.sid for s in a} == {s.sid for s in b}
    assert {s.content_hash for s in a} == {s.content_hash for s in b}


def test_reference_text_is_hashed_as_written():
    a = make_sid("operations.g.hypothesis", "ann_1", "{q4.n} patients, mean {q4.mean_risk}")
    b = make_sid("operations.g.hypothesis", "ann_1", "{q5.n} patients, mean {q5.mean_risk}")
    assert a != b                                           # another query is another statement


def test_split_before_a_sentence_starting_with_a_digit_and_not_after_abbreviations():
    assert split_sentences("It counts rows. 2134 is a Thailand indicator.") == \
        ["It counts rows.", "2134 is a Thailand indicator."]
    assert len(split_sentences("Mostly older, e.g. Above 46 years it rises.")) == 1


def test_content_hash_follows_cited_claims():
    lay = layer({"lead": [{"text": "About 20% for a typical patient.", "claims": ["C01"]}]})
    a = extract(None, lay, [])[0]
    lay2 = copy.deepcopy(lay)
    lay2["claims"][0]["risk"]["to"] = 0.45
    b = extract(None, lay2, [])[0]
    assert a.sid == b.sid and a.content_hash != b.content_hash


# ----------------------------------------------------------------------------- verdicts and the diff
def test_verdict_persists_across_a_redraft(tmp_path):
    s_keep = {"text": "It acts on 117 of 242 patients.", "claims": ["C01"]}
    s_num = {"text": "From about 2% to about 4% without one.", "claims": ["C01"]}
    lay1 = layer({"readdown": {"ann_79": {"evidence": [s_keep, s_num]}}})
    sents1 = extract(None, lay1, [])
    for s in sents1:
        vd.record("t", s, "confirmed", note="ok", directory=tmp_path, context=rdiff.context_of(None, lay1, []),
                  now="2026-11-12T10:00:00+13:00")
    # re-draft: the paragraph is reordered and one number is rounded twice (E10's mechanism)
    lay2 = layer({"readdown": {"ann_79": {"evidence": [dict(s_num, text="From about 2% to about 5% without one."),
                                                         s_keep]}}}, drafted_at="2026-11-13T00:00:00")
    sents2 = extract(None, lay2, [])
    lines = vd.load("t", tmp_path)
    assert len(lines) == 2
    d = rdiff.diff(sents2, lines, None, lay2, [])
    states = sorted(d["states"].values())
    assert states == ["changed", "unchanged"]
    ch = d["changes"][0]
    assert ch["numbers"] == [["4%", "5%"]]
    assert any(w.startswith("re-draft") for w in ch["why"])
    assert "[-4%-]" in ch["word_diff"] and "{+5%+}" in ch["word_diff"]
    # the changed sentence is a gate in the queue
    st = statuses(sents2, None, set(), lay2)
    q = {i.sid: i for i in build_queue(sents2, st, {}, d["states"])}
    assert q[ch["sid"]].tier == "G" and q[ch["sid"]].gates == ["changed"]


def test_verdict_store_is_append_only_and_latest_wins(tmp_path):
    s = extract(None, None, ops("Mean risk 0.13 for 35 patients."))[0]
    vd.record("t", s, "unsure", note="?", directory=tmp_path)
    vd.record("t", s, "confirmed", note="checked", directory=tmp_path)
    lines = vd.load("t", tmp_path)
    assert [ln["verdict"] for ln in lines] == ["unsure", "confirmed"]
    assert vd.latest(lines)[s.sid]["verdict"] == "confirmed" and lines[1]["supersedes"] == lines[0]["id"]
    with pytest.raises(ValueError):
        vd.record("t", s, "wrong", note="", directory=tmp_path)       # a wrong needs a note


# ----------------------------------------------------------------------------- badges
def _drafted(text, claims=("C01",)):
    return Sentence(make_sid("drafted.lead", None, text), "c-x", "drafted.lead[0]", 0, None, "drafted", text, text,
                    list(claims))


REPORT = {"claims": {"stated": 1}, "findings": [], "freetext": {"checked": [], "unchecked_statements": []}}


def test_badges_for_drafted_text():
    s = _drafted("From about 20% to about 40%.")
    claims = {"C01": {"id": "C01", "verified": True, "risk": {"from": 0.2, "to": 0.4}}}
    assert sentence_status(s, REPORT, claims=claims).badge == "R"
    assert sentence_status(s, None, claims=claims).badge == "U"                 # no recheck: unchecked
    bad = dict(REPORT, findings=[{"check": "prose_number", "location": "drafted.lead[0]", "severity": "mismatch",
                                  "stated": "40%"}])
    assert sentence_status(s, bad, claims=claims).badge == "F"
    unc = dict(REPORT, findings=[{"check": "prose_uncited", "location": "drafted.lead[0]", "severity": "warn",
                                  "stated": "40%"}])
    st = sentence_status(s, unc, claims=claims)
    assert st.badge == "U" and st.features["citation_gap"]
    assert sentence_status(_drafted("It ignores resting blood pressure."), REPORT, claims=claims).badge == "n/a"


def test_badges_for_free_text():
    s = extract(None, None, ops("Gate 290 = ReLU(1.2*age - 0.5). It is active for 48%."))
    rep = {"findings": [], "freetext": {"checked": [
        {"kind": "formula", "text": "Gate 290 = ReLU(1.2*age - 0.5)", "status": "ok", "location": "operations.gate.hypothesis"},
        {"kind": "active_share", "text": "active for 48%", "status": "mismatch", "location": "operations.gate.hypothesis"}],
        "unchecked_statements": []}}
    a = sentence_status(s[0], rep, node_ids_of(ops()))
    b = sentence_status(s[1], rep, node_ids_of(ops()))
    assert a.badge == "R" and a.features.get("formula") == "checked"
    assert b.badge == "F"
    c = sentence_status(extract(None, None, ops("Mean risk 0.13 for 35 patients."))[0], rep)
    assert c.badge == "U" and c.features["u_atoms"] == 2


def test_c_badge_needs_a_provenance_store():
    src = "Mean predicted risk {q4.mean_risk} for {q4.n} patients."
    s = Sentence(make_sid("operations.g.hypothesis", "ann_1", src), "c-x", "operations.g.hypothesis", 0, "ann_1",
                 "operations", "Mean predicted risk 0.18 for 35 patients.", src)
    assert sentence_status(s, REPORT).badge == "U"                              # no store yet: never C
    ok = DictProvenance({"refs": {"q4.mean_risk": {"reproduce": "match"}, "q4.n": {"reproduce": "match"}}})
    s_rendered_as_refs = Sentence(s.sid, s.content_hash, s.location, 0, "ann_1", "operations", src, src)
    assert sentence_status(s_rendered_as_refs, REPORT, provenance=ok).badge == "C"
    bad = DictProvenance({"refs": {"q4.mean_risk": {"reproduce": "mismatch"}, "q4.n": {"reproduce": "match"}}})
    assert sentence_status(s_rendered_as_refs, REPORT, provenance=bad).badge == "F"
    scope = DictProvenance({"refs": ok.data["refs"], "scope_mismatch": [s.sid]})
    st = sentence_status(s_rendered_as_refs, REPORT, provenance=scope)
    assert st.badge == "F"
    q = build_queue([s_rendered_as_refs], {s.sid: st})
    assert q[0].gates == ["failed", "scope_mismatch"]


# ----------------------------------------------------------------------------- the queue on the errata
def _snap(lab):
    d = SNAP / lab
    rd = lambda n: json.loads((d / n).read_text())
    o = rd("operations.json")
    return rd("record.json"), rd("plain_language.json"), (o.get("operations") if isinstance(o, dict) else o)


# label, needle in the sentence, location prefix, the recheck v2 sweep's finding on it (if any)
ERRATA = {
    "E1": ("heart-s5", "mean predicted risk 0.13", None, None),
    "E2": ("adult-s5", "27 rows", None, None),
    "E3": ("adult-s5", "2990=relu(1.446", None, ("freetext", "formula", "2990=relu(1.446 age-0.48)", "mismatch")),
    "E4": ("adult-s5", "50 hours", "drafted.readdown.ann_125.evidence[0]", ("prose_uncited", "50")),
    "E5": ("churn100-s4", "zero in area 415", None, None),
    "E6": ("churn100-s4", "0.15", None, None),
    "E7": ("heart-s2", "reversible de", "drafted.blocks[1:ann_31][2]", None),
    "E8": ("churn100-s1", "2.37", "operations", None),
    "E9": ("churn100-s1", "2.67", "operations", None),
    "E10": ("churn100-s1", "about 5%", "drafted.readdown.ann_79.evidence[2]", ("prose_rounding", "5%")),
    "E12": ("churn100-s7", "about 15%", "drafted.readdown.ann_80.evidence[2]", ("prose_rounding", "15%")),
    "E13": ("heart100-s0", "about 20%", "drafted.lead[0]", ("prose_rounding", "20%")),
    "E14": ("heart-s5-c1", "about 28% at 118", None, ("prose_rounding", "28%")),
    "E15": ("heart100-s7", "168 beats", "drafted.readdown.ann_38.evidence[2]", ("prose_uncited", "168")),
    "E16": ("heart-s5", "ONLY through these gates", None, None),
}


def _errata_tier(e):
    lab, needle, locf, finding = ERRATA[e]
    rec, lay, op = _snap(lab)
    ss = extract(rec, lay, op)
    s = next(x for x in ss if needle in x.text and (locf is None or x.ref.startswith(locf)))
    rep = {"claims": {"stated": 1}, "findings": [], "freetext": {"checked": [], "unchecked_statements": []}}
    if finding and finding[0] == "freetext":
        rep["freetext"]["checked"].append({"kind": finding[1], "text": finding[2], "status": finding[3],
                                           "location": s.location})
    elif finding:
        rep["findings"].append({"check": finding[0], "location": s.location, "severity": "warn", "stated": finding[1]})
    st = statuses(ss, rep, node_ids_of(op), lay)
    ov, _ = overlay(lab, ss, rec.get("genome_id"), operations=op)   # benchmark run 1, through its items
    it = next(i for i in build_queue(ss, st, ov, cap=10 ** 6) if i.sid == s.sid)
    return it


def test_queue_on_the_errata_flags_the_genuine_errors_and_keeps_e13_in_b():
    tiers = {e: _errata_tier(e) for e in ERRATA}
    # E11: a hand-typed payload dict in monk2-s0's root evidence (not in the snapshot)
    e11_ops = [{"seq": 40, "type": "annotate", "result": {}, "params": {
        "name": "upper_edge_count_classifier", "hypothesis": "", "subgraph_nodes": [], "entry_nodes": [],
        "exit_nodes": [], "evidence": {"records": [{"type": "empirical", "narrative": "",
                                                    "payload": {"a1=3": "108/108 pos"}}]}}}]
    s11 = extract(None, None, e11_ops)
    st11 = statuses(s11, REPORT, set())
    tiers["E11"] = build_queue(s11, st11)[0]
    genuine = [e for e in tiers if e != "E13"]
    assert len(genuine) == 15
    flagged = [e for e in genuine if tiers[e].tier in ("G", "A")]
    assert len(flagged) >= 12, {e: (tiers[e].tier, tiers[e].score) for e in genuine}
    assert tiers["E13"].tier == "B"                       # the correct sentence the false erratum changed
    # mechanisms the review says no checker reached land in G/A: subgroup, scope words, direct vs whole
    for e in ("E1", "E2", "E5", "E6", "E8", "E9", "E11", "E16"):
        assert tiers[e].tier in ("G", "A"), (e, tiers[e].tier, tiers[e].factors)
    assert "scope word (only)" in tiers["E16"].factors and "verifier_refuted" in tiers["E16"].gates
    assert tiers["E3"].gates == ["failed"]


def test_overlay_skips_items_with_another_structure():
    rec, lay, op = _snap("heart-s5")
    ss = extract(rec, lay, op)
    ov, summ = overlay("heart-s5", ss, rec.get("genome_id"), operations=op)
    assert summ["attached"] > 0
    op2 = copy.deepcopy(op)
    next(o for o in op2 if o["type"] == "annotate")["params"]["subgraph_nodes"].append("999999")
    ov2, summ2 = overlay("heart-s5", ss, rec.get("genome_id"), operations=op2)
    assert summ2["attached"] == 0 and summ2["dropped"] > 0


def test_score_rules():
    s, f = score_features({"u_atoms": 5, "prominent": True})
    assert s == 22.5 and "unchecked number x3" in f            # capped at 3, x1.5
    assert score_features({"u_atoms": 4, "payload": True})[0] == 5   # a payload counts once
    assert score_features({"r_atoms": 6})[0] == 2                    # R counts once per sentence
    assert score_features({})[0] == 0


def test_tier_a_cap_and_order():
    sents = [Sentence(f"s-{k:010d}", "c", "operations.g.hypothesis", k, "ann_1", "operations", "x", "x")
             for k in range(30)]

    class St:
        def __init__(self, u):
            self.badge, self.reasons, self.features = "U", [], {"u_atoms": u}
    st = {s.sid: St(1 + (k % 3)) for k, s in enumerate(sents)}
    q = build_queue(sents, st, cap=25)
    assert [i.tier for i in q].count("A") == 25 and [i.tier for i in q].count("A'") == 5
    a = [i for i in q if i.tier == "A"]
    assert all(a[k].score >= a[k + 1].score for k in range(len(a) - 1))


# ----------------------------------------------------------------------------- signoff-v1
def _signoff_case(n_a=3, n_b=25, n_g=1):
    sents, items = [], []
    for tier, n in (("G", n_g), ("A", n_a), ("B", n_b)):
        for k in range(n):
            s = Sentence(f"s-{tier}{k:08d}", f"c-{tier}{k}", "operations.g.hypothesis", k, "ann_1", "operations",
                         f"{tier} sentence {k}", f"{tier} sentence {k}")
            sents.append(s)
            items.append(Item(s.sid, s.ref, s.text, "U", 6.0 if tier != "B" else 3.0, tier,
                              gates=["failed"] if tier == "G" else []))
    return sents, items


REP_OK = {"findings": [], "target_hash": "t-x"}


def _verdict_all(tmp_path, sents, sids, verdict="confirmed", note="ok", disclosed=False):
    by = {s.sid: s for s in sents}
    for sid in sids:
        vd.record("L", by[sid], verdict, note=note, directory=tmp_path, disclosed=disclosed)


def test_signoff_passes_with_gates_tier_a_and_the_sample(tmp_path):
    sents, items = _signoff_case()
    res = so.check("L", items, sents, [], REP_OK, "t-x")
    assert not res["passed"] and len(res["missing"]) == 1 + 3 + 10
    drawn = res["sample"]["BC"]["drawn"]
    assert drawn == so.sample_order("L", [i.sid for i in items if i.tier == "B"])[:10]
    _verdict_all(tmp_path, sents, [i.sid for i in items if i.tier in ("G", "A")] + drawn)
    res = so.check("L", items, sents, vd.load("L", tmp_path), REP_OK, "t-x")
    assert res["passed"], res
    assert res["sample_n"] == 10 and res["sample_wrong"] == 0
    assert res["error_bound_95"] == pytest.approx(0.2589, abs=1e-4)
    assert res["counts"] == {"confirmed": 14, "wrong": 0, "unsure": 0}


def test_signoff_extends_the_sample_after_a_wrong_and_fails_on_a_second(tmp_path):
    sents, items = _signoff_case()
    by = {s.sid: s for s in sents}
    order = so.sample_order("L", [i.sid for i in items if i.tier == "B"])
    _verdict_all(tmp_path, sents, [i.sid for i in items if i.tier in ("G", "A")] + order[1:10])
    vd.record("L", by[order[0]], "wrong", note="27 rows, there are 26", directory=tmp_path)
    res = so.check("L", items, sents, vd.load("L", tmp_path), REP_OK, "t-x")
    assert res["sample"]["BC"]["extended"] and len(res["sample"]["BC"]["drawn"]) == 20
    assert any("marked wrong" in p for p in res["problems"])
    # fix the text (new content hash), confirm it, review the extension: passes, bound on 1 of 20
    fixed = Sentence(order[0], "c-fixed", by[order[0]].location, 0, "ann_1", "operations", "fixed", "fixed")
    sents2 = [fixed if s.sid == order[0] else s for s in sents]
    vd.record("L", fixed, "confirmed", note="now 26", directory=tmp_path)
    _verdict_all(tmp_path, sents2, order[10:20])
    res = so.check("L", items, sents2, vd.load("L", tmp_path), REP_OK, "t-x")
    assert res["passed"], res
    assert (res["sample_n"], res["sample_wrong"]) == (20, 1)
    assert res["error_bound_95"] == pytest.approx(so.clopper_pearson_upper(1, 20), abs=1e-4) == pytest.approx(0.2161, abs=1e-4)
    assert res["wrong_fixed"] == 1
    # a wrong in the extension fails unless the whole stratum is reviewed
    vd.record("L", {s.sid: s for s in sents2}[order[12]], "wrong", note="bad", directory=tmp_path)
    vd.record("L", {s.sid: s for s in sents2}[order[12]], "confirmed", note="after fix", directory=tmp_path)
    res = so.check("L", items, sents2, vd.load("L", tmp_path), REP_OK, "t-x")
    assert not res["passed"] and any("extension sample" in p for p in res["problems"])
    _verdict_all(tmp_path, sents2, order[20:])
    res = so.check("L", items, sents2, vd.load("L", tmp_path), REP_OK, "t-x")
    assert res["passed"], res["problems"]


def test_signoff_unsure_must_be_disclosed_and_recheck_must_be_current(tmp_path):
    sents, items = _signoff_case(n_b=5)
    gsid = items[0].sid
    _verdict_all(tmp_path, sents, [i.sid for i in items if i.sid != gsid])
    _verdict_all(tmp_path, sents, [gsid], verdict="unsure", note="cannot tell")
    res = so.check("L", items, sents, vd.load("L", tmp_path), REP_OK, "t-x")
    assert not res["passed"] and any("disclosed" in p for p in res["problems"])
    _verdict_all(tmp_path, sents, [gsid], verdict="unsure", note="cannot tell", disclosed=True)
    assert so.check("L", items, sents, vd.load("L", tmp_path), REP_OK, "t-x")["passed"]
    stale = so.check("L", items, sents, vd.load("L", tmp_path), dict(REP_OK, target_hash="t-other"), "t-x")
    assert any("another version" in p for p in stale["problems"])
    outside = dict(REP_OK, findings=[{"check": "structural_coverage", "location": "record.structural_coverage",
                                      "severity": "mismatch"}])
    assert not so.check("L", items, sents, vd.load("L", tmp_path), outside, "t-x")["passed"]
    assert not so.check("L", items, sents, vd.load("L", tmp_path), None, "t-x")["passed"]


def test_a_changed_sentence_invalidates_its_verdict(tmp_path):
    sents, items = _signoff_case(n_b=0)
    _verdict_all(tmp_path, sents, [i.sid for i in items])
    assert so.check("L", items, sents, vd.load("L", tmp_path), REP_OK, "t-x")["passed"]
    changed = [Sentence(s.sid, s.content_hash + "!", s.location, 0, "ann_1", "operations", s.text, s.text)
               if s.sid == items[1].sid else s for s in sents]
    res = so.check("L", items, changed, vd.load("L", tmp_path), REP_OK, "t-x")
    assert not res["passed"] and res["missing"][0]["stale"]


def test_clopper_pearson_values():
    assert so.clopper_pearson_upper(0, 300) == pytest.approx(0.0099, abs=1e-4)
    assert so.clopper_pearson_upper(1, 300) == pytest.approx(0.0157, abs=1e-4)
    assert so.clopper_pearson_lower(9, 12) == pytest.approx(0.473, abs=1e-3)
    assert so.clopper_pearson_upper(0, 0) == 1.0


def test_report_hash_changes_with_any_sentence():
    a = extract(None, None, ops("Mean risk 0.13 for 35 patients."))
    b = extract(None, None, ops("Mean risk 0.18 for 35 patients."))
    assert report_hash(a) != report_hash(b) and report_hash(a) == report_hash(list(reversed(a)))
