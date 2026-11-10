"""The verifier-benchmark scorer on a synthetic key and synthetic verifier outputs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vb_score", ROOT / "experiments" / "verifier_bench" / "score.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

KEY = {"items": [
    {"item_id": "vb-c1", "variant": "control", "faults": [],
     "preexisting": [{"parts": ["ann_9"], "claims": ["C09"], "fields": ["drafted.readdown.ann_9.evidence"], "level": "L1"}]},
    {"item_id": "vb-p1", "variant": "paraphrase", "faults": [], "preexisting": []},
    {"item_id": "vb-f1", "variant": "faulted", "preexisting": [], "faults": [
        {"fault_id": "vb-f1-f1", "type": "changed_number_claim", "level": "L1", "parts": ["ann_3"],
         "claims": ["C41"], "fields": ["drafted.readdown.ann_3", "claims.C41"]},
        {"fault_id": "vb-f1-f2", "type": "wrong_interaction_partner", "level": "L3", "parts": ["ann_5"],
         "claims": [], "fields": ["drafted.readdown.ann_5.hypothesis"]}]},
    {"item_id": "vb-f2", "variant": "faulted", "preexisting": [], "faults": [
        {"fault_id": "vb-f2-f1", "type": "structural_missing_node", "level": "L1", "parts": ["ann_7"],
         "claims": [], "fields": ["operations.gate", "structure", "record.structural_coverage"]}]},
    {"item_id": "vb-f3", "variant": "faulted", "preexisting": [], "faults": [
        {"fault_id": "vb-f3-f1", "type": "restates_closed_form", "level": "L4", "parts": ["ann_2"],
         "claims": [], "fields": ["drafted.readdown.ann_2.hypothesis"]}]},
]}


def finding(i, grade, level, location, part=None, claims=()):
    return {"id": i, "grade": grade, "level": level, "location": location, "part": part, "claims": list(claims)}


VERDICTS = {
    # control: a weakened finding at the known pre-existing issue (not a false positive) and an upheld one
    "vb-c1": {"item_id": "vb-c1", "minutes": 20, "overall": {"grade": "weakened"}, "findings": [
        finding("V1", "weakened", "L1", "drafted.readdown.ann_9.evidence[0]", "ann_9", ["C09"]),
        finding("V2", "upheld", "L1", "record.structural_coverage")]},
    # paraphrase: a refuted finding: a false positive
    "vb-p1": {"item_id": "vb-p1", "started_at": "2026-11-10T09:00:00+13:00",
              "finished_at": "2026-11-10T09:30:00+13:00", "overall": "refuted", "findings": [
        finding("V1", "refuted", "L2", "drafted.readdown.ann_4.hypothesis[0]", "ann_4")]},
    # faulted: the claim fault refuted at L1 by claim id; the partner fault only weakened at part level
    "vb-f1": {"item_id": "vb-f1", "minutes": 45, "overall": {"grade": "refuted"}, "findings": [
        finding("V1", "refuted", "L1", "claims.C41", "ann_3", ["C41"]),
        finding("V2", "weakened", "L3", "operations.other_name.hypothesis", "ann_5"),
        finding("V3", "refuted", "L2", "drafted.readdown.ann_8.hypothesis[0]", "ann_8")]},
    # faulted: structural fault caught at L1 via the record field
    "vb-f2": {"item_id": "vb-f2", "minutes": 10, "overall": {"grade": "refuted"}, "findings": [
        finding("V1", "refuted", "L1", "record.structural_coverage", None)]},
    # vb-f3 not run
    "vb-zz": {"item_id": "vb-zz", "findings": []},
}


@pytest.fixture()
def res():
    return S.score(KEY, VERDICTS)


def items(res):
    return {i["item_id"]: i for i in res["items"]}


def test_field_and_claim_matching():
    f = KEY["items"][2]["faults"][0]
    assert S.match_kind(finding("x", "refuted", "L1", "drafted.readdown.ann_3.evidence[1]"), f) == "field"
    assert S.match_kind(finding("x", "refuted", "L1", "drafted.blocks[2]", claims=["c41"]), f) == "claim"
    assert S.match_kind(finding("x", "refuted", "L1", "somewhere", part="ann_3"), f) == "part"
    # ann_33 must not match ann_3
    assert S.match_kind(finding("x", "refuted", "L1", "drafted.readdown.ann_33.hypothesis[0]", part="ann_33"), f) is None
    assert S.match_kind(finding("x", "refuted", "L1", "plain_language.claims.C41"), f) == "field"


def test_detection_strict_lenient_and_level(res):
    it = items(res)
    f1, f2 = it["vb-f1"]["faults"]
    assert f1["detected_strict"] and f1["detected_lenient"] and f1["caught_level"] == "L1"
    assert f1["match"] in ("field", "claim")
    assert not f2["detected_strict"] and f2["detected_lenient"] and f2["match"] == "part"
    assert f2["caught_level"] == "L3"
    s = it["vb-f2"]["faults"][0]
    assert s["detected_strict"] and s["match"] == "field"
    assert it["vb-f3"]["run"] is False and it["vb-f3"]["faults"][0]["not_run"]


def test_false_positives_and_preexisting(res):
    it = items(res)
    assert it["vb-c1"]["fp_lenient"] is False and it["vb-c1"]["fp_strict"] is False
    assert it["vb-p1"]["fp_strict"] is True
    assert [e["id"] for e in it["vb-f1"]["extra_findings"]] == ["V3"]
    fp = res["summary"]["false_positives"]
    assert fp["control"]["lenient"]["k"] == 0 and fp["paraphrase"]["strict"]["k"] == 1
    assert fp["all_clean"]["strict"]["rate"] == pytest.approx(0.5)


def test_summary_rates_levels_and_time(res):
    s = res["summary"]
    assert s["items"] == {"total": 5, "run": 4, "not_run": 1,
                          "by_variant": {"control": 1, "paraphrase": 1, "faulted": 3}}
    d = s["detection"]
    assert d["all"]["strict"]["n"] == 3 and d["all"]["strict"]["k"] == 2
    assert d["all"]["lenient"]["k"] == 3
    assert d["by_level"]["L3"]["strict"]["rate"] == 0.0 and d["by_level"]["L3"]["lenient"]["rate"] == 1.0
    assert d["by_type"]["structural_missing_node"]["strict"]["rate"] == 1.0
    assert s["caught_at_level"]["L1"] == {"L1": 2}
    assert s["time_minutes"]["n"] == 4 and s["time_minutes"]["median"] == pytest.approx(25.0)
    assert res["unknown_verdicts"] == ["vb-zz"]


def test_markdown_and_cli(tmp_path):
    runs = tmp_path / "runs"
    for iid, v in VERDICTS.items():
        (runs / iid).mkdir(parents=True)
        (runs / iid / "verdict.json").write_text(json.dumps(v))
    key = tmp_path / "key.json"
    key.write_text(json.dumps(KEY))
    hf = tmp_path / "KEY_SHA256"
    hf.write_text(hashlib.sha256(key.read_bytes()).hexdigest() + "  key.json\n")
    out, md = tmp_path / "score.json", tmp_path / "score.md"
    S.main(["--runs", str(runs), "--key", str(key), "--hash-file", str(hf), "--out", str(out), "--md", str(md)])
    r = json.loads(out.read_text())
    assert r["summary"]["items"]["run"] == 4
    text = md.read_text()
    assert "False positives" in text and "structural_missing_node" in text and "claude" not in text.lower()
    hf.write_text("0" * 64 + "  key.json\n")
    with pytest.raises(SystemExit):
        S.main(["--runs", str(runs), "--key", str(key), "--hash-file", str(hf)])


def test_committed_hash_matches_the_private_key_when_present():
    key = S.DEFAULT_KEY
    if not key.is_file():
        pytest.skip("answer key not on this machine")
    S.check_key(key)
    k = json.loads(key.read_text())
    on_disk = {p.name for p in (ROOT / "experiments" / "verifier_bench" / "items").iterdir() if p.is_dir()}
    assert on_disk == {e["item_id"] for e in k["items"]}
