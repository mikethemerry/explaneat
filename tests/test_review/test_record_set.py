"""Pointing the review workflow at another record set (the v4 records and layers)."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def v4(tmp_path, monkeypatch):
    rec, lay = tmp_path / "explanations_v4", tmp_path / "plain_language_v4"
    rec.mkdir()
    lay.mkdir()
    (rec / "heart-s5.json").write_text(json.dumps({"label": "heart-s5", "genome_id": "g"}))
    (lay / "heart-s5.json").write_text(json.dumps({"label": "heart-s5", "genome_id": "g", "claims": [], "parts": {}}))
    for k in ("EXPLANEAT_RECORDS_DIR", "EXPLANEAT_LAYERS_DIR"):   # restored after the test
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    return rec, lay


def test_layers_follow_the_records_and_resolve_reads_both(v4, monkeypatch):
    from explaneat.analysis.explanation_records import record_path, use_record_set
    from explaneat.analysis.plain_language_store import PLAIN_DIR, layer_path, layers_dir
    from explaneat.verify.artefacts import resolve
    rec, lay = v4
    assert layers_dir() == PLAIN_DIR
    use_record_set(str(rec))
    assert layers_dir() == lay.resolve() and record_path("heart-s5").parent == rec.resolve()
    assert layer_path("heart-s5").parent == lay.resolve()
    t = resolve("heart-s5")
    assert t.record["label"] == "heart-s5" and t.layer is not None
    assert t.paths["record"].startswith(str(rec.resolve())) and t.paths["layer"].startswith(str(lay.resolve()))
    other = rec.parent / "layers_elsewhere"
    other.mkdir()
    use_record_set(None, str(other))
    assert layers_dir() == other.resolve()
    with pytest.raises(FileNotFoundError):
        use_record_set(str(rec.parent / "missing"))


def test_signoff_writes_the_selected_record(v4):
    from explaneat.analysis.explanation_records import use_record_set, write_verification
    rec, _ = v4
    use_record_set(str(rec))
    write_verification("heart-s5", True, minutes=12, signoff={"protocol": "signoff-v1"})
    mv = json.loads((rec / "heart-s5.json").read_text())["mike_verification"]
    assert mv["verified"] and mv["protocol"] == "signoff-v1"


def test_verdicts_do_not_mix_record_sets(tmp_path):
    from explaneat.review.verdicts import load, record
    s = SimpleNamespace(sid="s-1", ref="lead[0]", content_hash="h", text="t")
    record("x", s, "confirmed", context={"records_dir": "experiments/10seed/explanations_v4"}, directory=tmp_path)
    record("x", s, "confirmed", context={"records_dir": "experiments/10seed/explanations_v4"}, directory=tmp_path)
    with pytest.raises(ValueError, match="explanations_v4"):
        record("x", s, "wrong", note="n", context={"records_dir": "experiments/10seed/explanations"},
               directory=tmp_path)
    assert len(load("x", tmp_path)) == 2
