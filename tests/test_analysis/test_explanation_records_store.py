"""Explanation records across study folders and in the database.

Files: several record folders (``EXPLANEAT_RECORDS_DIR`` with ``os.pathsep`` and the discovered
``experiments/*/explanations*`` folders), label clashes, write-back to the record's own file,
the layer beside the record. Database: import (idempotent, differs/force, label clash),
verification with history, export back to the files, the API routes, the migration.
"""

import json
import os
import uuid
from pathlib import Path

import pytest

from explaneat.analysis import explanation_records as er


def _rec(d: Path, name: str, label: str, gid: str, **extra) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps({"label": label, "genome_id": gid, "summary": f"{label} summary", **extra}, indent=2))
    return p


@pytest.fixture
def studies(tmp_path, monkeypatch):
    """A fake ``experiments/`` tree: a 10-seed set and its v4 set (same labels), a case study, a
    nested shape study (``<label>/record.json`` + ``report.json``) and their layers."""
    root = tmp_path / "experiments"
    ten, ten4 = root / "10seed" / "explanations", root / "10seed" / "explanations_v4"
    _rec(ten, "heart-s5.json", "heart-s5", "g-heart")
    _rec(ten4, "heart-s5.json", "heart-s5", "g-heart")
    _rec(ten4, "adult-s0.json", "adult-s0", "g-adult")
    case = root / "churn_case" / "explanations"
    _rec(case, "churn-case-s10.json", "churn-case-s10", "g-churn")
    cox = root / "survival" / "cox_case" / "explanations"
    _rec(cox, "cox-whas500-s0.json", "cox-whas500-s0", "g-whas", structural_coverage=1.0)
    shape = root / "ushape_saddle" / "explanations" / "shape-C"
    shape.mkdir(parents=True)
    (shape / "record.json").write_text(json.dumps({"label": "shape-C", "genome_id": "g-shape", "summary": "short"}))
    (shape / "report.json").write_text(json.dumps({"label": "shape-C", "genome_id": "g-shape", "summary": "long",
                                                   "hidden_nodes": 7}))
    (shape / "recheck_summary.json").write_text(json.dumps({"rule": "x"}))     # not a record
    for d, label in ((root / "churn_case" / "plain_language", "churn-case-s10"),
                     (root / "10seed" / "plain_language", "heart-s5"),
                     (root / "10seed" / "plain_language_v4", "heart-s5")):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{label}.json").write_text(json.dumps({"label": label, "where": str(d)}))
    monkeypatch.setattr(er, "DISCOVER_ROOT", root)
    monkeypatch.setattr(er, "_DISCOVERED", (0.0, []))
    for k in ("EXPLANEAT_RECORDS_DIR", "EXPLANEAT_LAYERS_DIR", "EXPLANEAT_RECORDS_DISCOVER", "EXPLANEAT_RECORDS_DB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", str(ten4))
    return {"root": root, "ten": ten, "ten4": ten4, "case": case, "cox": cox, "shape": shape}


# ----------------------------------------------------------------------------- files
def test_genomes_are_found_in_every_study_folder(studies):
    assert er.find_record_for_genome("g-churn")["label"] == "churn-case-s10"
    assert er.find_record_for_genome("g-whas")["label"] == "cox-whas500-s0"
    assert er.find_record_for_genome("g-adult")["label"] == "adult-s0"
    assert er.find_record_for_genome("nope") is None
    studies_seen = {r["study"] for r in er.list_records()}
    assert studies_seen == {"10seed-v4", "10seed", "churn_case", "cox_case", "ushape_saddle"}
    assert er.searched()["records_dirs"][0] == str(studies["ten4"].resolve())


def test_pathsep_list_and_discovery_switch(studies, monkeypatch):
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", os.pathsep.join([str(studies["case"]), str(studies["cox"])]))
    monkeypatch.setenv("EXPLANEAT_RECORDS_DISCOVER", "0")
    assert [r["label"] for r in er.list_records()] == ["churn-case-s10", "cox-whas500-s0"]
    assert er.records_dir() == studies["case"].resolve()
    assert er.find_record_for_genome("g-adult") is None


def test_label_clash_is_prefixed_with_the_study_deterministically(studies, monkeypatch):
    cat = {e.label: e for e in er.catalog()}
    assert cat["heart-s5"].study == "10seed-v4"                     # the configured set keeps its label
    assert cat["10seed.heart-s5"].source.endswith("10seed/explanations/heart-s5.json")
    assert er.load_record("10seed.heart-s5")["label"] == "10seed.heart-s5"
    assert er.find_record_for_genome("g-heart")["label"] == "heart-s5"     # the configured one first
    assert [e.label for e in er.catalog()] == [e.label for e in er.catalog()]
    monkeypatch.delenv("EXPLANEAT_RECORDS_DIR")                      # the default set: discovery order
    monkeypatch.setattr(er, "DEFAULT_DIR", studies["ten"].resolve())
    cat = {e.label: e for e in er.catalog()}
    assert cat["heart-s5"].study == "10seed" and cat["10seed-v4.heart-s5"].study == "10seed-v4"


def test_verification_is_written_to_the_records_own_file(studies):
    er.write_verification("churn-case-s10", True, minutes=7, entered_by="ui")
    mv = json.loads((studies["case"] / "churn-case-s10.json").read_text())["mike_verification"]
    assert mv["verified"] and mv["minutes"] == 7 and mv["entered_by"] == "ui"
    assert not (studies["ten4"] / "churn-case-s10.json").exists()
    er.write_verification("10seed.heart-s5", False, changes="none")
    assert "mike_verification" in json.loads((studies["ten"] / "heart-s5.json").read_text())
    assert "mike_verification" not in json.loads((studies["ten4"] / "heart-s5.json").read_text())
    with pytest.raises(FileNotFoundError):
        er.write_verification("missing", True)


def test_nested_record_beats_report_and_report_fills_the_summary(studies):
    e = er.entry("shape-C")
    assert e.source.endswith("shape-C/record.json") and e.report_path.endswith("report.json")
    s = next(r for r in er.list_records() if r["label"] == "shape-C")
    assert s["summary"] == "short" and s["hidden_nodes"] == 7


def test_layer_is_resolved_next_to_the_record(studies, monkeypatch):
    from explaneat.analysis.plain_language_store import PLAIN_DIR, layer_path, load_layer
    assert load_layer("churn-case-s10")["where"].endswith("churn_case/plain_language")
    assert load_layer("heart-s5")["where"].endswith("plain_language_v4")
    assert load_layer("10seed.heart-s5")["where"].endswith(os.path.join("10seed", "plain_language"))
    assert layer_path("cox-whas500-s0").parent.name == "plain_language_v4"   # no sibling folder: the layers dir
    assert layer_path("heart-paper").parent.name == "plain_language_v4"
    other = studies["root"] / "elsewhere"
    other.mkdir()
    (other / "churn-case-s10.json").write_text(json.dumps({"where": "explicit"}))
    monkeypatch.setenv("EXPLANEAT_LAYERS_DIR", str(other))              # an explicit layers dir wins
    assert load_layer("churn-case-s10")["where"] == "explicit"
    assert PLAIN_DIR.name == "plain_language"


def test_verify_resolve_reads_a_case_study_label(studies, monkeypatch):
    from explaneat.verify.artefacts import resolve
    t = resolve("churn-case-s10")
    assert t.genome_id == "g-churn" and t.layer["where"].endswith("churn_case/plain_language")
    t = resolve(str(studies["cox"] / "cox-whas500-s0.json"))
    assert t.record["label"] == "cox-whas500-s0"


# ----------------------------------------------------------------------------- database
@pytest.fixture
def dbrecords(studies, test_db, monkeypatch):
    from explaneat.records.sync import import_records
    with test_db.session_scope() as s:
        res = import_records(s)
    return studies, test_db, res


def test_import_is_idempotent_and_keeps_labels(dbrecords):
    studies, db, res = dbrecords
    acts = {r["label"]: r["action"] for r in res}
    assert set(acts.values()) == {"new"} and "10seed.heart-s5" in acts and "heart-s5" in acts
    assert er.db_in_use()
    assert {e.storage for e in er.catalog()} == {"db"}
    from explaneat.records.sync import import_records
    with db.session_scope() as s:
        again = import_records(s)
    assert {r["action"] for r in again} == {"unchanged"}
    p = studies["case"] / "churn-case-s10.json"
    d = json.loads(p.read_text())
    d["summary"] = "edited on disk"
    p.write_text(json.dumps(d))
    with db.session_scope() as s:
        res = {r["label"]: r for r in import_records(s)}
    assert res["churn-case-s10"]["action"] == "differs"
    assert er.load_record("churn-case-s10")["summary"] == "churn-case-s10 summary"     # the database's
    with db.session_scope() as s:
        assert {r["label"]: r["action"] for r in import_records(s, force=True)}["churn-case-s10"] == "updated"
    assert er.load_record("churn-case-s10")["summary"] == "edited on disk"
    assert [ev["kind"] for ev in er.verification_history("churn-case-s10")] == ["import", "reimport"]


def test_new_file_record_whose_label_the_database_holds_is_prefixed(dbrecords):
    studies, db, _ = dbrecords
    _rec(studies["root"] / "other_study" / "explanations", "churn-case-s10.json", "churn-case-s10", "g-other")
    from explaneat.records.sync import import_records
    with db.session_scope() as s:
        res = [r for r in import_records(s) if r["action"] == "new"]
    assert [r["label"] for r in res] == ["other_study.churn-case-s10"]


def test_verification_goes_to_the_database_with_history(dbrecords):
    studies, db, _ = dbrecords
    before = (studies["cox"] / "cox-whas500-s0.json").read_text()
    er.write_verification("cox-whas500-s0", True, minutes=4, entered_by="ui")
    er.write_verification("cox-whas500-s0", False, notes="second look", entered_by="chat")
    assert (studies["cox"] / "cox-whas500-s0.json").read_text() == before        # file untouched
    rec = er.find_record_for_genome("g-whas")
    assert rec["mike_verification"]["verified"] is False and rec["mike_verification"]["entered_by"] == "chat"
    hist = er.verification_history("cox-whas500-s0")
    assert [h["kind"] for h in hist] == ["import", "verification", "verification"]
    assert hist[2]["previous"]["minutes"] == 4                                      # nothing overwritten
    assert er.record_location("cox-whas500-s0").startswith("database")
    assert er.load_record_file(studies["cox"] / "cox-whas500-s0.json")["mike_verification"]["notes"] == "second look"

    from explaneat.records.sync import export_records
    with db.session_scope() as s:
        res = {r["label"]: r["action"] for r in export_records(s)}
    assert res["cox-whas500-s0"] == "written" and res["churn-case-s10"] == "unchanged"
    on_disk = json.loads((studies["cox"] / "cox-whas500-s0.json").read_text())
    assert on_disk["mike_verification"]["notes"] == "second look"
    assert list(on_disk)[:3] == ["label", "genome_id", "summary"]                  # key order kept
    with db.session_scope() as s:
        assert {r["action"] for r in export_records(s)} == {"unchanged"}
    from explaneat.records.sync import import_records
    with db.session_scope() as s:                                                  # exported = in sync
        assert {r["action"] for r in import_records(s)} == {"unchanged"}


def test_records_are_matched_to_their_explanation(studies, test_db, test_genome):
    from explaneat.db.models import Explanation, ExplanationRecord
    from explaneat.records.sync import import_records
    _rec(studies["case"], "linked.json", "linked", str(test_genome.id))
    with test_db.session_scope() as s:
        s.add(Explanation(genome_id=test_genome.id, operations=[]))
    with test_db.session_scope() as s:
        import_records(s, [studies["case"]])
        row = s.query(ExplanationRecord).filter_by(label="linked").one()
        assert row.explanation_id is not None and row.study == "churn_case"
        assert s.query(ExplanationRecord).filter_by(label="churn-case-s10").one().explanation_id is None


def test_files_are_used_until_the_table_exists(studies, monkeypatch):
    from explaneat.db import db
    db.init_db("sqlite:///:memory:")            # a database without the records table
    try:
        assert not er.db_in_use()
        assert er.find_record_for_genome("g-churn")["label"] == "churn-case-s10"
    finally:
        db.close()


def test_api_lists_and_verifies_database_records(dbrecords, monkeypatch):
    from fastapi.testclient import TestClient
    from explaneat.api.app import create_app
    studies, db, _ = dbrecords
    client = TestClient(create_app())
    body = client.get("/api/records").json()
    assert body["database"] is True and len(body["records_dirs"]) >= 5
    by = {r["label"]: r for r in body["records"]}
    assert by["cox-whas500-s0"]["study"] == "cox_case" and by["cox-whas500-s0"]["storage"] == "db"
    r = client.put("/api/records/churn-case-s10/verification", json={"verified": True, "minutes": 3})
    assert r.status_code == 200 and r.json()["stored_in"].startswith("database")
    assert er.verification_history("churn-case-s10")[-1]["entered_by"] == "ui"
    h = client.get("/api/records/churn-case-s10/history").json()
    assert [e["kind"] for e in h["events"]] == ["import", "verification"]


def test_api_reports_the_searched_folders_when_no_record(studies, monkeypatch):
    from fastapi.testclient import TestClient
    from explaneat.api.app import create_app
    client = TestClient(create_app())
    r = client.get(f"/api/genomes/{uuid.uuid4()}/tools/record").json()
    assert r["record"] is None and str(studies["cox"].resolve()) in r["records_dirs"]
    r = client.get("/api/genomes/g-churn/tools/record").json()
    assert r["record"]["label"] == "churn-case-s10" and r["source"]["study"] == "churn_case"


def test_migration_creates_the_models_tables():
    """The Alembic migration and the models agree (columns), on SQLite."""
    import importlib.util
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect
    import explaneat.db.sqlite_compat  # noqa: F401
    from explaneat.db.base import Base
    from explaneat.db import models

    eng = create_engine("sqlite:///:memory:")
    new = {"explanation_records", "explanation_record_events"}
    Base.metadata.create_all(eng, tables=[t for n, t in Base.metadata.tables.items() if n not in new])
    path = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "m6n7o8p9q0r1_add_explanation_records.py"
    spec = importlib.util.spec_from_file_location("mig", path)
    mig = importlib.util.module_from_spec(spec)
    with eng.begin() as conn:
        ctx = MigrationContext.configure(conn)
        import alembic.op as op_mod
        with Operations.context(ctx):
            spec.loader.exec_module(mig)
            mig.upgrade()
    insp = inspect(eng)
    for name, model in (("explanation_records", models.ExplanationRecord),
                        ("explanation_record_events", models.ExplanationRecordEvent)):
        assert {c["name"] for c in insp.get_columns(name)} == {c.name for c in model.__table__.columns}
    assert mig.down_revision == "l5m6n7o8p9q0"


def test_rewrites_keep_the_files_layout(tmp_path):
    for text in ('{\n "label": "a",\n "genome_id": "g",\n "note": "caf\\u00e9"\n}',
                 '{\n  "label": "a",\n  "genome_id": "g",\n  "note": "café"\n}\n'):
        assert er.dumps_like(json.loads(text), text) == text
    p = tmp_path / "a.json"
    p.write_text('{\n "label": "a",\n "genome_id": "g"\n}\n')
    er.write_verification("a", True, directory=tmp_path)
    t = p.read_text()
    assert t.startswith('{\n "label": "a"') and t.endswith("}\n") and '"mike_verification"' in t
