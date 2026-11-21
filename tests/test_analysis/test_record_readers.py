"""Readers of explanation records go through ``explanation_records`` (database first, file
fallback): when the database and the committed file differ, the database's verification wins.

Covers ``load_record_file`` (a record file, a ``report.json`` beside a ``record.json``), the
records' own database (``EXPLANEAT_RECORDS_DATABASE_URL``) and one thesis reader
(``thesis/ch7/exp2/make_assets.records``, which ``per_run`` and ``exp2_figures`` share) on the
committed v4 records imported into an in-memory database. Files are never written.
"""

import json
import sys
from pathlib import Path

import pytest

from explaneat.analysis import explanation_records as er

REPO = Path(__file__).resolve().parents[2]
V4 = REPO / "experiments" / "10seed" / "explanations_v4"


def _rec(p: Path, **doc) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2))
    return p


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No discovered study folders; records only where a test puts them."""
    monkeypatch.setattr(er, "DISCOVER_ROOT", tmp_path / "none")
    monkeypatch.setattr(er, "_DISCOVERED", (0.0, []))
    for k in ("EXPLANEAT_RECORDS_DIR", "EXPLANEAT_RECORDS_DB", "EXPLANEAT_RECORDS_DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EXPLANEAT_RECORDS_DISCOVER", "0")
    return tmp_path


def _set_db_verification(db, source, mv):
    from explaneat.db.models import ExplanationRecord as R
    with db.session_scope() as s:
        row = s.query(R).filter(R.source_path == source).one()
        row.mike_verification = mv
        row.record = dict(row.record, mike_verification=mv)


def test_reader_returns_the_databases_verification_when_db_and_file_differ(isolated, test_db, monkeypatch):
    from explaneat.records.sync import import_records
    d = isolated / "study" / "explanations"
    p = _rec(d / "x-s0.json", label="x-s0", genome_id="g-x", n_leaf_annotations=3,
             mike_verification={"verified": False, "minutes": 9})
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", str(d))
    with test_db.session_scope() as s:
        import_records(s, [d])
    before = p.read_text()
    _set_db_verification(test_db, er.rel(p), {"verified": True, "minutes": 4, "changes": ""})
    rec = er.load_record_file(p)
    assert rec["mike_verification"] == {"verified": True, "minutes": 4, "changes": ""}
    assert rec["n_leaf_annotations"] == 3
    assert er.load_record("x-s0")["mike_verification"]["minutes"] == 4
    assert p.read_text() == before                                     # the file is a cache, untouched
    monkeypatch.setenv("EXPLANEAT_RECORDS_DB", "0")                    # files forced: the file's
    assert er.load_record_file(p)["mike_verification"]["minutes"] == 9


def test_report_beside_a_record_carries_the_databases_verification(isolated, test_db):
    from explaneat.records.sync import import_records
    d = isolated / "c4" / "explanations"
    rp = _rec(d / "lines" / "record.json", label="lines", genome_id="g-l", summary="short")
    report = _rec(d / "lines" / "report.json", label="lines", genome_id="g-l", summary="long", hidden_nodes=7)
    with test_db.session_scope() as s:
        import_records(s, [d])
    _set_db_verification(test_db, er.rel(rp), {"verified": True, "minutes": 2})
    got = er.load_record_file(report)
    assert got["summary"] == "long" and got["hidden_nodes"] == 7
    assert got["mike_verification"]["minutes"] == 2
    assert er.load_record_file(rp)["summary"] == "short"


def test_files_when_the_database_has_no_records_table(isolated):
    from explaneat.db import db
    p = _rec(isolated / "s" / "explanations" / "y.json", label="y", genome_id="g-y",
             mike_verification={"verified": True, "minutes": 1})
    db.init_db("sqlite:///:memory:")
    try:
        assert er.load_record_file(p) == json.loads(p.read_text())
    finally:
        db.close()


def test_records_database_url_is_the_records_own_database(isolated, monkeypatch):
    """A script whose DATABASE_URL holds genomes but no records table reads the records from
    ``EXPLANEAT_RECORDS_DATABASE_URL``."""
    from explaneat.db import db
    from explaneat.db.base import Base, Database
    from explaneat.records.sync import import_records
    d = isolated / "s" / "explanations"
    p = _rec(d / "z.json", label="z", genome_id="g-z", mike_verification={"verified": False})
    url = f"sqlite:///{isolated / 'records.sqlite'}"
    rdb = Database(url)
    rdb.init_db()
    Base.metadata.create_all(rdb.engine)
    with rdb.session_scope() as s:
        import_records(s, [d])
    _set_db_verification(rdb, er.rel(p), {"verified": True, "minutes": 7})
    db.init_db("sqlite:///:memory:")                  # the process's database: no records table
    monkeypatch.setattr(er, "_RECORDS_DBS", {})
    try:
        assert er.load_record_file(p)["mike_verification"] == {"verified": False}   # the file's
        monkeypatch.setenv("EXPLANEAT_RECORDS_DATABASE_URL", url)
        assert er.load_record_file(p)["mike_verification"] == {"verified": True, "minutes": 7}
        er.write_verification("z", True, minutes=8, entered_by="cli")   # written to the records database
        assert er.load_record_file(p)["mike_verification"]["minutes"] == 8
        assert json.loads(p.read_text())["mike_verification"] == {"verified": False}
    finally:
        db.close()
        for x in er._RECORDS_DBS.values():
            x.close()
        rdb.close()


@pytest.mark.skipif(not V4.is_dir(), reason="needs the committed v4 records")
def test_thesis_reader_takes_the_databases_verification(test_db, monkeypatch):
    """make_assets.records (Experiment 2's tables and figures) on the committed v4 records, with one
    verification changed in the database only."""
    from explaneat.records.sync import import_records
    monkeypatch.setenv("EXPLANEAT_RECORDS_DISCOVER", "0")
    monkeypatch.delenv("EXPLANEAT_RECORDS_DATABASE_URL", raising=False)
    monkeypatch.delenv("EXPLANEAT_RECORDS_DB", raising=False)
    with test_db.session_scope() as s:
        import_records(s, [V4])
    f = V4 / "churn100-s3.json"
    before = f.read_bytes()
    _set_db_verification(test_db, er.rel(f), {"verified": True, "minutes": 123.5, "changes": "db only"})
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(REPO / "thesis" / "ch7" / "exp2"))
    monkeypatch.syspath_prepend(str(REPO / "thesis" / "ch7"))
    sys.modules.pop("make_assets", None)
    import make_assets
    recs = make_assets.records("churn100")
    assert recs[3]["mike_verification"]["changes"] == "db only"
    assert make_assets.mike_minutes({"churn100": recs}, "churn100")[3] == 123.5
    file_mv = json.loads(before).get("mike_verification") or {}
    assert all(recs[s].get("mike_verification") == (json.loads((V4 / f"churn100-s{s}.json").read_text())
                                                     .get("mike_verification")) for s in recs if s != 3)
    assert file_mv.get("changes") != "db only"
    assert f.read_bytes() == before
