"""Install smoke tests: `explaneat setup`, `doctor`, `demo --tiny`, and the API
serving the UI. Seconds each; CI runs them with ``pytest -m smoke``.

Every test points EXPLANEAT_HOME and the database at tmp_path, so nothing
touches ~/.explaneat or a real database. The SQLite path goes through
explaneat/db/sqlite_compat.py, as in a real install.
"""

import json

import pytest
from click.testing import CliRunner

from explaneat.cli import cli

pytestmark = pytest.mark.smoke


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield tmp_path / "home"
    from explaneat.db.base import db

    db.close()


def test_demo_tiny_in_memory(tmp_path, home):
    from explaneat.demo import run_demo

    result = run_demo(tmp_path / "out", tiny=True, use_db=False, log=lambda m: None)
    status = {s.key: s.status for s in result.stages}
    assert result.ok, [(s.key, s.status, s.notes) for s in result.stages]
    assert status["train"] == status["explain"] == status["metrics"] == "PASS"
    assert status["lineage"] == "SKIP"  # no database
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith(
        "# ExplaNEAT demo report: PASS")
    assert (tmp_path / "out" / "report.html").is_file()
    explain = next(s for s in result.stages if s.key == "explain")
    for m in explain.metrics.values():
        assert m["structural_coverage"] == 1.0 and m["compositional_coverage"] == 1.0
        assert max(m["max_abs_diff"].values()) < 1e-9


def test_demo_tiny_sqlite_records_runs(tmp_path, home):
    from explaneat.db.base import db
    from explaneat.db.models import Experiment, Explanation
    from explaneat.demo import run_demo
    from explaneat.settings import sqlite_url

    url = sqlite_url(tmp_path / "demo.db")
    result = run_demo(tmp_path / "out", tiny=True, database_url=url, log=lambda m: None)
    status = {s.key: s.status for s in result.stages}
    assert result.ok, [(s.key, s.status, s.notes) for s in result.stages]
    assert status["database"] == status["lineage"] == "PASS"
    db.init_db(url)
    with db.session_scope() as session:
        assert session.query(Experiment).count() == 4      # 2 seeds x (classification, survival)
        assert session.query(Explanation).count() == 4
    lineage = next(s for s in result.stages if s.key == "lineage")
    assert len(lineage.metrics) == 4
    assert all(v["lineage_length"] >= 1 for v in lineage.metrics.values())
    metrics = next(s for s in result.stages if s.key == "metrics").metrics
    assert {"harrell_c", "uno_c", "td_auc_mean"} <= set(metrics["survival_s0"])

    # The UI opens each experiment's best genome: it must carry the explanation.
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app

    client = TestClient(create_app())
    exps = client.get("/api/experiments?limit=10").json()["experiments"]
    assert len(exps) == 4
    for e in exps:
        best = client.get(f"/api/experiments/{e['id']}/best-genome").json()
        gid = best.get("id") or best.get("genome_id")
        ops = client.get(f"/api/genomes/{gid}/operations").json()["operations"]
        assert any(op["type"] == "annotate" for op in ops), e["name"]


def test_cli_setup_then_doctor(home):
    runner = CliRunner()
    r = runner.invoke(cli, ["setup", "--skip-ui"])
    assert r.exit_code == 0, r.output
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    assert cfg["database_url"].startswith("sqlite:///")
    assert (home / "explaneat.db").is_file()
    r = runner.invoke(cli, ["doctor"])
    assert "database" in r.output and "[OK  ] database" in r.output, r.output
    assert r.exit_code == 0, r.output


def test_cli_demo_tiny_exit_code(tmp_path, home):
    r = CliRunner().invoke(cli, ["demo", "--tiny", "--no-db", "--out", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "Demo PASSED" in r.output


def test_db_upgrade_refuses_sqlite(home, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    r = CliRunner().invoke(cli, ["db", "upgrade"])
    assert r.exit_code == 2
    assert "PostgreSQL-only" in r.output


def test_api_serves_ui_and_api(home):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.install import UI_STATIC

    client = TestClient(create_app())
    assert client.get("/api/health").json() == {"status": "healthy"}
    assert client.get("/api/no-such-route").status_code == 404
    if (UI_STATIC / "index.html").is_file():
        r = client.get("/")
        assert r.status_code == 200 and "<div id=\"root\">" in r.text
        assert client.get("/some/client/route").status_code == 200
