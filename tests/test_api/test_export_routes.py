"""Export API on a tiny seeded run recorded in SQLite: one genome, JSON + TeX; and graceful
degradation when PDF / DOCX tools are missing."""
import json
import time
import uuid

import pytest

from tests.test_db.test_population_ancestry import CONFIG_TEXT, _run


@pytest.fixture(scope="module")
def client_and_genome(tmp_path_factory):
    import torch
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.core.config_utils import load_neat_config
    from explaneat.core.explaneat import ExplaNEAT
    from explaneat.db import db
    from explaneat.db.models import Experiment, Explanation, Genome

    tmp = tmp_path_factory.mktemp("export_api")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, rows = _run(tmp, "exp")
    finally:
        torch.set_num_threads(n)
    db.init_db(f"sqlite:///{tmp / 'exp.db'}")
    best = max(rows, key=lambda r: (r[3] or 0, -r[0]))
    gid = str(best[1])
    with db.session_scope() as s:
        for e in s.query(Experiment):
            e.neat_config_text = CONFIG_TEXT
        g = s.get(Genome, uuid.UUID(gid))
        exp = g.population.experiment
        config = load_neat_config(exp.neat_config_text, exp.config_json)
        ph = ExplaNEAT(g.to_neat_genome(config), config).get_phenotype_network()
        nodes = [str(nd.id) for nd in ph.nodes]
        conns = [[str(c.from_node), str(c.to_node)] for c in ph.connections if c.enabled]
        op = {"seq": 0, "type": "annotate", "created_at": "2026-11-08T00:00:00",
              "params": {"name": "whole_model", "hypothesis": "The whole network as one function.",
                         "entry_nodes": [str(n) for n in ph.input_node_ids],
                         "exit_nodes": [str(n) for n in ph.output_node_ids],
                         "subgraph_nodes": nodes, "subgraph_connections": conns,
                         "evidence": {"records": [{"type": "analytical_formula", "payload": {"note": "x"},
                                                   "narrative": "closed form"}]}}}
        s.add(Explanation(genome_id=g.id, operations=[op]))
    yield TestClient(create_app()), gid, tmp
    db.close()


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/exports/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError("export job did not finish")


def test_single_genome_export_json_tex(client_and_genome, monkeypatch, tmp_path):
    client, gid, _ = client_and_genome
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))
    r = client.post(f"/api/genomes/{gid}/export", json={"formats": ["json", "tex"], "review_marks": True})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    m = job["manifest"]
    assert m["formats"] == {"json": "ok", "tex": "ok"}
    root = (tmp_path / "home" / "exports").resolve()
    assert m["path"].startswith(str(root))   # under $EXPLANEAT_HOME, not the repo
    names = {f["name"] for f in m["files"]}
    label = m["label"]
    assert f"{label}.json" in names and f"{label}.tex" in names
    url = next(f["url"] for f in m["files"] if f["name"] == f"{label}.json")
    doc = client.get(url).json()
    assert doc["provenance"]["genome"]["id"] == gid
    assert doc["coverage"]["matches_get_coverage"] is True
    assert doc["coverage"]["summary"]["structural_coverage"] == 1.0
    tex = client.get(next(f["url"] for f in m["files"] if f["name"] == f"{label}.tex")).text
    assert r"\begin{drafted}" in tex
    assert client.get(f"/api/exports/{m['export_id']}/manifest").json()["export_id"] == m["export_id"]
    # no path traversal
    assert client.get(f"/api/exports/{m['export_id']}/..%2F..%2Fsecret").status_code == 404


def test_unknown_format_rejected(client_and_genome):
    client, gid, _ = client_and_genome
    assert client.post(f"/api/genomes/{gid}/export", json={"formats": ["pptx"]}).status_code == 422


def test_pdf_and_docx_degrade_gracefully(client_and_genome, monkeypatch, tmp_path):
    """No LaTeX, no python-docx: JSON/TeX/TikZ still written, PDF/DOCX say what to install."""
    import builtins
    import shutil as _sh

    from explaneat.export import pack

    client, gid, _ = client_and_genome
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))
    real_which = _sh.which
    monkeypatch.setattr(pack.shutil, "which",
                        lambda c, *a, **k: None if c in ("latexmk", "pdflatex", "pdftoppm") else real_which(c, *a, **k))
    real_import = builtins.__import__

    def no_docx(name, *a, **k):
        if name == "docx":
            raise ImportError("no python-docx")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_docx)
    caps = client.get("/api/exports/capabilities").json()
    assert caps["json"]["available"] and caps["tex"]["available"]
    assert not caps["pdf"]["available"] and "install" in caps["pdf"]["reason"]
    assert not caps["docx"]["available"] and "install" in caps["docx"]["reason"]
    r = client.post(f"/api/genomes/{gid}/export", json={"formats": ["json", "tex", "tikz", "pdf", "docx"]})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    f = job["manifest"]["formats"]
    assert f["json"] == f["tex"] == f["tikz"] == "ok"
    assert f["pdf"].startswith("unavailable: install")
    assert f["docx"].startswith("unavailable: install")
    names = {x["name"] for x in job["manifest"]["files"]}
    assert not any(n.endswith((".pdf", ".docx")) for n in names)
    assert any(n.endswith("-fig-full.tex") for n in names)
