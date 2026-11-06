"""Lineage API routes on a tiny seeded run recorded in SQLite."""
import pytest

from tests.test_db.test_population_ancestry import CONFIG_TEXT, _run


@pytest.fixture(scope="module")
def client_and_run(tmp_path_factory):
    import torch
    from fastapi.testclient import TestClient

    from explaneat.analysis import lineage as L
    from explaneat.api.app import create_app
    from explaneat.db import db

    tmp = tmp_path_factory.mktemp("lineage_api")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, rows = _run(tmp, "on")
        _, rows_off = _run(tmp, "off", track_ancestry=False)
    finally:
        torch.set_num_threads(n)
    L.clear_cache()
    db.init_db(f"sqlite:///{tmp / 'on.db'}")
    from explaneat.db.models import Experiment

    with db.session_scope() as s:  # the run's config file was a temp file
        for e in s.query(Experiment):
            e.neat_config_text = CONFIG_TEXT
    yield TestClient(create_app()), rows, tmp, rows_off
    db.close()
    L.clear_cache()


def _final(rows):
    last = max(r[0] for r in rows)
    return max((r for r in rows if r[0] == last), key=lambda r: (r[3], -r[2]))


def test_lineage_route(client_and_run):
    client, rows, *_ = client_and_run
    final = _final(rows)
    body = client.get(f"/api/genomes/{final[1]}/lineage").json()
    assert body["available"]
    lin = body["lineage"]
    assert [r["generation"] for r in lin] == list(range(final[0] + 1))
    assert lin[-1]["genome_id"] == str(final[1])
    assert len(body["population"]) == final[0] + 1
    assert all("structural_change" in r for r in lin)


def test_jumps_ancestor_diff(client_and_run):
    client, rows, *_ = client_and_run
    final = _final(rows)
    gid = str(final[1])
    j = client.get(f"/api/genomes/{gid}/lineage/jumps?method=top_k&k=2").json()
    assert j["available"] and len(j["jumps"]) <= 2
    for jump in j["jumps"]:
        assert "diff_summary" in jump
    a = client.get(f"/api/genomes/{gid}/ancestor?generation=1").json()
    assert a["ancestor"]["generation"] == 1
    anc = a["ancestor"]["genome_id"]
    d = client.get(f"/api/genomes/{anc}/diff/{gid}").json()
    assert d["genome_b_id"] == gid and "summary" in d
    g = client.get(f"/api/genomes/{anc}/diff/{gid}/graph").json()
    statuses = {n["status"] for n in g["nodes"]} | {c["status"] for c in g["connections"]}
    assert statuses <= {"common", "added", "removed"}
    assert client.get(f"/api/genomes/{gid}/ancestor?generation=999").status_code == 400


def test_lineage_not_recorded(client_and_run, tmp_path):
    from explaneat.analysis import lineage as L
    from explaneat.db import db

    client, rows, tmp, rows_off = client_and_run
    db.close()
    L.clear_cache()
    db.init_db(f"sqlite:///{tmp / 'off.db'}")
    try:
        gid = str(_final(rows_off)[1])
        body = client.get(f"/api/genomes/{gid}/lineage").json()
        assert body["available"] is False
        assert "run the replay/backfill" in body["message"]
        assert client.get(f"/api/genomes/{gid}/lineage/jumps").json()["available"] is False
    finally:
        db.close()
        L.clear_cache()
        db.init_db(f"sqlite:///{tmp / 'on.db'}")
