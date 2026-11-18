"""Features-of-interest pins: the file store and the /api/features routes (no database)."""
import json

import pytest

from explaneat.analysis import feature_pins as fp

GID = "b4c5f225-ec5f-4e11-996d-27d32b7d1290"


def _pin_fields(**kw):
    base = {
        "genome_id": GID,
        "experiment_id": "exp-1",
        "experiment_name": "heart100 seed 8",
        "record_label": "heart100-s8",
        "scope": {"kind": "model"},
        "inputs": [{"node_id": "-4", "raw": "trestbps", "label": "resting blood pressure", "index": 3}],
        "viz": {"type": "line", "params": {"input_dim": 3, "output_dim": 0}, "split_choice": "both",
                "sample_fraction": 0.1, "view": "network"},
        "note": "five sub-shapes",
        "tags": ["Heart100", "bp", "bp"],
    }
    base.update(kw)
    return base


@pytest.fixture
def store(tmp_path, monkeypatch):
    d = tmp_path / "features"
    monkeypatch.setenv("EXPLANEAT_FEATURES_DIR", str(d))
    return d


def test_store_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv("EXPLANEAT_FEATURES_DIR", raising=False)
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))
    assert fp.store_dir() == (tmp_path / "home" / "features").resolve()


def test_create_list_and_index(store):
    pin = fp.create_pin(_pin_fields())
    assert pin["id"].startswith("fp-") and pin["version"] == 1 and pin["status"] == "open"
    assert pin["tags"] == ["heart100", "bp"]
    assert pin["dossier_command"] == f"uv run python -m explaneat dossier {GID} trestbps"
    assert (store / f"{pin['id']}.json").is_file()
    idx = json.loads((store / "index.json").read_text())
    assert [p["id"] for p in idx["pins"]] == [pin["id"]]
    assert [p["id"] for p in fp.list_pins(genome_id=GID)] == [pin["id"]]
    assert fp.list_pins(genome_id="other") == []
    assert len(fp.list_pins(tag="bp")) == 1 and fp.list_pins(tag="x") == []
    assert len(fp.list_pins(q="blood pressure")) == 1
    assert len(fp.list_pins(label="heart100-s8", status="open")) == 1


def test_update_appends_versions(store):
    pin = fp.create_pin(_pin_fields())
    pid = pin["id"]
    v2 = fp.update_pin(pid, {"status": "working", "note": "work through it"})
    assert v2["version"] == 2 and v2["status"] == "working" and v2["note"] == "work through it"
    assert v2["created_at"] == pin["created_at"]
    same = fp.update_pin(pid, {"status": "working"})  # no change: no new version
    assert same["version"] == 2
    full = fp.get_pin(pid, history=True)
    assert [v["version"] for v in full["history"]] == [1, 2]
    assert full["history"][0]["note"] == "five sub-shapes" and full["history"][0]["status"] == "open"
    assert full["viz"]["params"] == {"input_dim": 3, "output_dim": 0}  # carried over
    with pytest.raises(fp.PinError):
        fp.update_pin(pid, {"status": "bogus"})
    with pytest.raises(fp.PinError):
        fp.update_pin(pid, {"genome_id": "x"})
    assert fp.get_pin(pid)["version"] == 2


def test_index_rebuilt_when_missing_or_stale(store):
    a = fp.create_pin(_pin_fields())
    (store / "index.json").unlink()
    assert [p["id"] for p in fp.list_pins()] == [a["id"]]
    (store / "index.json").write_text("not json")
    assert len(fp.list_pins()) == 1


def test_validation(store):
    with pytest.raises(fp.PinError):
        fp.create_pin(_pin_fields(genome_id=""))
    with pytest.raises(fp.PinError):
        fp.create_pin(_pin_fields(scope={"kind": "galaxy"}))
    with pytest.raises(fp.PinError):
        fp.create_pin(_pin_fields(viz={"params": {}}))
    with pytest.raises(fp.PinError):
        fp.get_pin("../etc/passwd")


def test_dossier_link(store, tmp_path):
    d = tmp_path / "dossier"
    pin = fp.create_pin(_pin_fields(dossier_path=str(d)))
    assert pin["dossier_exists"] is False
    d.mkdir()
    (d / "report.md").write_text("# dossier")
    assert fp.get_pin(pin["id"])["dossier_exists"] is True
    assert fp.dossier_files(fp.get_pin(pin["id"])) == ["report.md"]


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app

    monkeypatch.setenv("EXPLANEAT_MARK_ORPHANS", "0")
    rec = tmp_path / "records"
    rec.mkdir()
    (rec / "heart100-s8.json").write_text(json.dumps({"label": "heart100-s8", "genome_id": GID}))
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", str(rec))
    return TestClient(create_app())


def test_routes(client, store, tmp_path):
    body = _pin_fields()
    body.pop("record_label")  # filled from the explanation records
    r = client.post("/api/features", json=body)
    assert r.status_code == 200, r.text
    pin = r.json()
    assert pin["record_label"] == "heart100-s8"
    pid = pin["id"]

    lst = client.get("/api/features", params={"genome_id": GID}).json()
    assert lst["features_dir"] == str(store.resolve())
    assert [p["id"] for p in lst["pins"]] == [pid]
    assert client.get("/api/features", params={"status": "dropped"}).json()["pins"] == []
    assert len(client.get("/api/features", params={"tag": "bp", "label": "heart100-s8"}).json()["pins"]) == 1

    r = client.patch(f"/api/features/{pid}", json={"status": "explained", "edit": "done"})
    assert r.status_code == 200 and r.json()["version"] == 2
    got = client.get(f"/api/features/{pid}").json()
    assert got["status"] == "explained" and got["history"][-1]["edit"] == "done"
    assert client.patch(f"/api/features/{pid}", json={"status": "nope"}).status_code == 422
    assert client.get("/api/features/fp-missing").status_code == 404

    d = tmp_path / "dossier"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "a.txt").write_text("hello")
    client.patch(f"/api/features/{pid}", json={"dossier_path": str(d)})
    info = client.get(f"/api/features/{pid}/dossier").json()
    assert info["exists"] and info["files"] == ["sub/a.txt"]
    assert client.get(f"/api/features/{pid}/dossier/sub/a.txt").text == "hello"
    assert client.get(f"/api/features/{pid}/dossier/../../x").status_code == 404
