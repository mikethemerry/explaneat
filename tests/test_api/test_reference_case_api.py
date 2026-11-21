"""Reference case routes and ``params.reference`` on viz-data (whole model and node scope)."""
import numpy as np
import pytest

NAMES = [f"f{i}" for i in range(7)] + ["c:a", "c:b", "c:c"]


@pytest.fixture
def api(test_db, db_session, test_genome):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.db import Dataset, DatasetSplit

    rng = np.random.default_rng(5)
    n = 90
    X = rng.normal(size=(n, 10))
    X[:, 4] = rng.integers(0, 2, size=n)
    X[:, 7:] = np.eye(3)[rng.choice(3, size=n, p=[0.5, 0.3, 0.2])]
    y = (np.arange(n) % 2).astype(float)
    ds = Dataset(name="case test", num_samples=n, num_features=10, num_classes=2, feature_names=NAMES)
    ds.set_data(X, y)
    db_session.add(ds)
    db_session.flush()
    mean, scale = X[:60].mean(axis=0), X[:60].std(axis=0)
    sp = DatasetSplit(dataset_id=ds.id, split_type="train_test",
                      train_indices=list(range(60)), test_indices=list(range(60, n)),
                      scaler_type="StandardScaler",
                      scaler_params={"mean": mean.tolist(), "scale": scale.tolist()})
    db_session.add(sp)
    db_session.commit()
    yield TestClient(create_app()), str(test_genome.id), str(sp.id), X, y


@pytest.mark.db
def test_describe_median_case(api):
    client, gid, sid, X, _ = api
    r = client.post(f"/api/genomes/{gid}/evidence/reference-case", json={"dataset_split_id": sid})
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(b["x"]) == 10 and b["rows_used"] == "train"
    assert b["x"][0] == pytest.approx(float(np.median(X[:60, 0])))
    assert b["x"][7:] == [1.0, 0.0, 0.0]
    assert b["groups"][0]["base"] == "c" and b["groups"][0]["support"] > 0
    assert len(b["nearest"]) == 5 and all(0 <= nb["row"] < 60 for nb in b["nearest"])
    assert 0 < b["prediction"]["outputs"][0]["value"] < 1
    assert b["entry"] is None

    pcm = client.post(f"/api/genomes/{gid}/evidence/reference-case",
                      json={"dataset_split_id": sid, "spec": {"base": "per_column_median"}}).json()
    assert pcm["groups"][0]["support"] == 0 and pcm["warnings"]


@pytest.mark.db
def test_row_case_and_node_entry(api):
    client, gid, sid, X, y = api
    r = client.post(f"/api/genomes/{gid}/evidence/reference-case",
                    json={"dataset_split_id": sid, "node_id": "0", "spec": {"base": "row", "row": 70}})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["x"] == pytest.approx(X[70].tolist())
    assert b["row"] == {"index": 70, "outcome": int(y[70]), "in_training": False}
    assert b["entry"] is not None and len(b["entry"]["values"]) == len(b["entry"]["nodes"])
    bad = client.post(f"/api/genomes/{gid}/evidence/reference-case",
                      json={"dataset_split_id": sid, "spec": {"base": "row", "row": 999}})
    assert bad.status_code == 400


@pytest.mark.db
def test_viz_holds_reference(api):
    client, gid, sid, X, _ = api
    url = f"/api/genomes/{gid}/evidence/viz-data"
    base = {"dataset_split_id": sid, "viz_type": "line", "split": "train", "params": {"input_dim": 0}}
    d0 = client.post(url, json=base).json()["data"]
    assert d0["held_rule"] == "median"
    case = X[3].tolist()
    d1 = client.post(url, json={**base, "params": {"input_dim": 0, "reference": {"x": case}}}).json()["data"]
    assert d1["held_rule"] == "reference"
    mean, scale = X[:60].mean(axis=0), X[:60].std(axis=0)
    assert d1["held"] == pytest.approx(((X[3] - mean) / scale).tolist())
    assert d1["grid_y"] != d0["grid_y"]
    # node scope: the entry values come from the forward pass at the case
    dn = client.post(url, json={**base, "node_id": "0",
                                "params": {"input_dim": 0, "reference": {"x": case}}}).json()["data"]
    assert dn["held_rule"] == "reference"
    pp = client.post(url, json={"dataset_split_id": sid, "viz_type": "pairs_plot", "split": "train",
                                "params": {"dims": [0, 1, 2], "reference": {"x": case}}}).json()["data"]
    assert pp["held_rule"] == "reference" and len(pp["held"]) == 10
    wrong = client.post(url, json={**base, "params": {"input_dim": 0, "reference": {"x": [1.0, 2.0]}}})
    assert wrong.status_code == 400


@pytest.mark.db
def test_case_rows(api):
    client, gid, sid, _, y = api
    r = client.post(f"/api/genomes/{gid}/evidence/case-rows",
                    json={"dataset_split_id": sid, "split": "test", "limit": 5, "outcome": 1})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["total"] == int(np.sum(y[60:] == 1)) and len(b["rows"]) == 5
    assert all(row["outcome"] == 1 and row["row"] >= 60 for row in b["rows"])
    p = client.post(f"/api/genomes/{gid}/evidence/case-rows",
                    json={"dataset_split_id": sid, "order": "prediction", "limit": 100}).json()
    preds = [row["prediction"][0] for row in p["rows"]]
    assert preds == sorted(preds)
