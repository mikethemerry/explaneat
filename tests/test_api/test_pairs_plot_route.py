"""POST /api/genomes/{id}/evidence/viz-data with viz_type "pairs_plot" (whole model)."""
import numpy as np
import pytest


@pytest.fixture
def api(test_db, db_session, test_genome):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.db import Dataset, DatasetSplit

    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 10))
    X[:, 4] = rng.integers(0, 2, size=80)       # a binary column
    y = (np.arange(80) % 2).astype(float)
    ds = Dataset(name="pairs test", num_samples=80, num_features=10, num_classes=2,
                 feature_names=[f"f{i}" for i in range(10)])
    ds.set_data(X, y)
    db_session.add(ds)
    db_session.flush()
    sp = DatasetSplit(dataset_id=ds.id, split_type="train_test",
                      train_indices=list(range(60)), test_indices=list(range(60, 80)))
    db_session.add(sp)
    db_session.commit()
    yield TestClient(create_app()), str(test_genome.id), str(sp.id)


@pytest.mark.db
def test_pairs_plot_whole_model(api):
    client, gid, sid = api
    url = f"/api/genomes/{gid}/evidence/viz-data"
    r = client.post(url, json={"dataset_split_id": sid, "viz_type": "pairs_plot", "split": "train"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "pairs_plot" in body["suggested_viz_types"]
    d = body["data"]
    assert len(d["dims"]) == 5 and len(d["pairs"]) == 10
    assert d["n_rows"] == 60 and d["fix"] == "median"
    assert d["color_domain"][1] == 0.5              # binary classifier: centred on 0.5
    assert len(d["ranking"]) == 10

    r = client.post(url, json={"dataset_split_id": sid, "viz_type": "pairs_plot", "split": "train",
                               "params": {"dims": [4, 0, 1], "fix": "average"}})
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["dims"] == [4, 0, 1] and d["fix"] == "average"
    assert d["inputs"][0]["kind"] == "binary" and len(d["inputs"][0]["grid"]) == 2
    # the test genome is linear in its inputs before the output activation: the
    # heatmaps hold some interaction only through the output's nonlinearity
    assert all(0.0 <= p["h"] <= 1.0 for p in d["pairs"])

    r = client.post(url, json={"dataset_split_id": sid, "viz_type": "pairs_plot", "split": "train",
                               "params": {"dims": [2]}})
    assert r.status_code == 400
