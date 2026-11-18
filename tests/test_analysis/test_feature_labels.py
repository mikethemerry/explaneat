"""Readable feature labels (the feature key) from the dataset codebooks, and the API route."""

import numpy as np
import pytest

from explaneat.analysis.feature_labels import both_names, codebook_for, feature_key, feature_label
from explaneat.analysis.plain_language_codebooks import ADULT, CHURN, UCI_HEART, generic_codebook


def test_one_hot_level_is_decoded_with_what_1_means():
    e = feature_label(ADULT, "marital-status:2")
    assert e["raw"] == "marital-status:2"
    assert e["label"] == "married (civilian spouse)"
    assert e["full_label"] == "marital status: married (civilian spouse)"
    assert e["one_means"] == "The marital status is married (civilian spouse)"
    assert e["level"] == "2" and e["decoded"]
    assert both_names(e) == "marital-status:2 · married (civilian spouse)"


def test_binary_and_continuous_columns():
    sex = feature_label(ADULT, "sex")
    assert sex["label"] == "sex" and sex["one_means"] == "The person is a man"
    plan = feature_label(CHURN, "international plan")
    assert plan["label"] == "international calling plan"
    assert plan["one_means"] == "The customer has an international plan"
    age = feature_label(ADULT, "age")
    assert age["label"] == "age" and age["unit"] == "years" and age["one_means"] is None


def test_undecoded_level_and_no_codebook_keep_the_raw_name():
    e = feature_label(UCI_HEART, "thal:4")      # not a UCI code: never guessed
    assert e["label"] == "thal:4" and not e["decoded"]
    g = feature_label(generic_codebook(["f1", "c:3"]), "c:3")
    assert g["label"] == "c:3" and not g["decoded"]
    assert both_names(g) == "c:3"


def test_feature_key_node_ids_and_used():
    rows = feature_key(ADULT, ["age", "marital-status:2"], used_columns={1})
    assert [(r["node_id"], r["used"]) for r in rows] == [("-1", False), ("-2", True)]


def test_codebook_for_falls_back_when_checks_fail_or_cannot_run():
    names = ["age", "sex:0", "sex:1", "cp:4", "chol"]
    raw = np.zeros((10, 5))
    assert codebook_for("Heart test", names, raw, np.array([0, 1] * 5))["id"] == "uci-heart-cleveland"
    bad = codebook_for("Heart test", names, raw, np.ones(10))    # positive share 1.0: check fails
    assert bad["id"] == "none" and any("not used" in g for g in bad["gaps"])
    assert codebook_for("my data", ["x", "y"])["id"] == "none"
    # an adult-named dataset missing the columns the adult checks need: generic, no crash
    assert codebook_for("adult small", ["age", "sex"], np.zeros((4, 2)), np.ones(4))["id"] == "none"


# ------------------------------------------------------------------ API route

@pytest.fixture
def api(test_db, db_session, test_genome):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.api.dependencies import get_db
    from explaneat.api.routes import genomes as genome_routes
    from explaneat.db import Dataset, DatasetSplit, Experiment, Population

    genome_routes._FEATURE_CB_CACHE.clear()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 10))
    y = (np.arange(60) % 2).astype(float)
    names = ["age", "sex:0", "sex:1", "cp:4", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal:7"]
    ds = Dataset(name="Heart Disease (test)", num_samples=60, num_features=10, num_classes=2, feature_names=names)
    ds.set_data(X, y)
    db_session.add(ds)
    db_session.flush()
    sp = DatasetSplit(dataset_id=ds.id, split_type="train_test", train_indices=list(range(40)),
                      test_indices=list(range(40, 60)))
    db_session.add(sp)
    db_session.flush()
    pop = db_session.get(Population, test_genome.population_id)
    exp = db_session.get(Experiment, pop.experiment_id)
    exp.split_id, exp.dataset_id = sp.id, ds.id
    db_session.commit()

    def _get_db():
        s = test_db.session_factory()
        try:
            yield s
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[get_db] = _get_db
    yield TestClient(app), str(test_genome.id)
    genome_routes._FEATURE_CB_CACHE.clear()


@pytest.mark.db
def test_feature_labels_route(api):
    client, gid = api
    r = client.get(f"/api/genomes/{gid}/feature-labels")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["codebook"]["id"] == "uci-heart-cleveland"
    by_node = {f["node_id"]: f for f in body["features"]}
    assert by_node["-4"]["raw"] == "cp:4"
    assert by_node["-4"]["label"] == "no chest pain (asymptomatic)"
    assert by_node["-4"]["one_means"] == "The patient has no chest pain symptoms"
    assert by_node["-1"]["label"] == "age"
    assert all("used" in f for f in body["features"])
    assert client.get("/api/genomes/00000000-0000-0000-0000-000000000000/feature-labels").status_code == 404
