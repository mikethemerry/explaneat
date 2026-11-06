"""Explanation tools (checks, linearisation, unit activity, candidates,
category offsets), explanation records, governance, and their API routes."""

import json
import uuid

import numpy as np
import pytest

from explaneat.analysis import explanation_records as er
from explaneat.analysis import explanation_tools as et
from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType


def _net():
    """x1, x2 (inputs -1, -2), a one-hot pair c:a / c:b (-3, -4).
    h = relu(x1 - 0.5) (hinge), d = relu(-x2 - 100) (dead), o = relu(x2 + 100) (always on),
    out = sigmoid(2h + d + 0.1 o + 3 c:a - 1)."""
    N, C = NetworkNode, NetworkConnection
    nodes = [
        N("-1", NodeType.INPUT), N("-2", NodeType.INPUT), N("-3", NodeType.INPUT), N("-4", NodeType.INPUT),
        N("1", NodeType.HIDDEN, bias=-0.5, activation="relu"),
        N("2", NodeType.HIDDEN, bias=-100.0, activation="relu"),
        N("3", NodeType.HIDDEN, bias=100.0, activation="relu"),
        N("0", NodeType.OUTPUT, bias=-1.0, activation="sigmoid"),
    ]
    conns = [
        C("-1", "1", 1.0, True), C("-2", "2", -1.0, True), C("-2", "3", 1.0, True),
        C("1", "0", 2.0, True), C("2", "0", 1.0, True), C("3", "0", 0.1, True), C("-3", "0", 3.0, True),
    ]
    return NetworkStructure(nodes, conns, ["-1", "-2", "-3", "-4"], ["0"])


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n), np.zeros(n), np.zeros(n)])
    level = rng.random(n) < 0.1
    X[:, 2] = np.where(level, 3.0, -0.33)  # z-scored one-hot
    X[:, 3] = np.where(level, -0.33, 3.0)
    z = 2 * np.maximum(X[:, 0] - 0.5, 0) - 1
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


def test_evaluator_matches_structure_network():
    from explaneat.analysis.activation_extractor import ActivationExtractor

    s = _net()
    X, _ = _data()
    _, out = ActivationExtractor.from_structure(s).extract(X, {"entry_nodes": ["0"], "exit_nodes": ["0"]})
    assert np.max(np.abs(et.StateEvaluator(s).output(X) - out.ravel())) < 1e-12


def test_unit_activity_statuses():
    X, _ = _data()
    st = {r["node_id"]: r["status"] for r in et.unit_activity(_net(), X)}
    assert st == {"1": "hinge", "2": "dead", "3": "always_on"}


def test_linearisation_restore_gain():
    s = _net()
    X, y = _data()
    ops = [
        {"seq": 0, "type": "annotate", "params": {"name": "hinge", "entry_nodes": ["-1"], "exit_nodes": ["1"],
                                                  "subgraph_nodes": ["-1", "1"], "subgraph_connections": [["-1", "1"]]}},
        {"seq": 1, "type": "annotate", "params": {"name": "relay", "entry_nodes": ["-2"], "exit_nodes": ["3"],
                                                  "subgraph_nodes": ["-2", "3"], "subgraph_connections": [["-2", "3"]]}},
    ]
    res = et.linearise_annotations(s, ops, X, y)
    assert res["scope"] == "leaf annotations"
    assert res["nonlinear_units"] == ["hinge"]
    assert res["units"]["relay"]["restore_gain"] == 0.0
    assert res["units"]["hinge"]["restore_gain"] == pytest.approx(res["full"] - res["all_linear"])
    assert res["units"]["hinge"]["max_abs_residual_fit"] > 0.1


def test_candidates_and_unconnected_inputs():
    c = et.candidate_subgraphs(_net())
    kinds = {(x["kind"], tuple(x["exit_nodes"])) for x in c["candidates"]}
    assert ("hidden_gate", ("1",)) in kinds
    assert [d["node_id"] for d in c["direct_inputs"]] == ["-3"]
    assert [u["node_id"] for u in c["unconnected_inputs"]] == ["-4"]
    # -2 feeds two hidden nodes: the gate of node 2 alone needs it split
    gate2 = next(x for x in c["candidates"] if x["exit_nodes"] == ["2"])
    assert gate2["splits_needed"] == ["-2"]


def test_category_offsets():
    X, y = _data()
    res = et.category_offsets(_net(), X, y, ["x1", "x2", "c:a", "c:b"])
    (g,) = res["groups"]
    assert g["base_feature"] == "c"
    (lvl,) = g["levels"]  # c:b has no path to the output
    assert lvl["feature"] == "c:a"
    assert lvl["model_logit_effect"] == pytest.approx(3.0 * (3.0 - -0.33))
    assert lvl["flag"] == "large_unsupported"  # y never depended on c


def test_hierarchy_check():
    ops = [
        {"type": "annotate", "params": {"name": "A", "subgraph_nodes": ["1"], "exit_nodes": ["1"]}},
        {"type": "annotate", "params": {"name": "R", "subgraph_nodes": ["0"], "exit_nodes": ["0"],
                                        "child_annotation_ids": ["A"]}},
        {"type": "annotate", "params": {"name": "B", "subgraph_nodes": ["2"], "exit_nodes": ["2"]}},
    ]
    h = et.hierarchy_check(ops, ["0"])
    assert h["roots"] == ["R", "B"] and not h["complete"]
    h = et.hierarchy_check(ops[:2], ["0"])
    assert h["complete"]


# ------------------------------------------------------------------ records


def test_write_verification_keeps_record(tmp_path):
    rec = {"label": "x-s0", "genome_id": "g", "summary": "s", "verified_by_orchestrator": {"full_coverage": True}}
    (tmp_path / "x-s0.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    er.write_verification("x-s0", True, 25, "renamed a leaf", directory=tmp_path)
    out = json.loads((tmp_path / "x-s0.json").read_text(encoding="utf-8"))
    assert out["mike_verification"]["verified"] is True
    assert out["mike_verification"]["minutes"] == 25
    assert out["mike_verification"]["changes"] == "renamed a leaf"
    assert {k: out[k] for k in rec} == rec
    assert list(out)[: len(rec)] == list(rec)
    (s,) = er.list_records(tmp_path)
    assert s["verified"] and s["label"] == "x-s0"
    assert er.find_record_for_genome("g", tmp_path)["label"] == "x-s0"
    with pytest.raises(ValueError):
        er.record_path("../etc", tmp_path)


# ------------------------------------------------------------------ governance


def test_governance(monkeypatch):
    from explaneat import governance as gv

    class D:
        name = "cohort"
        additional_metadata = {"restricted": True}

    monkeypatch.delenv("EXPLANEAT_RESTRICTED", raising=False)
    monkeypatch.delenv("EXPLANEAT_ALLOW_ASSISTED_RESTRICTED", raising=False)
    gv.require_external_network()
    gv.require_assisted_access(None)
    with pytest.raises(gv.GovernanceError):
        gv.require_assisted_access(D())
    monkeypatch.setenv("EXPLANEAT_ALLOW_ASSISTED_RESTRICTED", "1")
    gv.require_assisted_access(D())
    monkeypatch.setenv("EXPLANEAT_RESTRICTED", "1")
    with pytest.raises(gv.GovernanceError):
        gv.require_external_network()


# ------------------------------------------------------------------ API


@pytest.fixture
def api(test_db, db_session, test_genome, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.api.dependencies import get_db
    from explaneat.db import Dataset, DatasetSplit, Experiment, Explanation, Population

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 10))
    y = (X[:, 0] > 0).astype(float)
    ds = Dataset(name="d", num_samples=60, num_features=10, num_classes=2,
                 feature_names=[f"f{i}" for i in range(9)] + ["g:1"])
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
    ops = [{"seq": 0, "type": "annotate", "params": {
        "name": "lin", "entry_nodes": [str(i) for i in range(-10, 0)], "exit_nodes": ["0"],
        "subgraph_nodes": [str(i) for i in range(-10, 1)],
        "subgraph_connections": [[str(i), "0"] for i in range(-10, 0)]}, "result": {"annotation_id": "ann_0"}}]
    db_session.add(Explanation(genome_id=test_genome.id, operations=ops))
    db_session.commit()

    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    (rec_dir / "t-s0.json").write_text(json.dumps({"label": "t-s0", "genome_id": str(test_genome.id)}), encoding="utf-8")
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", str(rec_dir))

    def _get_db():
        s = test_db.session_factory()
        try:
            yield s
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[get_db] = _get_db
    return TestClient(app), str(test_genome.id), rec_dir


@pytest.mark.db
def test_api_tools(api):
    client, gid, rec_dir = api
    r = client.get(f"/api/genomes/{gid}/tools/check")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["function_preservation_max_abs_err"] == 0.0
    assert body["hierarchy"]["complete"]
    assert body["coverage"]["structural_coverage"] == 1.0
    assert body["full_coverage"]

    r = client.get(f"/api/genomes/{gid}/analyze/coverage")
    assert r.status_code == 200
    assert r.json()["compositional_coverage"] == 1.0  # old route returned the cached 0.0

    r = client.post(f"/api/genomes/{gid}/tools/linearise", json={})
    assert r.status_code == 200, r.text
    assert r.json()["nonlinear_units"] == ["lin"]  # its exit is the sigmoid output

    assert client.get(f"/api/genomes/{gid}/tools/unit-activity").status_code == 200
    assert client.get(f"/api/genomes/{gid}/tools/candidates").status_code == 200
    assert client.get(f"/api/genomes/{gid}/tools/category-offsets").status_code == 200
    assert client.get(f"/api/genomes/{gid}/tools/record").json()["record"]["label"] == "t-s0"

    doc = client.get(f"/api/genomes/{gid}/tools/operations/export").json()
    assert doc["format"] == "explaneat-explanation/1" and len(doc["operations"]) == 1
    prev = client.post(f"/api/genomes/{gid}/tools/operations/import-preview", json={"document": doc}).json()
    assert prev["identical_to_existing"] and prev["function_preserved"]
    r = client.post(f"/api/genomes/{gid}/tools/operations/import", json={"document": doc})
    assert r.status_code == 409  # refuses to overwrite without replace
    r = client.post(f"/api/genomes/{gid}/tools/operations/import", json={"document": doc, "replace": True})
    assert r.status_code == 200
    bad = {"operations": [{"seq": 0, "type": "split_node", "params": {"node_id": "999"}}]}
    assert client.post(f"/api/genomes/{gid}/tools/operations/import-preview", json=bad).status_code == 422

    recs = client.get("/api/records").json()
    assert recs["records"][0]["in_db"] and recs["n_verified"] == 0
    r = client.put("/api/records/t-s0/verification", json={"verified": True, "minutes": 12, "changes": ""})
    assert r.status_code == 200
    assert json.loads((rec_dir / "t-s0.json").read_text())["mike_verification"]["minutes"] == 12
    assert client.put("/api/records/nope/verification", json={"verified": True}).status_code == 404


@pytest.mark.db
def test_restricted_mode_blocks_downloads(api, monkeypatch):
    client, *_ = api
    monkeypatch.setenv("EXPLANEAT_RESTRICTED", "1")
    assert client.get("/api/datasets/search?q=heart").status_code == 403
    assert client.get("/api/governance").json()["restricted_server"] is True


def test_spa_fallback():
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app

    c = TestClient(create_app())
    assert c.get("/api/nope").status_code == 404
    r = c.get("/genome/abc")
    assert r.status_code == 200 and "html" in r.headers["content-type"]
