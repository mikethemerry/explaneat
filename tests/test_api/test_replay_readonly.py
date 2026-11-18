"""Replay route, read-only mode and the static appendix exporter (explaneat export site)."""

import json

import numpy as np
import pytest

from explaneat.analysis.replay import describe_op, group_steps
from explaneat.api.readonly import is_allowed
from explaneat.export.site_keys import get_key, model_key, post_key, safe

OPS = [
    {"seq": 0, "type": "rename_node", "params": {"node_id": "-1", "display_name": "age"}},
    {"seq": 1, "type": "rename_node", "params": {"node_id": "-2", "display_name": "sex"}},
    {"seq": 2, "type": "add_identity_node",
     "params": {"target_node": "0", "connections": [["-1", "0"], ["-2", "0"]], "new_node_id": "identity_0"}},
    {"seq": 3, "type": "annotate", "params": {
        "name": "pair", "hypothesis": "age and sex add up", "entry_nodes": ["-1", "-2"],
        "exit_nodes": ["identity_0"], "subgraph_nodes": ["-1", "-2", "identity_0"],
        "subgraph_connections": [["-1", "identity_0"], ["-2", "identity_0"]]},
     "result": {"annotation_id": "ann_3"}},
    {"seq": 4, "type": "annotate", "params": {
        "name": "whole", "hypothesis": "", "entry_nodes": [str(i) for i in range(-10, 0)],
        "exit_nodes": ["0"], "subgraph_nodes": [str(i) for i in range(-10, 1)] + ["identity_0"],
        "subgraph_connections": [], "child_annotation_ids": ["pair"]},
     "result": {"annotation_id": "ann_4"}},
]


def test_group_steps_and_descriptions():
    assert group_steps(OPS) == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert "identity node identity_0" in describe_op(OPS[2])
    assert describe_op(OPS[4]).startswith("Annotate whole (composition of 1 part")


def test_read_only_allow_list():
    assert is_allowed("GET", "/api/genomes/x/model")
    assert is_allowed("POST", "/api/genomes/x/evidence/viz-data")
    assert is_allowed("POST", "/api/genomes/x/operations/validate")
    for m, p in [("POST", "/api/genomes/x/operations"), ("DELETE", "/api/genomes/x/operations/3"),
                 ("POST", "/api/genomes/x/evidence/snapshot"), ("PUT", "/api/genomes/x/evidence/narrative"),
                 ("POST", "/api/features"), ("PATCH", "/api/features/p1"), ("POST", "/api/review/l/verdict"),
                 ("PUT", "/api/records/l/verification"), ("POST", "/api/datasets/pmlb"),
                 ("POST", "/api/portal/batches"), ("POST", "/api/genomes/x/tools/operations/import"),
                 ("PUT", "/api/genomes/x/tools/annotations/a/hypothesis"), ("POST", "/api/experiments/run"),
                 ("POST", "/api/genomes/x/export")]:
        assert not is_allowed(m, p), (m, p)


def test_site_keys_match_the_explorer_rules():
    # the same examples are what web/react-explorer/src/api/staticData.ts produces
    assert safe("a b/ü") == "a~20b~2f~fc"
    assert get_key("experiments", {"limit": 200, "offset": 0}) == "experiments.json"
    assert get_key("genomes/g1/evidence/formula", {"annotation_id": "ann_3", "force": "true"}) == \
        "genomes/g1/evidence/formula@annotation_id=ann_3.json"
    assert get_key("genomes/g1/tools/check", {"closed_form": "false"}) == "genomes/g1/tools/check.json"
    assert get_key("features", {"genome_id": "g1"}) == "features@genome_id=g1.json"
    assert get_key("genomes/g1/evidence/node-info", {"node_id": "-1_a"}) == \
        "genomes/g1/evidence/node-info@node_id=-1_a.json"
    assert model_key("g1") == "genomes/g1/model.json"
    assert model_key("g1", 7) == "genomes/g1/model@upto=7.json"
    body = {"annotation_id": "ann_3", "viz_type": "ice", "params": {"input_dim": 2, "output_dim": 1},
            "split": "both", "sample_fraction": 0.1, "view": "network"}
    assert post_key("genomes/g1/evidence/viz-data", body) == "genomes/g1/viz/ann-ann_3/ice@i2_o1.json"
    assert post_key("genomes/g1/evidence/viz-data", {"viz_type": "pairs_plot", "params": {"output_dim": 0, "fix": "median"}}) \
        == "genomes/g1/viz/model/pairs_plot@o0_median.json"
    assert post_key("genomes/g1/evidence/viz-data", {"viz_type": "pairs_plot", "params": {"dims": [3, 1], "fix": "average"}}) \
        == "genomes/g1/viz/model/pairs_plot@o0_average_d3-1.json"
    assert post_key("genomes/g1/evidence/viz-data", {"node_id": "5", "viz_type": "line", "view": "source"}) \
        == "genomes/g1/viz/node-5/line@i0_o0_source.json"
    assert post_key("genomes/g1/evidence/performance", {"split": "test"}) == "genomes/g1/performance@split=test.json"
    assert post_key("genomes/g1/evidence/input-distribution", {"feature_indices": [4]}) == \
        "genomes/g1/input-distribution@f=4.json"
    assert post_key("genomes/g1/tools/linearise", {"annotation_names": None}) == "genomes/g1/tools/linearise.json"
    assert post_key("genomes/g1/evidence/shap", {}) is None


@pytest.fixture
def api(test_db, db_session, test_genome, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.db import Dataset, DatasetSplit, Experiment, Explanation, Population

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 10))
    y = (X[:, 0] > 0).astype(float)
    ds = Dataset(name="d", num_samples=60, num_features=10, num_classes=2,
                 feature_names=[f"f{i}" for i in range(10)])
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
    db_session.add(Explanation(genome_id=test_genome.id, operations=OPS))
    db_session.commit()

    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    (rec_dir / "t-s0.json").write_text(json.dumps({
        "label": "t-s0", "genome_id": str(test_genome.id), "hidden_nodes": 1,
        "n_leaf_annotations": 1, "n_composition_annotations": 1,
        "mike_verification": {"verified": True, "minutes": 3}}), encoding="utf-8")
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", str(rec_dir))
    monkeypatch.delenv("EXPLANEAT_READ_ONLY", raising=False)
    return TestClient(create_app()), str(test_genome.id), rec_dir, str(exp.id)


@pytest.mark.db
def test_replay_route(api):
    client, gid, *_ = api
    r = client.get(f"/api/genomes/{gid}/replay")
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["n_operations"] == 5 and rep["n_steps"] == 5
    steps = rep["steps"]
    assert steps[0]["upto"] == 0 and steps[0]["type"] == "start"
    assert steps[1]["type"] == "rename" and steps[1]["changes"]["nodes_renamed"] == ["-1", "-2"]
    assert steps[2]["changes"]["nodes_added"] == ["identity_0"]
    assert ["-1", "identity_0"] in steps[2]["changes"]["connections_added"]
    assert steps[3]["annotation"]["name"] == "pair" and steps[3]["annotation"]["hypothesis"]

    m0 = client.get(f"/api/genomes/{gid}/model", params={"upto": 0}).json()
    assert m0["metadata"]["is_original"] and "identity_0" not in {n["id"] for n in m0["nodes"]}
    m3 = client.get(f"/api/genomes/{gid}/model", params={"upto": 3}).json()
    assert "identity_0" in {n["id"] for n in m3["nodes"]}
    names = {n["id"]: n.get("display_name") for n in m3["nodes"]}
    assert names["-1"] == "age"
    # collapse at a replay step uses the annotations made so far
    m4 = client.get(f"/api/genomes/{gid}/model", params={"upto": 4, "collapsed": "pair,whole"}).json()
    assert "fn_pair" in {n["id"] for n in m4["nodes"]} and "fn_whole" not in {n["id"] for n in m4["nodes"]}
    full = client.get(f"/api/genomes/{gid}/model").json()
    assert {n["id"] for n in full["nodes"]} == {n["id"] for n in client.get(
        f"/api/genomes/{gid}/model", params={"upto": 99}).json()["nodes"]}


@pytest.mark.db
def test_get_never_creates_an_explanation(test_db, db_session, test_genome, monkeypatch):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.db import Explanation

    monkeypatch.delenv("EXPLANEAT_READ_ONLY", raising=False)
    client = TestClient(create_app())
    gid = str(test_genome.id)
    assert client.get(f"/api/genomes/{gid}/explanation").status_code == 404
    assert client.get(f"/api/genomes/{gid}/model").status_code == 200
    assert client.get(f"/api/genomes/{gid}/replay").json()["n_steps"] == 1
    assert client.post(f"/api/genomes/{gid}/operations/validate",
                       json={"type": "rename_node", "params": {"node_id": "-1", "display_name": "x"}}).status_code == 200
    with test_db.session_scope() as s:
        assert s.query(Explanation).count() == 0
    # a write creates it
    r = client.post(f"/api/genomes/{gid}/operations",
                    json={"type": "rename_node", "params": {"node_id": "-1", "display_name": "x"}})
    assert r.status_code == 200, r.text
    with test_db.session_scope() as s:
        assert s.query(Explanation).count() == 1
    assert client.get(f"/api/genomes/{gid}/explanation").status_code == 200


@pytest.mark.db
def test_read_only_mode(api, monkeypatch):
    client, gid, rec_dir, _ = api
    assert client.get("/api/config").json() == {"read_only": False, "static": False}
    monkeypatch.setenv("EXPLANEAT_READ_ONLY", "1")
    assert client.get("/api/config").json()["read_only"] is True
    writes = [
        ("post", f"/api/genomes/{gid}/operations", {"type": "rename_node", "params": {"node_id": "-1"}}),
        ("delete", f"/api/genomes/{gid}/operations/0", None),
        ("post", f"/api/genomes/{gid}/evidence/snapshot", {}),
        ("put", f"/api/genomes/{gid}/evidence/narrative", {}),
        ("put", f"/api/genomes/{gid}/tools/annotations/pair/hypothesis", {"hypothesis": "x"}),
        ("post", "/api/features", {"genome_id": gid}),
        ("put", "/api/records/t-s0/verification", {"verified": False}),
        ("post", "/api/review/t-s0/verdict", {}),
        ("post", "/api/datasets/pmlb", {"name": "iris"}),
        ("post", "/api/portal/batches", {}),
        ("put", f"/api/genomes/{gid}/explanation", {"name": "x"}),
    ]
    for method, url, body in writes:
        r = getattr(client, method)(url, **({"json": body} if body is not None else {}))
        assert r.status_code == 403, (method, url, r.status_code)
    assert json.loads((rec_dir / "t-s0.json").read_text())["mike_verification"]["verified"] is True
    # reads and compute-only routes still work
    assert client.get(f"/api/genomes/{gid}/replay").status_code == 200
    assert client.post(f"/api/genomes/{gid}/operations/validate",
                       json={"type": "rename_node", "params": {"node_id": "-1", "display_name": "y"}}).status_code == 200


@pytest.mark.db
def test_export_site(api, tmp_path, monkeypatch):
    client, gid, rec_dir, eid = api
    from explaneat.export.site import export_site, resolve_targets
    from explaneat.export.site_landing import load_records, write_landing

    monkeypatch.setenv("EXPLANEAT_READ_ONLY", "0")  # restored after the test; export_site sets it to 1
    out = tmp_path / "site"
    res = export_site(resolve_targets(["t-s0"], rec_dir), out, records_dir=rec_dir, build=False,
                      log=lambda m: None)
    data = out / "explorer" / "data"
    g = gid
    for rel in ["experiments.json", f"experiments/{eid}.json", f"experiments/{eid}/split.json",
                f"experiments/{eid}/best-genome.json", f"genomes/{g}.json", f"genomes/{g}/model.json",
                f"genomes/{g}/model@upto=0.json", f"genomes/{g}/model@upto=5.json", f"genomes/{g}/replay.json",
                f"genomes/{g}/operations.json", f"genomes/{g}/annotations.json",
                f"genomes/{g}/collapse-meta.json", f"genomes/{g}/tools/check.json",
                f"genomes/{g}/evidence/formula@annotation_id=ann_3.json",
                f"genomes/{g}/performance@split=test.json",
                f"genomes/{g}/viz/ann-ann_3/line@i0_o0.json", f"genomes/{g}/viz/ann-ann_3/ice@i1_o0.json",
                f"genomes/{g}/viz/ann-ann_4/pairs_plot@o0_median.json",
                f"genomes/{g}/viz/model/line@i0_o0.json", f"genomes/{g}/_files.json", "_files.json",
                "records.json", "manifest.json"]:
        assert (data / rel).is_file(), rel
    best = json.loads((data / f"experiments/{eid}/best-genome.json").read_text())
    assert best["genome_id"] == g
    listed = set(json.loads((data / f"genomes/{g}/_files.json").read_text()))
    assert "viz/ann-ann_3/line@i0_o0.json" in listed and "model.json" in listed
    meta = json.loads((data / f"genomes/{g}/collapse-meta.json").read_text())
    assert {a["name"]: a["parent"] for a in meta["annotations"]} == {"pair": "whole", "whole": None}
    ice = json.loads((data / f"genomes/{g}/viz/ann-ann_3/ice@i0_o0.json").read_text())
    assert len(ice["data"]["grid_x"]) <= 51 and len(ice["data"]["ice_curves"]) <= 30
    assert res["networks"][0]["n_plots"] > 0
    landing = write_landing(out, res["networks"], records=load_records(rec_dir, ["t-s0"]),
                            stats={"t": {"seeds": [0], "test_auc_by_seed": {"0": 0.9},
                                         "test_auc": {"median": 0.9, "q1": 0.9, "q3": 0.9}}})
    text = landing.read_text()
    assert f"explorer/#/genome/{g}?experiment={eid}" in text and "test AUC 0.900" in text
