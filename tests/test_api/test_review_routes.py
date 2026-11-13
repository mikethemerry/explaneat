"""The review page's API (/api/review): the queue with paragraphs, appending sentence and leaf
verdicts to the append-only store, the records_dir guard, read-back and the signoff-v1 sign-off.
No database: the operations loader and the recheck are stubbed; records, layers and the review
store live in a temporary directory."""

import json
import uuid

import pytest

GID = str(uuid.uuid4())


def ops():
    return [
        {"seq": 3, "type": "annotate", "result": {"annotation_id": "ann_3"},
         "params": {"name": "age_gate", "hypothesis": "Gate on age above 50. It adds 0.4 to the log-odds for 117 patients.",
                    "entry_nodes": ["-1"], "exit_nodes": ["7"], "subgraph_nodes": ["-1", "7"],
                    "subgraph_connections": [["-1", "7"]]}},
        {"seq": 4, "type": "annotate", "result": {"annotation_id": "ann_4"},
         "params": {"name": "linear_terms", "hypothesis": "Additive terms. The largest is 1.5 per sd of ca.", "entry_nodes": ["-2"],
                    "exit_nodes": ["8"], "subgraph_nodes": ["-2", "8"], "subgraph_connections": [["-2", "8"]]}},
        {"seq": 5, "type": "annotate", "result": {"annotation_id": "ann_5"},
         "params": {"name": "root", "hypothesis": "Sum of the parts.", "entry_nodes": ["7", "8"], "exit_nodes": ["0"],
                    "subgraph_nodes": ["7", "8", "0"], "subgraph_connections": [],
                    "child_annotation_ids": ["ann_3", "ann_4"]}},
    ]


def record():
    return {"label": "t-s0", "genome_id": GID, "structural_coverage": 1.0, "compositional_coverage": 1.0,
            "n_leaf_annotations": 2, "n_composition_annotations": 1, "hidden_nodes": 2,
            "summary": "The model adds an age gate to 3 linear terms.", "verified_by_orchestrator": {"full_coverage": True}}


def layer():
    return {"genome_id": GID, "label": "t-s0", "claims": [{"id": "C01", "risk": {"from": 0.2, "to": 0.4}, "verified": True}],
            "claims_hash": "h1", "parts": {},
            "drafted": {"drafted_at": "2026-11-01T00:00:00", "titles": {"ann_3": "Age gate"},
                        "lead": [{"text": "About 20% for a typical patient.", "claims": ["C01"]}],
                        "readdown": {"ann_3": {"hypothesis": [{"text": "Older patients have a higher risk.", "claims": ["C01"]}],
                                               "evidence": [{"text": "It acts on 117 of 242 patients.", "claims": ["C01"]},
                                                            {"text": "Its formula reproduces the part exactly.", "claims": ["C01"]}]}}}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    rec_dir, lay_dir, rev_dir = tmp_path / "records", tmp_path / "layers", tmp_path / "review"
    for d in (rec_dir, lay_dir, rev_dir):
        d.mkdir()
    (rec_dir / "t-s0.json").write_text(json.dumps(record(), indent=2))
    (lay_dir / "t-s0.json").write_text(json.dumps(layer()))
    monkeypatch.setenv("EXPLANEAT_RECORDS_DIR", str(rec_dir))
    monkeypatch.setenv("EXPLANEAT_LAYERS_DIR", str(lay_dir))
    monkeypatch.setenv("EXPLANEAT_REVIEW_DIR", str(rev_dir))
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))          # the recheck cache
    monkeypatch.setenv("EXPLANEAT_LEVELS_DIR", str(tmp_path / "levels"))  # no probe sidecar
    import explaneat.verify.model as vm
    import explaneat.verify.recheck as rc
    monkeypatch.setattr(vm, "load_db_operations", lambda gid, db=None: ops())
    monkeypatch.setattr(rc, "recheck", lambda target, log=None: {
        "summary": {"verdict": "stub"}, "findings": [], "claims": {"C01": {"verified": True}},
        "freetext": {"checked": [], "unchecked_statements": []}})
    import explaneat.api.routes.review as rv
    monkeypatch.setattr(rv, "_experiment_of", lambda gid: {"in_db": False})
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    c = TestClient(create_app())
    c.dirs = {"records": rec_dir, "layers": lay_dir, "review": rev_dir}
    return c


def test_queue_with_paragraphs_leaves_and_live_signoff(client):
    r = client.get("/api/review/t-s0")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"] == "t-s0" and d["genome_id"] == GID and d["protocol"] == "signoff-v1"
    assert d["queue"], "the hypotheses' typed numbers make tier-A items"
    paras = {p["id"]: p for p in d["document"]}
    for it in d["queue"]:
        assert it["paragraph_id"] in paras
        assert any(s["sid"] == it["sid"] for s in paras[it["paragraph_id"]]["sentences"])
        assert it["section"] in ("gate", "tier A", "sample B/C", "sample A'") and it["verdict"] is None
    # the drafted read-down keeps its paragraph (several sentences, one title from the layer)
    ev = paras["drafted.readdown.ann_3.evidence"]
    assert ev["title"] == "Age gate: evidence" and len(ev["sentences"]) == 2
    assert [lf["name"] for lf in d["leaves"]] == ["age_gate", "linear_terms"]
    assert all(lf["verdict"] is None and lf["probe"] is None for lf in d["leaves"])
    assert d["signoff"]["passed"] is False and len(d["signoff"]["missing"]) == len(d["queue"])
    assert d["verdicts"] == {"n": 0, "n_leaf": 0, "first_at": None, "last_at": None, "seconds": 0}
    assert not list(client.dirs["review"].iterdir()), "a read writes nothing"


def test_append_verdict_read_back_and_guards(client):
    d = client.get("/api/review/t-s0").json()
    it = d["queue"][0]
    r = client.post("/api/review/t-s0/verdict", json={"sid": it["sid"], "verdict": "confirmed", "note": "checked", "seconds": 9})
    assert r.status_code == 200, r.text
    assert r.json()["verdict"]["verdict"] == "confirmed" and r.json()["verdict"]["current"] is True
    store = client.dirs["review"] / "t-s0.verdicts.jsonl"
    lines = [json.loads(x) for x in store.read_text().splitlines()]
    assert len(lines) == 1
    ln = lines[0]
    assert ln["id"] == "v-000001" and ln["source"] == "ui" and ln["by"] == "mike" and ln["seconds"] == 9
    assert ln["target"] == {"kind": "sentence", "sid": it["sid"], "ref": it["ref"],
                            "content_hash": ln["target"]["content_hash"], "text": it["text"]}
    assert ln["context"]["records_dir"] == str(client.dirs["records"])
    # read back: the item carries its current verdict; the progress list sees the label
    d2 = client.get("/api/review/t-s0").json()
    q = {i["sid"]: i for i in d2["queue"]}
    assert q[it["sid"]]["verdict"]["verdict"] == "confirmed" and q[it["sid"]]["verdict"]["note"] == "checked"
    assert d2["verdicts"]["n"] == 1 and d2["verdicts"]["first_at"] == ln["at"]
    assert len(d2["signoff"]["missing"]) == len(d["queue"]) - 1
    p = client.get("/api/review").json()
    assert p["labels"]["t-s0"]["n_verdicts"] == 1 and p["labels"]["t-s0"]["signoff_passed"] is None
    # a 'wrong' needs a note; bad verdicts, unknown sids and bad labels are refused
    assert client.post("/api/review/t-s0/verdict", json={"sid": it["sid"], "verdict": "wrong", "note": ""}).status_code == 409
    assert client.post("/api/review/t-s0/verdict", json={"sid": it["sid"], "verdict": "maybe"}).status_code == 400
    assert client.post("/api/review/t-s0/verdict", json={"sid": "s-nothing", "verdict": "confirmed"}).status_code == 404
    assert client.post("/api/review/.hidden/verdict", json={"sid": it["sid"], "verdict": "confirmed"}).status_code == 400
    assert client.get("/api/review/no-such-label").status_code == 404
    # the second verdict supersedes the first; the store is append-only
    r = client.post("/api/review/t-s0/verdict", json={"sid": it["sid"], "verdict": "unsure", "note": "?", "disclose": True})
    assert r.status_code == 200
    lines = [json.loads(x) for x in store.read_text().splitlines()]
    assert [x["verdict"] for x in lines] == ["confirmed", "unsure"] and lines[1]["supersedes"] == "v-000001"
    assert lines[1]["disclosed"] is True


def test_records_dir_guard_refuses_mixed_record_sets(client):
    d = client.get("/api/review/t-s0").json()
    store = client.dirs["review"] / "t-s0.verdicts.jsonl"
    other = {"id": "v-000001", "target": {"kind": "sentence", "sid": "s-x", "content_hash": "c-x", "text": "x"},
             "verdict": "confirmed", "note": "", "at": "2026-11-01T00:00:00+13:00", "by": "mike",
             "context": {"records_dir": "experiments/10seed/explanations"}}
    store.write_text(json.dumps(other) + "\n")
    r = client.post("/api/review/t-s0/verdict", json={"sid": d["queue"][0]["sid"], "verdict": "confirmed"})
    assert r.status_code == 409 and "records_dir" in r.text.lower().replace("explaneat_", "") or "record" in r.text
    assert store.read_text().count("\n") == 1, "nothing appended"
    r = client.post("/api/review/t-s0/leaf", json={"leaf": "age_gate", "verdict": "understood"})
    assert r.status_code == 409
    r = client.post("/api/review/t-s0/signoff", json={"write": False})
    assert r.status_code == 409


def test_leaf_verdicts_share_the_store(client):
    r = client.post("/api/review/t-s0/leaf", json={"leaf": "age_gate", "verdict": "partial", "note": "threshold unclear"})
    assert r.status_code == 200, r.text
    assert r.json()["leaf"] == "age_gate" and r.json()["line"]["target"]["annotation_id"] == "ann_3"
    r = client.post("/api/review/t-s0/leaf", json={"leaf": "ann_4", "verdict": "understood"})
    assert r.status_code == 200
    assert client.post("/api/review/t-s0/leaf", json={"leaf": "nothing", "verdict": "understood"}).status_code == 404
    assert client.post("/api/review/t-s0/leaf", json={"leaf": "age_gate", "verdict": "yes"}).status_code == 400
    d = client.get("/api/review/t-s0").json()
    lv = {lf["name"]: lf["verdict"] for lf in d["leaves"]}
    assert lv["age_gate"]["verdict"] == "partial" and lv["age_gate"]["note"] == "threshold unclear" and lv["age_gate"]["current"]
    assert lv["linear_terms"]["verdict"] == "understood"
    assert d["verdicts"]["n"] == 0 and d["verdicts"]["n_leaf"] == 2
    assert d["signoff"]["leaf_verdicts"]["graded"] == 2 and d["signoff"]["leaf_verdicts"]["understood_mike"] == 0.5
    lines = [json.loads(x) for x in (client.dirs["review"] / "t-s0.verdicts.jsonl").read_text().splitlines()]
    assert [ln["target"]["kind"] for ln in lines] == ["leaf", "leaf"] and lines[0]["source"] == "ui"


def test_signoff_passes_after_every_item_and_writes_only_on_request(client):
    d = client.get("/api/review/t-s0").json()
    for it in d["queue"]:
        r = client.post("/api/review/t-s0/verdict", json={"sid": it["sid"], "verdict": "confirmed", "seconds": 60})
        assert r.status_code == 200, r.text
    rec_path = client.dirs["records"] / "t-s0.json"
    before = rec_path.read_text()
    r = client.post("/api/review/t-s0/signoff", json={"write": False})
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["passed"] is True and res["missing"] == [] and res["problems"] == []
    assert res["counts"]["confirmed"] == len(d["queue"]) and res["sample_wrong"] == 0
    assert 0 < res["error_bound_95"] <= 1 and res["minutes"] == len(d["queue"])
    assert (client.dirs["review"] / "t-s0.signoff.json").is_file() and r.json()["record_written"] is False
    assert rec_path.read_text() == before, "the check alone leaves the record alone"
    r = client.post("/api/review/t-s0/signoff", json={"write": True, "minutes": 7, "changes": "none", "notes": "ui"})
    assert r.status_code == 200 and r.json()["record_written"] is True
    rec = json.loads(rec_path.read_text())
    mv = rec["mike_verification"]
    assert mv["verified"] is True and mv["protocol"] == "signoff-v1" and mv["minutes"] == 7 and mv["changes"] == "none"
    assert mv["signoff"]["counts"]["confirmed"] == len(d["queue"]) and mv["signoff"]["content_hash"] == d["target_hash"]
    assert mv["signoff"]["leaf_verdicts"]["graded"] == 0
    assert sorted(p.name for p in client.dirs["records"].iterdir()) == ["t-s0.json"], "nothing else written"
    d2 = client.get("/api/review/t-s0").json()
    assert d2["record"]["mike_verification"]["verified"] is True
    assert client.get("/api/review").json()["labels"]["t-s0"]["signoff_passed"] is True


def test_signoff_fails_on_an_unfixed_wrong(client):
    d = client.get("/api/review/t-s0").json()
    for k, it in enumerate(d["queue"]):
        body = {"sid": it["sid"], "verdict": "wrong", "note": "off by one"} if k == 0 else {"sid": it["sid"], "verdict": "confirmed"}
        assert client.post("/api/review/t-s0/verdict", json=body).status_code == 200
    r = client.post("/api/review/t-s0/signoff", json={"write": True, "minutes": 3})
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["passed"] is False and any("marked wrong" in p for p in res["problems"])
    assert r.json()["record_written"] is False
    assert "mike_verification" not in json.loads((client.dirs["records"] / "t-s0.json").read_text())


def test_csv_command_and_route(client):
    from explaneat.api.routes.review import csv_command
    cmd = csv_command("experiments/10seed/explanations_v4")
    assert cmd is not None and cmd[-4:] == ["--dir", "explanations_v4", "--suffix", "_v4"]
    assert csv_command("experiments/10seed/explanations")[-2:] == ["--dir", "explanations"]
    assert csv_command("/tmp/elsewhere") is None and csv_command("experiments/10seed/plain_language_v4") is None
    r = client.post("/api/review/t-s0/rebuild-csv")
    assert r.status_code == 200
    assert r.json()["ran"] is False and "build_explanations_csv.py" in r.json()["command"]


# ----------------------------------------------------------------------------- hierarchy
def test_tree_kinds_parents_sids_and_levels(client, monkeypatch):
    r = client.get("/api/review/t-s0/tree")
    assert r.status_code == 200, r.text
    d = r.json()
    nodes = {n["id"]: n for n in d["nodes"]}
    assert d["roots"] == ["ann_5"] and d["genome_id"] == GID
    assert nodes["ann_5"]["kind"] == "root" and nodes["ann_5"]["children"] == ["ann_3", "ann_4"] and nodes["ann_5"]["parent"] is None
    assert nodes["ann_3"]["kind"] == "leaf" and nodes["ann_3"]["parent"] == "ann_5" and nodes["ann_3"]["children"] == []
    assert nodes["ann_3"]["hypothesis"].startswith("Gate on age") and nodes["ann_3"]["entry_nodes"] == ["-1"]
    # the leaf's sentences: its hypothesis (2 sentences) and the drafted read-down (3)
    assert len(nodes["ann_3"]["sids"]) == 5 and len(nodes["ann_4"]["sids"]) == 2
    assert nodes["ann_3"]["probe"] is None and nodes["ann_3"]["verdict"] is None
    assert d["coverage"]["structural_coverage"] == 1.0 and d["coverage"]["n_leaf_annotations"] == 2
    assert d["levels"] is None
    # a leaf verdict and a levels sidecar show up on the tree
    assert client.post("/api/review/t-s0/leaf", json={"leaf": "age_gate", "verdict": "understood"}).status_code == 200
    lv = client.dirs["review"].parent / "levels"
    lv.mkdir()
    (lv / "t-s0.json").write_text(json.dumps({"levels": {"faithful": {"passed": True}},
                                              "leaves": {"age_gate": {"meaning": {"grade": "meaningful", "concept": "age"}}}}))
    d = client.get("/api/review/t-s0/tree").json()
    nodes = {n["id"]: n for n in d["nodes"]}
    assert nodes["ann_3"]["verdict"]["verdict"] == "understood" and nodes["ann_3"]["probe"] == "meaningful"
    assert d["levels"]["levels"] == {"faithful": {"passed": True}}


def test_tree_resolves_children_by_name_or_parent_id():
    from explaneat.api.routes.review import annotation_tree
    o = ops()
    o[2]["params"]["child_annotation_ids"] = ["age_gate", "ann_4"]          # by name and by id
    t = annotation_tree(o)
    root = next(n for n in t["nodes"] if n["id"] == "ann_5")
    assert root["children"] == ["ann_3", "ann_4"] and root["kind"] == "root"
    o[2]["params"].pop("child_annotation_ids")
    o[0]["params"]["parent_annotation_id"] = "root"
    o[1]["params"]["parent_annotation_id"] = "ann_5"
    t = annotation_tree(o)
    kinds = {n["id"]: (n["kind"], n["parent"]) for n in t["nodes"]}
    assert kinds == {"ann_3": ("leaf", "ann_5"), "ann_4": ("leaf", "ann_5"), "ann_5": ("root", None)}
    # a composition between root and leaves
    o = ops()
    o.append({"seq": 6, "type": "annotate", "result": {"annotation_id": "ann_6"},
              "params": {"name": "top", "hypothesis": "", "entry_nodes": ["0"], "exit_nodes": ["0"], "subgraph_nodes": ["0"],
                         "subgraph_connections": [], "child_annotation_ids": ["root"]}})
    t = annotation_tree(o)
    assert {n["id"]: n["kind"] for n in t["nodes"]} == {"ann_3": "leaf", "ann_4": "leaf", "ann_5": "composition", "ann_6": "root"}
    assert t["roots"] == ["ann_6"]


def test_node_card_evidence_paragraphs_and_viz_flag(client, monkeypatch):
    import explaneat.api.routes.review as rv
    monkeypatch.setattr(rv, "_display_names", lambda gid, ids: [f"name:{i}" for i in ids])
    r = client.get("/api/review/t-s0/node/ann_3")
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["node"]["id"] == "ann_3" and c["node"]["kind"] == "leaf"
    assert c["entry_names"] == ["name:-1"] and c["exit_names"] == ["name:7"] and c["dimensionality"] == [1, 1]
    assert c["compute_viz"] is True and c["suggested_viz_types"][0] == "line"
    assert c["paragraphs"] == ["drafted.readdown.ann_3.evidence", "drafted.readdown.ann_3.hypothesis"]
    assert len(c["sids"]) == 5 and c["findings"] == [] and c["verdict"] is None and c["meaning"] is None
    assert c["evidence"] == []                      # the synthetic ops store none
    # by name, and the root has no on-demand viz (not a leaf)
    root = client.get("/api/review/t-s0/node/root").json()
    assert root["node"]["id"] == "ann_5" and root["compute_viz"] is False and root["dimensionality"] == [2, 1]
    assert client.get("/api/review/t-s0/node/ann_99").status_code == 404


def test_evidence_records_are_normalised():
    from explaneat.api.routes.review import _evidence_records
    ann = {"evidence": {"records": [
        {"type": "analytical_formula", "payload": {"latex": "y = x", "sympy_srepr": "..."}, "narrative": "Closed form."},
        {"type": "subgroup", "narrative": "Rows where the gate is on.", "provenance": ["q_1"],
         "payload": {"query_id": "q_1", "tool": "query_subgroup", "split": "train", "n": 242,
                     "query": {"population": "rows where node_7 > 0"},
                     "values": {"subgroup": {"n": 117, "mean_risk": 0.41234}, "rest": {"n": 125}, "curve": list(range(20))}},
         "history": [{"reason": "amended"}]},
    ], "visualizations": [{"viz_config": {"viz_type": "line"}, "svg_data": "PHN2Zz4=", "narrative": "old snapshot"}]}}
    ev = _evidence_records(ann)
    assert [e["type"] for e in ev] == ["analytical_formula", "subgroup", "visualizations"]
    assert ev[0]["latex"] == "y = x" and ev[0]["rows"] == [] and ev[0]["provenance"] == []
    assert ev[1]["provenance"] == ["q_1"] and ev[1]["tool"] == "query_subgroup" and ev[1]["n"] == 242 and ev[1]["history"] == 1
    assert ["subgroup.n", "117"] in ev[1]["rows"] and ["subgroup.mean_risk", "0.4123"] in ev[1]["rows"]
    assert ["curve", "[20 values]"] in ev[1]["rows"] and ev[1]["query"] == {"population": "rows where node_7 > 0"}
    assert ev[2]["svg_data"] == "PHN2Zz4=" and ev[2]["viz_config"] == {"viz_type": "line"}
