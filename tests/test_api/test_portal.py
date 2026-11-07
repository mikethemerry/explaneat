"""Portal workflow end to end on SQLite with synthetic data:
CSV import (survival + classification), a 2-seed batch in worker processes,
progress, cancel/resume, results with the comparator, hazard ratios,
restricted-storage checks. The CSVs live in tmp_path (outside the repo)."""
import json
import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _write_survival_csv(path, n=300, seed=0):
    import pandas as pd

    from explaneat.survival.simulate import simulate

    X, t, e, _ = simulate(n=n, d=4, mode="nonlinear", seed=seed)
    df = pd.DataFrame(X, columns=["age", "sbp", "chol", "bmi"])
    df["smoker"] = np.where(X[:, 3] > 0, "yes", "no")
    df["fu_years"] = t
    df["cvd_event"] = e.astype(int)
    df.to_csv(path, index=False)


def _write_class_csv(path, n=300, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (np.abs(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n) > 0.8).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["label"] = y
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import explaneat.db.sqlite_compat  # noqa: F401
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.batch import jobs
    from explaneat.db import db

    tmp = tmp_path_factory.mktemp("portal")
    url = f"sqlite:///{tmp / 'portal.db'}"
    db.init_db(url)
    db.create_all()
    mp = pytest.MonkeyPatch()
    mp.setenv("EXPLANEAT_HOME", str(tmp / "home"))
    mgr = jobs.BatchManager(root=tmp / "home" / "jobs", poll_seconds=0.2)
    mp.setattr(jobs, "manager", mgr)
    yield TestClient(create_app()), tmp, url
    mp.undo()
    db.close()


def _wait(client, batch_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = client.get(f"/api/portal/batches/{batch_id}").json()
        if b["done"]:
            return b
        time.sleep(0.5)
    raise AssertionError(f"batch did not finish: {b}")


def test_survival_workflow(env):
    client, tmp, url = env
    csv = tmp / "cohort.csv"
    _write_survival_csv(csv)

    prev = client.post("/api/portal/csv/preview", json={"path": str(csv)}).json()
    sugg = {c["name"]: c["suggest"] for c in prev["columns"]}
    assert sugg["fu_years"] == "time" and sugg["cvd_event"] == "event" and sugg["smoker"] == "categorical"
    assert "rows" not in prev  # no raw rows

    r = client.post("/api/portal/datasets/csv", json={
        "path": str(csv), "name": "synthetic cohort", "task": "survival", "time_col": "fu_years",
        "event_col": "cvd_event", "categorical": ["smoker"], "time_unit": "years", "restricted": True})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["restricted"] and ds["val"] > 0 and "smoker:yes" in ds["feature_names"]

    tpl = client.get("/api/portal/templates").json()["templates"]
    assert any(t["name"] == "survival-cox-v3" and "num_inputs" in t["ini_text"] for t in tpl)

    b = client.post("/api/portal/batches", json={
        "task": "survival", "dataset_split_id": ds["split_id"], "seeds": [0, 1], "name_prefix": "synth",
        "generations": 3, "pop_size": 12, "epochs": 2, "max_concurrent": 2}).json()
    b = _wait(client, b["batch_id"])
    assert [r["status"] for r in b["runs"]] == ["completed", "completed"], b
    assert all(r["selected"]["genome_id"] for r in b["runs"])

    res = client.get(f"/api/portal/batches/{b['batch_id']}/results").json()
    assert res["task"] == "survival" and len(res["rows"]) == 2, res
    row = res["rows"][0]
    for k in ("harrell_c", "uno_c", "td_auc_mean", "ibs", "calibration_eo"):
        assert k in row["metrics"]
    assert row["primary_ci"][0] is not None and row["comparator_primary"] is not None
    assert next(iter(res["comparators"].values()))["info"]["model"].startswith("ridge Cox")
    assert "harrell_c" in res["summary"]
    csv_text = client.get(f"/api/portal/batches/{b['batch_id']}/results.csv").text
    assert csv_text.splitlines()[0].startswith("experiment_name")
    rep = client.get(f"/api/portal/batches/{b['batch_id']}/report.md")
    assert rep.status_code == 200 and rep.headers["content-type"].startswith("text/markdown")
    for heading in ("## Test-fold metrics per seed", "## Per-horizon metrics", "## Calibration at the horizon",
                    "ridge Cox", "cohort.csv"):
        assert heading in rep.text, heading
    assert str(tmp) not in rep.text  # the file's base name only, never its path

    gid = row["selected"]["genome_id"]
    hr = client.get(f"/api/portal/genomes/{gid}/hazard-ratios")
    assert hr.status_code == 200, hr.text
    assert "log_risk_spread" in hr.json()
    perf = client.post(f"/api/genomes/{gid}/evidence/performance",
                       json={"dataset_split_id": ds["split_id"], "split": "train"}).json()
    assert perf["survival"]["harrell_c"] is not None
    lin = client.post(f"/api/genomes/{gid}/tools/linearise", json={})
    assert lin.status_code == 400  # no annotations yet

    # The MCP (Claude-assisted) path refuses the restricted dataset
    from mcp_server.tools import models as mcp_models
    from mcp_server import server as mcp_srv
    from explaneat.db import db

    from explaneat.governance import GovernanceError

    old = mcp_srv._db
    mcp_srv._db = db
    try:
        with pytest.raises(GovernanceError, match="restricted"):
            mcp_models.get_phenotype(gid)
    finally:
        mcp_srv._db = old


def test_cancel_and_resume(env):
    client, tmp, url = env
    csv = tmp / "cohort2.csv"
    _write_survival_csv(csv, seed=3)
    ds = client.post("/api/portal/datasets/csv", json={
        "path": str(csv), "name": "c2", "task": "survival", "time_col": "fu_years",
        "event_col": "cvd_event", "categorical": ["smoker"]}).json()
    b = client.post("/api/portal/batches", json={
        "task": "survival", "dataset_split_id": ds["split_id"], "seeds": [5], "generations": 40,
        "pop_size": 12, "epochs": 1}).json()
    bid = b["batch_id"]
    t0 = time.time()
    while time.time() - t0 < 120:
        r = client.get(f"/api/portal/batches/{bid}").json()["runs"][0]
        if (r.get("generations_done") or 0) >= 1:
            break
        time.sleep(0.3)
    client.post(f"/api/portal/batches/{bid}/cancel")
    b = _wait(client, bid)
    run = b["runs"][0]
    assert run["status"] == "interrupted" and run["experiment_id"], run
    done = run["generations_done"]
    # resume with a short horizon: the spec's generation total is kept, so lower it
    from explaneat.batch import jobs

    spec_p = jobs.manager.root / bid / "s5" / "spec.json"
    spec = json.loads(spec_p.read_text())
    spec["generations"] = done + 2
    spec_p.write_text(json.dumps(spec))
    client.post(f"/api/portal/batches/{bid}/resume")
    b = _wait(client, bid)
    run = b["runs"][0]
    assert run["status"] == "completed", run
    assert run["generations_done"] == done + 2


def test_classification_batch(env):
    client, tmp, url = env
    csv = tmp / "cls.csv"
    _write_class_csv(csv)
    ds = client.post("/api/portal/datasets/csv", json={
        "path": str(csv), "name": "cls", "task": "classification", "target_col": "label"}).json()
    assert ds["task"] == "classification"
    bad = client.post("/api/portal/batches", json={"task": "survival", "dataset_split_id": ds["split_id"],
                                                   "seeds": [0]})
    assert bad.status_code == 400
    b = client.post("/api/portal/batches", json={
        "task": "classification", "dataset_split_id": ds["split_id"], "seeds": [0], "generations": 2,
        "pop_size": 10, "epochs": 2}).json()
    b = _wait(client, b["batch_id"])
    assert b["runs"][0]["status"] == "completed", b
    res = client.get(f"/api/portal/batches/{b['batch_id']}/results").json()
    assert res["primary"] == "auc" and res["rows"][0]["comparator_primary"] is not None


def test_restricted_storage_refuses_repo_paths(env, tmp_path):
    from explaneat.batch.common import REPO_ROOT, StorageError, check_restricted_storage

    with pytest.raises(StorageError):
        check_restricted_storage(csv_path=REPO_ROOT / "x.csv")
    with pytest.raises(StorageError):
        check_restricted_storage(database_url=f"sqlite:///{REPO_ROOT / 'x.db'}")
    check_restricted_storage(csv_path=tmp_path / "x.csv", database_url=f"sqlite:///{tmp_path / 'x.db'}")
