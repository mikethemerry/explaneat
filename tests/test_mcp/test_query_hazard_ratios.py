"""``query_hazard_ratios``: provenance-stamped hazard-ratio readings of a Cox (identity-output)
model state, and the verifier's base data for a survival target."""
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

MODULES = ["experiments", "models", "operations", "evidence", "coverage", "queries", "analysis_tools"]


@pytest.fixture
def survival_world(test_db, db_session, test_genome):
    """test_genome (10 inputs straight to the output) on a survival dataset with an identity output."""
    from explaneat.db import Dataset, DatasetSplit, Experiment, Population
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 10))
    t = np.exp(0.5 * X[:, 0] + rng.normal(size=60)) + 0.1
    e = (rng.random(60) < 0.7).astype(float)
    ds = Dataset(name="synthetic-surv", num_samples=60, num_features=10, num_classes=None,
                 feature_names=[f"f{i}" for i in range(10)], target_name="time,event",
                 additional_metadata={"task_type": "survival", "target_columns": ["time", "event"]})
    ds.set_data(X, np.column_stack([t, e]))
    db_session.add(ds)
    db_session.flush()
    tr = list(range(40))
    sp = DatasetSplit(dataset_id=ds.id, split_type="train_test", train_indices=tr, test_indices=list(range(40, 60)),
                      scaler_type="StandardScaler",
                      scaler_params={"mean": X[tr].mean(axis=0).tolist(), "scale": X[tr].std(axis=0).tolist()})
    db_session.add(sp)
    db_session.flush()
    pop = db_session.get(Population, test_genome.population_id)
    exp = db_session.get(Experiment, pop.experiment_id)
    exp.split_id, exp.dataset_id = sp.id, ds.id
    cfg = dict(exp.config_json or {})
    cfg["training"] = {**cfg.get("training", {}), "output_activation": "identity"}
    exp.config_json = cfg
    db_session.commit()
    with ExitStack() as st:
        for m in MODULES:
            st.enter_context(patch(f"mcp_server.tools.{m}.get_db", return_value=test_db))
        yield {"db": test_db, "gid": str(test_genome.id), "split": str(sp.id), "X": X, "e": e}


def test_survival_target_becomes_the_event_indicator(survival_world):
    from explaneat.verify.model import base_from_session
    with survival_world["db"].session_scope() as s:
        base = base_from_session(s, survival_world["gid"])
    assert base.y_train.shape == (40,) and base.y_train.tolist() == survival_world["e"][:40].tolist()
    assert len(base.meta["survival"]["time_train"]) == 40 and len(base.meta["survival"]["time_test"]) == 20


def test_query_hazard_ratios_reads_direct_edges_with_provenance(survival_world):
    from mcp_server.tools.queries import get_query, query_hazard_ratios
    r = json.loads(query_hazard_ratios(survival_world["gid"]))
    assert "error" not in r, r
    assert r["output"]["activation"] == "identity" and r["n_terms"] == 10
    first = r["readings"][0]
    assert {"node", "feature", "log_hr_per_sd", "hr_per_sd", "hr_per_unit", "spread"} <= set(first)
    assert first["hr_per_sd"] == pytest.approx(np.exp(first["log_hr_per_sd"]))
    assert r["split"]["split"] == "train" and r["split"]["n"] == 40
    assert r["provenance"]["id"].startswith("q_")
    stored = json.loads(get_query(survival_world["gid"], r["provenance"]["id"]))
    assert stored["tool"] == "query_hazard_ratios"


def test_query_hazard_ratios_refuses_a_sigmoid_output(test_db, db_session, test_genome):
    from mcp_server.tools.queries import query_hazard_ratios
    with ExitStack() as st:
        for m in MODULES:
            st.enter_context(patch(f"mcp_server.tools.{m}.get_db", return_value=test_db))
        r = json.loads(query_hazard_ratios(str(test_genome.id)))
    assert "error" in r


def test_survival_genome_uses_the_horizon_risk_scale(survival_world):
    from mcp_server.tools.queries import _vm
    vm = _vm(survival_world["gid"])
    assert vm.m.risk_scale.kind == "survival"
    g = vm.m.logit(vm.m.X)
    r = vm.m.risk(g)
    assert np.all((r >= 0) & (r <= 1))
    assert np.all(np.diff(r[np.argsort(g)]) >= 0)   # monotone in the log-risk
