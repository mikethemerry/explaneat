"""Survival metrics, the criterion cache, and the survival compute_performance path."""

import copy
import json

import numpy as np
import pytest
import torch

pytest.importorskip("sksurv")

from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402
from sksurv.metrics import (  # noqa: E402
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv  # noqa: E402

from explaneat.survival import BreslowBaseline, CoxCriterion, CoxRiskSets, cox_ph_loss  # noqa: E402
from explaneat.survival import metrics as M  # noqa: E402
from explaneat.survival.data import survival_split_indices  # noqa: E402
from explaneat.survival.evidence import survival_performance  # noqa: E402
from explaneat.survival.simulate import simulate  # noqa: E402


@pytest.fixture(scope="module")
def linear_fold():
    X, t, e, g = simulate(n=1500, mode="linear", seed=4)
    sp = survival_split_indices(e, test_size=0.3, seed=0)
    tr, te = sp["train"], sp["test"]
    cox = CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron").fit(X[tr], Surv.from_arrays(e[tr], t[tr]))
    return X, t, e, tr, te, cox


def test_metrics_match_sksurv(linear_fold):
    X, t, e, tr, te, cox = linear_fold
    r = cox.predict(X[te])
    assert M.harrell_c(r, t[te], e[te]) == pytest.approx(
        concordance_index_censored(e[te], t[te], r)[0], abs=1e-12)
    assert M.concordance(r, t[te], e[te]) == pytest.approx(M.harrell_c(r, t[te], e[te]), abs=1e-12)
    tau = M.default_tau(t[tr], e[tr], t[te], e[te])
    assert M.uno_c(r, t[tr], e[tr], t[te], e[te]) == pytest.approx(
        concordance_index_ipcw(Surv.from_arrays(e[tr], t[tr]), Surv.from_arrays(e[te], t[te]),
                               r, tau=tau)[0], abs=1e-12)
    hz = np.quantile(t[e], [0.25, 0.5, 0.75])
    td = M.time_dependent_auc(r, t[tr], e[tr], t[te], e[te], hz)
    m = M.ipcw_mask(t[tr], t[te])  # test rows past the last training time are excluded
    ref, ref_mean = cumulative_dynamic_auc(Surv.from_arrays(e[tr], t[tr]),
                                           Surv.from_arrays(e[te][m], t[te][m]), r[m], hz)
    np.testing.assert_allclose(td["auc"], ref)
    assert td["mean_auc"] == pytest.approx(ref_mean)


def test_well_specified_model_is_calibrated(linear_fold):
    X, t, e, tr, te, cox = linear_fold
    g_tr, g_te = cox.predict(X[tr]), cox.predict(X[te])
    bl = BreslowBaseline().fit(g_tr, t[tr], e[tr])
    h = float(np.median(t[e]))
    cal = M.calibration_at_horizon(bl.risk_at(g_te, h), t[te], e[te], h)
    assert cal["n_groups"] == 10
    assert 0.85 < cal["expected_over_observed"] < 1.15
    preds = [r["mean_predicted"] for r in cal["groups"]]
    assert preds == sorted(preds)
    obs = [r["observed_km"] for r in cal["groups"]]
    assert obs[-1] > obs[0] + 0.3  # deciles separate risk


def test_ipcw_metrics_drop_test_rows_beyond_train_followup():
    """sksurv raises for test times past the training maximum; we exclude them."""
    rng = np.random.default_rng(3)
    t_tr = rng.exponential(size=300) + 0.01
    e_tr = rng.uniform(size=300) < 0.6
    t_te = np.r_[rng.uniform(0.01, 0.9 * t_tr.max(), size=99), t_tr.max() + 1.0]
    e_te = np.r_[rng.uniform(size=99) < 0.6, False]
    r = rng.normal(size=100)
    grid = np.quantile(t_te[:99], [0.2, 0.4, 0.6])
    S = np.exp(-np.outer(np.exp(r), grid))
    assert M.ipcw_mask(t_tr, t_te).sum() == 99
    assert np.isfinite(M.integrated_brier(lambda g: np.exp(-np.outer(np.exp(r), g)),
                                          t_tr, e_tr, t_te, e_te, grid=grid))
    assert len(M.brier_at(S, t_tr, e_tr, t_te, e_te, grid)["brier"]) == 3
    assert np.isfinite(M.time_dependent_auc(r, t_tr, e_tr, t_te, e_te, grid)["mean_auc"])


def test_kaplan_meier_matches_sksurv():
    from sksurv.nonparametric import kaplan_meier_estimator

    rng = np.random.default_rng(0)
    t = rng.integers(1, 30, 200).astype(float)
    e = rng.uniform(size=200) < 0.6
    times, surv = kaplan_meier_estimator(e, t)
    for at in (5.0, 12.5, 29.0):
        k = np.searchsorted(times, at, side="right") - 1
        assert M.kaplan_meier(t, e, at) == pytest.approx(surv[k], abs=1e-12)


def test_survival_performance_bundle(linear_fold):
    X, t, e, tr, te, cox = linear_fold
    y = np.column_stack([t, e.astype(float)])
    out = survival_performance(cox.predict, X[tr], y[tr], X[te], y[te],
                               meta={"time_unit": "years"}, calibration_horizon=float(np.median(t)))
    json.dumps(out)  # JSON-safe
    for k in ("harrell_c", "uno_c", "td_auc", "ibs", "brier", "calibration",
              "cox_loss", "null_cox_loss", "ph_caveat"):
        assert k in out
    assert out["cox_loss"] < out["null_cox_loss"]
    assert out["harrell_c"] > 0.7 and out["ibs"] < 0.25
    assert out["task_type"] == "survival" and out["time_unit"] == "years"


def test_criterion_cache_is_content_keyed():
    rng = np.random.default_rng(1)
    a = torch.tensor(np.column_stack([rng.exponential(size=50), rng.uniform(size=50) < 0.5]),
                     dtype=torch.float64)
    b = a.clone()
    b[[0, 1], 0] = b[[1, 0], 0]  # same sums, same first/last only if lucky: different content
    crit = CoxCriterion("efron")
    g = torch.tensor(rng.normal(size=50))
    la = crit(g, a)
    rs_a = crit._risk_sets
    assert crit(g, a.clone()) == la and crit._risk_sets is rs_a  # cache hit
    lb = crit(g, b)
    assert float(lb) == pytest.approx(float(cox_ph_loss(g, CoxRiskSets(b[:, 0], b[:, 1]))))
    with pytest.raises(ValueError):
        crit(g, a[:, :1])


# --- MCP compute_performance on a survival dataset ---------------------------


def test_mcp_compute_performance_survival(neat_config, tmp_path, monkeypatch):
    from explaneat.core.neuralneat import set_output_activation
    from explaneat.db import db
    from explaneat.db.models import Genome
    from explaneat.db.population import DatabaseBackpropPopulation
    from explaneat.survival.data import save_survival_dataset_to_db, save_survival_split_to_db
    import mcp_server.tools.evidence as ev

    url = f"sqlite:///{tmp_path / 'mcp.db'}"
    db.init_db(url)
    db.create_all()
    try:
        X, t, e, _ = simulate(n=600, d=10, mode="linear", seed=2)
        y = np.column_stack([t, e.astype(float)])
        ds = save_survival_dataset_to_db("sim", X, y, time_unit="years",
                                         metadata={"horizons": [2.0, 5.0], "calibration_horizon": 5.0})
        sp = survival_split_indices(e, test_size=0.3, seed=0)
        split = save_survival_split_to_db(ds.id, sp, seed=0, test_size=0.3)

        cfg = copy.deepcopy(neat_config)
        set_output_activation(cfg, "identity")
        pop = DatabaseBackpropPopulation(cfg, X[sp["train"]], y[sp["train"]], experiment_name="m",
                                         database_url=url, criterion=CoxCriterion())
        pop._save_genomes(pop._save_population_state(0))
        with db.session_scope() as s:
            gid = str(s.query(Genome).first().id)

        monkeypatch.setattr(ev, "get_db", lambda: db)
        out = json.loads(ev.compute_performance(gid, str(split.id), split="test"))
        assert out["task_type"] == "survival"
        assert out["n_test"] == len(sp["test"])
        assert out["horizons"] == [2.0, 5.0]
        assert out["calibration"]["horizon"] == 5.0
        assert "accuracy" not in out and "auc_roc" not in out
        assert 0.0 <= out["harrell_c"] <= 1.0
    finally:
        db.close()
