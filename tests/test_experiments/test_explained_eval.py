"""Tests for experiments/churn/explained_eval.py (linearisation / restore gains)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import explained_eval as ee
import genome_surgery as gs


def _ops():
    return [
        {"seq": 0, "type": "split_node", "params": {"node_id": "-1"}},
        {"seq": 1, "type": "split_node", "params": {"node_id": "-3"}},
        {"seq": 2, "type": "add_identity_node",
         "params": {"target_node": "0", "connections": [["-1_a", "0"], ["-3_a", "0"]],
                    "new_node_id": "identity_l"}},
        {"seq": 3, "type": "annotate", "params": {
            "name": "hinge", "entry_nodes": ["-1_b", "-2", "-3_b"], "exit_nodes": ["6"],
            "subgraph_nodes": ["-1_b", "-2", "-3_b", "5", "6"],
            "subgraph_connections": [["-1_b", "5"], ["-2", "5"], ["5", "6"], ["-3_b", "6"]]}},
        {"seq": 4, "type": "annotate", "params": {
            "name": "lin", "entry_nodes": ["-1_a", "-3_a"], "exit_nodes": ["identity_l"],
            "subgraph_nodes": ["-1_a", "-3_a", "identity_l"],
            "subgraph_connections": [["-1_a", "identity_l"], ["-3_a", "identity_l"]]}},
        {"seq": 5, "type": "annotate", "params": {
            "name": "root", "entry_nodes": ["6", "identity_l"], "exit_nodes": ["0"],
            "subgraph_nodes": ["0"], "subgraph_connections": [["6", "0"], ["identity_l", "0"]],
            "child_annotation_ids": ["hinge", "lin"]}},
    ]


@pytest.fixture
def setup(small_config, donor_genome):
    engine, ops = gs.replay_model_state(_ops(), donor_genome, small_config)
    anns = ee.annotations_from_ops(ops)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    p = gs.predict_proba(donor_genome, small_config, X)
    y = (rng.uniform(size=600) < p).astype(float)
    return engine.current_state, anns, X, y


def test_evaluator_matches_neuralneat(small_config, donor_genome, setup):
    state, _, X, _ = setup
    p_nn = gs.predict_proba(donor_genome, small_config, X)
    p_ev = ee.StateEvaluator(state).proba(X)
    np.testing.assert_allclose(p_ev, p_nn, atol=1e-12)


def test_root_and_top_level_children(setup):
    _, anns, _, _ = setup
    root, children = ee.root_and_children(anns)
    assert root["name"] == "root"
    assert [c["name"] for c in children] == ["hinge", "lin"]
    assert [a["name"] for a in ee.leaf_annotations(anns)] == ["hinge", "lin"]


def test_linear_fit_and_nonlinearity_flag(setup):
    state, anns, X, _ = setup
    ev = ee.StateEvaluator(state)
    _, children = ee.root_and_children(anns)
    lin = ee.Linearisation(ev, {c["name"]: c for c in children}, X)
    assert lin.fits["lin"].nonlinear is False
    assert lin.fits["hinge"].nonlinear is True
    # linearising an exactly linear unit changes nothing
    np.testing.assert_allclose(lin.proba(X, ["lin"]), ev.proba(X), atol=1e-10)
    # the hinge's linear replacement is the least-squares fit of its exit on its entries
    vals = ev.forward(X)
    A = np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1], X[:, 2]])
    coef, *_ = np.linalg.lstsq(A, vals["6"], rcond=None)
    logit = 2.0 * (A @ coef) + (0.3 * X[:, 0] - 0.4 * X[:, 2]) + 0.1
    np.testing.assert_allclose(lin.proba(X, ["hinge"]), 1 / (1 + np.exp(-logit)), atol=1e-10)


def test_fit_is_applied_unchanged_to_new_data(setup):
    state, anns, X, _ = setup
    ev = ee.StateEvaluator(state)
    _, children = ee.root_and_children(anns)
    lin = ee.Linearisation(ev, {c["name"]: c for c in children}, X[:300])
    lin2 = ee.Linearisation(ev, {c["name"]: c for c in children}, X[:300])
    np.testing.assert_array_equal(lin.proba(X[300:], ["hinge"]), lin2.proba(X[300:], ["hinge"]))
    full = ee.Linearisation(ev, {c["name"]: c for c in children}, X)
    assert not np.allclose(lin.fits["hinge"].coef, full.fits["hinge"].coef)


def test_performance_sources(setup):
    state, anns, X, y = setup
    ev = ee.StateEvaluator(state)
    _, children = ee.root_and_children(anns)
    lin = ee.Linearisation(ev, {c["name"]: c for c in children}, X)
    res = ee.performance_sources(lin, X, y)
    assert res["full_auc"] == pytest.approx(roc_auc_score(y, ev.proba(X)))
    assert res["nonlinear_units"] == ["hinge"]
    h = res["units"]["hinge"]
    # with one nonlinear unit, restoring it alone is the full model
    assert h["restore_alone_auc"] == pytest.approx(res["full_auc"])
    assert h["restore_gain"] == pytest.approx(res["full_auc"] - res["all_linear_auc"])
    assert res["units"]["lin"]["restore_gain"] == 0.0


def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.uniform(size=500), 2)  # ties
    assert ee.fast_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_paired_bootstrap_delta():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    a = y + rng.normal(0, 1.0, 300)
    idx = np.stack([rng.integers(0, 300, 300) for _ in range(200)])
    lo, hi, n = ee.bootstrap_delta_ci(y, a, a, idx)
    assert lo == hi == 0.0 and n == 200
    lo, hi, _ = ee.bootstrap_auc_ci(y, a, idx)
    assert lo < roc_auc_score(y, a) < hi
