"""annotation_to_genome_subgraph against stored churn100 explanations (read-only).

Skipped when the Postgres dev database is not reachable. Every session runs in
a READ ONLY transaction and is rolled back.
"""

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

import genome_surgery as gs

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "experiments" / "10seed" / "explanations"


@pytest.fixture(scope="module")
def ro_session():
    from explaneat.db.base import Database

    url = os.environ.get("DATABASE_URL", "postgresql://localhost/explaneat_dev")
    if not url.startswith("postgresql"):
        pytest.skip("needs the Postgres dev database")
    database = Database(url)
    try:
        database.init_db()
        s = database.Session()
        s.execute(text("SET TRANSACTION READ ONLY"))
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Postgres not reachable: {e}")
    yield s
    s.rollback()
    s.close()
    database.close()


def _load(session, seed):
    from explaneat.db.models import Explanation
    from mcp_server.helpers import load_genome_and_config

    gid = json.loads((RECORDS / f"churn100-s{seed}.json").read_text())["genome_id"]
    genome, config, _ = load_genome_and_config(session, gid)
    ops = session.query(Explanation).filter(Explanation.genome_id == uuid.UUID(gid)).one().operations
    return genome, config, list(ops)


@pytest.mark.db
def test_s7_usage_pathway(ro_session):
    genome, config, ops = _load(ro_session, 7)
    sub = gs.annotation_to_genome_subgraph(ops, "usagePathway", genome, config)
    assert set(sub.hidden_nodes) == {1406, 7558, 879}
    assert set(sub.entry_inputs) == {-61, -37, -62, -59, -65, -57}
    assert sub.exit_nodes == [879]
    # 879 -> identity_usage -> 0 in the model state is genome edge 879 -> 0
    assert sub.readout_connections == [(879, 0)]
    # 879 also feeds the composition's own nodes: egress, not copied
    assert set(sub.egress_connections) == {(879, 4344), (879, 14351)}
    assert len(sub.internal_connections) == 8
    for k in sub.all_connections:
        assert genome.connections[k].enabled


@pytest.mark.db
def test_s7_identity_exit_collects_direct_input_edges(ro_session):
    genome, config, ops = _load(ro_session, 7)
    sub = gs.annotation_to_genome_subgraph(ops, "stateOffsets", genome, config)
    # direct state -> identity_state edges are genome input -> output edges
    assert (-3, 0) in sub.input_output_connections
    assert (-37, 13164) in sub.internal_connections  # via consolidated -37_cd
    assert (-37, 0) in sub.input_output_connections
    for k in sub.all_connections:
        assert genome.connections[k].enabled


@pytest.mark.db
def test_s7_composition_rejected(ro_session):
    genome, config, ops = _load(ro_session, 7)
    with pytest.raises(ValueError, match="leaf"):
        gs.annotation_to_genome_subgraph(ops, "usageResponse", genome, config)


@pytest.mark.db
@pytest.mark.parametrize("seed", range(10))
def test_every_churn100_leaf_resolves(ro_session, seed):
    """Every leaf of every churn100 explanation resolves; the leaves' hidden
    nodes are disjoint, and every resolved key is an enabled genome edge."""
    from explaneat.analysis.composition import annotation_dicts_from_operations

    genome, config, ops = _load(ro_session, seed)
    seen = {}
    for ann in annotation_dicts_from_operations(ops):
        if ann["child_annotation_ids"]:
            continue
        sub = gs.annotation_to_genome_subgraph(ops, ann["name"], genome, config)
        for n in sub.hidden_nodes:
            seen.setdefault(n, []).append(ann["name"])
        for k in sub.all_connections:
            assert genome.connections[k].enabled, (ann["name"], k)
    # split hidden nodes may legitimately appear in two leaves; report them
    shared = {n: v for n, v in seen.items() if len(v) > 1}
    assert all(len(v) <= 2 for v in shared.values()), shared


@pytest.mark.db
def test_graft_s8_feature_into_s9_builds(ro_session):
    genome8, config, ops8 = _load(ro_session, 8)
    genome9, _, _ = _load(ro_session, 9)
    sub = gs.annotation_to_genome_subgraph(ops8, "csc_lowusage_hinge", genome8, config)
    new, rec = gs.graft(genome9, config, genome8, sub)
    assert rec["acyclic"]
    X = np.random.default_rng(0).normal(size=(64, 71))
    p_old = gs.predict_proba(genome9, config, X)
    p_new = gs.predict_proba(new, config, X)
    assert np.all(np.isfinite(p_new)) and not np.allclose(p_old, p_new)


# ------------------------------------------------- explained_eval on real records

@pytest.mark.db
@pytest.mark.parametrize("seed,unit,expected_all_linear,expected_gain", [
    (3, "day_eve_usage", 0.8400, 0.0493),          # record: restore_gain
    (5, "csc_low_day_gate", 0.8352, 0.0497),       # record: restore_gain_alone_onto_all_linear
    (7, "usageResponse", 0.8341, 0.0852),          # composition, linearised over raw entries
    (8, "csc_lowusage_hinge", 0.8390, 0.0555),
])
def test_train_restore_gains_reproduce_records(ro_session, seed, unit, expected_all_linear,
                                               expected_gain):
    import explained_eval as ee
    from mcp_server.helpers import load_split_data

    genome, config, ops = _load(ro_session, seed)
    X, y, *_ = load_split_data(ro_session, "45962ea6-3c78-49f9-a01e-3665f6809b1e", "train",
                               sample_fraction=1.0, max_samples=10**9)
    engine, replayed = gs.replay_model_state(ops, genome, config)
    state = engine.current_state
    anns = ee.annotations_from_ops(replayed)
    ev = ee.StateEvaluator(state)
    np.testing.assert_allclose(ev.proba(X), gs.predict_proba(genome, config, X), atol=1e-12)
    _, children = ee.root_and_children(anns)
    lin = ee.Linearisation(ev, {c["name"]: ee.as_unit(c, anns, state) for c in children}, X)
    res = ee.performance_sources(lin, X, y)
    assert res["all_linear_auc"] == pytest.approx(expected_all_linear, abs=6e-4)
    assert res["units"][unit]["restore_gain"] == pytest.approx(expected_gain, abs=2e-4)
