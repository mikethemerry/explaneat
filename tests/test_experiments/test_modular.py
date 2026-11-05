"""Tests for the modular-crossover study (experiments/modular, PLAN.md)."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from neat.genes import DefaultConnectionGene, DefaultNodeGene

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "experiments" / "modular"
for p in (str(MOD), str(REPO / "experiments" / "churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

import chunks as ck  # noqa: E402
import operators as ops  # noqa: E402
import tasks  # noqa: E402
from islands import Study, make_config  # noqa: E402
from module_genome import ModuleGenome  # noqa: E402
from registry import ModuleLibrary  # noqa: E402


@pytest.fixture(scope="module")
def mux():
    return tasks.load_mux11()


@pytest.fixture
def config(mux):
    return make_config(mux, 20)


def mgenome(key, nodes, conns, tags=None):
    g = ModuleGenome(key)
    for nid, bias in nodes.items():
        n = DefaultNodeGene(nid)
        n.bias, n.response, n.activation, n.aggregation = float(bias), 1.0, "relu", "sum"
        g.nodes[nid] = n
    for k, w in conns.items():
        c = DefaultConnectionGene(k)
        c.weight, c.enabled = float(w), True
        g.connections[k] = c
    g.module_tags = dict(tags or {})
    g.fitness = 0.5
    return g


def donor(config):
    # g0-like unit: address 0 (a0 = a1 = a2 = -1) and d0 = +1
    # hidden 101 = relu(-a0 - a1 - a2 + d0 - 3) ; second chunk 102 = relu(d1)
    conns = {(-1, 101): -1.0, (-2, 101): -1.0, (-3, 101): -1.0, (-4, 101): 1.0,
             (101, 0): 3.0, (-5, 102): 1.0, (102, 103): 0.7, (103, 0): -0.4,
             (-6, 0): 0.2}
    g = mgenome(7, {0: -1.0, 101: -3.0, 102: 0.0, 103: 0.1}, conns)
    config.genome_config.node_indexer = None
    config.genome_config.get_new_node_key({200: None})  # ids start above any test id
    return g


def recipient():
    conns = {(-7, 0): 0.5, (-8, 104): 1.0, (104, 0): 0.9, (-1, 0): -0.3}
    return mgenome(8, {0: 0.2, 104: 0.0}, conns)


# ------------------------------------------------------------------ chunk finder

def test_chunk_closure_and_boundary(config, mux):
    g = donor(config)
    X, y = mux.final.X_train, mux.final.y_train
    found = ck.find_chunks(g, config, X, y, probe_X=X, keep_all=True)
    sets = sorted(tuple(c.hidden_nodes) for c in found)
    assert sets == [(101,), (102, 103)]
    c101 = [c for c in found if c.hidden_nodes == [101]][0]
    assert c101.entry_inputs == [-4, -3, -2, -1]
    assert c101.readout_keys == [(101, 0)]
    # an egress edge from 102 to 101's cone merges the closures
    g2 = copy.deepcopy(g)
    g2.connections[(102, 101)] = copy.deepcopy(g.connections[(-1, 101)])
    g2.connections[(102, 101)].key = (102, 101)
    sets2 = sorted(tuple(c.hidden_nodes) for c in ck.find_chunks(g2, config, X, y,
                                                                  keep_all=True))
    assert sets2 == [(101, 102, 103)]


def test_restore_gain_positive_for_true_feature(config, mux):
    g = donor(config)
    st = mux.stages[0]
    found = ck.find_chunks(g, config, st.X_train, st.y_train, keep_all=True)
    c101 = [c for c in found if c.hidden_nodes == [101]][0]
    assert c101.nonlinear and c101.gain > 0


# ------------------------------------------------------------------ graft + refit

def test_graft_refit_preserves_donor_contribution(config, mux):
    g = donor(config)
    X, y = mux.final.X_train, mux.final.y_train
    ch = [c for c in ck.find_chunks(g, config, X, y, probe_X=X, keep_all=True)
          if c.hidden_nodes == [101]][0]
    r = recipient()
    new, ro, idmap = ops.graft_chunk(r, config, ch, module_id=1, weight_mode="explained")
    # before refit: grafted contribution == donor's contribution exactly
    _, contrib = ck.readout_terms(new, config, X)
    s_new = sum(contrib[k] for k in ro)
    np.testing.assert_allclose(s_new, ch.s_probe, rtol=0, atol=1e-12)
    # and the recipient's own logit is unchanged apart from the added term
    z_r, _ = ck.readout_terms(r, config, X)
    z_n, _ = ck.readout_terms(new, config, X)
    np.testing.assert_allclose(z_n - s_new, z_r, rtol=0, atol=1e-12)
    # refit changes only readout weight(s) + output bias; the grafted exit
    # activations (the donor feature) are identical, train logloss not worse
    fit, info = ops.refit(new, config, X, y, ro)
    assert info["train_logloss_after"] <= info["train_logloss_before"] + 1e-12
    changed = {k for k in new.connections
               if new.connections[k].weight != fit.connections[k].weight}
    assert changed <= set(ro)
    assert all(new.nodes[n].bias == fit.nodes[n].bias for n in new.nodes if n != 0)
    w_old = new.connections[ro[0]].weight
    w_new = fit.connections[ro[0]].weight
    _, c_fit = ck.readout_terms(fit, config, X)
    np.testing.assert_allclose(c_fit[ro[0]] / w_new, s_new / w_old, rtol=1e-10, atol=1e-12)
    assert fit.module_tags == {idmap[101]: 1}


def test_sham_and_topology_modes(config, mux):
    g = donor(config)
    X, y = mux.final.X_train, mux.final.y_train
    ch = [c for c in ck.find_chunks(g, config, X, y, probe_X=X, keep_all=True)
          if c.hidden_nodes == [101]][0]
    r = recipient()
    rng = np.random.default_rng(0)
    sham, ro, idm = ops.graft_chunk(r, config, ch, 1, "sham", rng)
    assert sham.nodes[idm[101]].bias != g.nodes[101].bias
    assert sham.connections[(-1, idm[101])].weight != g.connections[(-1, 101)].weight
    topo, ro_t, _ = ops.graft_chunk(r, config, ch, 1, "topology", np.random.default_rng(0))
    assert all(topo.connections[k].weight == 0.0 for k in ro_t)
    np.testing.assert_allclose(ck.readout_terms(topo, config, X)[0],
                               ck.readout_terms(r, config, X)[0], rtol=0, atol=1e-12)


# ------------------------------------------------------------------ alignment

def test_module_alignment(config, mux):
    g = donor(config)
    X, y = mux.final.X_train, mux.final.y_train
    found = ck.find_chunks(g, config, X, y, probe_X=X, keep_all=True)
    c101 = [c for c in found if c.hidden_nodes == [101]][0]
    c102 = [c for c in found if c.hidden_nodes == [102, 103]][0]
    lib = ModuleLibrary()
    v1 = lib.add(c101, island=0, generation=24, genome_key=7)
    v2 = lib.add(c102, island=0, generation=24, genome_key=7)
    assert (v1.module_id, v2.module_id) == (1, 2)
    # the same feature grafted into another genome (another island) and
    # re-found there aligns to module 1 (an alias)
    new, ro, _ = ops.graft_chunk(recipient(), config, c101, 1, "explained")
    fit, _ = ops.refit(new, config, X, y, ro)
    again = [c for c in ck.find_chunks(fit, config, X, y, probe_X=X, keep_all=True)
             if set(c.readout_keys) == set(ro)][0]
    v3 = lib.add(again, island=2, generation=49, genome_key=8)
    assert v3.module_id == 1 and lib.history[-1]["align_r"] >= 0.9
    assert lib.best_variant(1, exclude_island=0).island == 2
    # an anti-correlated signal is not an alias
    neg = copy.deepcopy(c101)
    neg.s_probe = -c101.s_probe
    assert lib.add(neg, island=1, generation=49, genome_key=9).module_id == 3


# ------------------------------------------------------------------ operators

@pytest.mark.parametrize("mode", ["explained", "sham", "topology"])
def test_operators_valid_feedforward(config, mux, mode):
    g = donor(config)
    X, y = mux.final.X_train, mux.final.y_train
    found = ck.find_chunks(g, config, X, y, probe_X=X, keep_all=True)
    c101 = [c for c in found if c.hidden_nodes == [101]][0]
    rng = np.random.default_rng(1)
    # insertion
    ins, ro, idm = ops.graft_chunk(recipient(), config, c101, 1, mode, rng)
    if mode != "topology":
        ins, _ = ops.refit(ins, config, X, y, ro)
    assert ops.valid_feedforward(ins, config)
    # replacement: knock out module 1 in `ins`, graft an alias
    ko, removed = ops.knockout_module(ins, 1)
    assert removed == [idm[101]] and not ko.module_ids()
    rep, ro2, _ = ops.graft_chunk(ko, config, c101, 1, mode, rng)
    assert ops.valid_feedforward(rep, config)
    assert rep.module_ids() == {1} and not set(rep.module_nodes(1)) & set(removed)
    # crossover: child of `rep` (module 1, variant A) and `ins2` carrying a
    # different variant of module 1 plus module 2
    g.module_tags = {101: 1, 102: 2, 103: 2}
    g.fitness, rep.fitness = 0.6, 0.7
    child = ModuleGenome(99)
    child.configure_crossover(rep, g, config.genome_config)
    assert set(child.module_tags.values()) == {1}
    diff = ops.differing_modules(child, g)
    assert diff == [1, 2]
    for mid in diff:
        child, ro3 = ops.add_module_keep_ids(child, config, g, mid, mode, rng)
    assert ops.valid_feedforward(child, config)
    assert set(child.module_nodes(1)) == {101} and set(child.module_nodes(2)) == {102, 103}
    if mode == "explained":
        assert child.connections[(101, 0)].weight == g.connections[(101, 0)].weight


# ------------------------------------------------------------------ speciation

def test_distance_counts_module_as_one_unit(config):
    gc = config.genome_config
    a = mgenome(1, {0: 0.0, 104: 0.0}, {(-1, 0): 1.0, (-2, 104): 1.0, (104, 0): 1.0})
    b = copy.deepcopy(a)
    import neat
    plain_a, plain_b = neat.DefaultGenome(1), neat.DefaultGenome(2)
    plain_a.nodes, plain_a.connections = a.nodes, a.connections
    plain_b.nodes, plain_b.connections = b.nodes, b.connections
    assert a.distance(b, gc) == plain_a.distance(plain_b, gc) == 0.0
    # graft a 3-node module into b: untagged it would be 3 disjoint nodes
    for n in (300, 301, 302):
        ng = DefaultNodeGene(n)
        ng.bias, ng.response, ng.activation, ng.aggregation = 0.0, 1.0, "relu", "sum"
        b.nodes[n] = ng
    for k in [(-3, 300), (300, 301), (301, 302), (302, 0), (-4, 301)]:
        c = DefaultConnectionGene(k)
        c.weight, c.enabled = 1.0, True
        b.connections[k] = c
    untagged = a.distance(b, gc)
    b.module_tags = {300: 5, 301: 5, 302: 5}
    tagged = a.distance(b, gc)
    assert tagged < untagged
    # one disjoint unit over (2 untagged + 1 module) nodes, and one over
    # (3 untagged + 1 module) connections
    assert tagged == pytest.approx(1.0 / 3 + 1.0 / 4)


# ------------------------------------------------------------------ reproducibility

def _run(out):
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1")
    cmd = [sys.executable, str(MOD / "run.py"), "mux11", "modules", "0", "--generations", "21",
           "--pop-size", "50", "--stage-switch", "15", "--K", "10", "--quiet", "--out", str(out)]
    subprocess.run(cmd, check=True, env=env, cwd=REPO, capture_output=True)
    s = json.loads((out / "mux11-modules-s0.json").read_text())
    s.pop("wall_s")
    d = (out / "mux11-modules-s0.detail.json").read_text()
    return s, d


@pytest.mark.slow
def test_seeded_runs_bit_identical(tmp_path):
    s1, d1 = _run(tmp_path / "a")
    s2, d2 = _run(tmp_path / "b")
    assert s1 == s2
    assert d1 == d2
    det = json.loads(d1)
    assert any(r.get("applied") for r in det["operators"])  # operators exercised
