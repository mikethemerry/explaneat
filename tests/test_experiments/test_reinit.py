"""Tests for study E (experiments/churn/RESCUE2_REINIT_PLAN.md "E"):
symmetry normalisation, Hungarian unit matching, dead-unit detection,
re-initialisation and checkpointed masked training."""

import copy

import numpy as np
import pytest

import genome_surgery as gs
import reinit as ri
from churn_test_helpers import build_genome


@pytest.fixture
def wide_genome():
    # hidden 1, 2, 3 all <- (-1, -2, -3) and -> 0 (one position group of 3);
    # hidden 4 <- (1, 2, 3) -> 0 ... no: keep 4 <- (-1) -> (0) as its own group
    nodes = {0: 0.1, 1: 0.3, 2: -0.2, 3: 0.5, 4: 0.7}
    conns = {}
    W = {1: (0.9, -0.4, 0.2), 2: (-0.3, 1.2, 0.6), 3: (0.5, 0.5, -1.1)}
    for h, ws in W.items():
        for i, w in zip((-1, -2, -3), ws):
            conns[(i, h)] = w
        conns[(h, 0)] = {1: 1.5, 2: -0.8, 3: 0.6}[h]
    conns[(-1, 4)] = 1.3
    conns[(4, 0)] = -0.9
    return build_genome(7, nodes, conns)


@pytest.fixture
def deep_genome():
    # 1 <- (-1, -2); 2 <- (1, -3); 0 <- (2, 1, -1)
    return build_genome(8, {0: -0.2, 1: 0.4, 2: -0.1},
                        {(-1, 1): 1.1, (-2, 1): -0.7, (1, 2): 0.9, (-3, 2): 0.5,
                         (2, 0): 1.4, (1, 0): -0.6, (-1, 0): 0.3})


def _X(n=400, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3))


def _with_params(genome, conns, bias):
    g = copy.deepcopy(genome)
    for k, w in conns.items():
        g.connections[k].weight = float(w)
    for n, b in bias.items():
        g.nodes[n].bias = float(b)
    return g


class TestSymmetryNormalisation:
    def test_normalisation_preserves_function(self, small_config, deep_genome):
        units, conns, bias = ri.network_params(deep_genome, small_config)
        assert units == [1, 2]
        w, b, scales = ri.normalise_units(units, conns, bias)
        X = _X()
        g2 = _with_params(deep_genome, w, b)
        np.testing.assert_allclose(gs.predict_proba(g2, small_config, X),
                                   gs.predict_proba(deep_genome, small_config, X), atol=1e-12)
        for u in units:
            v = [w[k] for k in w if k[1] == u] + [b[u]]
            assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_rescaled_unit_gives_cosine_one(self, small_config, deep_genome):
        units, conns, bias = ri.network_params(deep_genome, small_config)
        # rescale unit 1 by c (incoming and bias x c, outgoing / c), unit 2 by d
        c, d = 3.7, 0.21
        c2, b2 = dict(conns), dict(bias)
        for k in conns:
            if k[1] == 1:
                c2[k] *= c
            if k[0] == 1:
                c2[k] /= c
        b2[1] *= c
        for k in conns:
            if k[1] == 2:
                c2[k] *= d
            if k[0] == 2:
                c2[k] /= d
        b2[2] *= d
        X = _X()
        np.testing.assert_allclose(
            gs.predict_proba(_with_params(deep_genome, c2, b2), small_config, X),
            gs.predict_proba(deep_genome, small_config, X), atol=1e-12)
        m = ri.match_units(units, conns, bias, c2, b2)
        np.testing.assert_allclose(m["cosines"], 1.0, atol=1e-12)
        assert m["median"] == pytest.approx(1.0)
        assert m["share_ge"] == 1.0
        # without normalisation unit 2's vector differs (it carries unit 1's scale)
        raw = ri.match_units(units, conns, bias, c2, b2, normalise=False)
        assert dict((a, cs) for a, _, cs in raw["pairs"])[2] < 0.999

    def test_different_weights_give_lower_cosine(self, small_config, deep_genome):
        units, conns, bias = ri.network_params(deep_genome, small_config)
        other = ri.reinit_all(deep_genome, 1000)
        _, c2, b2 = ri.network_params(other, small_config)
        m = ri.match_units(units, conns, bias, c2, b2)
        assert m["median"] < 0.999


class TestHungarianMatching:
    def test_groups(self, small_config, wide_genome):
        units, conns, bias = ri.network_params(wide_genome, small_config)
        groups = ri.position_groups(units, conns)
        assert sorted(map(sorted, groups)) == [[1, 2, 3], [4]]

    def test_permuted_toy_net(self, small_config, wide_genome):
        units, conns, bias = ri.network_params(wide_genome, small_config)
        # permute hidden units 1 -> 2 -> 3 -> 1 (swap their whole parameter sets),
        # and rescale each: same function, different ids
        perm = {1: 2, 2: 3, 3: 1}
        scale = {1: 2.0, 2: 0.5, 3: 4.0}
        c2, b2 = dict(conns), dict(bias)
        for old, new in perm.items():
            s = scale[old]
            for i in (-1, -2, -3):
                c2[(i, new)] = conns[(i, old)] * s
            c2[(new, 0)] = conns[(old, 0)] / s
            b2[new] = bias[old] * s
        X = _X()
        np.testing.assert_allclose(
            gs.predict_proba(_with_params(wide_genome, c2, b2), small_config, X),
            gs.predict_proba(wide_genome, small_config, X), atol=1e-12)
        m = ri.match_units(units, conns, bias, c2, b2)
        pairs = {a: b for a, b, _ in m["pairs"]}
        assert pairs == {1: 2, 2: 3, 3: 1, 4: 4}
        np.testing.assert_allclose(m["cosines"], 1.0, atol=1e-12)
        assert m["n_nontrivial_groups"] == 1 and m["max_group_size"] == 3

    def test_permutation_carried_downstream(self, small_config):
        # 1, 2 <- (-1, -2) -> 3; 3 <- (1, 2) -> 0. Swap 1 and 2: unit 3's
        # incoming vector must be read through the matched permutation.
        conns = {(-1, 1): 0.9, (-2, 1): -0.2, (-1, 2): -0.4, (-2, 2): 1.1,
                 (1, 3): 0.7, (2, 3): -1.3, (3, 0): 1.0}
        g = build_genome(9, {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}, conns)
        units, c, b = ri.network_params(g, small_config)
        c2 = {(-1, 1): -0.4, (-2, 1): 1.1, (-1, 2): 0.9, (-2, 2): -0.2,
              (1, 3): -1.3, (2, 3): 0.7, (3, 0): 1.0}
        b2 = {0: 0.0, 1: 0.2, 2: 0.1, 3: 0.3}
        m = ri.match_units(units, c, b, c2, b2)
        assert {a: bb for a, bb, _ in m["pairs"]} == {1: 2, 2: 1, 3: 3}
        np.testing.assert_allclose(m["cosines"], 1.0, atol=1e-12)


class TestDeadUnits:
    def test_dead_unit_detected(self, small_config, deep_genome):
        g = copy.deepcopy(deep_genome)
        X = np.abs(_X())  # all inputs >= 0
        # unit 1: negative weights from non-negative inputs and negative bias -> dead
        g.connections[(-1, 1)].weight = -1.0
        g.connections[(-2, 1)].weight = -2.0
        g.nodes[1].bias = -0.1
        acts = ri.hidden_activations(g, small_config, X)
        assert set(acts) == {1, 2}
        assert ri.dead_units(acts, [1, 2]) == [1]

    def test_live_units(self, small_config, deep_genome):
        acts = ri.hidden_activations(deep_genome, small_config, _X())
        assert ri.dead_units(acts, [1, 2]) == []


class TestReinitAndTraining:
    def test_reinit_keeps_topology_and_disabled(self, small_config, deep_genome):
        g = copy.deepcopy(deep_genome)
        g.connections[(-1, 0)].enabled = False
        g.connections[(-1, 0)].weight = 0.123
        r = ri.reinit_all(g, 1000)
        assert set(r.connections) == set(g.connections)
        assert r.connections[(-1, 0)].weight == 0.123 and not r.connections[(-1, 0)].enabled
        rng = np.random.default_rng(1000)
        draws = rng.normal(0.0, 1.0, size=len(g.nodes) + 6)
        assert [r.nodes[n].bias for n in sorted(g.nodes)] == list(draws[:3])
        en = sorted(k for k, c in g.connections.items() if c.enabled)
        assert [r.connections[k].weight for k in en] == list(draws[3:])
        assert ri.reinit_all(g, 1000).nodes[1].bias == r.nodes[1].bias
        assert ri.reinit_all(g, 1001).nodes[1].bias != r.nodes[1].bias

    def test_checkpoints_equal_finetune(self, small_config, deep_genome):
        X = _X(300, 1)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
        out = ri.train_masked(deep_genome, small_config, X, y, [3, 7], seed=5)
        for E in (3, 7):
            ref = gs.finetune(deep_genome, small_config, X, y, E, 5, mask_phantom=True)
            for k in ref.connections:
                assert out[E][0].connections[k].weight == ref.connections[k].weight
            for n in ref.nodes:
                assert out[E][0].nodes[n].bias == ref.nodes[n].bias
