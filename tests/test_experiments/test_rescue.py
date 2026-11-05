"""Tests for study D (experiments/churn/RESCUE_PLAN.md): feature knockout,
grafting an alias into a knocked-out recipient, and key-collision handling."""

import copy

import numpy as np
import pytest

import genome_surgery as gs
from test_genome_surgery import _donor_ops


def _relu(v):
    return np.maximum(v, 0.0)


def _recipient_ops():
    # recipient_genome: hidden 1 <- (-1, -2); output <- (1, -3)
    return [
        {"seq": 0, "type": "annotate",
         "params": {"name": "rfeat", "entry_nodes": ["-1", "-2"], "exit_nodes": ["1"],
                    "subgraph_nodes": ["-1", "-2", "1"],
                    "subgraph_connections": [["-1", "1"], ["-2", "1"]]}},
    ]


def _z(genome, cfg, X):
    return gs.output_preactivation_terms(genome, cfg, X)["z"]


class TestFeatureKnockout:
    def test_knockout_zeroes_feature_contribution(self, small_config, donor_genome):
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
        X = np.random.default_rng(0).normal(size=(500, 3))
        s = gs.feature_contribution(donor_genome, small_config, X, sub)
        assert np.std(s) > 0
        ko, rec = gs.feature_knockout(donor_genome, small_config, sub)
        assert set(rec["disabled_connections"]) == {(-1, 5), (-2, 5), (5, 6), (-3, 6), (6, 0)}
        for k in rec["disabled_connections"]:
            assert not ko.connections[k].enabled
            assert donor_genome.connections[k].enabled  # original untouched
        # everything else unchanged
        for k, c in donor_genome.connections.items():
            if k not in rec["disabled_connections"]:
                assert ko.connections[k].enabled == c.enabled
                assert ko.connections[k].weight == c.weight
        assert gs.enabled_feature_readout(ko, sub) == ([], [])
        s_ko = gs.feature_contribution(ko, small_config, X, sub)
        assert np.all(s_ko == 0.0)
        z_base, z_ko = _z(donor_genome, small_config, X), _z(ko, small_config, X)
        np.testing.assert_allclose(z_ko, z_base - s, atol=1e-12)
        np.testing.assert_allclose(z_ko, 0.1 - 0.4 * X[:, 2] + 0.3 * X[:, 0], atol=1e-12)
        assert rec["other_outgoing"] == [] and rec["other_to_output"] == []

    def test_other_outgoing_edges_reported_not_disabled(self, small_config, donor_genome):
        g = copy.deepcopy(donor_genome)
        from churn_test_helpers import build_genome
        extra = build_genome(9, {7: 0.0}, {(5, 7): 0.5, (7, 0): 1.0, (-2, 7): 0.3})
        g.nodes[7] = extra.nodes[7]
        for k in ((5, 7), (7, 0), (-2, 7)):
            g.connections[k] = extra.connections[k]
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", g, small_config)
        ko, rec = gs.feature_knockout(g, small_config, sub)
        assert rec["other_outgoing"] == [(5, 7)]
        assert ko.connections[(5, 7)].enabled
        assert rec["external_incoming"] == []

    def test_knockout_requires_enabled_connections(self, small_config, donor_genome):
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
        g = gs.disable_connections(donor_genome, [(5, 6)])
        with pytest.raises(ValueError, match="not enabled"):
            gs.feature_knockout(g, small_config, sub)


class TestGraftIntoKnockout:
    def test_graft_builds_and_computes_expected_logit(self, small_config, recipient_genome,
                                                      donor_genome):
        rsub = gs.annotation_to_genome_subgraph(_recipient_ops(), "rfeat", recipient_genome,
                                                small_config)
        assert set(rsub.all_connections) == {(-1, 1), (-2, 1), (1, 0)}
        dsub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome,
                                                small_config)
        ko, krec = gs.feature_knockout(recipient_genome, small_config, rsub)
        new, rec = gs.graft_into_knockout(ko, small_config, donor_genome, dsub,
                                          knocked_out_keys=krec["disabled_connections"])
        assert set(rec["id_map"].values()).isdisjoint(recipient_genome.nodes)
        X = np.random.default_rng(1).normal(size=(400, 3))
        h5 = _relu(1.1 * X[:, 0] + 0.9 * X[:, 1] + 0.4)
        h6 = _relu(-1.4 * h5 + 0.6 * X[:, 2] - 0.2)
        expected = -0.3 + 0.7 * X[:, 2] + 2.0 * h6
        np.testing.assert_allclose(_z(new, small_config, X), expected, atol=1e-12)
        s_d = gs.feature_contribution(donor_genome, small_config, X, dsub)
        np.testing.assert_allclose(_z(new, small_config, X), _z(ko, small_config, X) + s_d,
                                   atol=1e-12)
        # the knocked-out genes stay disabled; masked fine-tuning keeps them out
        ft = gs.finetune(new, small_config, X, (X[:, 0] > 0).astype(float), 3, 0,
                         mask_phantom=True)
        for k in krec["disabled_connections"]:
            assert not new.connections[k].enabled and not ft.connections[k].enabled
        # sham: same topology, different weights
        sham, srec = gs.graft_into_knockout(ko, small_config, donor_genome, dsub,
                                            knocked_out_keys=krec["disabled_connections"],
                                            sham_seed=3)
        assert set(srec["added_connections"]) == set(rec["added_connections"])


class TestCollisions:
    def _io_donor(self):
        # one hidden node 5 <- -1, readout 5 -> 0, and an input->output edge (-3, 0)
        from churn_test_helpers import build_genome
        g = build_genome(3, {0: 0.0, 5: 0.1}, {(-1, 5): 1.0, (5, 0): 1.5, (-3, 0): 0.8})
        sub = gs.DonorSubgraph(annotation_name="iofeat", genome_key=3, hidden_nodes=[5],
                               entry_inputs=[-1, -3], exit_nodes=[5],
                               internal_connections=[(-1, 5)], readout_connections=[(5, 0)],
                               input_output_connections=[(-3, 0)])
        return g, sub

    def _recipient_with_io_feature(self, recipient_genome):
        # the recipient's feature includes its input->output edge (-3, 0)
        return gs.DonorSubgraph(annotation_name="rio", genome_key=1, hidden_nodes=[1],
                                entry_inputs=[-1, -2, -3], exit_nodes=[1],
                                internal_connections=[(-1, 1), (-2, 1)],
                                readout_connections=[(1, 0)],
                                input_output_connections=[(-3, 0)])

    def test_collision_with_knocked_out_key_fails(self, small_config, recipient_genome):
        dg, dsub = self._io_donor()
        rsub = self._recipient_with_io_feature(recipient_genome)
        ko, krec = gs.feature_knockout(recipient_genome, small_config, rsub)
        assert not ko.connections[(-3, 0)].enabled
        assert gs.graft_collisions(ko, small_config, dsub) == [
            {"key": (-3, 0), "recipient_enabled": False}]
        with pytest.raises(gs.GraftCollisionError, match=r"\(-3, 0\) \(disabled by the knockout\)"):
            gs.graft_into_knockout(ko, small_config, dg, dsub,
                                   knocked_out_keys=krec["disabled_connections"])
        # never re-enabled or overwritten
        assert not ko.connections[(-3, 0)].enabled
        assert ko.connections[(-3, 0)].weight == recipient_genome.connections[(-3, 0)].weight

    def test_collision_with_originally_disabled_key_fails(self, small_config,
                                                          recipient_genome):
        dg, dsub = self._io_donor()
        g = gs.disable_connections(recipient_genome, [(-3, 0)])
        with pytest.raises(gs.GraftCollisionError,
                           match="disabled in the original recipient"):
            gs.graft_into_knockout(g, small_config, dg, dsub, knocked_out_keys=[(1, 0)])

    def test_collision_with_enabled_key_fails(self, small_config, recipient_genome):
        dg, dsub = self._io_donor()
        assert gs.graft_collisions(recipient_genome, small_config, dsub) == [
            {"key": (-3, 0), "recipient_enabled": True}]
        with pytest.raises(gs.GraftCollisionError, match="enabled in the recipient"):
            gs.graft_into_knockout(recipient_genome, small_config, dg, dsub)
        # the plain graft refuses the same key
        with pytest.raises(gs.GraftCollisionError):
            gs.graft(recipient_genome, small_config, dg, dsub)

    def test_no_collision_without_io_edges(self, small_config, recipient_genome,
                                           donor_genome):
        dsub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome,
                                                small_config)
        assert gs.graft_collisions(recipient_genome, small_config, dsub) == []
