"""Tests for experiments/churn/genome_surgery.py (EXTENSIONS_PLAN.md studies B and C)."""

import copy
import uuid

import numpy as np
import pytest
import torch
from sklearn.metrics import log_loss

from churn_test_helpers import build_genome  # noqa: F401
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.neuralneat import NeuralNeat

import genome_surgery as gs


def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * np.maximum(X[:, 2], 0) - 0.2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return X, y


def _logit_of(genome, config, X):
    p = gs.predict_proba(genome, config, X)
    return np.log(p) - np.log1p(-p)


def _relu(v):
    return np.maximum(v, 0.0)


# ------------------------------------------------------------------ finetune

class TestFinetune:
    def test_returns_copy_and_leaves_input_untouched(self, small_config, recipient_genome):
        X, y = _toy_data()
        before = copy.deepcopy(recipient_genome)
        out = gs.finetune(recipient_genome, small_config, X, y, epochs=5, seed=0)
        assert out is not recipient_genome
        for k, c in before.connections.items():
            assert recipient_genome.connections[k].weight == c.weight
        for k, n in before.nodes.items():
            assert recipient_genome.nodes[k].bias == n.bias

    def test_improves_train_bce(self, small_config, recipient_genome):
        X, y = _toy_data()
        p0 = gs.predict_proba(recipient_genome, small_config, X)
        out = gs.finetune(recipient_genome, small_config, X, y, epochs=50, seed=0)
        p1 = gs.predict_proba(out, small_config, X)
        assert log_loss(y, p1) < log_loss(y, p0) - 1e-3

    def test_same_seed_same_result(self, small_config, recipient_genome):
        X, y = _toy_data()
        a = gs.finetune(recipient_genome, small_config, X, y, epochs=10, seed=3)
        b = gs.finetune(recipient_genome, small_config, X, y, epochs=10, seed=3)
        for k in a.connections:
            assert a.connections[k].weight == b.connections[k].weight
        for k in a.nodes:
            assert a.nodes[k].bias == b.nodes[k].bias

    def test_frozen_weights_and_biases_bit_identical(self, small_config, recipient_genome):
        X, y = _toy_data()
        frozen = [(-1, 1), (1, 0)]
        out = gs.finetune(recipient_genome, small_config, X, y, epochs=30, seed=0,
                          frozen_conn_keys=frozen, frozen_node_keys=[1])
        for k in frozen:
            assert out.connections[k].weight == recipient_genome.connections[k].weight
        assert out.nodes[1].bias == recipient_genome.nodes[1].bias
        # the unfrozen parameters did move
        assert out.connections[(-2, 1)].weight != recipient_genome.connections[(-2, 1)].weight
        assert out.connections[(-3, 0)].weight != recipient_genome.connections[(-3, 0)].weight
        assert out.nodes[0].bias != recipient_genome.nodes[0].bias

    def test_frozen_node_keeps_its_function_during_training(self, small_config, recipient_genome):
        """Freezing a node also masks the phantom (non-connection) entries of its
        weight column, so the frozen unit computes the same function at every
        epoch, not only after the write-back."""
        X, y = _toy_data()
        info = gs.finetune(recipient_genome, small_config, X, y, epochs=30, seed=0,
                           frozen_conn_keys=[(-1, 1), (-2, 1)], frozen_node_keys=[1],
                           return_info=True)[1]
        assert info["max_abs_phantom_into_frozen"] == 0.0

    def test_literal_primitive_trains_phantom_entries(self, small_config, recipient_genome):
        """The primitive trains NeuralNeat's dense zero entries (e.g. -3 -> 1, no
        gene) and drops them at write-back: the stored genome != trained net."""
        X, y = _toy_data()
        _, info = gs.finetune(recipient_genome, small_config, X, y, epochs=30, seed=0,
                              return_info=True)
        assert info["max_abs_pred_diff_trained_vs_written_back"] > 1e-6

    def test_mask_phantom_written_back_genome_is_trained_net(self, small_config,
                                                             recipient_genome):
        X, y = _toy_data()
        g, info = gs.finetune(recipient_genome, small_config, X, y, epochs=30, seed=0,
                              mask_phantom=True, return_info=True)
        assert info["max_abs_pred_diff_trained_vs_written_back"] < 1e-12
        assert info["loss_after_written_back"] < info["loss_before"] - 1e-3
        assert info["mask_phantom"] is True

    def test_frozen_unknown_enabled_key_raises(self, small_config, recipient_genome):
        X, y = _toy_data()
        with pytest.raises(KeyError):
            gs.finetune(recipient_genome, small_config, X, y, epochs=1, seed=0,
                        frozen_conn_keys=[(-3, 1)])

    def test_info_reports_losses(self, small_config, recipient_genome):
        X, y = _toy_data()
        g, info = gs.finetune(recipient_genome, small_config, X, y, epochs=5, seed=0,
                              return_info=True)
        assert info["epochs"] == 5 and len(info["losses"]) == 5
        assert info["loss_after_written_back"] == pytest.approx(
            log_loss(y, gs.predict_proba(g, small_config, X)), abs=1e-9)


# ------------------------------------------------------------------ disable

class TestDisableConnections:
    def test_copy_with_disabled_keys(self, small_config, recipient_genome):
        out = gs.disable_connections(recipient_genome, [(-2, 1)])
        assert out.connections[(-2, 1)].enabled is False
        assert recipient_genome.connections[(-2, 1)].enabled is True
        ph = ExplaNEAT(out, small_config).get_phenotype_network()
        edges = {(c.from_node, c.to_node) for c in ph.connections if c.enabled}
        assert ("-2", "1") not in edges and ("-1", "1") in edges
        NeuralNeat(out, small_config)  # still builds

    def test_orphaned_unit_dropped_and_net_builds(self, small_config, recipient_genome):
        out = gs.disable_connections(recipient_genome, [(-1, 1), (-2, 1)])
        ph = ExplaNEAT(out, small_config).get_phenotype_network()
        assert "1" not in {n.id for n in ph.nodes}
        X, _ = _toy_data(50)
        p = gs.predict_proba(out, small_config, X)
        lin = out.nodes[0].bias + 0.7 * X[:, 2]
        np.testing.assert_allclose(p, 1 / (1 + np.exp(-lin)), atol=1e-12)

    def test_missing_key_raises(self, recipient_genome):
        with pytest.raises(KeyError):
            gs.disable_connections(recipient_genome, [(-3, 1)])


# ------------------------------------------------------------------ id mapping

def _donor_ops(with_identity=False):
    ops = [
        {"seq": 0, "type": "split_node", "params": {"node_id": "-1"}},
        {"seq": 1, "type": "split_node", "params": {"node_id": "-3"}},
        {"seq": 2, "type": "rename_node", "params": {"node_id": "-2", "display_name": "x2"}},
    ]
    exit_node = "6"
    sub_nodes = ["-1_b", "-2", "-3_b", "5", "6"]
    sub_conns = [["-1_b", "5"], ["-2", "5"], ["5", "6"], ["-3_b", "6"]]
    if with_identity:
        ops.append({"seq": 3, "type": "add_identity_node",
                    "params": {"target_node": "0", "connections": [["6", "0"]],
                               "new_node_id": "identity_f"}})
        exit_node = "identity_f"
        sub_nodes = sub_nodes + ["identity_f"]
        sub_conns = sub_conns + [["6", "identity_f"]]
    ops.append({"seq": len(ops), "type": "annotate",
                "params": {"name": "feat", "entry_nodes": ["-1_b", "-2", "-3_b"],
                           "exit_nodes": [exit_node], "subgraph_nodes": sub_nodes,
                           "subgraph_connections": sub_conns}})
    ops.append({"seq": len(ops), "type": "annotate",
                "params": {"name": "root", "entry_nodes": [exit_node, "-1_a", "-3_a"],
                           "exit_nodes": ["0"], "subgraph_nodes": ["0", "-1_a", "-3_a"],
                           "subgraph_connections": [[exit_node, "0"], ["-1_a", "0"],
                                                    ["-3_a", "0"]],
                           "child_annotation_ids": ["feat"]}})
    return ops


class TestAnnotationToGenomeSubgraph:
    @pytest.mark.parametrize("with_identity", [False, True])
    def test_maps_to_original_ids(self, small_config, donor_genome, with_identity):
        sub = gs.annotation_to_genome_subgraph(_donor_ops(with_identity), "feat",
                                               donor_genome, small_config)
        assert sub.annotation_name == "feat"
        assert set(sub.hidden_nodes) == {5, 6}
        assert set(sub.entry_inputs) == {-1, -2, -3}
        assert set(sub.exit_nodes) == {6}
        assert set(sub.internal_connections) == {(-1, 5), (-2, 5), (5, 6), (-3, 6)}
        assert set(sub.readout_connections) == {(6, 0)}
        assert sub.input_output_connections == []
        for k in sub.all_connections:
            assert donor_genome.connections[k].enabled

    def test_non_input_entry_raises(self, small_config, donor_genome):
        ops = _donor_ops()[:3] + [{"seq": 3, "type": "annotate", "params": {
            "name": "upper", "entry_nodes": ["5", "-3_b"], "exit_nodes": ["6"],
            "subgraph_nodes": ["5", "-3_b", "6"],
            "subgraph_connections": [["5", "6"], ["-3_b", "6"]]}}]
        with pytest.raises(gs.NonInputEntryError, match="5"):
            gs.annotation_to_genome_subgraph(ops, "upper", donor_genome, small_config)

    def test_composite_raises(self, small_config, donor_genome):
        with pytest.raises(ValueError, match="leaf"):
            gs.annotation_to_genome_subgraph(_donor_ops(), "root", donor_genome, small_config)

    def test_unknown_name_raises(self, small_config, donor_genome):
        with pytest.raises(KeyError):
            gs.annotation_to_genome_subgraph(_donor_ops(), "nope", donor_genome, small_config)


# ------------------------------------------------------------------ graft

class TestGraft:
    def _sub(self, cfg, donor):
        return gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor, cfg)

    def test_output_is_recipient_logit_plus_donor_readout(self, small_config,
                                                          recipient_genome, donor_genome):
        sub = self._sub(small_config, donor_genome)
        new, rec = gs.graft(recipient_genome, small_config, donor_genome, sub)
        X, _ = _toy_data(200, seed=5)
        h5 = _relu(1.1 * X[:, 0] + 0.9 * X[:, 1] + 0.4)
        h6 = _relu(-1.4 * h5 + 0.6 * X[:, 2] - 0.2)
        expected = _logit_of(recipient_genome, small_config, X) + 2.0 * h6
        np.testing.assert_allclose(_logit_of(new, small_config, X), expected, atol=1e-9)
        # fresh ids above the recipient max, record complete
        assert set(rec["id_map"]) == {5, 6}
        assert min(rec["id_map"].values()) > max(recipient_genome.nodes)
        assert len(rec["added_connections"]) == 5
        assert all(k not in recipient_genome.connections for k in rec["added_connections"])
        assert rec["acyclic"] is True
        # attributes copied
        for d, r in rec["id_map"].items():
            assert new.nodes[r].bias == donor_genome.nodes[d].bias
            assert new.nodes[r].activation == donor_genome.nodes[d].activation
            assert new.nodes[r].response == donor_genome.nodes[d].response
            assert new.nodes[r].aggregation == donor_genome.nodes[d].aggregation
        # recipient untouched
        assert set(recipient_genome.nodes) == {0, 1}

    def test_sham_same_topology_redrawn_weights(self, small_config, recipient_genome,
                                                donor_genome):
        sub = self._sub(small_config, donor_genome)
        real, rec = gs.graft(recipient_genome, small_config, donor_genome, sub)
        sham, srec = gs.graft(recipient_genome, small_config, donor_genome, sub, sham_seed=4)
        sham2, _ = gs.graft(recipient_genome, small_config, donor_genome, sub, sham_seed=4)
        assert set(sham.connections) == set(real.connections)
        assert set(sham.nodes) == set(real.nodes)
        assert srec["sham_seed"] == 4
        diffs = [sham.connections[k].weight != real.connections[k].weight
                 for k in srec["added_connections"]]
        assert all(diffs)
        for k in srec["added_connections"]:
            assert sham.connections[k].weight == sham2.connections[k].weight
        for k in recipient_genome.connections:  # recipient weights untouched
            assert sham.connections[k].weight == recipient_genome.connections[k].weight

    def test_input_output_collision_reported(self, small_config, recipient_genome,
                                             donor_genome):
        sub = self._sub(small_config, donor_genome)
        sub = gs.DonorSubgraph(**{**sub.__dict__,
                                  "input_output_connections": [(-3, 0)]})
        with pytest.raises(gs.GraftCollisionError, match=r"\(-3, 0\)"):
            gs.graft(recipient_genome, small_config, donor_genome, sub)

    def test_grafted_genome_finetunes_and_freezes(self, small_config, recipient_genome,
                                                  donor_genome):
        sub = self._sub(small_config, donor_genome)
        new, rec = gs.graft(recipient_genome, small_config, donor_genome, sub)
        X, y = _toy_data()
        out = gs.finetune(new, small_config, X, y, epochs=10, seed=0,
                          frozen_conn_keys=rec["added_connections"],
                          frozen_node_keys=list(rec["id_map"].values()))
        for k in rec["added_connections"]:
            assert out.connections[k].weight == new.connections[k].weight


# ------------------------------------------------------------------ storage

class TestSaveDerivedGenome:
    def test_payload(self, small_config, recipient_genome):
        prov = {"source_genome_id": "abc", "edits": []}
        p = gs.derived_experiment_payload(
            recipient_genome, small_config, "churn-repair-x-s0", prov,
            split_id=str(uuid.uuid4()), dataset_id=str(uuid.uuid4()),
            source_config_json={"pop_size": 10}, neat_config_text="[NEAT]")
        assert p["experiment"]["status"] == "completed"
        assert p["experiment"]["random_seed"] is None
        assert p["experiment"]["config_json"]["provenance"] == prov
        assert p["experiment"]["config_json"]["pop_size"] == 10
        assert p["population"]["generation"] == 0

    @pytest.mark.db
    def test_roundtrip_sqlite(self, test_db, small_config, recipient_genome):
        from explaneat.db.models import Experiment, Genome, Population
        from explaneat.db.serialization import serialize_population_config

        g = copy.deepcopy(recipient_genome)
        g.fitness = 0.75
        ids = gs.save_derived_genome(
            g, small_config, "churn-repair-test-s0", {"source_genome_id": "abc"},
            split_id=None, dataset_id=None,
            source_config_json=serialize_population_config(small_config),
            neat_config_text="", database=test_db)
        with test_db.session_scope() as s:
            exp = s.get(Experiment, ids["experiment_id"])
            assert exp.name == "churn-repair-test-s0" and exp.status == "completed"
            assert exp.config_json["provenance"]["source_genome_id"] == "abc"
            pop = s.get(Population, ids["population_id"])
            assert pop.generation == 0 and pop.experiment_id == exp.id
            row = s.get(Genome, ids["genome_id"])
            back = row.to_neat_genome(small_config)
            assert row.fitness == 0.75
            for k, c in g.connections.items():
                assert back.connections[k].weight == c.weight

    @pytest.mark.db
    def test_refuses_duplicate_name(self, test_db, small_config, recipient_genome):
        from explaneat.db.serialization import serialize_population_config
        kw = dict(split_id=None, dataset_id=None,
                  source_config_json=serialize_population_config(small_config),
                  neat_config_text="", database=test_db)
        gs.save_derived_genome(recipient_genome, small_config, "dup", {}, **kw)
        with pytest.raises(ValueError, match="exists"):
            gs.save_derived_genome(recipient_genome, small_config, "dup", {}, **kw)
