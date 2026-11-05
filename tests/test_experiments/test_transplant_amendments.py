"""Tests for the transplant / repair amendments (EXTENSIONS_PLAN.md, Amendments):
base features + Jaccard rule for the complementary donor, the readout refit,
and per-E derived-genome naming."""

import copy

import numpy as np
import pytest
from sklearn.metrics import log_loss

import genome_surgery as gs

FEATURES = ["state:AK", "total day minutes", "total day charge", "total eve minutes",
            "total eve charge", "total day calls", "number customer service calls",
            "total intl charge", "international plan"]


def col(name):
    return -(FEATURES.index(name) + 1)  # input id <-> data column


INPUTS = [-(i + 1) for i in range(len(FEATURES))]


# ------------------------------------------------------------------ base features

class TestBaseFeatures:
    def test_charge_and_minutes_of_a_period_collapse(self):
        assert gs.base_feature_name("total day minutes") == "day usage"
        assert gs.base_feature_name("total day charge") == "day usage"
        assert gs.base_feature_name("total intl charge") == "intl usage"
        assert gs.base_feature_name("total night minutes") == "night usage"
        # calls and other features stay distinct
        assert gs.base_feature_name("total day calls") == "total day calls"
        assert gs.base_feature_name("state:AK") == "state:AK"

    def test_split_copies_collapse_to_input(self):
        d = str(col("total day minutes"))
        ops = [{"type": "split_node", "params": {"node_id": d},
                "result": {"created_nodes": [f"{d}_a", f"{d}_b"]}}]
        res = gs.ModelStateIdResolver(ops)
        s = gs.entry_base_features([f"{d}_a", f"{d}_b", str(col("total day charge"))],
                                   res, FEATURES, INPUTS)
        assert s == frozenset({"day usage"})

    def test_non_input_entry_is_own_token(self):
        res = gs.ModelStateIdResolver([])
        s = gs.entry_base_features(["104", str(col("international plan"))], res, FEATURES,
                                   INPUTS)
        assert s == frozenset({"node:104", "international plan"})

    def test_jaccard(self):
        assert gs.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
        assert gs.jaccard({"a"}, {"a"}) == 1.0
        assert gs.jaccard(set(), {"a"}) == 0.0


def _cand(seed, name, gain, feats, nonlinear=True, qualifies=True, all_inputs=True):
    return {"seed": seed, "name": name, "train_restore_gain": gain, "nonlinear": nonlinear,
            "qualifies": qualifies, "entries_all_inputs": all_inputs,
            "entry_base_features": frozenset(feats)}


class TestComplementaryDonor:
    def test_charge_minutes_overlap_blocks_donor(self):
        # recipient 9 already has a nonlinear leaf on day minutes + service calls
        rec = _cand(9, "r_nl", 0.0, {"day usage", "number customer service calls"})
        # donor on day *charge* + service calls is the same base features: J = 1
        same = _cand(2, "csc_day", 0.20, {"day usage", "number customer service calls"})
        other = _cand(3, "intl", 0.05, {"intl usage", "international plan"})
        best, scored = gs.complementary_donor([rec, same, other], 9)
        assert best[0]["name"] == "intl"
        assert dict((c["name"], j) for c, j in scored)["csc_day"] == 1.0

    def test_must_differ_from_every_nonlinear_leaf(self):
        rec1 = _cand(9, "a", 0.0, {"intl usage", "international plan"})
        rec2 = _cand(9, "b", 0.0, {"day usage"})
        rec_lin = _cand(9, "lin", 0.0, {"eve usage"}, nonlinear=False)  # ignored
        d = _cand(1, "eve", 0.1, {"eve usage", "day usage"})  # J with b = 0.5: not < 0.5
        e = _cand(1, "eve2", 0.03, {"eve usage", "total day calls"})  # J = 0 with both
        best, _ = gs.complementary_donor([rec1, rec2, rec_lin, d, e], 9)
        assert best[0]["name"] == "eve2"

    def test_gain_threshold_own_seed_and_qualification(self):
        rec = _cand(9, "r", 0.0, {"day usage"})
        cands = [rec,
                 _cand(9, "own", 0.5, {"intl usage"}),             # recipient's own seed
                 _cand(1, "small", 0.01, {"intl usage"}),          # gain not > 0.01
                 _cand(2, "hidden_entry", 0.3, {"node:5"}, all_inputs=False),
                 _cand(3, "noqual", 0.3, {"eve usage"}, qualifies=False)]
        best, _ = gs.complementary_donor(cands, 9)
        assert best is None

    def test_tie_break_lowest_seed(self):
        rec = _cand(9, "r", 0.0, {"day usage"})
        a = _cand(4, "x", 0.05, {"intl usage"})
        b = _cand(2, "y", 0.05, {"eve usage"})
        best, _ = gs.complementary_donor([rec, a, b], 9)
        assert best[0]["name"] == "y"


# ------------------------------------------------------------------ readout refit

def _toy(n=600, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    h = np.maximum(-1.4 * np.maximum(1.1 * X[:, 0] + 0.9 * X[:, 1] + 0.4, 0)
                   + 0.6 * X[:, 2] - 0.2, 0)
    logit = 1.3 * np.maximum(0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2, 0) + 0.7 * X[:, 2] \
        - 0.6 * h + 0.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return X, y


class TestRefitReadout:
    def test_only_readout_and_output_bias_change_and_loss_improves(
            self, small_config, recipient_genome, donor_genome):
        from test_genome_surgery import _donor_ops
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
        grafted, rec = gs.graft(recipient_genome, small_config, donor_genome, sub)
        X, y = _toy()
        readout = [tuple(k) for k in rec["readout_connections"]]
        assert readout and all(k in grafted.connections for k in readout)
        new, info = gs.refit_readout(grafted, small_config, X, y, readout)

        changed_c = {k for k in grafted.connections
                     if new.connections[k].weight != grafted.connections[k].weight}
        changed_n = {k for k in grafted.nodes if new.nodes[k].bias != grafted.nodes[k].bias}
        assert changed_c <= set(readout) and changed_c
        assert changed_n <= {0}
        assert set(new.connections) == set(grafted.connections)
        assert all(new.connections[k].enabled == grafted.connections[k].enabled
                   for k in grafted.connections)

        before = log_loss(y, gs.predict_proba(grafted, small_config, X))
        after = log_loss(y, gs.predict_proba(new, small_config, X))
        assert after < before - 1e-3
        assert info["train_logloss_before"] == pytest.approx(before, abs=1e-9)
        assert info["train_logloss_after"] == pytest.approx(after, abs=1e-9)
        assert info["max_abs_pred_diff_fit_vs_genome"] < 1e-9
        assert info["converged"]
        # true readout of the donor feature in the toy problem is -0.6
        assert info["readout_weights_after"][0] < 0

    def test_refit_is_optimal_for_its_parameters(self, small_config, recipient_genome,
                                                 donor_genome):
        from test_genome_surgery import _donor_ops
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
        grafted, rec = gs.graft(recipient_genome, small_config, donor_genome, sub)
        X, y = _toy(seed=3)
        readout = [tuple(k) for k in rec["readout_connections"]]
        new, _ = gs.refit_readout(grafted, small_config, X, y, readout, l2=0.0)
        base = log_loss(y, gs.predict_proba(new, small_config, X))
        for k, dv in [(readout[0], 1e-3), (readout[0], -1e-3)]:
            g = copy.deepcopy(new)
            g.connections[k].weight += dv
            assert log_loss(y, gs.predict_proba(g, small_config, X)) >= base - 1e-12
        for dv in (1e-3, -1e-3):
            g = copy.deepcopy(new)
            g.nodes[0].bias += dv
            assert log_loss(y, gs.predict_proba(g, small_config, X)) >= base - 1e-12

    def test_scaled_group_scales_input_output_weights_together(self, small_config,
                                                               recipient_genome):
        g = copy.deepcopy(recipient_genome)
        X, y = _toy(seed=4)
        new, info = gs.refit_readout(g, small_config, X, y, [], scaled_groups=[[(-3, 0)]])
        s = info["group_scales_after"][0]
        assert new.connections[(-3, 0)].weight == pytest.approx(0.7 * s)
        assert new.connections[(1, 0)].weight == g.connections[(1, 0)].weight
        assert info["train_logloss_after"] <= info["train_logloss_before"]


# ------------------------------------------------------------------ naming

class TestPerEpochNaming:
    def test_repair_names(self):
        import run_repair as rr
        assert rr.name_for("prune-ft", 0, 500) == "churn-repair-prune-ft-e500-s0"
        assert rr.name_for("prune-ft", 0, 50) == "churn-repair-prune-ft-e50-s0"
        assert rr.name_for("prune-noft", 3) == "churn-repair-prune-noft-s3"
        names = [rr.name_for(c, s, e) for s in range(10) for c, e in rr.planned(
            [500, 50])]
        assert len(names) == len(set(names))
        assert sum(1 for c, e in rr.planned([500, 50]) if c == "ftonly") == 2

    def test_transplant_names(self):
        import run_transplant as rt
        assert rt.name_for("graft-ft", 9, "d2", 500) == "churn-graft-graft-ft-e500-r9-d2"
        assert rt.name_for("graft-noft", 9, "c3") == "churn-graft-graft-noft-r9-c3"
        plan = rt.planned([500, 50], ["d2", "c3"])
        names = [rt.name_for(c, 9, d, e) for c, d, e in plan if c != "base"]
        assert len(names) == len(set(names))
        # every graft condition has its matching sham, per donor and E
        for c, d, e in plan:
            if c.startswith("graft"):
                assert (rt.SHAM_OF[c], d, e) in plan

    def test_parse_epochs(self):
        import ext_common as ec
        assert ec.parse_epochs("500") == [500]
        assert ec.parse_epochs("500,50") == [500, 50]
        with pytest.raises(ValueError):
            ec.parse_epochs("500,500")
