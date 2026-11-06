"""Shared F/G helpers (experiments/churn/graft_fg.py, GRAFT_F_G_PLAN.md)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

import genome_surgery as gs
import graft_fg as fg
from test_genome_surgery import _donor_ops


def _pairs(incs, graftable=None):
    graftable = graftable or [True] * len(incs)
    return [{"donor_seed": i, "donor_annotation": f"a{i}", "inc": v, "graftable": g}
            for i, (v, g) in enumerate(zip(incs, graftable))]


class TestPickArms:
    def test_maxinc_and_random_distinct_and_reproducible(self):
        p = _pairs([0.01, 0.05, 0.02, 0.03])
        a = fg.pick_arms(p, (20261106, 3, 10))
        b = fg.pick_arms(p, (20261106, 3, 10))
        assert a["maxinc"]["donor_seed"] == 1
        assert a["random"] is not a["maxinc"]
        assert a["random"]["donor_seed"] == b["random"]["donor_seed"]

    def test_ungraftable_excluded(self):
        p = _pairs([0.09, 0.05, 0.02], graftable=[False, True, True])
        a = fg.pick_arms(p, (1, 2, 3))
        assert a["maxinc"]["donor_seed"] == 1
        assert a["random"]["donor_seed"] == 2

    def test_single_or_no_donor(self):
        assert fg.pick_arms(_pairs([0.1]), (1,))["random"] is None
        assert fg.pick_arms(_pairs([0.1], [False]), (1,)) == {"maxinc": None, "random": None}

    def test_ties_lowest_seed(self):
        a = fg.pick_arms(_pairs([0.05, 0.05]), (1,))
        assert a["maxinc"]["donor_seed"] == 0


class TestStats:
    def test_sign_test(self):
        s = fg.sign_test([0.1, 0.2, -0.1, 0.0, float("nan")])
        assert (s["n"], s["n_positive"], s["n_negative"], s["n_zero"]) == (4, 2, 1, 1)
        assert s["median"] == pytest.approx(0.05)

    def test_median_ci_contains_median(self):
        rng = np.random.default_rng(0)
        v = rng.normal(0.01, 0.01, size=60)
        lo, hi = fg.median_ci(v, n_boot=500)
        assert lo < np.median(v) < hi

    def test_spearman_ci(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=80)
        y = x + rng.normal(scale=0.5, size=80)
        r = fg.spearman_ci(x, y, n_boot=300)
        assert r["rho"] == pytest.approx(spearmanr(x, y)[0])
        assert r["ci_lo"] < r["rho"] < r["ci_hi"] and r["ci_lo"] > 0.5
        rc = fg.spearman_ci(x, y, clusters=np.arange(80) % 8, n_boot=300)
        assert rc["rho"] == r["rho"] and rc["ci_lo"] < rc["ci_hi"]

    def test_partial_spearman_removes_confounder(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=400)
        x = z + 0.3 * rng.normal(size=400)
        y = z + 0.3 * rng.normal(size=400)
        assert spearmanr(x, y)[0] > 0.7
        assert abs(fg.partial_spearman(x, y, z)) < 0.2

    def test_about_zero(self):
        assert fg.about_zero(-0.004, 0.003)
        assert not fg.about_zero(-0.006, 0.003)
        assert not fg.about_zero(float("nan"), 0.0)


class TestGraftHelpers:
    def test_contribution_and_survival(self, small_config, recipient_genome, donor_genome):
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        s, F, t0 = gs.readout_contribution(donor_genome, small_config, X,
                                           sub.readout_connections,
                                           sub.input_output_connections)
        donor = {"genome": donor_genome, "sub": sub, "s": s}
        G = fg.make_grafts(recipient_genome, small_config, donor, 3, X)
        gg, rec = G["graft"]
        np.testing.assert_allclose(fg.contribution(gg, small_config, rec, X), s, atol=1e-12)
        assert fg.survival_r(gg, small_config, rec, X, s) == pytest.approx(1.0)
        y = (X[:, 0] > 0).astype(float)
        rd = {"regime": "e500", "X_refit": X, "y_refit": y}
        g1, info = fg.finetune(gg, small_config, rd, 0, rec=rec, P=5)
        conns, nodes = gs.graft_protect_keys(rec)
        assert info["protect_epochs"] == 5
        assert any(g1.connections[k].weight != gg.connections[k].weight for k in conns)

    def test_val_split_rule_stratified(self):
        y = np.array([0] * 80 + [1] * 20)
        f, v = fg.val_split_rule(y)
        assert len(v) == 20 and len(f) == 80 and sorted(f + v) == list(range(100))
        assert y[v].sum() == 4
