"""Tests for transplant study C2 (experiments/churn/TRANSPLANT2_PLAN.md): the
signal-novelty measure, the incremental-value refit and donor selection."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import genome_surgery as gs


# ------------------------------------------------------------------ novelty

class TestLinearNovelty:
    def test_linear_function_of_basis_has_zero_novelty(self):
        rng = np.random.default_rng(0)
        B = rng.normal(size=(2000, 12))
        B[:, 5] = np.maximum(B[:, 0] + 0.3 * B[:, 1], 0)  # a "hidden activation"
        s = 2.0 * B[:, 5] - 0.7 * B[:, 2] + 3.0
        r = gs.linear_novelty(s, B)
        assert r["nov"] == pytest.approx(0.0, abs=1e-10)
        assert r["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_independent_signal_has_novelty_near_one(self):
        rng = np.random.default_rng(1)
        B = rng.normal(size=(3000, 10))
        s = rng.normal(size=3000)
        r = gs.linear_novelty(s, B)
        assert 0.98 < r["nov"] <= 1.0

    def test_rank_deficient_basis(self):
        rng = np.random.default_rng(2)
        n = 1500
        onehot = np.eye(4)[rng.integers(0, 4, size=n)]  # sums to 1: collinear with intercept
        X = rng.normal(size=(n, 3))
        B = np.column_stack([X, onehot, X[:, 0] * 2.0, np.zeros(n), np.ones(n)])
        s = X[:, 0] - onehot[:, 2] * 1.5
        r = gs.linear_novelty(s, B)
        assert r["nov"] == pytest.approx(0.0, abs=1e-10)
        assert r["rank"] < r["n_cols"]
        noise = rng.normal(size=n)
        r2 = gs.linear_novelty(noise, B)
        assert 0.98 < r2["nov"] <= 1.0

    def test_partial_novelty(self):
        rng = np.random.default_rng(3)
        B = rng.normal(size=(20000, 4))
        u = rng.normal(size=20000)
        s = B[:, 0] + u  # half the variance is new
        assert gs.linear_novelty(s, B)["nov"] == pytest.approx(0.5, abs=0.02)

    def test_constant_signal_is_nan(self):
        B = np.random.default_rng(4).normal(size=(100, 3))
        r = gs.linear_novelty(np.full(100, 2.5), B)
        assert np.isnan(r["nov"]) and r["reason"]


class TestNoveltyOnGenomes:
    """Basis from a NeuralNeat forward pass; donor contribution from the donor genome."""

    def _sub(self, cfg, donor):
        from test_genome_surgery import _donor_ops
        return gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor, cfg)

    def test_basis_columns_are_inputs_then_hidden(self, small_config, recipient_genome):
        X = np.random.default_rng(5).normal(size=(50, 3))
        B, labels = gs.signal_basis(recipient_genome, small_config, X)
        assert labels == ["in:-1", "in:-2", "in:-3", "h:1"]
        np.testing.assert_allclose(B[:, :3], X)
        np.testing.assert_allclose(B[:, 3], np.maximum(0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2, 0),
                                   atol=1e-12)

    def test_donor_signal_and_self_novelty(self, small_config, donor_genome, recipient_genome):
        sub = self._sub(small_config, donor_genome)
        X = np.random.default_rng(6).normal(size=(800, 3))
        s, F, t0 = gs.readout_contribution(donor_genome, small_config, X,
                                           sub.readout_connections)
        h5 = np.maximum(1.1 * X[:, 0] + 0.9 * X[:, 1] + 0.4, 0)
        h6 = np.maximum(-1.4 * h5 + 0.6 * X[:, 2] - 0.2, 0)
        np.testing.assert_allclose(s, 2.0 * h6, atol=1e-12)
        # a recipient that already computes the donor's exit: novelty ~0
        Bd, _ = gs.signal_basis(donor_genome, small_config, X)
        assert gs.linear_novelty(s, Bd)["nov"] == pytest.approx(0.0, abs=1e-10)
        # the toy recipient does not compute it: clearly novel
        Br, _ = gs.signal_basis(recipient_genome, small_config, X)
        assert gs.linear_novelty(s, Br)["nov"] > 0.05

    def test_identity_exit_group_column(self, small_config, donor_genome):
        X = np.random.default_rng(7).normal(size=(40, 3))
        s, F, t0 = gs.readout_contribution(donor_genome, small_config, X, [(6, 0)],
                                           io_keys=[(-3, 0), (-1, 0)])
        assert F.shape == (40, 2) and list(t0[1:]) == [1.0]
        np.testing.assert_allclose(F[:, 1], -0.4 * X[:, 2] + 0.3 * X[:, 0], atol=1e-12)


class TestIncrementalRefit:
    def test_matches_graft_plus_refit_readout(self, small_config, recipient_genome,
                                              donor_genome):
        from test_genome_surgery import _donor_ops
        from test_transplant_amendments import _toy
        sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
        X, y = _toy(seed=8)
        grafted, rec = gs.graft(recipient_genome, small_config, donor_genome, sub)
        refit, info = gs.refit_readout(grafted, small_config, X, y,
                                       [tuple(k) for k in rec["readout_connections"]])
        terms = gs.output_preactivation_terms(recipient_genome, small_config, X)
        _, F, t0 = gs.readout_contribution(donor_genome, small_config, X,
                                           sub.readout_connections)
        z, inf2 = gs.refit_added_signal(terms["z"], terms["b"], F, t0, y)
        z_graft = gs.output_preactivation_terms(refit, small_config, X)["z"]
        np.testing.assert_allclose(z, z_graft, atol=1e-8)
        assert inf2["readout_params_after"] == pytest.approx(info["readout_weights_after"],
                                                             abs=1e-8)
        inc = roc_auc_score(y, z) - roc_auc_score(y, terms["z"])
        assert np.isfinite(inc)


# ------------------------------------------------------------------ selection

def _p(ds, name, nov, inc, rs=0):
    return {"recipient_seed": rs, "donor_seed": ds, "donor_annotation": name,
            "nov": nov, "inc": inc}


class TestSelectNovelDonor:
    def test_largest_inc_among_novel(self):
        pairs = [_p(1, "a", 0.9, 0.01), _p(2, "b", 0.6, 0.03), _p(3, "c", 0.4, 0.10)]
        best, mx = gs.select_novel_donor(pairs, 0)
        assert (best["donor_seed"], best["donor_annotation"]) == (2, "b")
        assert mx == 0.9

    def test_threshold_is_inclusive(self):
        best, _ = gs.select_novel_donor([_p(1, "a", 0.5, 0.01)], 0)
        assert best is not None

    def test_tie_break_lowest_seed_then_name(self):
        pairs = [_p(4, "a", 0.8, 0.02), _p(2, "z", 0.7, 0.02), _p(2, "m", 0.9, 0.02),
                 _p(3, "a", 0.9, 0.02)]
        best, _ = gs.select_novel_donor(pairs, 0)
        assert (best["donor_seed"], best["donor_annotation"]) == (2, "m")

    def test_own_seed_excluded(self):
        pairs = [_p(0, "own", 0.99, 0.5), _p(1, "x", 0.6, 0.001)]
        best, mx = gs.select_novel_donor(pairs, 0)
        assert best["donor_seed"] == 1 and mx == 0.6

    def test_none_when_nothing_novel(self):
        pairs = [_p(1, "a", 0.3, 0.05), _p(2, "b", 0.45, 0.02), _p(3, "c", float("nan"), 0.1),
                 _p(4, "d", 0.2, 0.0, rs=1)]
        best, mx = gs.select_novel_donor(pairs, 0)
        assert best is None
        assert mx == 0.45  # max novelty recorded with "none"

    def test_none_when_no_pairs(self):
        best, mx = gs.select_novel_donor([], 0)
        assert best is None and np.isnan(mx)


# ------------------------------------------------------------------ runner helpers

class TestRunnerHelpers:
    def test_names_and_ft_conditions(self):
        import run_transplant2 as r2
        assert r2.name_for("graft-refit-ft", 4, 5, 500) == "churn-graft2-graft-refit-ft-e500-r4-d5"
        assert r2.name_for("ftonly", 0, None, 500) == "churn-graft2-ftonly-e500-r0-dnone"
        assert r2.name_for("sham-refit-ft", 9, 5, 500, r2.PREFIX["judged"]) == \
            "churn-graft2j-sham-refit-ft-e500-r9-d5"
        assert [c for c in ["base", "ftonly"] + r2.GRAFT_CONDS if r2.is_ft(c)] == \
            ["ftonly", "graft-refit-ft", "sham-refit-ft"]

    def test_judged_pairs_skip_none(self, tmp_path):
        import run_transplant2 as r2
        f = tmp_path / "p.csv"
        f.write_text("recipient_seed,donor_seed,donor_annotation,x\n0,,none,a\n"
                     "2,6,heavy_day_eve_usage,b\n4,5,csc_threshold_low_day,c\n")
        got = [(a, b, c) for a, b, c, _ in r2.read_judged_pairs(f)]
        assert got == [(2, 6, "heavy_day_eve_usage"), (4, 5, "csc_threshold_low_day")]
