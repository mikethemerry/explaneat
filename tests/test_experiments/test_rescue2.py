"""Tests for study D2 (experiments/churn/RESCUE2_REINIT_PLAN.md): the alias-pair
rule and the wrong-family donor rule (genome_surgery.alias_pairs /
wrong_family_donor), and D2's derived-genome naming."""

import numpy as np

import genome_surgery as gs

rng = np.random.default_rng(0)
BASE = rng.normal(size=400)
NOISE = rng.normal(size=400)
OTHER = rng.normal(size=400)


def _leaf(seed, name, s, n_hidden=1, bf=("day usage",), drop=0.01, gain=0.03):
    return {"seed": seed, "name": name, "s": np.asarray(s, dtype=float),
            "n_hidden": n_hidden, "base_features": frozenset(bf),
            "ko_train_auc_drop": drop, "train_restore_gain": gain}


def _pair(rows, rs, rn, ds, dn):
    return next(r for r in rows if (r["recipient_seed"], r["recipient_feature"],
                                     r["donor_seed"], r["donor_feature"]) == (rs, rn, ds, dn))


class TestAliasPairs:
    def test_all_ordered_pairs_across_seeds(self):
        leaves = [_leaf(0, "a", BASE), _leaf(0, "b", OTHER), _leaf(1, "c", BASE),
                  _leaf(2, "d", NOISE)]
        rows = gs.alias_pairs(leaves)
        keys = [(r["recipient_seed"], r["recipient_feature"], r["donor_seed"],
                 r["donor_feature"]) for r in rows]
        # same-seed pairs excluded; both orders present
        assert (0, "a", 0, "b") not in keys and (0, "b", 0, "a") not in keys
        assert (0, "a", 1, "c") in keys and (1, "c", 0, "a") in keys
        assert len(keys) == 4 * 3 - 2 and len(set(keys)) == len(keys)
        assert keys == sorted(keys)

    def test_r_threshold_is_signed_and_inclusive(self):
        s_hi = BASE + 0.1 * NOISE  # r ~ 0.995
        s_neg = -BASE
        leaves = [_leaf(0, "f", BASE, n_hidden=2), _leaf(1, "g", s_hi),
                  _leaf(2, "h", s_neg), _leaf(3, "u", OTHER)]
        rows = gs.alias_pairs(leaves)
        assert _pair(rows, 0, "f", 1, "g")["alias"]
        neg = _pair(rows, 0, "f", 2, "h")
        assert neg["r"] < -0.99 and not neg["r_ok"] and not neg["alias"]
        assert not _pair(rows, 0, "f", 3, "u")["alias"]
        # exactly at the threshold counts
        r = _pair(rows, 0, "f", 1, "g")["r"]
        at = _pair(gs.alias_pairs(leaves, min_r=r), 0, "f", 1, "g")
        assert at["r_ok"]

    def test_representation_must_differ(self):
        same = [_leaf(0, "f", BASE, n_hidden=2, bf=("day usage", "eve usage")),
                _leaf(1, "g", BASE + 0.01 * NOISE, n_hidden=2, bf=("eve usage", "day usage"))]
        rows = gs.alias_pairs(same)
        assert all(not r["different_representation"] and not r["alias"] for r in rows)
        by_hidden = [same[0], _leaf(1, "g", BASE, n_hidden=3, bf=("day usage", "eve usage"))]
        assert all(r["alias"] for r in gs.alias_pairs(by_hidden))
        by_entries = [same[0], _leaf(1, "g", BASE, n_hidden=2, bf=("day usage",))]
        assert all(r["alias"] for r in gs.alias_pairs(by_entries))

    def test_knockout_cost_is_the_recipients(self):
        leaves = [_leaf(0, "f", BASE, n_hidden=2, drop=0.005),
                  _leaf(1, "g", BASE, drop=0.0049)]
        rows = gs.alias_pairs(leaves)
        fg, gf = _pair(rows, 0, "f", 1, "g"), _pair(rows, 1, "g", 0, "f")
        assert fg["alias"] and fg["ko_train_auc_drop"] == 0.005
        assert not gf["alias"] and not gf["knockout_cost_ok"] and gf["r_ok"]

    def test_failed_knockout_or_constant_signal_never_qualifies(self):
        leaves = [_leaf(0, "f", BASE, n_hidden=2, drop=None), _leaf(1, "g", BASE),
                  _leaf(2, "c", np.zeros(400), n_hidden=4)]
        rows = gs.alias_pairs(leaves)
        assert not _pair(rows, 0, "f", 1, "g")["alias"]
        assert _pair(rows, 1, "g", 0, "f")["alias"]
        c = _pair(rows, 1, "g", 2, "c")
        assert np.isnan(c["r"]) and not c["alias"]


class TestWrongFamilyDonor:
    def test_excludes_recipient_and_alias_seeds_and_correlated(self):
        F = _leaf(0, "f", BASE)
        leaves = [F, _leaf(1, "alias", BASE, gain=0.9),
                  _leaf(2, "corr", BASE + 0.5 * NOISE, gain=0.5),     # |r| >= 0.3
                  _leaf(3, "low", OTHER, gain=0.04),
                  _leaf(4, "best", NOISE, gain=0.06),
                  _leaf(0, "own", OTHER, gain=0.99)]
        L, r, n = gs.wrong_family_donor(leaves, F, exclude_seeds={0, 1})
        assert L["name"] == "best" and abs(r) < 0.3 and n == 2

    def test_negative_correlation_counts_by_absolute_value(self):
        F = _leaf(0, "f", BASE)
        leaves = [F, _leaf(2, "neg", -BASE + 0.1 * NOISE, gain=0.5),
                  _leaf(3, "ok", OTHER, gain=0.02)]
        L, _, n = gs.wrong_family_donor(leaves, F, exclude_seeds={0, 1})
        assert L["name"] == "ok" and n == 1

    def test_ties_lowest_seed_then_name(self):
        F = _leaf(0, "f", BASE)
        leaves = [F, _leaf(5, "a", OTHER, gain=0.05), _leaf(3, "z", NOISE, gain=0.05),
                  _leaf(3, "b", OTHER, gain=0.05)]
        L, _, _ = gs.wrong_family_donor(leaves, F, exclude_seeds={0, 1})
        assert (L["seed"], L["name"]) == (3, "b")

    def test_none_when_nothing_eligible(self):
        F = _leaf(0, "f", BASE)
        leaves = [F, _leaf(1, "a", BASE), _leaf(2, "b", BASE + 0.2 * NOISE)]
        L, r, n = gs.wrong_family_donor(leaves, F, exclude_seeds={0, 1})
        assert L is None and np.isnan(r) and n == 0


class TestNames:
    def test_name_format_and_length(self):
        import run_rescue2 as r2
        n = r2.name_for("ko-alias-refit-ft", 4, "day_eve_minutes_hinge", 6,
                        "heavy_day_eve_usage")
        assert n == ("churn-rescue2-ko-alias-refit-ft-r4-day-eve-minutes-hinge-d6-"
                     "heavy-day-eve-usage")
        long = r2.name_for("ko-alias-refit-ft", 4, "x" * 200, 6, "y" * 200)
        assert len(long) <= r2.NAME_MAX
        assert long != r2.name_for("ko-alias-refit-ft", 4, "x" * 200, 6, "y" * 199)
