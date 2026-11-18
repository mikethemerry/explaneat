"""Pairs plot data (viz_data.compute_pairs): shapes, binary inputs, the interaction measure."""
import numpy as np
import pytest

from explaneat.analysis.viz_data import (
    PAIRS_MAX_INPUTS, compute_pairs, pair_interaction, suggest_viz_types,
)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(size=n),            # 0 continuous
        rng.uniform(-2, 2, size=n),    # 1 continuous
        rng.integers(0, 2, size=n),    # 2 binary
        rng.normal(size=n),            # 3 continuous
        rng.integers(0, 4, size=n),    # 4 discrete (4 levels)
    ]).astype(float)


def _col(f):
    return lambda x: f(x).reshape(-1, 1)


def test_shapes_and_defaults():
    X = _data()
    f = _col(lambda x: x[:, 0] + 2 * x[:, 1] + x[:, 2] + 0.1 * x[:, 3] + 0.5 * x[:, 4])
    r = compute_pairs(f, X, f(X), grid_size=25, max_scatter=200)
    k = len(r["dims"])
    assert k == 5 and set(r["dims"]) == set(range(5))
    assert len(r["pairs"]) == k * (k - 1) // 2
    assert r["dims"][0] == 1            # highest sensitivity first (slope 2)
    assert r["n_rows"] == 600 and r["n_scatter"] == 200 and len(r["color"]) == 200
    for inp in r["inputs"]:
        assert len(inp["pd"]) == len(inp["grid"]) == len(inp["weights"])
        assert len(inp["scatter"]) == 200
        assert abs(sum(inp["weights"]) - 1) < 1e-9
    for p in r["pairs"]:
        z = np.asarray(p["z"])
        assert z.shape == (len(r["inputs"][p["a"]]["grid"]), len(r["inputs"][p["b"]]["grid"]))
    assert [p["rank"] for p in r["pairs"]] == list(range(1, len(r["pairs"]) + 1))
    lo, mid, hi = r["color_domain"]
    assert lo <= min(r["color"]) and hi >= max(r["color"]) and lo <= mid <= hi
    assert np.isclose(mid, r["reference"])     # centred on the prediction at the medians
    assert len(r["ranking"]) == 5


def test_binary_and_discrete_inputs_use_only_their_values():
    X = _data()
    f = _col(lambda x: x[:, 0] * x[:, 2] + x[:, 4])
    r = compute_pairs(f, X, f(X), dims=[0, 2, 4])
    by_dim = {i["dim"]: i for i in r["inputs"]}
    assert by_dim[2]["kind"] == "binary" and by_dim[2]["grid"] == [0.0, 1.0]
    assert by_dim[4]["kind"] == "discrete" and by_dim[4]["grid"] == [0.0, 1.0, 2.0, 3.0]
    assert by_dim[0]["kind"] == "continuous" and len(by_dim[0]["grid"]) == 25
    assert by_dim[2]["hist"]["levels"] == [0.0, 1.0] and sum(by_dim[2]["hist"]["counts"]) == 600
    pair02 = next(p for p in r["pairs"] if {r["dims"][p["a"]], r["dims"][p["b"]]} == {0, 2})
    assert np.asarray(pair02["z"]).shape == (25, 2)   # 2 columns for the binary input


def test_interaction_zero_for_additive_positive_for_product():
    X = _data()
    add = _col(lambda x: np.sin(x[:, 0]) + x[:, 1] ** 2 + 3 * x[:, 2] + np.maximum(x[:, 3], 0))
    for fix in ("median", "average"):
        r = compute_pairs(add, X, add(X), dims=[0, 1, 2, 3], fix=fix)
        assert max(p["h"] for p in r["pairs"]) < 1e-6, fix
        assert max(p["h_pd"] for p in r["pairs"]) < 1e-6, fix
    prod = _col(lambda x: x[:, 0] * x[:, 1] + x[:, 3])
    r = compute_pairs(prod, X, prod(X), dims=[0, 1, 3])
    top = r["pairs"][0]
    assert {r["dims"][top["a"]], r["dims"][top["b"]]} == {0, 1}
    assert top["h"] > 0.5 and top["interaction_sd"] > 0.1
    others = [p for p in r["pairs"] if p is not top]
    assert all(p["h"] < 1e-6 for p in others)


def test_pair_interaction_bounds():
    gi, gj = np.linspace(-1, 1, 9), np.linspace(-1, 1, 7)
    w_i, w_j = np.full(9, 1 / 9), np.full(7, 1 / 7)
    pure = np.outer(gi, gj)                      # pure interaction surface
    res = pair_interaction(pure, np.zeros(9), np.zeros(7), w_i, w_j)
    assert np.isclose(res["h"], 1.0)
    flat = np.zeros((9, 7))
    assert pair_interaction(flat, np.zeros(9), np.zeros(7), w_i, w_j)["h"] == 0.0


def test_dims_capped_and_validated():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 12))
    f = _col(lambda x: x.sum(axis=1))
    r = compute_pairs(f, X, f(X), dims=list(range(12)))
    assert len(r["dims"]) == PAIRS_MAX_INPUTS
    with pytest.raises(ValueError):
        compute_pairs(f, X, f(X), dims=[3])


def test_suggested_only_with_three_inputs():
    assert "pairs_plot" in suggest_viz_types(3, 1)
    assert "pairs_plot" not in suggest_viz_types(2, 1)


def test_impossible_combinations_do_not_count():
    """Two levels of one category (never both 1) and a split copy against itself."""
    rng = np.random.default_rng(2)
    level = rng.integers(0, 3, size=400)
    X = np.column_stack([(level == 0), (level == 1), rng.normal(size=400), (level == 0)]).astype(float)
    # an interaction only where both one-hot columns are 1, a combination no row has
    f = _col(lambda x: 3 * x[:, 0] * x[:, 1] + x[:, 0] + x[:, 2])
    r = compute_pairs(f, X, f(X), dims=[0, 1, 2, 3])
    by = {frozenset((r["dims"][p["a"]], r["dims"][p["b"]])): p for p in r["pairs"]}
    p01 = by[frozenset((0, 1))]
    assert np.asarray(p01["counts"])[1, 1] == 0
    assert p01["h"] < 1e-6
    p03 = by[frozenset((0, 3))]
    assert p03["same_input"] and r["pairs"][-1]["same_input"]
