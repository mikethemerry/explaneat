"""The fix-at-median viz types hold the other inputs at a given reference (``fixed``)."""
import numpy as np
import pytest

from explaneat.analysis import viz_data as vd


def _fn(x):
    x = np.asarray(x, dtype=float)
    return (x[:, 0] * 2 + x[:, 1] * x[:, 2] - x[:, 3]).reshape(-1, 1)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 4))
    return X, _fn(X)


def test_default_is_per_column_median(data):
    X, Y = data
    out = vd.compute_line_plot(_fn, X, Y, input_dim=0)
    med = np.median(X, axis=0)
    assert out["held"] == pytest.approx(med.tolist())
    assert out["held_rule"] == "median"
    g = np.asarray(out["grid_x"])
    expect = 2 * g + med[1] * med[2] - med[3]
    assert np.allclose(out["grid_y"], expect)


def test_line_uses_fixed(data):
    X, Y = data
    fixed = np.array([0.0, 1.5, -2.0, 0.25])
    out = vd.compute_line_plot(_fn, X, Y, input_dim=0, fixed=fixed)
    g = np.asarray(out["grid_x"])
    assert np.allclose(out["grid_y"], 2 * g + 1.5 * -2.0 - 0.25)
    assert out["held"] == pytest.approx(fixed.tolist())
    assert out["held_rule"] == "reference"


def test_heatmap_and_pd_use_fixed(data):
    X, Y = data
    fixed = np.array([1.0, 0.0, 0.0, 3.0])
    hm = vd.compute_heatmap(_fn, X, Y, input_dims=(1, 2), fixed=fixed, grid_size=5)
    xr, yr = np.asarray(hm["x_range"]), np.asarray(hm["y_range"])
    z = np.asarray(hm["z_grid"])
    assert np.allclose(z, 2 * 1.0 + yr[:, None] * xr[None, :] - 3.0)
    pd1 = vd.compute_partial_dependence(_fn, X, vary_dims=[3], fixed=fixed, grid_points=7)
    assert np.allclose(pd1["grid_y"], 2.0 - np.asarray(pd1["grid_x"]))
    assert pd1["held_rule"] == "reference"
    fo = vd.compute_feature_output_scatter(_fn, X, Y, input_dim=0, fixed=fixed)
    assert np.allclose(fo["pd_y"], 2 * np.asarray(fo["pd_x"]) - 3.0)


def test_pairs_uses_fixed_only_at_median(data):
    X, Y = data
    fixed = np.array([0.5, 1.0, 1.0, 0.0])
    out = vd.compute_pairs(_fn, X, Y, dims=[0, 1, 2], fixed=fixed, grid_size=5)
    assert out["reference"] == pytest.approx(2 * 0.5 + 1.0)
    assert out["held"] == pytest.approx(fixed.tolist())
    assert out["held_rule"] == "reference"
    avg = vd.compute_pairs(_fn, X, Y, dims=[0, 1, 2], fixed=fixed, fix="average", grid_size=5)
    assert avg["held_rule"] == "average"


def test_fixed_wrong_length(data):
    X, Y = data
    with pytest.raises(ValueError):
        vd.compute_line_plot(_fn, X, Y, fixed=np.zeros(3))
