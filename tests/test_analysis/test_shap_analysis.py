"""Tests for SHAP analysis integration."""
import numpy as np
import pytest
from explaneat.analysis.shap_analysis import compute_shap_values


class TestShapAnalysis:
    def test_compute_shap_returns_values_and_labels(self):
        """SHAP values should be computed for a simple linear model."""
        X = np.random.randn(100, 3)
        # Simple linear model: y = 2*x0 + 0*x1 + 1*x2
        def predict(x):
            return x @ np.array([2.0, 0.0, 1.0])

        result = compute_shap_values(
            predict_fn=predict,
            X=X,
            feature_names=["pregWeight", "ID", "gestMonth"],
        )
        assert "shap_values" in result
        assert "feature_names" in result
        assert "mean_abs_shap" in result
        assert len(result["feature_names"]) == 3
        assert len(result["mean_abs_shap"]) == 3
        # pregWeight (coeff=2) should have highest importance
        assert result["mean_abs_shap"][0] > result["mean_abs_shap"][1]

    def test_feature_names_default_to_indices(self):
        X = np.random.randn(50, 2)
        result = compute_shap_values(lambda x: x[:, 0], X)
        assert result["feature_names"] == ["x_0", "x_1"]

    def test_multi_output_squeeze(self):
        """SHAP values for multi-dim output should be squeezed to flat list."""
        X = np.random.randn(50, 2)
        # Returns (n, 1) shape - simulates single-output with 2D return
        def predict(x):
            return (x @ np.array([[1.0], [0.5]]))

        result = compute_shap_values(predict, X)
        # mean_abs_shap should be a flat list of floats, not nested
        assert isinstance(result["mean_abs_shap"][0], float)
        assert len(result["mean_abs_shap"]) == 2


class TestShapManyFeatures:
    """With > 10 features, shap's default l1_reg='num_features(10)' keeps
    only 10 features per explained row and assigns every other feature an
    exact 0 -- small-but-real effects vanish from mean |SHAP|."""

    def test_no_feature_zeroed_by_feature_selection(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 16))
        w = np.linspace(1.0, 0.1, 16)  # every feature matters, some only a little

        result = compute_shap_values(lambda x: x @ w, X, max_samples=40)
        mean_abs = np.array(result["mean_abs_shap"])
        assert np.all(mean_abs > 1e-3), mean_abs
        # linear model: mean|SHAP_j| ~= |w_j| * mean|x_j - E[x_j]|
        expected = np.abs(w) * np.mean(np.abs(X - X.mean(axis=0)), axis=0)
        np.testing.assert_allclose(mean_abs, expected, rtol=0.05, atol=1e-3)

    def test_no_progress_bar_on_stderr(self, capsys):
        X = np.random.default_rng(1).normal(size=(20, 3))
        compute_shap_values(lambda x: x.sum(axis=1), X)
        captured = capsys.readouterr()
        assert "it/s" not in captured.err and "%|" not in captured.err

    def test_weak_features_beyond_top10_not_exactly_zero(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 16))
        w = np.array([5.0] * 10 + [0.3] * 6)
        result = compute_shap_values(lambda x: x @ w, X, max_samples=30)
        weak = np.array(result["mean_abs_shap"][10:])
        assert np.all(weak > 0.05), weak
