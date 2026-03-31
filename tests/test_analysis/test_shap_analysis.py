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
