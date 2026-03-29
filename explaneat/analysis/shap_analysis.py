"""SHAP analysis for ExplaNEAT models.

Computes Shapley values to assess variable importance — complementary
to the structural explanation. Helps identify variables that can be
safely removed vs. those that are critical to the model.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def compute_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values for a prediction function.

    Args:
        predict_fn: (n_samples, n_features) -> (n_samples,) or (n_samples, n_out)
        X: Background/reference data for SHAP
        feature_names: Names for each feature column
        max_samples: Max background samples for KernelExplainer

    Returns:
        Dict with shap_values (n_samples x n_features), feature_names,
        mean_abs_shap (per-feature importance), and base_value.
    """
    import shap

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    # Subsample background if needed
    bg = X if len(X) <= max_samples else shap.sample(X, max_samples)

    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(X)

    # Handle multi-output: take first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.mean(np.abs(shap_values), axis=0).tolist()

    return {
        "shap_values": shap_values.tolist(),
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
        "base_value": float(explainer.expected_value)
        if not isinstance(explainer.expected_value, np.ndarray)
        else float(explainer.expected_value[0]),
    }
