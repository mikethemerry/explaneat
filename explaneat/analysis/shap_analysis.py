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
    output_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute SHAP values for a prediction function.

    Args:
        predict_fn: (n_samples, n_features) -> (n_samples,) or (n_samples, n_out)
        X: Background/reference data for SHAP
        feature_names: Names for each feature column
        max_samples: Max background samples for KernelExplainer
        output_names: Names for each output dimension (for multi-output)

    Returns:
        Dict with shap_values, feature_names, mean_abs_shap, base_value.
        For multi-output, also includes ``outputs`` — a list of per-output
        dicts each containing mean_abs_shap, base_value, and output_name.
    """
    import shap

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    # Subsample background if needed
    bg = X if len(X) <= max_samples else shap.sample(X, max_samples)

    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(X)

    # Normalise to list-of-arrays (one per output)
    if isinstance(shap_values, list):
        per_output = [np.asarray(sv) for sv in shap_values]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            # (n_samples, n_features, n_outputs) -> list of (n_samples, n_features)
            per_output = [sv[:, :, i] for i in range(sv.shape[2])]
        else:
            per_output = [sv]

    # Normalise expected_value to list
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray):
        base_values = ev.tolist()
    elif isinstance(ev, (list, tuple)):
        base_values = [float(v) for v in ev]
    else:
        base_values = [float(ev)]

    # Pad base_values if shorter than outputs
    while len(base_values) < len(per_output):
        base_values.append(0.0)

    n_outputs = len(per_output)

    if n_outputs == 1:
        # Single-output: flat response (backwards compatible)
        sv0 = per_output[0]
        mean_abs = np.mean(np.abs(sv0), axis=0).flatten().tolist()
        return {
            "shap_values": sv0.tolist(),
            "feature_names": feature_names,
            "mean_abs_shap": mean_abs,
            "base_value": base_values[0],
        }
    else:
        # Multi-output: per-output breakdown + aggregate
        if output_names is None:
            output_names = [f"output_{i}" for i in range(n_outputs)]

        outputs = []
        agg_abs = np.zeros(n_features)
        for i, sv_i in enumerate(per_output):
            mean_abs_i = np.mean(np.abs(sv_i), axis=0).flatten()
            agg_abs += mean_abs_i
            outputs.append({
                "output_name": output_names[i] if i < len(output_names) else f"output_{i}",
                "mean_abs_shap": mean_abs_i.tolist(),
                "base_value": base_values[i],
            })

        # Aggregate: average across outputs
        agg_abs = (agg_abs / n_outputs).tolist()

        return {
            "shap_values": per_output[0].tolist(),  # first output for compat
            "feature_names": feature_names,
            "mean_abs_shap": agg_abs,
            "base_value": base_values[0],
            "outputs": outputs,
        }
