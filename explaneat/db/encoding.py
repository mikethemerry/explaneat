"""Pure encoding functions for one-hot and ordinal feature preparation.

These functions have no database dependencies and operate on numpy arrays.
They support three encoding strategies:
- Categorical features: one-hot encoding
- Ordinal features: either rank-mapped integers or one-hot (opt-in)
- Passthrough: numeric/continuous features copied as-is
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def build_encoding_config(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: List[str],
    ordinal_onehot: Optional[Set[str]] = None,
    ordinal_orders: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Build an encoding configuration from data and feature metadata.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        feature_names: Name for each column in X.
        feature_types: Type for each column — "categorical", "ordinal",
            or anything else (treated as passthrough).
        ordinal_onehot: Set of ordinal feature names that should be
            one-hot encoded instead of rank-mapped.
        ordinal_orders: Dict mapping ordinal feature names to their
            explicit value ordering (as string lists). If not provided,
            unique values are auto-detected and sorted.

    Returns:
        Dict with keys: "categorical", "ordinal_as_ranked",
        "ordinal_as_onehot", "passthrough". Each maps feature names
        to their encoding metadata (list of category strings, or
        original type for passthrough).
    """
    if ordinal_onehot is None:
        ordinal_onehot = set()
    if ordinal_orders is None:
        ordinal_orders = {}

    config: Dict = {
        "categorical": {},
        "ordinal_as_ranked": {},
        "ordinal_as_onehot": {},
        "passthrough": {},
    }

    for col_idx, (name, ftype) in enumerate(zip(feature_names, feature_types)):
        column = X[:, col_idx]

        if ftype == "categorical":
            unique_vals = sorted(set(str(int(v)) for v in column))
            config["categorical"][name] = unique_vals

        elif ftype == "ordinal":
            if name in ordinal_orders:
                values = ordinal_orders[name]
            else:
                values = sorted(set(str(int(v)) for v in column))

            if name in ordinal_onehot:
                config["ordinal_as_onehot"][name] = values
            else:
                config["ordinal_as_ranked"][name] = values

        else:
            config["passthrough"][name] = ftype

    return config


def prepare_dataset_arrays(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: List[str],
    encoding_config: Dict,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Apply encoding config to expand/transform the feature matrix.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        feature_names: Name for each column in X.
        feature_types: Type string for each column.
        encoding_config: Config dict from build_encoding_config.

    Returns:
        Tuple of:
        - X_encoded: Transformed feature matrix.
        - new_feature_names: List of column names in the encoded matrix.
        - new_feature_types: Dict mapping each new column name to its type
          ("binary" for one-hot, "ordinal" for ranked, or original type).
    """
    n_samples = X.shape[0]
    columns: List[np.ndarray] = []
    new_names: List[str] = []
    new_types: Dict[str, str] = {}

    for col_idx, (name, ftype) in enumerate(zip(feature_names, feature_types)):
        column = X[:, col_idx]

        if name in encoding_config["categorical"]:
            _one_hot_encode(
                column, name, encoding_config["categorical"][name],
                n_samples, columns, new_names, new_types,
            )

        elif name in encoding_config["ordinal_as_onehot"]:
            _one_hot_encode(
                column, name, encoding_config["ordinal_as_onehot"][name],
                n_samples, columns, new_names, new_types,
            )

        elif name in encoding_config["ordinal_as_ranked"]:
            order = encoding_config["ordinal_as_ranked"][name]
            value_to_rank = {str(v): rank for rank, v in enumerate(order)}
            ranked = np.array([
                value_to_rank.get(str(int(v)), -1) for v in column
            ], dtype=float)
            columns.append(ranked.reshape(-1, 1))
            new_names.append(name)
            new_types[name] = "ordinal"

        else:
            # Passthrough
            columns.append(column.reshape(-1, 1))
            new_names.append(name)
            new_types[name] = ftype

    X_encoded = np.hstack(columns) if columns else np.empty((n_samples, 0))
    return X_encoded, new_names, new_types


def _one_hot_encode(
    column: np.ndarray,
    feature_name: str,
    categories: List[str],
    n_samples: int,
    columns: List[np.ndarray],
    new_names: List[str],
    new_types: Dict[str, str],
) -> None:
    """One-hot encode a column and append results to the output lists.

    Unknown values (not in categories) produce all-zero rows.
    """
    value_to_idx = {str(v): i for i, v in enumerate(categories)}
    one_hot = np.zeros((n_samples, len(categories)), dtype=float)

    for row_idx, val in enumerate(column):
        str_val = str(int(val))
        if str_val in value_to_idx:
            one_hot[row_idx, value_to_idx[str_val]] = 1.0

    for cat_idx, cat_val in enumerate(categories):
        col_name = f"{feature_name}:{cat_val}"
        columns.append(one_hot[:, cat_idx].reshape(-1, 1))
        new_names.append(col_name)
        new_types[col_name] = "binary"
