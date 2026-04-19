"""PMLB dataset loader - downloads and stores PMLB datasets in the database."""
import numpy as np
from typing import Optional

import pmlb

from .dataset_utils import save_dataset_to_db
from .models import Dataset


def download_and_store_pmlb_dataset(
    name: str,
    version: Optional[str] = None,
) -> Dataset:
    """Download a PMLB dataset and store it in the database.

    Args:
        name: PMLB dataset name (e.g., 'iris', 'breast_cancer', 'xor')
        version: Optional version string

    Returns:
        Dataset model instance
    """
    # Try the exact name first, then alternate separators (- vs _)
    # The PMLB catalog lists names that sometimes don't match the GitHub repo
    attempts = [name]
    if "_" in name:
        attempts.append(name.replace("_", "-"))
    elif "-" in name:
        attempts.append(name.replace("-", "_"))

    df = None
    last_error = None
    for attempt in attempts:
        try:
            df = pmlb.fetch_data(attempt)
            name = attempt  # use the name that worked
            break
        except ValueError as e:
            last_error = e

    if df is None:
        raise ValueError(
            f"Dataset '{name}' not found in PMLB (tried: {', '.join(attempts)}). "
            f"This dataset may have been removed from the PMLB repository. "
            f"Try searching UCI instead."
        ) from last_error

    # PMLB convention: last column is 'target'
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # Build metadata
    classification_datasets = set(pmlb.classification_dataset_names)
    is_classification = name in classification_datasets

    class_names = None
    if is_classification:
        class_names = [str(c) for c in sorted(np.unique(y).astype(int))]

    dataset = save_dataset_to_db(
        name=name,
        X=X,
        y=y,
        source="PMLB",
        version=version,
        source_url=f"https://github.com/EpistasisLab/pmlb/tree/master/datasets/{name}",
        description=f"PMLB dataset: {name}",
        feature_names=feature_cols,
        target_name=target_col,
        class_names=class_names,
        metadata={
            "pmlb_name": name,
            "task_type": "classification" if is_classification else "regression",
        },
    )

    return dataset
