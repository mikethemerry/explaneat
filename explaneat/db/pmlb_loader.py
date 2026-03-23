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
    df = pmlb.fetch_data(name)

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
