"""Shared helpers for the Heart regularisation test (read-only DB access)."""

import json
import sys
from pathlib import Path
from uuid import UUID

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

# Heart-100 dataset and split (experiments/10seed/run_seed.py DATASETS["heart100"]).
DATASET_ID = "eea247c2-bdb7-41af-8ec1-9e26dcb5050d"
SPLIT_ID = "165cebbd-6609-46ca-867e-c09e17b2d6ec"
SOURCE_DB = "postgresql://localhost/explaneat_dev"
NEAT_CONFIG = REPO / "experiments/10seed/configs/heart.ini"
SPLIT_JSON = HERE / "validation_split.json"


def load_heart_raw():
    """X, y, train and test row indices of the paper split, in a READ ONLY transaction."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from explaneat.db.models import Dataset, DatasetSplit

    engine = create_engine(SOURCE_DB)
    with Session(engine) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        dataset = session.get(Dataset, UUID(DATASET_ID))
        split = session.get(DatasetSplit, UUID(SPLIT_ID))
        assert split.dataset_id == dataset.id
        X, y = dataset.get_data()
        train_idx = [int(i) for i in split.train_indices]
        test_idx = [int(i) for i in split.test_indices]
        session.rollback()
    engine.dispose()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    return X, y, train_idx, test_idx


def load_splits():
    """Standardised (scaler fit on the fit rows only) fit / validation / test arrays."""
    X, y, train_idx, test_idx = load_heart_raw()
    sp = json.loads(SPLIT_JSON.read_text())
    assert sorted(sp["fit_indices"] + sp["val_indices"]) == sorted(train_idx)
    assert sp["test_indices"] == test_idx
    fit, val, test = sp["fit_indices"], sp["val_indices"], sp["test_indices"]
    mean = X[fit].mean(axis=0)
    scale = X[fit].std(axis=0)
    scale[scale == 0.0] = 1.0

    def z(idx):
        return (X[idx] - mean) / scale

    return (z(fit), y[fit]), (z(val), y[val]), (z(test), y[test])
