"""Local CSV datasets for the portal: preview and import.

The file is read from a local path (it is never uploaded or copied).  The
dataset is stored in the configured database with:

* survival: ``additional_metadata.task_type = "survival"`` and ``y`` = [N, 2]
  (time, event) (``explaneat.survival.data``);
* classification: ``num_classes = 2`` and ``y`` = 0/1.

plus ``additional_metadata.restricted`` and the file's base name only.  A
train / (validation) / test split stratified by the event indicator (or the
class) is created with a StandardScaler fitted on the training rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import StorageError, check_restricted_storage, current_database_url

_TIME_HINTS = ("time", "fu", "follow", "duration", "days", "years", "months", "surv")
_EVENT_HINTS = ("event", "status", "death", "dead", "died", "outcome", "cvd", "fail")


def _read(path):
    from ..survival.data import read_table

    return read_table(path)


def preview_csv(path) -> Dict[str, Any]:
    """Column summary for the column picker. No raw rows are returned."""
    import pandas as pd

    df = _read(path)
    cols = []
    for c in df.columns:
        s = df[c]
        info: Dict[str, Any] = {
            "name": str(c),
            "dtype": str(s.dtype),
            "numeric": bool(pd.api.types.is_numeric_dtype(s)),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if info["numeric"] and s.notna().any():
            info["min"] = float(s.min())
            info["max"] = float(s.max())
        if info["n_unique"] <= 12:
            info["levels"] = [str(v) for v in sorted(s.dropna().unique().tolist(), key=str)][:12]
        lc = str(c).lower()
        info["suggest"] = (
            "time" if any(h in lc for h in _TIME_HINTS) and info["numeric"] and info["n_unique"] > 12
            else "event" if any(h in lc for h in _EVENT_HINTS) and info["n_unique"] == 2
            else "categorical" if not info["numeric"] or (2 < info["n_unique"] <= 12)
            else ""
        )
        cols.append(info)
    return {"file": Path(path).name, "n_rows": int(len(df)), "columns": cols}


def _encode(df, features: Sequence[str], categorical: Sequence[str], impute: Optional[str]):
    from ..survival.data import encode_features

    return encode_features(df, features, categorical, impute)


def import_csv(
    path,
    name: str,
    task: str,
    features: Optional[Sequence[str]] = None,
    categorical: Optional[Sequence[str]] = None,
    impute: Optional[str] = None,
    time_col: Optional[str] = None,
    event_col: Optional[str] = None,
    event_value: Any = 1,
    target_col: Optional[str] = None,
    positive_value: Any = 1,
    time_unit: Optional[str] = None,
    restricted: bool = False,
    test_size: float = 0.25,
    val_size: float = 0.20,
    split_seed: int = 0,
    horizons: Optional[Sequence[float]] = None,
    calibration_horizon: Optional[float] = None,
) -> Dict[str, Any]:
    """Load a local CSV/Parquet file into a dataset and a stratified split.

    Survival only: ``horizons`` (td-AUC / Brier times) and
    ``calibration_horizon`` are stored in the dataset metadata, where the
    results table and evidence read them (pre-register them, e.g. 1, 3, 5 and
    5 years). Without them the training event-time quartiles and their median
    are used.
    """
    from .. import governance

    restricted = bool(restricted or governance.restricted_server())
    if restricted:
        check_restricted_storage(csv_path=path, database_url=current_database_url())
    df = _read(path)
    if task == "survival":
        if not time_col or not event_col:
            raise ValueError("survival needs time_col and event_col")
        outcome = [time_col, event_col]
    elif task == "classification":
        if not target_col:
            raise ValueError("classification needs target_col")
        outcome = [target_col]
    else:
        raise ValueError("task must be 'survival' or 'classification'")
    for c in outcome:
        if c not in df.columns:
            raise ValueError(f"column {c!r} not in the file")
    if df[outcome].isna().any().any():
        raise ValueError(f"missing values in the outcome columns {outcome}")
    feats = list(features) if features else [c for c in df.columns if c not in outcome]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found: {missing}")
    leaked = [c for c in feats if c in outcome]
    if leaked:
        raise ValueError(f"outcome columns cannot be features: {leaked}")
    X, names = _encode(df, feats, categorical or [], impute)

    from ..survival.data import match_value as _match

    meta = {"restricted": restricted, "source_file": Path(path).name, "features": feats,
            "categorical": list(categorical or []), "impute": impute}
    from ..db import db
    from ..db.models import Dataset

    if task == "survival":
        from ..survival.data import (make_survival_targets, save_survival_dataset_to_db,
                                     save_survival_split_to_db, survival_split_indices)

        y = make_survival_targets(df[time_col].to_numpy(float), _match(df[event_col], event_value))
        meta.update(time_col=time_col, event_col=event_col, event_value=str(event_value))
        if horizons:
            hs = [float(h) for h in horizons]
            if any(not np.isfinite(h) or h <= 0 for h in hs):
                raise ValueError("horizons must be positive times")
            meta["horizons"] = sorted(hs)
        if calibration_horizon is not None:
            if not np.isfinite(float(calibration_horizon)) or float(calibration_horizon) <= 0:
                raise ValueError("calibration_horizon must be a positive time")
            meta["calibration_horizon"] = float(calibration_horizon)
        ds = save_survival_dataset_to_db(name, X, y, names, time_unit=time_unit, source="csv",
                                         description=f"Imported from {Path(path).name}",
                                         metadata=meta)
        splits = survival_split_indices(y[:, 1], test_size=test_size, val_size=val_size, seed=split_seed)
        split = save_survival_split_to_db(ds.id, splits, split_seed, test_size, X)
        n_pos = int(y[:, 1].sum())
    else:
        from sklearn.model_selection import train_test_split

        from ..db.models import DatasetSplit

        y = _match(df[target_col], positive_value).astype(np.float64)
        if y.min() == y.max():
            raise ValueError("the target has one class only")
        meta.update(target_col=target_col, positive_value=str(positive_value), task_type="classification")
        idx = np.arange(len(y))
        itr, ite = train_test_split(idx, test_size=test_size, random_state=split_seed, stratify=y)
        iva = None
        if val_size:
            itr, iva = train_test_split(itr, test_size=val_size, random_state=split_seed, stratify=y[itr])
        Xtr = X[itr]
        sd = Xtr.std(0)
        sd[sd == 0] = 1.0
        with db.session_scope() as s:
            ds = Dataset(name=name, source="csv", description=f"Imported from {Path(path).name}",
                         num_samples=int(X.shape[0]), num_features=int(X.shape[1]), num_classes=2,
                         feature_names=names, feature_descriptions={}, feature_types={},
                         target_name=target_col, class_names=["0", "1"], additional_metadata=meta)
            ds.set_data(X, y)
            s.add(ds)
            s.flush()
            split = DatasetSplit(
                dataset_id=ds.id, split_type="train_val_test" if iva is not None else "train_test",
                test_size=test_size, random_state=split_seed, shuffle=True, stratify=True,
                train_indices=sorted(int(i) for i in itr), test_indices=sorted(int(i) for i in ite),
                validation_indices=sorted(int(i) for i in iva) if iva is not None else None,
                scaler_type="StandardScaler",
                scaler_params={"mean": Xtr.mean(0).tolist(), "scale": sd.tolist()},
                preprocessing_steps=[{"step": "stratify", "by": target_col}],
                train_size=len(itr), test_size_actual=len(ite),
                validation_size=len(iva) if iva is not None else None,
            )
            s.add(split)
            s.flush()
            s.expunge(ds)
            s.expunge(split)
        n_pos = int(y.sum())
    return {
        "dataset_id": str(ds.id),
        "split_id": str(split.id),
        "task": task,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": names,
        "n_events_or_positives": n_pos,
        "restricted": restricted,
        "train": len(split.train_indices),
        "val": len(split.validation_indices or []),
        "test": len(split.test_indices),
    }


__all__ = ["preview_csv", "import_csv", "StorageError"]
