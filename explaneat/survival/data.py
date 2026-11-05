"""Survival data: targets, loaders, event-stratified splits, DB storage.

Target convention
-----------------
``y`` is ``[N, 2]`` float64: column 0 = observed time (> 0), column 1 = event
indicator (1 = event, 0 = right-censored).  ``make_survival_targets`` builds it.

Loaders
-------
* ``load_builtin(name)``: public datasets bundled with scikit-survival (no
  download): gbsg2, whas500, veterans, flchain, aids.  Categoricals are
  one-hot encoded, missing values median-imputed (flchain only).
* ``load_survival_csv(path, time_col, event_col, features)``: the hook for a
  real cohort file.  It never imputes silently: missing values raise unless
  ``impute`` is given explicitly, because the imputation rule has to be the
  pre-registered one, identical for the comparator Cox model.

Splits
------
``survival_split_indices`` makes train / validation / test index arrays
stratified by the event indicator.  Persist them (``save_split_indices``) and
have every model read the same file: the comparison is paired only if the
folds are identical index arrays, not "the same seed".

DB
--
``save_survival_dataset_to_db`` stores a dataset with
``additional_metadata["task_type"] = "survival"`` and ``y_data`` as the
``[N, 2]`` array (the ``LargeBinary`` column holds any ``np.save`` array, so no
schema change is needed); ``save_survival_split_to_db`` stores event-stratified
index splits.  Data stays in whatever database ``DATABASE_URL`` points at, so
for restricted cohorts that must be a local database inside the approved
environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TASK_TYPE = "survival"


def make_survival_targets(time, event) -> np.ndarray:
    """Pack (time, event) into the [N, 2] float64 target ExplaNEAT carries as ``ys``."""
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.float64).ravel()
    validate_survival_target(np.column_stack([t, e]))
    return np.column_stack([t, e])


def validate_survival_target(y) -> None:
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"survival target must be [N, 2] (time, event), got {y.shape}")
    t, e = y[:, 0], y[:, 1]
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("survival times must be finite and > 0")
    if not np.all(np.isin(e, (0, 1))):
        raise ValueError("event indicator must be 0 (censored) or 1 (event)")
    if e.sum() == 0:
        raise ValueError("no events")


def is_survival_target(y) -> bool:
    y = np.asarray(y)
    return y.ndim == 2 and y.shape[1] == 2


def is_survival_dataset(dataset) -> bool:
    """True if a ``Dataset`` row is tagged ``task_type: "survival"``."""
    meta = getattr(dataset, "additional_metadata", None) or {}
    return meta.get("task_type") == TASK_TYPE


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

BUILTIN = ("gbsg2", "whas500", "veterans", "flchain", "aids")


def load_builtin(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``X [n, p] float64, y [n, 2], feature_names`` for a bundled dataset."""
    from sksurv import datasets as sd
    from sksurv.preprocessing import OneHotEncoder

    loaders = {
        "gbsg2": sd.load_gbsg2,
        "whas500": sd.load_whas500,
        "veterans": sd.load_veterans_lung_cancer,
        "flchain": sd.load_flchain,
        "aids": sd.load_aids,
    }
    if name not in loaders:
        raise ValueError(f"unknown builtin {name!r}; choose from {sorted(loaders)}")
    X, y = loaders[name]()
    if name == "flchain":
        # chapter is the cause of death: it is outcome information, not a covariate.
        X = X.drop(columns=[c for c in ("chapter",) if c in X.columns])
    X = OneHotEncoder().fit_transform(X)
    X = X.astype(float)
    X = X.fillna(X.median(numeric_only=True))
    ev_field, t_field = y.dtype.names
    time = y[t_field].astype(float)
    event = y[ev_field].astype(bool)
    # A handful of public datasets carry t = 0 rows; nudge to the smallest positive time / 2.
    if np.any(time <= 0):
        time = np.where(time <= 0, time[time > 0].min() / 2.0, time)
    return X.to_numpy(float), make_survival_targets(time, event), list(X.columns)


def load_survival_csv(path, time_col: str, event_col: str,
                      features: Optional[Sequence[str]] = None,
                      categorical: Optional[Sequence[str]] = None,
                      impute: Optional[str] = None,
                      event_value=1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a survival CSV (or Parquet) file: the cohort hook.

    Parameters
    ----------
    time_col : column of follow-up time (any unit; keep it consistent with the
        horizon you evaluate at, e.g. years for a 5-year risk).
    event_col : column whose value ``event_value`` marks the event of interest;
        anything else is treated as censored.  For CVD outcomes with competing
        non-CVD death this is a cause-specific hazard: say so in the analysis.
    features : predictor columns (default: every other column).
    categorical : columns to one-hot encode (drop-first).  Non-numeric columns
        are always one-hot encoded.
    impute : None (default: missing values raise) or "median".
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
    for c in (time_col, event_col):
        if c not in df.columns:
            raise ValueError(f"column {c!r} not in {path.name}")
    feats = list(features) if features else [c for c in df.columns if c not in (time_col, event_col)]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found: {missing}")

    X = df[feats].copy()
    cat = set(categorical or []) | {c for c in feats if not pd.api.types.is_numeric_dtype(X[c])}
    if cat:
        X = pd.get_dummies(X, columns=[c for c in feats if c in cat], drop_first=True, dtype=float)
    if X.isna().any().any() or df[[time_col, event_col]].isna().any().any():
        if impute != "median":
            bad = [c for c in X.columns if X[c].isna().any()]
            raise ValueError(f"missing values in {bad or [time_col, event_col]}; "
                             "pass impute='median' explicitly or impute upstream")
        X = X.fillna(X.median(numeric_only=True))
    time = df[time_col].to_numpy(float)
    event = (df[event_col] == event_value).to_numpy()
    return X.to_numpy(float), make_survival_targets(time, event), list(X.columns)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def survival_split_indices(event, test_size: float = 0.25, val_size: float = 0.0,
                           seed: int = 0) -> Dict[str, List[int]]:
    """Train / (validation) / test indices stratified by the event indicator.

    ``val_size`` is a fraction of the training part (as in the reference
    benchmark: 25% test, then 20% of the rest for validation).
    """
    from sklearn.model_selection import train_test_split

    e = np.asarray(event).astype(int).ravel()
    idx = np.arange(len(e))
    itr, ite = train_test_split(idx, test_size=test_size, random_state=seed, stratify=e)
    out = {"train": itr, "test": ite}
    if val_size:
        itr, iva = train_test_split(itr, test_size=val_size, random_state=seed, stratify=e[itr])
        out["train"], out["val"] = itr, iva
    return {k: sorted(int(i) for i in v) for k, v in out.items()}


def save_split_indices(path, splits: Dict[str, List[int]], **meta) -> None:
    Path(path).write_text(json.dumps({**meta, "splits": splits}, indent=1))


def load_split_indices(path) -> Dict[str, List[int]]:
    return json.loads(Path(path).read_text())["splits"]


def standardise(X_train, *others):
    """Standardise on the training rows only; returns (mean, scale, X_train_s, *others_s)."""
    mu = X_train.mean(0)
    sd = X_train.std(0)
    sd[sd == 0] = 1.0
    return (mu, sd, (X_train - mu) / sd) + tuple((X - mu) / sd for X in others)


# --------------------------------------------------------------------------
# DB
# --------------------------------------------------------------------------


def save_survival_dataset_to_db(name: str, X: np.ndarray, y: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                time_unit: Optional[str] = None,
                                source: str = "custom", description: Optional[str] = None,
                                metadata: Optional[dict] = None):
    """Store a survival dataset: ``task_type = "survival"``, ``y_data`` = [N, 2].

    ``num_classes`` is left NULL and ``target_name`` is "time,event".  No
    schema change is needed.
    """
    from explaneat.db.base import db
    from explaneat.db.models import Dataset

    validate_survival_target(y)
    meta = {"task_type": TASK_TYPE, "target_columns": ["time", "event"],
            "n_events": int(np.asarray(y)[:, 1].sum())}
    if time_unit:
        meta["time_unit"] = time_unit
    meta.update(metadata or {})
    with db.session_scope() as session:
        ds = Dataset(
            name=name, source=source, description=description,
            num_samples=int(X.shape[0]), num_features=int(X.shape[1]), num_classes=None,
            feature_names=feature_names or [f"x{i}" for i in range(X.shape[1])],
            feature_descriptions={}, feature_types={},
            target_name="time,event", target_description="observed time; event (1) / censored (0)",
            additional_metadata=meta,
        )
        ds.set_data(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        session.add(ds)
        session.flush()
        session.expunge(ds)
        return ds


def save_survival_split_to_db(dataset_id, splits: Dict[str, List[int]], seed: int,
                              test_size: float, X: Optional[np.ndarray] = None):
    """Store event-stratified index splits; StandardScaler params from train rows if X given.

    Returns the (expunged, usable) ``DatasetSplit`` row.
    """
    import uuid as _uuid

    from explaneat.db.base import db
    from explaneat.db.models import DatasetSplit

    scaler_type = scaler_params = None
    if X is not None:
        Xtr = np.asarray(X, dtype=np.float64)[splits["train"]]
        sd = Xtr.std(0)
        sd[sd == 0] = 1.0
        scaler_type = "StandardScaler"
        scaler_params = {"mean": Xtr.mean(0).tolist(), "scale": sd.tolist()}
    if isinstance(dataset_id, str):
        dataset_id = _uuid.UUID(dataset_id)
    val = splits.get("val") or None
    with db.session_scope() as session:
        split = DatasetSplit(
            dataset_id=dataset_id,
            split_type="train_val_test" if val else "train_test",
            test_size=test_size, random_state=seed, shuffle=True, stratify=True,
            train_indices=list(splits["train"]), test_indices=list(splits["test"]),
            validation_indices=list(val) if val else None,
            scaler_type=scaler_type, scaler_params=scaler_params,
            preprocessing_steps=[{"step": "stratify", "by": "event"}],
            train_size=len(splits["train"]), test_size_actual=len(splits["test"]),
            validation_size=len(val) if val else None,
        )
        session.add(split)
        session.flush()
        session.expunge(split)
        return split
