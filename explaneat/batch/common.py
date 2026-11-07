"""Shared helpers for the portal workflow."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class StorageError(PermissionError):
    """Restricted data would be stored or read somewhere it must not be."""


def data_home() -> Path:
    """Local data directory: ``$EXPLANEAT_HOME`` or ``~/.explaneat`` (outside the repo)."""
    return Path(os.environ.get("EXPLANEAT_HOME") or (Path.home() / ".explaneat")).expanduser()


def jobs_dir() -> Path:
    d = data_home() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def inside_repo(path) -> bool:
    try:
        p = Path(path).expanduser().resolve()
    except OSError:
        return False
    root = REPO_ROOT.resolve()
    return p == root or root in p.parents


def sqlite_file(database_url: Optional[str]) -> Optional[Path]:
    if not database_url or not database_url.startswith("sqlite"):
        return None
    tail = database_url.split(":///", 1)[-1]
    if not tail or ":memory:" in tail:
        return None
    return Path(tail)


def check_restricted_storage(csv_path=None, database_url: Optional[str] = None) -> None:
    """Restricted data must never be read from or written into the repo tree.

    Refuses a source file inside the repository, a SQLite database file inside
    the repository, and a data directory (job files) inside it.
    """
    if csv_path is not None and inside_repo(csv_path):
        raise StorageError(
            "Restricted data must not live in the repository tree: move the file to a "
            "directory outside the repo (e.g. under $EXPLANEAT_HOME)."
        )
    f = sqlite_file(database_url)
    if f is not None and inside_repo(f):
        raise StorageError(
            "The database is a SQLite file inside the repository; restricted data cannot "
            "be stored there. Point DATABASE_URL at a database outside the repo."
        )
    if inside_repo(data_home()):
        raise StorageError("EXPLANEAT_HOME is inside the repository; set it outside the repo.")


def write_json_atomic(path: Path, obj, attempts: int = 50) -> None:
    """Write JSON via a temp file and ``os.replace``, safe on Windows.

    The temp name is unique per process (the API and a worker may write the
    same progress file). On Windows ``os.replace`` raises PermissionError while
    another process has the target open (the API polling ``progress.json``),
    so it is retried for up to ~2.5 s instead of failing the run.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def current_database_url() -> str:
    from ..db import db

    return db.database_url or os.environ.get("DATABASE_URL", "postgresql://localhost/explaneat_dev")


# ----------------------------------------------------------------------------- data


def load_split_arrays(session, split_id) -> Dict[str, Any]:
    """Scaled X / y for train, val and test of a split, plus dataset metadata.

    The split's stored scaler is applied (as the evidence pipeline does).
    """
    from ..db.models import Dataset, DatasetSplit

    split = session.get(DatasetSplit, uuid.UUID(str(split_id)))
    if split is None:
        raise KeyError(f"split {split_id} not found")
    ds = session.get(Dataset, split.dataset_id)
    X, y = ds.get_data()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if split.scaler_type == "StandardScaler" and split.scaler_params:
        mean = np.asarray(split.scaler_params["mean"])
        scale = np.asarray(split.scaler_params["scale"])
        scale = np.where(scale == 0, 1.0, scale)
        X = (X - mean) / scale
    out = {
        "dataset_id": str(ds.id),
        "dataset_name": ds.name,
        "feature_names": list(ds.feature_names or []),
        "meta": dict(ds.additional_metadata or {}),
        "scaler": split.scaler_params,
        "split_id": str(split.id),
    }
    for part, idx in (("train", split.train_indices), ("val", split.validation_indices),
                      ("test", split.test_indices)):
        idx = list(idx or [])
        out[f"X_{part}"] = X[idx] if idx else None
        out[f"y_{part}"] = y[idx] if idx else None
    return out


def task_of_dataset(dataset) -> str:
    meta = getattr(dataset, "additional_metadata", None) or {}
    return "survival" if meta.get("task_type") == "survival" else "classification"


# ----------------------------------------------------------------------------- templates

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "survival-cox-v3": {
        "task": "survival",
        "title": "Cox survival, protocol v3",
        "description": (
            "Identity output (the network output is the log-risk g(x)); full-batch "
            "backprop on the Efron Cox partial likelihood, 5 epochs per generation; "
            "phantom mask on; population 150, 60 generations, weight_init_stdev 0.1; "
            "fitness = -loss - lambda*k/D on the training fold (lambda = 1: per-event "
            "AIC; k = enabled connections + hidden nodes; D = training events); the "
            "selected genome is the generation champion with the lowest validation "
            "Cox loss; Breslow baseline on the training rows. Comparator: ridge Cox "
            "(Efron), alpha tuned on validation Cox loss. "
            "(experiments/survival/RESULTS.md section 2)"
        ),
        "ini": "survival.ini",
        "defaults": {"generations": 60, "pop_size": 150, "epochs": 5, "complexity_penalty": 1.0,
                     "test_size": 0.25, "val_size": 0.20},
    },
    "classification-propneat": {
        "task": "classification",
        "title": "Binary classification (10-seed protocol)",
        "description": (
            "Sigmoid output; backprop on BCE, 5 epochs per generation; phantom mask on; "
            "weights/biases learned by backprop only (experiments/10seed/configs/heart.ini); "
            "fitness = training AUC; the selected genome is the generation champion with the "
            "highest validation AUC (or the best training fitness without a validation fold). "
            "Comparator: L2 logistic regression, C tuned on validation AUC."
        ),
        "ini": "classification.ini",
        "defaults": {"generations": 40, "pop_size": 250, "epochs": 5, "complexity_penalty": 0.0,
                     "test_size": 0.20, "val_size": 0.20},
    },
}


def template_ini_text(name: str, n_inputs: int, pop_size: Optional[int] = None,
                      overrides: Optional[Dict[str, Any]] = None) -> str:
    t = TEMPLATES[name]
    text = (TEMPLATE_DIR / t["ini"]).read_text(encoding="utf-8")
    overrides = dict(overrides or {})
    if pop_size:
        overrides["pop_size"] = pop_size
    lines = []
    for line in text.splitlines():
        key = line.split("=")[0].strip()
        if key == "num_inputs":
            line = f"num_inputs = {n_inputs}"
        elif key in overrides:
            line = f"{key} = {overrides[key]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def list_templates():
    out = []
    for name, t in TEMPLATES.items():
        out.append({"name": name, **{k: v for k, v in t.items() if k != "ini"},
                    "ini_text": template_ini_text(name, 0)})
    return out


def json_safe(obj):
    import math

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def split_counts(y, task: str) -> Tuple[int, int]:
    y = np.asarray(y)
    if task == "survival":
        return int(len(y)), int(y[:, 1].sum())
    return int(len(y)), int((y.ravel() > 0.5).sum())
