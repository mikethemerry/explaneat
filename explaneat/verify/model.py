"""The model a verifier challenges: a genome's phenotype with an explanation's operations
replayed in memory.

The database is read only (``SET TRANSACTION READ ONLY`` on PostgreSQL). What it gives (the
phenotype, the split's data and scaler, the stored operations) is cached as a pickle under
``$EXPLANEAT_HOME/verify_cache/`` so repeated probes do not reload it; pass ``cache=False``
(``--no-cache``) to skip the cache. The operations that are replayed can come from anywhere,
for example a copied operations list in a benchmark item, so a structural change is checked
on a copy and never written back.
"""

from __future__ import annotations

import copy
import json
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CACHE_VERSION = 2   # 2: the explanation's operations are no longer part of the base


@dataclass
class BaseData:
    """Everything the database contributes, independent of the explanation being checked."""

    genome_id: str
    phenotype: Any                       # NetworkStructure
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    feature_names: List[str]
    raw_train: np.ndarray
    raw_test: Optional[np.ndarray]
    mean: np.ndarray
    scale: np.ndarray
    dataset_name: str
    meta: Dict[str, Any]
    db_operations: List[Dict[str, Any]] = field(default_factory=list)   # tests only; never filled from the DB
    codebook: Optional[Dict[str, Any]] = None   # None: the dataset's codebook (by name)


def cache_dir() -> Path:
    from .. import settings
    return settings.data_dir() / "verify_cache"


def _cache_path(genome_id: str) -> Path:
    return cache_dir() / f"{genome_id}.v{CACHE_VERSION}.pkl"


def load_base(genome_id: str, cache: bool = True, db=None) -> BaseData:
    """Phenotype and data of a genome (database read only). The explanation stored in the
    database is not read here: see :func:`load_db_operations`."""
    genome_id = str(genome_id)
    path = _cache_path(genome_id)
    if cache and path.is_file():
        try:
            with path.open("rb") as f:
                base = pickle.load(f)
            if isinstance(base, BaseData) and base.genome_id == genome_id:
                return base
        except Exception:
            pass
    base = _load_base_db(genome_id, db)
    if cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(base, f)
            tmp.replace(path)
        except Exception:
            pass
    return base


def _load_base_db(genome_id: str, db=None) -> BaseData:
    from sqlalchemy import text

    from mcp_server.helpers import load_genome_and_config, load_split_data
    from mcp_server.server import get_db

    from ..core.explaneat import ExplaNEAT
    from ..db.models import Dataset, DatasetSplit, Experiment, Genome, Population

    db = db or get_db()
    with db.session_scope() as s:
        if s.bind.dialect.name == "postgresql":
            s.execute(text("SET TRANSACTION READ ONLY"))
        g = s.get(Genome, uuid.UUID(genome_id))
        if g is None:
            raise ValueError(f"Genome not found: {genome_id}")
        pop = s.get(Population, g.population_id)
        exp = s.get(Experiment, pop.experiment_id)
        ds = s.get(Dataset, exp.dataset_id)
        split = s.get(DatasetSplit, exp.split_id)
        neat_genome, config, _ = load_genome_and_config(s, genome_id)
        phenotype = ExplaNEAT(neat_genome, config).get_phenotype_network()
        X_tr, y_tr, names, *_ = load_split_data(s, str(split.id), "train", 1.0, 10**9)
        try:
            X_te, y_te, *_ = load_split_data(s, str(split.id), "test", 1.0, 10**9)
        except ValueError:
            X_te, y_te = None, None
        X_full, _ = ds.get_data()
        raw_tr = np.asarray(X_full[split.train_indices], dtype=float)
        raw_te = np.asarray(X_full[split.test_indices], dtype=float) if split.test_indices else None
        if split.scaler_type != "StandardScaler":
            raise ValueError(f"scaler {split.scaler_type!r} is not supported (StandardScaler only)")
        meta = {"genome_id": genome_id,
                "dataset": {"id": str(ds.id), "name": ds.name,
                            "class_names": list(ds.class_names or [])},
                "split": {"id": str(split.id), "scaler_type": split.scaler_type},
                "experiment": {"id": str(exp.id), "name": exp.name}}
        return BaseData(
            genome_id=genome_id, phenotype=phenotype,
            X_train=np.asarray(X_tr, dtype=float), y_train=np.asarray(y_tr, dtype=float).ravel(),
            X_test=None if X_te is None else np.asarray(X_te, dtype=float),
            y_test=None if y_te is None else np.asarray(y_te, dtype=float).ravel(),
            feature_names=list(names or ds.feature_names), raw_train=raw_tr, raw_test=raw_te,
            mean=np.asarray(split.scaler_params["mean"], dtype=float),
            scale=np.asarray(split.scaler_params["scale"], dtype=float),
            dataset_name=ds.name, meta=meta)


def load_db_operations(genome_id: str, db=None) -> List[Dict[str, Any]]:
    """The operations of the explanation stored in the database (read only). Used only when a
    target has no operations of its own."""
    from sqlalchemy import text

    from mcp_server.server import get_db

    from ..db.models import Explanation

    db = db or get_db()
    with db.session_scope() as s:
        if s.bind.dialect.name == "postgresql":
            s.execute(text("SET TRANSACTION READ ONLY"))
        ex = s.query(Explanation).filter(Explanation.genome_id == uuid.UUID(str(genome_id))).first()
        ops = copy.deepcopy(list(ex.operations or [])) if ex is not None else []
    return json.loads(json.dumps(ops, default=str))


def build_engine_from(phenotype, operations: Sequence[Dict[str, Any]]):
    """A ``ModelStateEngine`` on a copy of the phenotype with ``operations`` replayed."""
    from ..core.model_state import ModelStateEngine

    engine = ModelStateEngine(copy.deepcopy(phenotype))
    engine.load_operations({"operations": copy.deepcopy(list(operations))})
    return engine


def codebook_for(base: BaseData):
    from ..analysis.plain_language_codebooks import (codebook_checks, find_codebook,
                                                     generic_codebook)
    if getattr(base, "codebook", None) is not None:
        return base.codebook
    names = base.feature_names
    cb = find_codebook(base.dataset_name, names) or generic_codebook(names)
    failed = [c for c in codebook_checks(cb, names, base.raw_train, base.y_train) if not c["passed"]]
    if failed:   # same rule as plain_language.load_model
        note = ("codebook " + str(cb.get("id")) + " not used: its checks failed ("
                + "; ".join(c["check"] for c in failed) + ")")
        cb = generic_codebook(names)
        cb["gaps"] = cb["gaps"] + [note]
    return cb


class VerifyModel:
    """A genome's model with an explanation's operations replayed (in memory)."""

    def __init__(self, base: BaseData, operations: Optional[Sequence[Dict[str, Any]]] = None):
        from ..analysis.plain_language import PlainModel

        self.base = base
        if operations is None:   # only now, and only for a target without its own operations
            operations = base.db_operations or load_db_operations(base.genome_id)
            self.operations_source = "database"
        else:
            self.operations_source = "given"
        self.operations = copy.deepcopy(list(operations))
        self.engine = build_engine_from(base.phenotype, self.operations)
        self.structure = self.engine.current_state
        self.m = PlainModel(self.structure, self.operations, base.X_train, base.y_train,
                            base.X_test, base.y_test, base.feature_names, base.mean, base.scale,
                            codebook_for(base), raw_train=base.raw_train, raw_test=base.raw_test)
        self._parts = None

    @property
    def genome_id(self) -> str:
        return self.base.genome_id

    @property
    def parts(self):
        """The annotations as plain-language parts (with baselines for ablation)."""
        if self._parts is None:
            from ..analysis.plain_language import build_parts
            self._parts = build_parts(self.m)
        return self._parts

    def part(self, ref: str, titles: Optional[Dict[str, str]] = None):
        """A part by id (``ann_31``), name, or plain title (case-insensitive)."""
        ref_l = str(ref).strip().lower()
        for p in self.parts:
            if ref_l in (p.id.lower(), p.name.lower()):
                return p
        for pid, t in (titles or {}).items():
            if str(t).strip().lower() == ref_l:
                for p in self.parts:
                    if p.id == pid:
                        return p
        known = ", ".join(f"{p.id} ({p.name})" for p in self.parts)
        raise KeyError(f"no part {ref!r}; parts: {known}")


def load(genome_id: str, operations: Optional[Sequence[Dict[str, Any]]] = None,
         cache: bool = True, db=None) -> VerifyModel:
    return VerifyModel(load_base(genome_id, cache=cache, db=db), operations)
