"""Portal workflow routes: local CSV datasets, seed batches, results tables,
and hazard-ratio readings for survival models (``explaneat.batch``)."""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ...batch import common
from ...batch.common import StorageError
from ...db import db

router = APIRouter()


def _err(e: Exception, code: int = 400):
    raise HTTPException(status_code=code, detail=str(e))


# ----------------------------------------------------------------- datasets


class PreviewRequest(BaseModel):
    path: str


@router.post("/csv/preview")
async def csv_preview(req: PreviewRequest):
    """Column summary of a local CSV/Parquet file (no raw rows)."""
    from ...batch.datasets import preview_csv

    try:
        return preview_csv(req.path)
    except FileNotFoundError as e:
        _err(e, 404)
    except Exception as e:
        _err(e)


class ImportRequest(BaseModel):
    path: str
    name: str
    task: str = Field(pattern="^(survival|classification)$")
    features: Optional[List[str]] = None
    categorical: Optional[List[str]] = None
    impute: Optional[str] = Field(default=None, pattern="^median$")
    time_col: Optional[str] = None
    event_col: Optional[str] = None
    event_value: Any = 1
    target_col: Optional[str] = None
    positive_value: Any = 1
    time_unit: Optional[str] = None
    restricted: bool = False
    test_size: float = Field(default=0.25, gt=0, lt=1)
    val_size: float = Field(default=0.20, ge=0, lt=1)
    split_seed: int = 0


@router.post("/datasets/csv")
async def csv_import(req: ImportRequest):
    """Import a local CSV as a dataset with a stratified train/val/test split."""
    from ...batch.datasets import import_csv

    try:
        return import_csv(**req.model_dump())
    except StorageError as e:
        _err(e, 403)
    except FileNotFoundError as e:
        _err(e, 404)
    except ValueError as e:
        _err(e)


@router.get("/datasets")
async def portal_datasets():
    """Datasets with their task, restricted flag and splits (for the batch form)."""
    from ...db.models import Dataset, DatasetSplit

    out = []
    with db.session_scope() as s:
        for ds in s.query(Dataset).order_by(Dataset.created_at.desc()).limit(300):
            meta = ds.additional_metadata or {}
            splits = s.query(DatasetSplit).filter(DatasetSplit.dataset_id == ds.id).all()
            out.append({
                "id": str(ds.id), "name": ds.name, "task": common.task_of_dataset(ds),
                "restricted": bool(meta.get("restricted")), "n_rows": ds.num_samples,
                "n_features": ds.num_features, "time_unit": meta.get("time_unit"),
                "splits": [{"id": str(sp.id), "train": sp.train_size, "val": sp.validation_size,
                            "test": sp.test_size_actual, "seed": sp.random_state,
                            "has_val": bool(sp.validation_indices)} for sp in splits],
            })
    return {"datasets": out}


@router.get("/templates")
async def templates():
    """Built-in experiment templates (survival Cox protocol v3, classification)."""
    return {"templates": common.list_templates()}


# ----------------------------------------------------------------- batches


class BatchRequest(BaseModel):
    task: str = Field(pattern="^(survival|classification)$")
    dataset_split_id: str
    seeds: List[int] = Field(min_length=1, max_length=100)
    template: Optional[str] = None
    name_prefix: str = "batch"
    generations: Optional[int] = Field(default=None, ge=1, le=10000)
    pop_size: Optional[int] = Field(default=None, ge=2, le=10000)
    epochs: Optional[int] = Field(default=None, ge=0, le=10000)
    complexity_penalty: Optional[float] = Field(default=None, ge=0)
    max_concurrent: int = Field(default=1, ge=1, le=16)
    config_overrides: Optional[Dict[str, Any]] = None
    fitness: str = Field(default="auc", pattern="^(auc|bce)$")


def _check_split_task(split_id: str, task: str):
    from ...db.models import Dataset, DatasetSplit

    with db.session_scope() as s:
        sp = s.get(DatasetSplit, uuid.UUID(split_id))
        if sp is None:
            raise HTTPException(status_code=404, detail="split not found")
        ds = s.get(Dataset, sp.dataset_id)
        t = common.task_of_dataset(ds)
        if t != task:
            raise HTTPException(status_code=400, detail=f"the split's dataset is {t}, not {task}")
        if (ds.additional_metadata or {}).get("restricted"):
            try:
                common.check_restricted_storage(database_url=common.current_database_url())
            except StorageError as e:
                raise HTTPException(status_code=403, detail=str(e))


@router.post("/batches")
async def create_batch(req: BatchRequest):
    """Launch one background process per seed, at most ``max_concurrent`` at a time."""
    from ...batch.jobs import manager

    _check_split_task(req.dataset_split_id, req.task)
    try:
        return manager.create(**req.model_dump())
    except ValueError as e:
        _err(e)


@router.get("/batches")
async def list_batches():
    from ...batch.jobs import manager

    return {"batches": manager.list(), "jobs_dir": str(manager.root)}


def _get(batch_id):
    from ...batch.jobs import manager

    try:
        return manager.get(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    return _get(batch_id)


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, run_id: Optional[str] = None):
    """Cancel queued runs and stop running ones at the next generation boundary."""
    from ...batch.jobs import manager

    _get(batch_id)
    return manager.cancel(batch_id, run_id)


@router.post("/batches/{batch_id}/resume")
async def resume_batch(batch_id: str, run_id: Optional[str] = None):
    """Re-queue interrupted/cancelled/failed runs (continuing from their last generation)."""
    from ...batch.jobs import manager

    _get(batch_id)
    return manager.resume(batch_id, run_id)


@router.get("/batches/{batch_id}/results")
async def batch_results(batch_id: str, refresh: bool = False):
    """Test metrics per seed with bootstrap CIs, the comparator on the same split, and the summary."""
    from ...batch.results import batch_results as compute

    b = _get(batch_id)
    ids = [r["experiment_id"] for r in b["runs"] if r.get("experiment_id") and r["status"] == "completed"]
    res = compute(ids, refresh=refresh)
    res["batch_id"] = batch_id
    res["pending"] = [r["run_id"] for r in b["runs"] if r["status"] != "completed"]
    return res


@router.get("/batches/{batch_id}/results.csv", response_class=PlainTextResponse)
async def batch_results_csv(batch_id: str):
    from ...batch.results import batch_results as compute, results_csv

    b = _get(batch_id)
    ids = [r["experiment_id"] for r in b["runs"] if r.get("experiment_id") and r["status"] == "completed"]
    return PlainTextResponse(results_csv(compute(ids)), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="results-{batch_id}.csv"'})


@router.get("/results")
async def results_for_experiments(experiment_ids: str = Query(..., description="comma-separated experiment ids")):
    """Results table for any set of experiments (e.g. runs launched elsewhere)."""
    from ...batch.results import batch_results as compute

    return compute([e for e in experiment_ids.split(",") if e])


# ----------------------------------------------------------------- hazard ratios


@router.get("/genomes/{genome_id}/hazard-ratios")
async def hazard_ratios(genome_id: str = Path(...)):
    """Hazard-ratio reading of every term at the output of a survival model
    (direct edges per SD / per unit; ReLU hinges with thresholds), with each
    term's spread on the training rows, and the annotations whose exits feed
    the output (their log-HR contribution as hazard ratios vs the median)."""
    import numpy as np
    import torch

    from ...core.structure_network import StructureNetwork
    from ...db.models import Genome
    from ...survival.explain import contribution_summary, explain_output
    from .evidence import _build_engine

    with db.session_scope() as s:
        g = s.get(Genome, uuid.UUID(genome_id))
        if g is None:
            raise HTTPException(status_code=404, detail="genome not found")
        exp = g.population.experiment
        if exp.split_id is None:
            raise HTTPException(status_code=400, detail="the genome's experiment has no split")
        data = common.load_split_arrays(s, exp.split_id)
        if data["meta"].get("task_type") != "survival":
            raise HTTPException(status_code=400, detail="hazard ratios need a survival dataset")
        engine = _build_engine(s, genome_id)
    state = engine.current_state
    X = data["X_train"]
    terms = explain_output(engine.original_phenotype, X, scaler=data["scaler"], feature_names=data["feature_names"])
    out_id = str(state.output_node_ids[0])
    sn = StructureNetwork(state)
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation("relu")
    sn.forward(torch.as_tensor(X, dtype=torch.float64))
    ann_terms = []
    for a in engine.annotations:
        for ex in a.exit_nodes:
            w = sum(c.weight for c in state.connections
                    if c.enabled and str(c.from_node) == str(ex) and str(c.to_node) == out_id)
            if w == 0:
                continue
            try:
                acts = sn.get_node_activation(str(ex))
            except Exception:
                continue
            ann_terms.append({"annotation": a.name, "exit": str(ex), "output_weight": float(w),
                              "spread": contribution_summary(np.asarray(acts), float(w))})
    g_all = np.asarray(sn.get_node_activation(out_id)).ravel()
    return common.json_safe({
        "genome_id": genome_id, "time_unit": data["meta"].get("time_unit"),
        "log_risk_spread": contribution_summary(g_all, 1.0),
        "terms": terms, "annotation_terms": sorted(ann_terms, key=lambda t: -t["spread"]["log_hr_sd"]),
        "note": ("The output is the log-risk g(x): each additive term is a log hazard ratio, constant over "
                 "follow-up (proportional hazards, inherited from the Cox loss, not tested by it). "
                 "Spreads are on the training rows; HRs are against the median contribution."),
    })
