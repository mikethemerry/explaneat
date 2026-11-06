"""Explanation tools: automatic checks, linearisation / restore gain, unit
activity, candidate subgraphs, one-hot category offsets, and explanation
import/export.

All routes except ``POST /operations/import`` are read-only: they replay the
operations in memory (``evidence._build_engine`` never creates rows).
"""

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...analysis import explanation_tools as et
from ...analysis import explanation_records as er
from ...analysis.coverage_report import coverage_report
from ...core.explaneat import ExplaNEAT
from ...core.model_state import ModelStateEngine
from ...db import db
from ...db.models import Explanation, Genome
from .evidence import _build_engine, _load_genome_and_config, _load_split_data

router = APIRouter()

MAX_ROWS = 10**9


def _split_for(session, genome_id: str, split_id: Optional[str]) -> str:
    if split_id:
        return split_id
    g = session.query(Genome).filter_by(id=uuid.UUID(genome_id)).first()
    if not g:
        raise HTTPException(status_code=404, detail="Genome not found")
    exp = g.population.experiment
    if not exp.split_id:
        raise HTTPException(
            status_code=400,
            detail="The genome's experiment has no dataset split; pass dataset_split_id",
        )
    return str(exp.split_id)


def _operations(session, genome_id: str) -> List[Dict[str, Any]]:
    ex = session.query(Explanation).filter(Explanation.genome_id == uuid.UUID(genome_id)).first()
    return list(ex.operations or []) if ex else []


def _data(session, genome_id, split_id, split, max_samples=MAX_ROWS):
    sid = _split_for(session, genome_id, split_id)
    X, y, feature_names, *_ = _load_split_data(session, sid, split, 1.0, max_samples)
    return sid, X, y, feature_names or []


# ---------------------------------------------------------------- checks


@router.get("/check")
async def check(
    genome_id: str = Path(...),
    dataset_split_id: Optional[str] = None,
    split: str = Query("train", pattern="^(train|test|val|both)$"),
    closed_form: bool = True,
):
    """Function preservation, per-leaf closed-form error, hierarchy and
    coverage (the checks of ``experiments/10seed/check_explanation.py`` plus
    ``get_coverage``), on the training split by default."""
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        ops = _operations(session, genome_id)
        sid, X, y, _ = _data(session, genome_id, dataset_split_id, split)
        res = et.check_explanation(engine, ops, X, closed_form=closed_form)
        cov = coverage_report(engine.current_state, engine.annotations)
    res["split"] = split
    res["dataset_split_id"] = sid
    res["coverage"] = cov
    res["full_coverage"] = bool(
        cov["structural_coverage"] == 1.0
        and cov["compositional_coverage"] == 1.0
        and res["hierarchy"]["complete"]
        and res["function_preservation_max_abs_err"] < 1e-9
    )
    return res


class LineariseRequest(BaseModel):
    annotation_names: Optional[List[str]] = None
    dataset_split_id: Optional[str] = None
    split: str = "train"


@router.post("/linearise")
async def linearise(req: LineariseRequest, genome_id: str = Path(...)):
    """Restore-gain table: AUC with every nonlinear unit replaced by its
    least-squares affine fit (all linear), each unit linearised alone, and each
    unit's nonlinearity restored alone. Fits are made on the requested split."""
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        ops = _operations(session, genome_id)
        sid, X, y, _ = _data(session, genome_id, req.dataset_split_id, req.split)
    if not any(op.get("type") == "annotate" for op in ops):
        raise HTTPException(status_code=400, detail="The genome has no annotations")
    try:
        res = et.linearise_annotations(engine.current_state, ops, X, y, req.annotation_names or None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    res["metric"] = "auc"
    res["split"] = req.split
    res["dataset_split_id"] = sid
    return res


@router.get("/unit-activity")
async def unit_activity(
    genome_id: str = Path(...),
    dataset_split_id: Optional[str] = None,
    split: str = Query("train", pattern="^(train|test|val|both)$"),
):
    """Per hidden node: dead / rarely active / always on (affine) / hinge."""
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        sid, X, y, _ = _data(session, genome_id, dataset_split_id, split)
    owners = {}
    for a in engine.annotations:
        for n in a.subgraph_nodes:
            owners[str(n)] = a.name
    rows = et.unit_activity(engine.current_state, X)
    for r in rows:
        r["annotation"] = owners.get(r["node_id"])
    return {"split": split, "dataset_split_id": sid, "n_rows": int(X.shape[0]), "units": rows}


@router.get("/candidates")
async def candidates(genome_id: str = Path(...)):
    """Candidate leaf subgraphs from the topology, with the splits each needs."""
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
    return et.candidate_subgraphs(engine.current_state, engine.annotations)


@router.get("/category-offsets")
async def category_offsets(
    genome_id: str = Path(...),
    dataset_split_id: Optional[str] = None,
    split: str = Query("train", pattern="^(train|test|val|both)$"),
    large_effect: float = 1.0,
):
    """Per one-hot level: the model's logit effect vs the empirical offset,
    flagging large effects the data does not support (rare-category switches)."""
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        sid, X, y, names = _data(session, genome_id, dataset_split_id, split)
    if not names:
        raise HTTPException(status_code=400, detail="The dataset has no feature names")
    res = et.category_offsets(engine.current_state, X, y, names, large_effect=large_effect)
    res["split"] = split
    res["dataset_split_id"] = sid
    return res


# ---------------------------------------------------------------- records


@router.get("/record")
async def genome_record(genome_id: str = Path(...)):
    """The Claude-produced explanation record for this genome, if any."""
    rec = er.find_record_for_genome(genome_id)
    return {"record": rec, "records_dir": str(er.records_dir())}


# ---------------------------------------------------------------- import / export


@router.get("/operations/export")
async def export_operations(genome_id: str = Path(...)):
    """The explanation as a JSON document (operations event stream + record link)."""
    with db.session_scope() as session:
        g = session.query(Genome).filter_by(id=uuid.UUID(genome_id)).first()
        if not g:
            raise HTTPException(status_code=404, detail="Genome not found")
        ops = _operations(session, genome_id)
        exp = g.population.experiment
        doc = {
            "format": "explaneat-explanation/1",
            "genome_id": genome_id,
            "experiment_id": str(exp.id),
            "experiment_name": exp.name,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "operations": ops,
        }
    rec = er.find_record_for_genome(genome_id)
    if rec:
        doc["record_label"] = rec.get("label")
    return JSONResponse(
        doc,
        headers={"Content-Disposition": f'attachment; filename="explanation-{genome_id[:8]}.json"'},
    )


class ImportRequest(BaseModel):
    operations: Optional[List[Dict[str, Any]]] = None
    document: Optional[Dict[str, Any]] = None  # an exported document
    replace: bool = False
    dataset_split_id: Optional[str] = None


def _ops_from_request(req: ImportRequest) -> List[Dict[str, Any]]:
    ops = req.operations
    if ops is None and req.document is not None:
        ops = req.document.get("operations")
    if not isinstance(ops, list):
        raise HTTPException(status_code=400, detail="Provide operations (a list) or an exported document")
    return copy.deepcopy(ops)


def _replay(session, genome_id: str, ops):
    _, neat_genome, config = _load_genome_and_config(session, genome_id)
    phenotype = ExplaNEAT(neat_genome, config).get_phenotype_network()
    engine = ModelStateEngine(phenotype)
    try:
        engine.load_operations({"operations": ops})
        engine.current_state  # replay now, so errors surface here
    except Exception as e:  # invalid stream for this genome
        raise HTTPException(status_code=422, detail=f"Operations do not replay on this genome: {e}")
    return engine


@router.post("/operations/import-preview")
async def import_preview(req: ImportRequest, genome_id: str = Path(...)):
    """Replay imported operations in memory and run the checks. Nothing is saved."""
    ops = _ops_from_request(req)
    with db.session_scope() as session:
        engine = _replay(session, genome_id, ops)
        current = _operations(session, genome_id)
        sid, X, y, _ = _data(session, genome_id, req.dataset_split_id, "train")
        res = et.check_explanation(engine, ops, X)
        cov = coverage_report(engine.current_state, engine.annotations)
    res["coverage"] = cov
    res["n_operations"] = len(ops)
    res["n_existing_operations"] = len(current)
    res["identical_to_existing"] = _strip(current) == _strip(ops)
    return res


def _strip(ops):
    return [{"type": o.get("type"), "params": o.get("params")} for o in ops]


@router.post("/operations/import")
async def import_operations(req: ImportRequest, genome_id: str = Path(...)):
    """Save imported operations as the genome's explanation. Refuses to
    overwrite a non-empty explanation unless ``replace`` is true."""
    from ...db.explanation_lock import lock_explanation

    ops = _ops_from_request(req)
    with db.session_scope() as session:
        engine = _replay(session, genome_id, ops)
        ex = lock_explanation(session, genome_id, create=True)
        if ex.operations and not req.replace:
            raise HTTPException(
                status_code=409,
                detail=f"The genome already has {len(ex.operations)} operations; pass replace=true to overwrite",
            )
        ex.operations = [op.to_dict() for op in engine.operations]
        n = len(ex.operations)
    return {"status": "imported", "n_operations": n}


# ---------------------------------------------------------------- hypothesis


def _match(op, annotation_id):
    p = op.get("params", {}) or {}
    aid = (op.get("result") or {}).get("annotation_id") or f"ann_{op.get('seq', 0)}"
    return op.get("type") == "annotate" and annotation_id in (aid, p.get("name")), aid, p


@router.get("/annotations/{annotation_id}/hypothesis")
async def get_hypothesis(annotation_id: str, genome_id: str = Path(...)):
    """The annotation's hypothesis text and its evidence categories."""
    with db.session_scope() as session:
        for op in _operations(session, genome_id):
            ok, aid, p = _match(op, annotation_id)
            if ok:
                ev = p.get("evidence") or {}
                return {"annotation_id": aid, "name": p.get("name"),
                        "hypothesis": p.get("hypothesis") or "",
                        "evidence_counts": {k: len(v) for k, v in ev.items() if isinstance(v, list)}}
    raise HTTPException(status_code=404, detail=f"Annotation {annotation_id!r} not found")


class HypothesisRequest(BaseModel):
    hypothesis: str


@router.put("/annotations/{annotation_id}/hypothesis")
async def put_hypothesis(annotation_id: str, req: HypothesisRequest, genome_id: str = Path(...)):
    """Overwrite the annotation's hypothesis (as MCP ``update_narrative``)."""
    from sqlalchemy.orm.attributes import flag_modified

    from ...db.explanation_lock import lock_explanation

    with db.session_scope() as session:
        ex = lock_explanation(session, genome_id)
        ops = list(ex.operations or []) if ex else []
        for op in ops:
            ok, aid, p = _match(op, annotation_id)
            if ok:
                p["hypothesis"] = req.hypothesis
                op["params"] = p
                ex.operations = ops
                flag_modified(ex, "operations")
                session.flush()
                return {"status": "ok", "annotation_id": aid}
    raise HTTPException(status_code=404, detail=f"Annotation {annotation_id!r} not found")
