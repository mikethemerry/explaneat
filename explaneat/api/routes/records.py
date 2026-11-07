"""Explanation records (Claude-produced explanation reports) and Mike's
verification of them. Records are JSON files, not DB rows; see
``explaneat.analysis.explanation_records``."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...analysis import explanation_records as er
from ...db import db
from ...db.models import Genome

router = APIRouter()


def _experiment_of(genome_ids):
    out = {}
    with db.session_scope() as s:
        for gid in genome_ids:
            try:
                g = s.get(Genome, uuid.UUID(str(gid)))
            except (ValueError, TypeError):
                g = None
            if g is not None:
                exp = g.population.experiment
                out[str(gid)] = {"experiment_id": str(exp.id), "experiment_name": exp.name, "in_db": True}
    return out


@router.get("")
async def list_records():
    recs = er.list_records()
    try:
        exps = _experiment_of([r["genome_id"] for r in recs])
    except Exception:  # DB unavailable: records are still listable
        exps = {}
    for r in recs:
        r.update(exps.get(str(r["genome_id"]), {"in_db": False}))
    return {
        "records_dir": str(er.records_dir()),
        "records": recs,
        "n_verified": sum(1 for r in recs if r["verified"]),
    }


@router.get("/{label}")
async def get_record(label: str):
    try:
        return er.load_record(label)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Record {label!r} not found")


class VerificationRequest(BaseModel):
    verified: bool
    minutes: Optional[float] = Field(default=None, ge=0)
    changes: str = ""
    notes: str = ""


@router.put("/{label}/verification")
async def put_verification(label: str, req: VerificationRequest):
    """Write ``mike_verification`` into ``<records_dir>/<label>.json``.
    Rebuild the CSVs afterwards with ``experiments/10seed/build_explanations_csv.py``."""
    try:
        rec = er.write_verification(label, req.verified, req.minutes, req.changes, req.notes)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Record {label!r} not found")
    return {"label": label, "mike_verification": rec["mike_verification"]}
