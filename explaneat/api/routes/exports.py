"""Export packs over the API: the same ``explaneat.export`` code as the CLI.

    POST /api/genomes/{genome_id}/export        {formats: [...], review_marks: bool} -> job
    POST /api/experiments/{experiment_id}/export {formats: [...], review_marks: bool} -> job
    GET  /api/exports/jobs/{job_id}             -> status, progress, manifest (download links)
    GET  /api/exports/capabilities              -> which formats this machine can produce
    GET  /api/exports/{export_id}/manifest
    GET  /api/exports/{export_id}/{file}        -> download

Output lives under $EXPLANEAT_HOME/exports (never in the repository). Exports hold the
model, the explanation and aggregate metrics only, and stay on this machine.
"""

from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...db import db
from ...export import service
from ...export.pack import ALL_FORMATS, capabilities

router = APIRouter()


class ExportRequest(BaseModel):
    formats: List[str] = Field(default_factory=lambda: ["json", "tex", "tikz", "pdf", "docx"])
    review_marks: bool = True


def _formats(req: ExportRequest) -> List[str]:
    bad = [f for f in req.formats if f not in ALL_FORMATS]
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown format(s): {bad}; choose from {list(ALL_FORMATS)}")
    fs = [f for f in ALL_FORMATS if f in req.formats]
    return fs if "json" in fs else ["json"] + fs   # the JSON is always written: it is the record


def _db():
    if not db.session_factory:
        db.init_db()
    return db


@router.post("/genomes/{genome_id}/export")
async def export_genome(genome_id: str, req: ExportRequest):
    job = service.start_job("genome", service.export_genome, genome_id=genome_id,
                            formats=_formats(req), review_marks=req.review_marks, db=_db())
    return {"job_id": job["job_id"], "status": job["status"]}


@router.post("/experiments/{experiment_id}/export")
async def export_experiment(experiment_id: str, req: ExportRequest):
    job = service.start_job("experiment", service.export_experiment, experiment_id=experiment_id,
                            formats=_formats(req), review_marks=req.review_marks, db=_db())
    return {"job_id": job["job_id"], "status": job["status"]}


@router.get("/exports/capabilities")
async def export_capabilities():
    return capabilities()


@router.get("/exports/jobs/{job_id}")
async def export_job(job_id: str):
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="export job not found")
    return job


@router.get("/exports/{export_id}/manifest")
async def export_manifest(export_id: str):
    import json
    try:
        p = service.file_for(export_id, "manifest.json")
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="export not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/exports/{export_id}/{file_path:path}")
async def export_file(export_id: str, file_path: str):
    try:
        p = service.file_for(export_id, file_path)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(p), filename=p.name)
