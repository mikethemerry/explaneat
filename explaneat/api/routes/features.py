"""Features of interest ("pins") made while exploring a model. File-backed, never the DB:
see ``explaneat.analysis.feature_pins``."""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...analysis import feature_pins as fp

router = APIRouter()


class Scope(BaseModel):
    kind: str = "model"  # model | annotation | node | input
    annotation_id: Optional[str] = None
    annotation_name: Optional[str] = None
    node_id: Optional[str] = None


class PinInput(BaseModel):
    node_id: Optional[str] = None
    raw: Optional[str] = None
    label: Optional[str] = None
    index: Optional[int] = None  # position among the scope's entry nodes when pinned


class VizView(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    split_choice: Optional[str] = None
    sample_fraction: Optional[float] = None
    view: Optional[str] = None
    dataset_split_id: Optional[str] = None
    input_nodes: Optional[List[str]] = None  # node ids behind the input dims (reopen by node, not index)
    output_node: Optional[str] = None
    description: Optional[str] = None


class PinCreate(BaseModel):
    genome_id: str
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    record_label: Optional[str] = None
    scope: Scope = Field(default_factory=Scope)
    inputs: List[PinInput] = Field(default_factory=list)
    viz: Optional[VizView] = None
    also_views: List[VizView] = Field(default_factory=list)
    note: str = ""
    tags: List[str] = Field(default_factory=list)
    status: str = "open"
    dossier_path: Optional[str] = None


class PinUpdate(BaseModel):
    note: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    dossier_path: Optional[str] = None
    viz: Optional[VizView] = None
    also_views: Optional[List[VizView]] = None
    inputs: Optional[List[PinInput]] = None
    scope: Optional[Scope] = None
    record_label: Optional[str] = None
    edit: str = ""  # optional description of the edit, stored on the new version


def _fill_context(fields: Dict[str, Any]) -> None:
    """Record label from the explanation records, experiment from the DB, when not given."""
    gid = fields["genome_id"]
    if not fields.get("record_label"):
        try:
            from ...analysis import explanation_records as er

            rec = er.find_record_for_genome(gid)
            if rec:
                fields["record_label"] = rec.get("label")
        except Exception:
            pass
    if not fields.get("experiment_id") or not fields.get("experiment_name"):
        try:  # read-only lookup
            from ...db import db
            from ...db.models import Genome

            with db.session_scope() as s:
                g = s.get(Genome, uuid.UUID(str(gid)))
                if g is not None:
                    exp = g.population.experiment
                    fields["experiment_id"] = fields.get("experiment_id") or str(exp.id)
                    fields["experiment_name"] = fields.get("experiment_name") or exp.name
        except Exception:
            pass


@router.get("")
async def list_features(genome_id: Optional[str] = None, label: Optional[str] = None,
                        tag: Optional[str] = None, status: Optional[str] = None,
                        q: Optional[str] = None):
    pins = fp.list_pins(genome_id=genome_id, label=label, tag=tag, status=status, q=q)
    out = {"features_dir": str(fp.store_dir()), "statuses": list(fp.STATUSES), "pins": pins}
    if genome_id:  # the genome's explanation-record label, for new pins and dossier paths
        ctx = {"genome_id": genome_id, "experiment_id": "-", "experiment_name": "-"}
        _fill_context(ctx)
        out["record_label"] = ctx.get("record_label")
    return out


@router.post("")
async def create_feature(req: PinCreate):
    fields = req.model_dump()
    _fill_context(fields)
    try:
        return fp.create_pin(fields)
    except fp.PinError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{pin_id}")
async def get_feature(pin_id: str, history: bool = True):
    try:
        return fp.get_pin(pin_id, history=history)
    except (FileNotFoundError, fp.PinError):
        raise HTTPException(status_code=404, detail=f"Pin {pin_id!r} not found")


@router.patch("/{pin_id}")
async def update_feature(pin_id: str, req: PinUpdate):
    changes = req.model_dump(exclude_unset=True)
    edit = changes.pop("edit", "") or ""
    try:
        return fp.update_pin(pin_id, changes, edit=edit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pin {pin_id!r} not found")
    except fp.PinError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{pin_id}/dossier")
async def get_dossier(pin_id: str):
    try:
        pin = fp.get_pin(pin_id)
    except (FileNotFoundError, fp.PinError):
        raise HTTPException(status_code=404, detail=f"Pin {pin_id!r} not found")
    d = fp.dossier_dir(pin)
    return {"path": pin.get("dossier_path"), "resolved": str(d) if d else None,
            "exists": pin["dossier_exists"], "files": fp.dossier_files(pin),
            "command": pin["dossier_command"]}


@router.get("/{pin_id}/dossier/{file_path:path}")
async def get_dossier_file(pin_id: str, file_path: str):
    try:
        pin = fp.get_pin(pin_id)
    except (FileNotFoundError, fp.PinError):
        raise HTTPException(status_code=404, detail=f"Pin {pin_id!r} not found")
    d = fp.dossier_dir(pin)
    if not d or not d.is_dir():
        raise HTTPException(status_code=404, detail="No dossier yet")
    target = (d / file_path).resolve()
    if d.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target)
