"""Config template CRUD endpoints."""
import uuid
from fastapi import APIRouter, HTTPException

from ...db.base import db
from ...db.models import ConfigTemplate
from ..schemas import (
    ConfigTemplateResponse,
    ConfigTemplateListResponse,
    ConfigTemplateCreateRequest,
    ConfigTemplateUpdateRequest,
)


router = APIRouter()


@router.get("", response_model=ConfigTemplateListResponse)
async def list_templates():
    with db.session_scope() as session:
        templates = (
            session.query(ConfigTemplate)
            .order_by(ConfigTemplate.created_at.desc())
            .all()
        )
        return ConfigTemplateListResponse(
            templates=[ConfigTemplateResponse(**t.to_dict()) for t in templates],
            total=len(templates),
        )


@router.get("/{template_id}", response_model=ConfigTemplateResponse)
async def get_template(template_id: str):
    with db.session_scope() as session:
        template = (
            session.query(ConfigTemplate)
            .filter_by(id=uuid.UUID(template_id))
            .first()
        )
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return ConfigTemplateResponse(**template.to_dict())


@router.post("", response_model=ConfigTemplateResponse)
async def create_template(request: ConfigTemplateCreateRequest):
    with db.session_scope() as session:
        template = ConfigTemplate(
            name=request.name,
            description=request.description,
            config=request.config,
        )
        session.add(template)
        session.flush()
        return ConfigTemplateResponse(**template.to_dict())


@router.patch("/{template_id}", response_model=ConfigTemplateResponse)
async def update_template(template_id: str, request: ConfigTemplateUpdateRequest):
    with db.session_scope() as session:
        template = (
            session.query(ConfigTemplate)
            .filter_by(id=uuid.UUID(template_id))
            .first()
        )
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        if request.name is not None:
            template.name = request.name
        if request.description is not None:
            template.description = request.description
        if request.config is not None:
            template.config = request.config
        session.flush()
        return ConfigTemplateResponse(**template.to_dict())


@router.delete("/{template_id}")
async def delete_template(template_id: str):
    with db.session_scope() as session:
        template = (
            session.query(ConfigTemplate)
            .filter_by(id=uuid.UUID(template_id))
            .first()
        )
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        session.delete(template)
        return {"status": "deleted"}
