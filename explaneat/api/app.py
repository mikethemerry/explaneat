"""
FastAPI application for ExplaNEAT.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import experiments, genomes, operations, analysis, datasets, evidence, training, config_templates


logger = logging.getLogger(__name__)


def mark_orphaned_experiments_interrupted() -> int:
    """Mark any experiments stuck in 'running' state as 'interrupted'.

    Called on app startup to recover from server crashes. Returns the
    number of experiments updated.
    """
    from datetime import datetime
    from ..db.base import db
    from ..db.models import Experiment

    with db.session_scope() as session:
        orphans = session.query(Experiment).filter_by(status="running").all()
        for exp in orphans:
            exp.status = "interrupted"
            exp.end_time = datetime.utcnow()
        return len(orphans)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # FastAPI's Swagger / ReDoc pages load their JavaScript and CSS from a
    # public CDN, so they are off unless EXPLANEAT_API_DOCS=1. Restricted
    # environments get a portal that makes no external requests; the schema
    # itself stays available at /api/openapi.json.
    import os

    docs_on = os.environ.get("EXPLANEAT_API_DOCS", "0") == "1"
    app = FastAPI(
        title="ExplaNEAT API",
        description="REST API for managing explanations of NEAT neural networks",
        version="0.1.0",
        docs_url="/api/docs" if docs_on else None,
        redoc_url="/api/redoc" if docs_on else None,
        openapi_url="/api/openapi.json",
    )

    # Configure CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        experiments.router,
        prefix="/api/experiments",
        tags=["experiments"],
    )
    app.include_router(
        genomes.router,
        prefix="/api/genomes",
        tags=["genomes"],
    )
    app.include_router(
        operations.router,
        prefix="/api/genomes/{genome_id}",
        tags=["operations"],
    )
    app.include_router(
        analysis.router,
        prefix="/api/genomes/{genome_id}/analyze",
        tags=["analysis"],
    )
    app.include_router(
        datasets.router,
        prefix="/api/datasets",
        tags=["datasets"],
    )
    app.include_router(
        evidence.router,
        prefix="/api/genomes/{genome_id}/evidence",
        tags=["evidence"],
    )
    app.include_router(
        training.router,
        prefix="/api/genomes/{genome_id}",
        tags=["training"],
    )
    app.include_router(
        config_templates.router,
        prefix="/api/config-templates",
        tags=["config-templates"],
    )

    @app.on_event("startup")
    async def on_startup():
        # Experiments launched by other processes (e.g. experiments/10seed/
        # run_seed.py) are also status "running"; marking them interrupted
        # would corrupt a live run. `explaneat api` turns this off by default
        # (EXPLANEAT_MARK_ORPHANS=0); pass --mark-orphans to opt in.
        import os
        if os.environ.get("EXPLANEAT_MARK_ORPHANS", "1") == "0":
            logger.info("Skipping orphaned-experiment check (EXPLANEAT_MARK_ORPHANS=0)")
            return
        count = mark_orphaned_experiments_interrupted()
        if count:
            logger.info("Marked %d orphaned experiments as interrupted", count)

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    _mount_react_explorer(app)

    return app


def _mount_react_explorer(app: FastAPI) -> None:
    """Serve the built React Explorer (explaneat/static/react_explorer) at /.

    ``explaneat setup`` deploys ``web/react-explorer/dist`` there. Every asset
    is local: the page loads nothing from a CDN. Paths under /api are left to
    the routers (unknown ones still 404 as JSON).
    """
    from pathlib import Path

    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    ui_dir = Path(__file__).resolve().parent.parent / "static" / "react_explorer"
    index = ui_dir / "index.html"
    if not index.is_file():
        logger.info("React Explorer not built (%s missing); serving the API only", index)
        return
    if (ui_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=ui_dir / "assets"), name="ui-assets")

    @app.get("/", include_in_schema=False)
    async def ui_index():
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    async def ui_fallback(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (ui_dir / path).resolve()
        if candidate.is_file() and ui_dir.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)  # client-side routes


# Create default app instance
app = create_app()
