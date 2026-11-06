"""
FastAPI application for ExplaNEAT.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import experiments, genomes, operations, analysis, datasets, evidence, training, config_templates
from .routes import explain_tools, records, lineage


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

    app = FastAPI(
        title="ExplaNEAT API",
        description="REST API for managing explanations of NEAT neural networks",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
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

    app.include_router(
        explain_tools.router,
        prefix="/api/genomes/{genome_id}/tools",
        tags=["explanation-tools"],
    )
    app.include_router(records.router, prefix="/api/records", tags=["records"])
    app.include_router(lineage.router, prefix="/api/genomes", tags=["lineage"])

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/api/governance")
    async def governance_status():
        """Data-governance settings (restricted mode, assisted-path opt-in)."""
        from .. import governance

        return governance.status()

    _mount_frontend(app)
    return app


STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "react_explorer"


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React Explorer at / (SPA: unknown paths get index.html).

    Build it with ``npm run build`` in web/react-explorer and copy
    ``web/react-explorer/dist/*`` into ``explaneat/static/react_explorer/``.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="Not found")
        index = STATIC_DIR / "index.html"
        if full_path:
            candidate = (STATIC_DIR / full_path).resolve()
            if STATIC_DIR.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(
            "<h1>ExplaNEAT API</h1><p>The React Explorer is not built. Run "
            "<code>npm install && npm run build</code> in <code>web/react-explorer</code>, then copy "
            "<code>web/react-explorer/dist/*</code> into <code>explaneat/static/react_explorer/</code>. "
            "API docs: "
            "<a href='/api/docs'>/api/docs</a>.</p>",
            status_code=200,
        )


# Create default app instance
app = create_app()
