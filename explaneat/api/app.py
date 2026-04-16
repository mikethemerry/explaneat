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
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
        count = mark_orphaned_experiments_interrupted()
        if count:
            logger.info("Marked %d orphaned experiments as interrupted", count)

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


# Create default app instance
app = create_app()
