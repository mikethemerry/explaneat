"""
FastAPI application for ExplaNEAT.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import genomes, operations, analysis


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

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


# Create default app instance
app = create_app()
