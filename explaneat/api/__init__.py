"""
FastAPI application for ExplaNEAT.

Provides REST API for managing explanations, operations, and model state.
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
