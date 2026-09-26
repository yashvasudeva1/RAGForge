"""
Master API router registering all endpoints under /api.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    documents,
    executions,
    generation,
    health,
    models,
    pipelines,
    retrieval,
)

api_router = APIRouter(prefix="/api")

# Attach all sub-routers
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(pipelines.router)
api_router.include_router(executions.router)
api_router.include_router(documents.router)
api_router.include_router(retrieval.router)
api_router.include_router(generation.router)
