"""
RAGForge FastAPI application entry point.

Drag-and-drop RAG pipeline builder and experimentation engine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import (
    NodeNotFoundError,
    PipelineNotFoundError,
    PipelineValidationError,
    RAGForgeError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application startup and shutdown tasks.
    """
    settings = get_settings()

    # 1. Import all node classes to trigger self-registration in registry
    import app.nodes  # noqa: F401

    from app.core.registry import registry
    logger.info(
        "RAGForge starting up — %d nodes registered across 11 categories",
        len(registry.all_types()),
    )

    # 2. Ensure data directories exist
    pipeline_dir = Path(settings.pipeline_storage_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    upload_dir = pipeline_dir.parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    chroma_dir = pipeline_dir.parent / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("RAGForge shutting down.")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Visual RAG Pipeline Builder & Experimentation Engine. "
            "Build, test, and benchmark retrieval-augmented generation pipelines visually."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom exception handlers
    @app.exception_handler(PipelineNotFoundError)
    async def pipeline_not_found_handler(request: Request, exc: PipelineNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": exc.message, "details": exc.details},
        )

    @app.exception_handler(PipelineValidationError)
    async def pipeline_validation_handler(request: Request, exc: PipelineValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": exc.message, "details": exc.details},
        )

    @app.exception_handler(NodeNotFoundError)
    async def node_not_found_handler(request: Request, exc: NodeNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": exc.message, "details": exc.details},
        )

    @app.exception_handler(RAGForgeError)
    async def ragforge_error_handler(request: Request, exc: RAGForgeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": exc.message, "details": exc.details},
        )

    # Attach master API router
    app.include_router(api_router)

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "api": "/api",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
