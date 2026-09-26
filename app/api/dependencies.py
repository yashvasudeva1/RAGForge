"""
API dependencies for FastAPI routes.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.ingestion.manager import IngestionManager
from app.pipelines.compiler import PipelineCompiler
from app.pipelines.executor import PipelineExecutor
from app.pipelines.serializer import PipelineSerializer
from app.pipelines.validator import PipelineValidator


@lru_cache
def get_pipeline_serializer() -> PipelineSerializer:
    """Return the cached PipelineSerializer singleton."""
    return PipelineSerializer()


@lru_cache
def get_pipeline_validator() -> PipelineValidator:
    """Return the cached PipelineValidator singleton."""
    return PipelineValidator()


@lru_cache
def get_pipeline_compiler() -> PipelineCompiler:
    """Return the cached PipelineCompiler singleton."""
    return PipelineCompiler()


@lru_cache
def get_pipeline_executor() -> PipelineExecutor:
    """Return a PipelineExecutor instance."""
    settings = get_settings()
    return PipelineExecutor(timeout_seconds=settings.pipeline_timeout_seconds)


@lru_cache
def get_ingestion_manager() -> IngestionManager:
    """Return the IngestionManager singleton."""
    return IngestionManager()
