"""
Health check endpoints.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.registry import registry
from app.pipelines.templates import ALL_TEMPLATES

router = APIRouter(prefix="/health", tags=["Health"])

_START_TIME = time.time()


@router.get("", summary="System health check")
async def health_check() -> dict[str, Any]:
    """
    Return operational status, uptime, app version, and component metrics.
    """
    settings = get_settings()
    uptime_seconds = int(time.time() - _START_TIME)

    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": uptime_seconds,
        "registered_nodes_count": len(registry.all_types()),
        "templates_count": len(ALL_TEMPLATES),
    }
