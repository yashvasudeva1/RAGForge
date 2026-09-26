"""
Pydantic schemas for execution endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.pipelines.models import PipelineConfig, PipelineExecutionRecord


class DirectExecutionRequest(BaseModel):
    """Run an ad-hoc or unsaved pipeline configuration directly."""

    pipeline: PipelineConfig
    inputs: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecordResponse(BaseModel):
    """Response containing an execution's status and details."""

    execution_id: str
    pipeline_id: str
    pipeline_name: str
    status: str
    started_at: str
    completed_at: str | None = None
    duration_ms: float | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    node_records: dict[str, Any] = Field(default_factory=dict)
