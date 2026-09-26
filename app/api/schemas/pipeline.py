"""
Pydantic request and response schemas for pipeline endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.pipelines.models import EdgeConfig, NodeConfig, PipelineConfig


class PipelineCreateRequest(BaseModel):
    """Payload to create a new pipeline."""

    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    nodes: list[NodeConfig] = Field(default_factory=list)
    edges: list[EdgeConfig] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PipelineUpdateRequest(BaseModel):
    """Payload to update an existing pipeline."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    nodes: list[NodeConfig] | None = None
    edges: list[EdgeConfig] | None = None
    tags: list[str] | None = None


class PipelineValidationWarningResponse(BaseModel):
    code: str
    message: str
    node_id: str | None = None


class PipelineValidationResponse(BaseModel):
    """Result of pipeline validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[PipelineValidationWarningResponse] = Field(default_factory=list)


class PipelineExecuteRequest(BaseModel):
    """Input payload to trigger a pipeline run."""

    query: str | None = Field(default=None, description="User question / query string.")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Arbitrary port/node input values.")


class PipelineSummaryResponse(BaseModel):
    """Summary representation for pipeline lists."""

    id: str
    name: str
    description: str
    node_count: int
    edge_count: int
    tags: list[str]
    is_template: bool
    created_at: str
    updated_at: str
