"""
Pipeline management and execution endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import (
    get_pipeline_compiler,
    get_pipeline_executor,
    get_pipeline_serializer,
    get_pipeline_validator,
)
from app.api.schemas.pipeline import (
    PipelineCreateRequest,
    PipelineExecuteRequest,
    PipelineSummaryResponse,
    PipelineUpdateRequest,
    PipelineValidationResponse,
)
from app.core.exceptions import PipelineNotFoundError, PipelineValidationError
from app.domain.pipelines.models import PipelineConfig
from app.pipelines.compiler import PipelineCompiler
from app.pipelines.executor import PipelineExecutor
from app.pipelines.serializer import PipelineSerializer
from app.pipelines.templates import ALL_TEMPLATES
from app.pipelines.validator import PipelineValidator

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])


@router.get("", summary="List all saved pipelines")
async def list_pipelines(
    include_templates: bool = True,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
) -> list[dict[str, Any]]:
    """Return summaries of all saved pipelines and templates."""
    summaries = serializer.list_all(include_templates=include_templates)
    return [s.to_dict() for s in summaries]


@router.get("/templates", summary="List built-in starter templates")
async def list_templates() -> list[dict[str, Any]]:
    """Return all built-in pipeline templates."""
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "node_count": len(t.nodes),
            "edge_count": len(t.edges),
            "tags": t.tags,
            "is_template": True,
            "pipeline": t.model_dump(),
        }
        for t in ALL_TEMPLATES
    ]


@router.get("/templates/{template_id}", summary="Get a built-in template by ID")
async def get_template(template_id: str) -> dict[str, Any]:
    """Return full pipeline config for a built-in template."""
    for t in ALL_TEMPLATES:
        if t.id == template_id:
            return t.model_dump()
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Template with id '{template_id}' not found.",
    )


@router.post("", summary="Save a new pipeline", status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    payload: PipelineCreateRequest,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
) -> dict[str, Any]:
    """Create and persist a new pipeline configuration."""
    pipeline = PipelineConfig(
        name=payload.name,
        description=payload.description,
        nodes=payload.nodes,
        edges=payload.edges,
        tags=payload.tags,
    )
    serializer.save(pipeline)
    return pipeline.model_dump()


@router.get("/{pipeline_id}", summary="Load a pipeline by ID")
async def get_pipeline(
    pipeline_id: str,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
) -> dict[str, Any]:
    """Load and return a saved pipeline configuration."""
    try:
        pipeline = serializer.load(pipeline_id)
        return pipeline.model_dump()
    except PipelineNotFoundError:
        # Check templates as fallback
        for t in ALL_TEMPLATES:
            if t.id == pipeline_id:
                return t.model_dump()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{pipeline_id}' not found.",
        )


@router.put("/{pipeline_id}", summary="Update a pipeline")
async def update_pipeline(
    pipeline_id: str,
    payload: PipelineUpdateRequest,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
) -> dict[str, Any]:
    """Update fields on an existing pipeline."""
    try:
        pipeline = serializer.load(pipeline_id)
    except PipelineNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{pipeline_id}' not found.",
        )

    if payload.name is not None:
        pipeline.name = payload.name
    if payload.description is not None:
        pipeline.description = payload.description
    if payload.nodes is not None:
        pipeline.nodes = payload.nodes
    if payload.edges is not None:
        pipeline.edges = payload.edges
    if payload.tags is not None:
        pipeline.tags = payload.tags

    serializer.save(pipeline)
    return pipeline.model_dump()


@router.delete("/{pipeline_id}", summary="Delete a pipeline", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(
    pipeline_id: str,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
) -> None:
    """Delete a pipeline configuration from disk."""
    try:
        serializer.delete(pipeline_id)
    except PipelineNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{pipeline_id}' not found.",
        )


@router.post("/validate", summary="Validate a pipeline configuration")
async def validate_pipeline(
    pipeline: PipelineConfig,
    validator: PipelineValidator = Depends(get_pipeline_validator),
) -> PipelineValidationResponse:
    """Check a pipeline configuration for cycles, missing connections, and type mismatches."""
    result = validator.validate(pipeline)
    return PipelineValidationResponse(
        is_valid=result.is_valid,
        errors=result.errors,
        warnings=[
            {"code": w.code, "message": w.message, "node_id": w.node_id}
            for w in result.warnings
        ],
    )


@router.post("/{pipeline_id}/execute", summary="Execute a saved pipeline")
async def execute_pipeline(
    pipeline_id: str,
    payload: PipelineExecuteRequest,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
    validator: PipelineValidator = Depends(get_pipeline_validator),
    compiler: PipelineCompiler = Depends(get_pipeline_compiler),
    executor: PipelineExecutor = Depends(get_pipeline_executor),
) -> dict[str, Any]:
    """
    Run a pipeline by ID with optional runtime inputs.
    Returns the execution record with all outputs and per-node execution timings.
    """
    try:
        pipeline = serializer.load(pipeline_id)
    except PipelineNotFoundError:
        # Check templates
        template = next((t for t in ALL_TEMPLATES if t.id == pipeline_id), None)
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline '{pipeline_id}' not found.",
            )
        pipeline = template

    # Validate
    validation = validator.validate(pipeline)
    validation.raise_if_invalid()

    # Compile
    compiled = compiler.compile(pipeline)

    # Prepare inputs
    inputs = dict(payload.inputs)
    if payload.query:
        inputs["query"] = payload.query

    # Execute
    record = await executor.execute(compiled, inputs=inputs)
    return record.model_dump()


@router.get("/{pipeline_id}/export/yaml", summary="Export pipeline as YAML")
async def export_yaml(
    pipeline_id: str,
    serializer: PipelineSerializer = Depends(get_pipeline_serializer),
) -> Response:
    """Export pipeline definition in clean YAML format."""
    try:
        pipeline = serializer.load(pipeline_id)
    except PipelineNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{pipeline_id}' not found.",
        )
    yaml_content = serializer.to_yaml(pipeline)
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{pipeline.name.lower().replace(" ", "_")}.yaml"'},
    )
