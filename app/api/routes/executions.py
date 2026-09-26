"""
Execution history and real-time streaming endpoints.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status

from app.api.dependencies import (
    get_pipeline_compiler,
    get_pipeline_executor,
    get_pipeline_validator,
)
from app.api.schemas.execution import DirectExecutionRequest, ExecutionRecordResponse
from app.domain.pipelines.models import PipelineExecutionRecord
from app.pipelines.compiler import PipelineCompiler
from app.pipelines.executor import EventType, ExecutionEvent, PipelineExecutor
from app.pipelines.validator import PipelineValidator

router = APIRouter(prefix="/executions", tags=["Executions"])

# In-memory execution store: execution_id -> PipelineExecutionRecord (LRU cap 100)
_EXECUTION_STORE: OrderedDict[str, PipelineExecutionRecord] = OrderedDict()
_MAX_STORED_EXECUTIONS = 100

# Active websocket subscriptions: execution_id -> set of asyncio.Queue
_EVENT_LISTENERS: dict[str, set[asyncio.Queue]] = {}


def store_execution_record(record: PipelineExecutionRecord) -> None:
    """Store an execution record in the in-memory cache."""
    _EXECUTION_STORE[record.execution_id] = record
    if len(_EXECUTION_STORE) > _MAX_STORED_EXECUTIONS:
        _EXECUTION_STORE.popitem(last=False)


def broadcast_event(event: ExecutionEvent) -> None:
    """Send an event to all connected WebSocket subscribers for this execution."""
    queues = _EVENT_LISTENERS.get(event.execution_id, set())
    for q in queues:
        q.put_nowait(event.to_dict())


@router.get("", summary="List recent executions")
async def list_executions(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent pipeline execution records, newest first."""
    records = list(_EXECUTION_STORE.values())[::-1][:limit]
    return [r.model_dump() for r in records]


@router.get("/{execution_id}", summary="Get execution record by ID")
async def get_execution(execution_id: str) -> dict[str, Any]:
    """Retrieve full execution record including node timings and error details."""
    record = _EXECUTION_STORE.get(execution_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found.",
        )
    return record.model_dump()


@router.post("/run", summary="Execute an unsaved pipeline configuration directly")
async def run_direct(
    payload: DirectExecutionRequest,
    validator: PipelineValidator = Depends(get_pipeline_validator),
    compiler: PipelineCompiler = Depends(get_pipeline_compiler),
    executor: PipelineExecutor = Depends(get_pipeline_executor),
) -> dict[str, Any]:
    """
    Directly execute an in-memory pipeline config without saving to disk first.
    Ideal for 'Run' button on the canvas while designing.
    """
    validation = validator.validate(payload.pipeline)
    validation.raise_if_invalid()

    compiled = compiler.compile(payload.pipeline)

    record: PipelineExecutionRecord | None = None
    async for event in executor.stream_execute(compiled, inputs=payload.inputs):
        broadcast_event(event)
        if event.event_type in (EventType.PIPELINE_COMPLETED, EventType.PIPELINE_FAILED):
            record = event.data.get("record")

    if record:
        store_execution_record(record)
        return record.model_dump()

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Pipeline execution finished without producing a record.",
    )


@router.websocket("/ws/{execution_id}")
async def websocket_execution_stream(websocket: WebSocket, execution_id: str):
    """
    WebSocket channel streaming ExecutionEvent objects in real time for a running pipeline.
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()

    if execution_id not in _EVENT_LISTENERS:
        _EVENT_LISTENERS[execution_id] = set()
    _EVENT_LISTENERS[execution_id].add(queue)

    try:
        while True:
            event_data = await queue.get()
            await websocket.send_json(event_data)
            if event_data.get("type") in ("pipeline_completed", "pipeline_failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if execution_id in _EVENT_LISTENERS:
            _EVENT_LISTENERS[execution_id].discard(queue)
            if not _EVENT_LISTENERS[execution_id]:
                del _EVENT_LISTENERS[execution_id]
