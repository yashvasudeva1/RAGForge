"""
Pipeline executor for RAGForge.

The executor takes a ``CompiledPipeline`` and runs it node by node,
collecting outputs and routing them to downstream nodes via port bindings.

Features
--------
- Async execution — each node's ``execute()`` is awaited.
- Streaming events — after each node completes (or fails), an
  ``ExecutionEvent`` is emitted.  The API layer pushes these over WebSocket
  so the canvas can update node status badges in real time.
- Timeout enforcement — the entire pipeline may not exceed
  ``settings.pipeline_timeout_seconds``.
- Per-node error isolation — a failed node records its error and marks
  all downstream nodes as SKIPPED; execution continues for independent
  branches.
- Pipeline-level inputs (e.g. ``{"query": "..."}`` from the frontend)
  are injected into QueryInputNode configs before execution starts.

Usage
-----
    executor = PipelineExecutor()

    async for event in executor.stream_execute(compiled, inputs={"query": "What is RAG?"}):
        # event.type is one of: NODE_STARTED, NODE_COMPLETED, NODE_FAILED,
        #                        PIPELINE_COMPLETED, PIPELINE_FAILED
        print(event)

    # Or, non-streaming:
    record = await executor.execute(compiled, inputs={"query": "..."})
"""

from __future__ import annotations

import asyncio
import time
from asyncio import Task
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator

from app.core.constants import ExecutionStatus, NodeExecutionStatus
from app.core.exceptions import NodeExecutionError, PipelineExecutionError
from app.core.logging import get_logger
from app.domain.pipelines.models import (
    NodeExecutionRecord,
    PipelineExecutionRecord,
)
from app.pipelines.compiler import CompiledNode, CompiledPipeline

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Execution events (streamed over WebSocket)
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    PIPELINE_STARTED = "pipeline_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"


class ExecutionEvent:
    """
    A single streaming event emitted during pipeline execution.

    The API layer serialises these as JSON and sends them over WebSocket.
    """

    def __init__(
        self,
        event_type: EventType,
        execution_id: str,
        node_id: str | None = None,
        node_type: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.event_type = event_type
        self.execution_id = execution_id
        self.node_id = node_id
        self.node_type = node_type
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.data: dict[str, Any] = data or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def __repr__(self) -> str:
        return f"ExecutionEvent({self.event_type.value}, node={self.node_id})"


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class PipelineExecutor:
    """
    Runs a ``CompiledPipeline`` and streams ``ExecutionEvent`` objects.

    Parameters
    ----------
    timeout_seconds:
        Maximum wall-clock time for the entire pipeline.
        Defaults to 300 seconds.
    """

    def __init__(self, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        compiled: CompiledPipeline,
        inputs: dict[str, Any] | None = None,
    ) -> PipelineExecutionRecord:
        """
        Run the pipeline synchronously (consuming all streamed events).

        Returns the final ``PipelineExecutionRecord`` with per-node timing
        and outputs.
        """
        record: PipelineExecutionRecord | None = None
        async for event in self.stream_execute(compiled, inputs=inputs):
            if event.event_type in (
                EventType.PIPELINE_COMPLETED,
                EventType.PIPELINE_FAILED,
            ):
                record = event.data.get("record")
        if record is None:
            raise PipelineExecutionError("Execution did not produce a final record.")
        return record

    async def stream_execute(
        self,
        compiled: CompiledPipeline,
        inputs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ExecutionEvent, None]:
        """
        Execute the pipeline and yield ``ExecutionEvent`` objects as each
        node starts, completes, or fails.

        Parameters
        ----------
        compiled:
            The output of ``PipelineCompiler.compile()``.
        inputs:
            Pipeline-level inputs injected before execution.
            Typically ``{"query": "<user question>"}`` from the API.
        """
        inputs = inputs or {}
        record = PipelineExecutionRecord(
            pipeline_id=compiled.pipeline_id,
            pipeline_name=compiled.pipeline_name,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            inputs=inputs,
        )

        # Initialise per-node records
        for node_id in compiled.execution_order:
            compiled_node = compiled.nodes[node_id]
            record.node_records[node_id] = NodeExecutionRecord(
                node_id=node_id,
                node_type=compiled_node.node_type,
                status=NodeExecutionStatus.WAITING,
            )

        yield ExecutionEvent(
            event_type=EventType.PIPELINE_STARTED,
            execution_id=record.execution_id,
            data={"pipeline_id": compiled.pipeline_id, "name": compiled.pipeline_name},
        )

        # Inject pipeline-level inputs into QueryInputNode configs
        self._inject_inputs(compiled, inputs)

        # Node output store — keyed by node_id -> port_name -> value
        node_outputs: dict[str, dict[str, Any]] = {}

        # Track which nodes failed so we can skip their descendants
        failed_ancestors: set[str] = set()

        try:
            async with asyncio.timeout(self.timeout_seconds):
                for node_id in compiled.execution_order:
                    compiled_node = compiled.nodes[node_id]
                    node_record = record.node_records[node_id]

                    # Skip if an upstream node failed
                    upstream_ids = {b.source_node_id for b in compiled_node.input_bindings}
                    if upstream_ids & failed_ancestors:
                        node_record.status = NodeExecutionStatus.SKIPPED
                        yield ExecutionEvent(
                            event_type=EventType.NODE_SKIPPED,
                            execution_id=record.execution_id,
                            node_id=node_id,
                            node_type=compiled_node.node_type,
                            data={"reason": "Upstream node failed."},
                        )
                        failed_ancestors.add(node_id)
                        continue

                    # Assemble inputs from upstream outputs
                    node_inputs = self._resolve_inputs(compiled_node, node_outputs)

                    # --- Node starts ---
                    node_record.status = NodeExecutionStatus.RUNNING
                    node_record.started_at = datetime.now(timezone.utc)
                    yield ExecutionEvent(
                        event_type=EventType.NODE_STARTED,
                        execution_id=record.execution_id,
                        node_id=node_id,
                        node_type=compiled_node.node_type,
                    )

                    try:
                        outputs = await compiled_node.instance.execute(**node_inputs)
                        node_outputs[node_id] = outputs or {}

                        node_record.status = NodeExecutionStatus.COMPLETED
                        node_record.completed_at = datetime.now(timezone.utc)
                        node_record.output_preview = self._make_preview(outputs)

                        yield ExecutionEvent(
                            event_type=EventType.NODE_COMPLETED,
                            execution_id=record.execution_id,
                            node_id=node_id,
                            node_type=compiled_node.node_type,
                            data={
                                "duration_ms": node_record.duration_ms,
                                "output_preview": node_record.output_preview,
                            },
                        )

                    except Exception as exc:  # noqa: BLE001
                        error_msg = str(exc)
                        logger.exception(
                            "Node execution failed",
                            extra={"node_id": node_id, "node_type": compiled_node.node_type},
                        )
                        node_record.status = NodeExecutionStatus.FAILED
                        node_record.completed_at = datetime.now(timezone.utc)
                        node_record.error = error_msg
                        failed_ancestors.add(node_id)

                        yield ExecutionEvent(
                            event_type=EventType.NODE_FAILED,
                            execution_id=record.execution_id,
                            node_id=node_id,
                            node_type=compiled_node.node_type,
                            data={"error": error_msg},
                        )

        except TimeoutError:
            record.status = ExecutionStatus.FAILED
            record.error = (
                f"Pipeline timed out after {self.timeout_seconds} seconds."
            )
            record.completed_at = datetime.now(timezone.utc)
            yield ExecutionEvent(
                event_type=EventType.PIPELINE_FAILED,
                execution_id=record.execution_id,
                data={"error": record.error, "record": record},
            )
            return

        except Exception as exc:  # noqa: BLE001
            record.status = ExecutionStatus.FAILED
            record.error = str(exc)
            record.completed_at = datetime.now(timezone.utc)
            logger.exception("Pipeline execution failed unexpectedly")
            yield ExecutionEvent(
                event_type=EventType.PIPELINE_FAILED,
                execution_id=record.execution_id,
                data={"error": record.error, "record": record},
            )
            return

        # Determine final status
        has_failures = any(
            r.status == NodeExecutionStatus.FAILED
            for r in record.node_records.values()
        )
        record.status = ExecutionStatus.FAILED if has_failures else ExecutionStatus.COMPLETED
        record.completed_at = datetime.now(timezone.utc)

        # Collect final outputs (from answer_output node if present)
        for node_id, outputs in node_outputs.items():
            if compiled.nodes[node_id].node_type == "answer_output":
                record.outputs = outputs
                break
        if not record.outputs and node_outputs:
            # Fallback: use the last node's outputs
            last_node_id = compiled.execution_order[-1]
            record.outputs = node_outputs.get(last_node_id, {})

        yield ExecutionEvent(
            event_type=EventType.PIPELINE_COMPLETED if not has_failures else EventType.PIPELINE_FAILED,
            execution_id=record.execution_id,
            data={
                "duration_ms": record.duration_ms,
                "outputs": record.outputs,
                "record": record,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _inject_inputs(
        self, compiled: CompiledPipeline, inputs: dict[str, Any]
    ) -> None:
        """
        Inject pipeline-level inputs into QueryInputNode configs.

        If the user supplies ``{"query": "What is RAG?"}`` in the API call,
        that value overrides the hardcoded query in the node's config.
        """
        for node_id, compiled_node in compiled.nodes.items():
            if compiled_node.node_type == "query_input":
                if "query" in inputs:
                    compiled_node.instance.config["query"] = inputs["query"]

    def _resolve_inputs(
        self,
        compiled_node: CompiledNode,
        node_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build the ``**inputs`` dict to pass into a node's ``execute()`` call
        by reading from the outputs of upstream nodes.
        """
        resolved: dict[str, Any] = {}
        for binding in compiled_node.input_bindings:
            upstream_outputs = node_outputs.get(binding.source_node_id, {})
            if binding.source_port in upstream_outputs:
                resolved[binding.target_port] = upstream_outputs[binding.source_port]
        return resolved

    @staticmethod
    def _make_preview(outputs: dict[str, Any]) -> dict[str, Any]:
        """
        Create a truncated, JSON-serialisable preview of node outputs
        for the execution log panel.
        """
        preview: dict[str, Any] = {}
        for key, value in (outputs or {}).items():
            if isinstance(value, str):
                preview[key] = value[:300] + "..." if len(value) > 300 else value
            elif isinstance(value, list):
                preview[key] = f"list[{len(value)} items]"
            elif hasattr(value, "answer"):
                # GenerationResult
                preview[key] = {"answer": str(value.answer)[:300]}
            elif hasattr(value, "model_dump"):
                try:
                    preview[key] = str(value.model_dump())[:300]
                except Exception:  # noqa: BLE001
                    preview[key] = str(value)[:300]
            else:
                preview[key] = str(value)[:300]
        return preview
