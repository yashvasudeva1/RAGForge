"""
RAGForge pipelines package.

Exports the core pipeline engine components:
  - PipelineValidator  — validates a PipelineConfig before execution
  - PipelineCompiler   — topologically sorts and instantiates nodes
  - PipelineExecutor   — runs the compiled pipeline, streams events
  - PipelineSerializer — saves/loads/exports pipeline configs
"""

from app.pipelines.compiler import CompiledPipeline, PipelineCompiler
from app.pipelines.executor import ExecutionEvent, EventType, PipelineExecutor
from app.pipelines.serializer import PipelineSerializer, PipelineSummary
from app.pipelines.validator import PipelineValidator, ValidationResult

__all__ = [
    "PipelineValidator",
    "ValidationResult",
    "PipelineCompiler",
    "CompiledPipeline",
    "PipelineExecutor",
    "ExecutionEvent",
    "EventType",
    "PipelineSerializer",
    "PipelineSummary",
]
