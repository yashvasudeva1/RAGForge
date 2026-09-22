"""
Pipeline templates package.

Exports all built-in pipeline templates so the API can serve them
without reading from disk.
"""

from app.pipelines.templates.basic_rag import BASIC_RAG_TEMPLATE
from app.pipelines.templates.hybrid_rag import HYBRID_RAG_TEMPLATE
from app.pipelines.templates.agentic_rag import AGENTIC_RAG_TEMPLATE

ALL_TEMPLATES = [
    BASIC_RAG_TEMPLATE,
    HYBRID_RAG_TEMPLATE,
    AGENTIC_RAG_TEMPLATE,
]

__all__ = [
    "BASIC_RAG_TEMPLATE",
    "HYBRID_RAG_TEMPLATE",
    "AGENTIC_RAG_TEMPLATE",
    "ALL_TEMPLATES",
]
