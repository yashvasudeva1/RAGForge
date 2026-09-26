"""
Pydantic schemas for document and ingestion endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.documents.models import Document


class DocumentUploadResponse(BaseModel):
    """Response after a file is uploaded to the server."""

    filename: str
    file_path: str
    size_bytes: int
    content_type: str | None = None


class DocumentLoadRequest(BaseModel):
    """Request to parse a file or URL into documents."""

    source: str = Field(..., description="File path or URL to load.")


class DocumentLoadResponse(BaseModel):
    """Loaded documents response."""

    source: str
    count: int
    documents: list[Document]


class ChunkTestRequest(BaseModel):
    """Test chunking on raw text or document."""

    text: str
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ChunkTestResponse(BaseModel):
    """Result of chunking test."""

    chunk_count: int
    total_characters: int
    chunks: list[dict[str, Any]]
