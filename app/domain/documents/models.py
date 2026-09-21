"""
Domain models for documents.

A ``Document`` represents a single source file or URL after it has been
loaded by a document loader.  It is the primary unit of input to the
chunking stage of the pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.core.constants import FileType


class DocumentMetadata(BaseModel):
    """
    Flexible metadata container for a document.

    Loaders populate the fields they can extract; all fields are optional.
    Additional arbitrary key-value pairs can be stored in ``extra``.
    """

    source: str | None = Field(
        default=None,
        description="The original source path or URL of the document.",
    )
    file_type: FileType = Field(
        default=FileType.UNKNOWN,
        description="The detected file type of the source.",
    )
    file_name: str | None = Field(
        default=None,
        description="The base filename (e.g. 'report.pdf').",
    )
    file_size_bytes: int | None = Field(
        default=None,
        description="File size in bytes, if available.",
    )
    page_count: int | None = Field(
        default=None,
        description="Number of pages (for PDFs and PPTX files).",
    )
    author: str | None = Field(
        default=None,
        description="Document author, if available in metadata.",
    )
    title: str | None = Field(
        default=None,
        description="Document title, if available in metadata.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Document creation timestamp.",
    )
    modified_at: datetime | None = Field(
        default=None,
        description="Document last-modified timestamp.",
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'fr').",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Loader-specific or user-defined additional metadata.",
    )

    model_config = {"extra": "allow"}


class Document(BaseModel):
    """
    A loaded document ready for chunking.

    Attributes
    ----------
    id:
        Unique identifier for this document (UUID4 by default).
    content:
        The full extracted text content of the document.
    metadata:
        Structured metadata about the source.
    loaded_at:
        UTC timestamp of when this document was loaded.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier (UUID4).",
    )
    content: str = Field(
        description="The full extracted text content of the document.",
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Structured metadata about the document source.",
    )
    loaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when this document was loaded.",
    )

    @property
    def source(self) -> str | None:
        """Shortcut to ``metadata.source``."""
        return self.metadata.source

    @property
    def file_type(self) -> FileType:
        """Shortcut to ``metadata.file_type``."""
        return self.metadata.file_type

    @property
    def char_count(self) -> int:
        """Number of characters in the document content."""
        return len(self.content)

    def truncated_preview(self, max_chars: int = 200) -> str:
        """Return a truncated preview of the document content."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id!r}, source={self.source!r}, "
            f"chars={self.char_count})"
        )
