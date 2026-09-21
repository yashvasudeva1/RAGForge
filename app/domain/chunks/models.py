"""
Domain models for text chunks.

A ``Chunk`` is a segment of a ``Document`` produced by a chunker node.
It carries the original document reference and positional metadata so
retrieval results can be traced back to their source.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChunkMetadata(BaseModel):
    """
    Metadata attached to a chunk.

    Inherits all fields from the parent document's metadata and adds
    chunk-specific positional information.
    """

    document_id: str = Field(description="ID of the parent Document.")
    document_source: str | None = Field(
        default=None,
        description="Source path or URL of the parent document.",
    )
    chunk_index: int = Field(
        description="Zero-based position of this chunk within the document.",
    )
    start_char: int | None = Field(
        default=None,
        description="Character offset where this chunk starts in the original document.",
    )
    end_char: int | None = Field(
        default=None,
        description="Character offset where this chunk ends in the original document.",
    )
    page_number: int | None = Field(
        default=None,
        description="Page number this chunk originates from (for paginated sources).",
    )
    section_title: str | None = Field(
        default=None,
        description="Section or heading title if the chunker is header-aware.",
    )
    parent_chunk_id: str | None = Field(
        default=None,
        description=(
            "ID of the parent chunk when using parent-child chunking. "
            "The parent contains the broader context; this chunk is the "
            "smaller unit used for retrieval."
        ),
    )
    chunker_type: str | None = Field(
        default=None,
        description="The chunker strategy that produced this chunk.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunker-specific or user-defined additional metadata.",
    )

    model_config = {"extra": "allow"}


class Chunk(BaseModel):
    """
    A text segment derived from a ``Document``.

    Attributes
    ----------
    id:
        Unique identifier for this chunk (UUID4 by default).
    content:
        The text content of this chunk.
    metadata:
        Positional and source metadata.
    token_count:
        Approximate token count (populated by tokenisation-aware chunkers).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier (UUID4).",
    )
    content: str = Field(description="The text content of this chunk.")
    metadata: ChunkMetadata
    token_count: int | None = Field(
        default=None,
        description="Approximate token count, populated when a tokenisation-aware chunker is used.",
    )

    @model_validator(mode="after")
    def set_token_count_if_missing(self) -> "Chunk":
        """Estimate token count by whitespace splitting if not already set."""
        if self.token_count is None:
            # Simple whitespace-based approximation; accurate chunkers
            # (TokenChunker) will override this with the real count.
            self.token_count = len(self.content.split())
        return self

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @property
    def document_id(self) -> str:
        return self.metadata.document_id

    @property
    def chunk_index(self) -> int:
        return self.metadata.chunk_index

    @property
    def char_count(self) -> int:
        return len(self.content)

    def truncated_preview(self, max_chars: int = 200) -> str:
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.id!r}, doc={self.document_id!r}, "
            f"index={self.chunk_index}, chars={self.char_count})"
        )
