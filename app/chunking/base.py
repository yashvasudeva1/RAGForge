"""
BaseChunker — abstract base class for all text chunkers.

Every chunker must implement ``split(document) -> list[Chunk]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.logging import get_logger
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document

logger = get_logger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.

    Subclasses implement ``split()``, which takes a single ``Document``
    and returns a list of ``Chunk`` objects with positional metadata.
    """

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """
        Split ``document`` into a list of ``Chunk`` objects.

        Parameters
        ----------
        document:
            The ``Document`` to split.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks. Each chunk carries a reference to
            ``document.id`` and a zero-based ``chunk_index``.
        """

    def split_many(self, documents: list[Document]) -> list[Chunk]:
        """
        Convenience method — split multiple documents and return all chunks.

        Chunks from different documents are concatenated in document order.
        """
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.split(doc))
        return chunks

    # ------------------------------------------------------------------ #
    # Helpers shared by subclasses
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_chunk(
        document: Document,
        content: str,
        index: int,
        start_char: int | None = None,
        end_char: int | None = None,
        chunker_type: str | None = None,
        parent_chunk_id: str | None = None,
        **extra_meta,
    ) -> Chunk:
        """
        Build a ``Chunk`` from a document and a text slice.

        Centralises the repetitive metadata construction so subclasses
        stay concise.
        """
        from app.domain.chunks.models import ChunkMetadata

        return Chunk(
            content=content,
            metadata=ChunkMetadata(
                document_id=document.id,
                document_source=document.metadata.source,
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
                chunker_type=chunker_type,
                parent_chunk_id=parent_chunk_id,
                extra=extra_meta,
            ),
        )
