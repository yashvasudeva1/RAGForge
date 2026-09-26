"""
BaseVectorStore — abstract base class for all vector store connectors.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.domain.embeddings.models import EmbeddedChunk
from app.domain.retrieval.models import RetrievalResult


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store connectors.

    Subclasses implement ``add()`` and ``search()`` (sync).
    Async variants default to thread-pool wrappers.
    """

    @abstractmethod
    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Index a list of embedded chunks into the store."""

    @abstractmethod
    def search(self, query_vector: list[float], k: int = 5, **kwargs) -> list[RetrievalResult]:
        """Search for the k nearest neighbours of ``query_vector``."""

    async def aadd(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Async wrapper around ``add``."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.add, embedded_chunks)

    async def asearch(
        self, query: str, k: int = 5, **kwargs
    ) -> list[RetrievalResult]:
        """
        Async search.

        Subclasses that need a query *string* (to embed on the fly inside the
        store client) should override this directly.  The default raises
        ``NotImplementedError`` since a query vector is needed for the sync
        ``search()``.
        """
        raise NotImplementedError(
            "Subclasses must implement asearch() or provide a query-embedding step."
        )

    @staticmethod
    def _to_retrieval_result(
        chunk_content: str,
        chunk_id: str,
        source: str,
        score: float,
        rank: int,
        metadata: dict | None = None,
    ) -> RetrievalResult:
        """Helper to build a RetrievalResult (with a synthetic Chunk) from raw store output."""
        from app.domain.chunks.models import Chunk, ChunkMetadata
        from app.domain.retrieval.models import RetrievalResult as RR

        chunk = Chunk(
            id=chunk_id,
            content=chunk_content,
            metadata=ChunkMetadata(
                document_id=str((metadata or {}).get("document_id", "")),
                document_source=source,
                chunk_index=int((metadata or {}).get("chunk_index", 0)),
            ),
        )
        return RR(
            chunk=chunk,
            score=score,
            rank=rank,
            metadata=metadata or {},
        )
