"""
Domain models for retrieval results.

A ``RetrievalResult`` represents a single retrieved chunk together with
its relevance score and rank.  The list of retrieval results is the output
of retriever nodes and the input to reranker and context builder nodes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.chunks.models import Chunk


class RetrievalResult(BaseModel):
    """
    A single chunk retrieved in response to a query.

    Attributes
    ----------
    chunk:
        The retrieved ``Chunk`` object.
    score:
        Relevance score as returned by the retriever.  Semantics vary by
        strategy: cosine similarity for dense retrievers, BM25 score for
        sparse retrievers, reciprocal rank for hybrid (RRF), etc.
        Higher is always better.
    rank:
        Zero-based rank within the retrieved list (0 = most relevant).
    retriever_type:
        The retrieval strategy that produced this result.
    metadata:
        Optional extra info from the retriever (e.g. MMR diversity score).
    """

    chunk: Chunk = Field(description="The retrieved chunk.")
    score: float = Field(description="Relevance score (higher = more relevant).")
    rank: int = Field(description="Zero-based rank within the result list.")
    retriever_type: str | None = Field(
        default=None,
        description="Identifier of the retriever strategy that produced this result.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra metadata from the retriever.",
    )

    @property
    def content(self) -> str:
        """Shortcut to the chunk's text content."""
        return self.chunk.content

    @property
    def document_id(self) -> str:
        """Shortcut to the originating document ID."""
        return self.chunk.document_id

    @property
    def source(self) -> str | None:
        """Shortcut to the originating document source path or URL."""
        return self.chunk.metadata.document_source

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, "
            f"chunk_id={self.chunk.id!r})"
        )


class RetrievalResultSet(BaseModel):
    """
    The full set of results from a single retrieval call.

    Attributes
    ----------
    results:
        Ordered list of retrieval results (most relevant first).
    query:
        The query string that produced these results.
    retriever_type:
        The retrieval strategy used.
    total_candidates:
        Total number of candidates examined before filtering to ``k``.
    """

    results: list[RetrievalResult] = Field(default_factory=list)
    query: str = Field(description="The query that produced these results.")
    retriever_type: str | None = Field(default=None)
    total_candidates: int | None = Field(
        default=None,
        description="Total candidates considered before top-k selection.",
    )

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def top(self) -> RetrievalResult | None:
        """Return the highest-ranked result, or None if empty."""
        return self.results[0] if self.results else None
