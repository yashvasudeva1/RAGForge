"""
VoyageAI reranker runner.
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.reranking.base import BaseReranker

logger = get_logger(__name__)


class VoyageReranker(BaseReranker):
    """Rerank using VoyageAI's rerank API."""

    def __init__(
        self,
        model: str = "rerank-2",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("VOYAGE_API_KEY")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("voyageai is required. Run: pip install voyageai") from exc

        client = voyageai.Client(api_key=self._api_key)
        documents = [r.content for r in results]

        response = client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=min(top_n, len(results)),
        )

        reranked: list[RetrievalResult] = []
        for new_rank, item in enumerate(response.results, start=1):
            original = results[item.index]
            reranked.append(
                RetrievalResult(
                    content=original.content,
                    chunk_id=original.chunk_id,
                    source=original.source,
                    score=float(item.relevance_score),
                    rank=new_rank,
                    metadata=original.metadata,
                )
            )

        return reranked
