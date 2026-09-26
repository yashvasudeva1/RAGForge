"""
Cohere Rerank v3 runner.
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.reranking.base import BaseReranker

logger = get_logger(__name__)


class CohereReranker(BaseReranker):
    """
    Rerank using Cohere Rerank v3 API.

    Parameters
    ----------
    model:
        Cohere rerank model ID.
    api_key:
        Cohere API key. Falls back to COHERE_API_KEY.
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("COHERE_API_KEY")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        try:
            import cohere  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("cohere is required. Run: pip install cohere") from exc

        client = cohere.Client(api_key=self._api_key)
        documents = [r.content for r in results]

        response = client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=min(top_n, len(results)),
        )

        reranked: list[RetrievalResult] = []
        for new_rank, hit in enumerate(response.results, start=1):
            original = results[hit.index]
            reranked.append(
                RetrievalResult(
                    content=original.content,
                    chunk_id=original.chunk_id,
                    source=original.source,
                    score=float(hit.relevance_score),
                    rank=new_rank,
                    metadata=original.metadata,
                )
            )

        logger.debug("CohereReranker: %d results reranked", len(reranked))
        return reranked
