"""
BaseReranker — abstract base class for all reranker runners.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.domain.retrieval.models import RetrievalResult


class BaseReranker(ABC):
    """
    Abstract base class for reranker runners.

    Subclasses implement ``rerank()`` (sync) and optionally
    ``arerank()`` (async, defaults to thread-pool wrapper).
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        """
        Rerank ``results`` with respect to ``query``.

        Parameters
        ----------
        query:
            The user's question.
        results:
            Candidate retrieval results to rerank.
        top_n:
            Number of top results to return after reranking.

        Returns
        -------
        list[RetrievalResult]
            Top ``top_n`` results with updated ranks and scores.
        """

    async def arerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        """Async wrapper — runs ``rerank`` in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.rerank(query, results, top_n)
        )
