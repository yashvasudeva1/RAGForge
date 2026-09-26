"""
BaseRetriever — abstract base class for all retrieval runners.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.domain.retrieval.models import RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval runners.

    Subclasses implement ``retrieve()`` (sync) and optionally
    ``aretrieve()`` (async, defaults to thread-pool wrapper).
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for ``query``.

        Parameters
        ----------
        query:
            The user's question or search string.
        k:
            Number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by descending relevance score.
        """

    async def aretrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        """Async wrapper — runs ``retrieve`` in a thread pool by default."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(query, k, **kwargs))
