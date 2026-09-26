"""
BaseEmbedder — abstract base class for all embedding runners.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.core.constants import EmbeddingProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding runners.

    Subclasses implement ``embed_texts()`` (sync) and optionally
    ``aembed_texts()`` (async). The default async implementation runs
    the sync method in a thread pool.

    Attributes
    ----------
    model_name:
        The model identifier string.
    provider:
        The ``EmbeddingProvider`` enum value.
    """

    model_name: str = ""
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings and return a list of embedding vectors.

        Parameters
        ----------
        texts:
            Strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input string, in the same order.
        """

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper — runs embed_texts in a thread pool by default."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    async def aembed_query(self, query: str) -> list[float]:
        """Async single-query embedding."""
        results = await self.aembed_texts([query])
        return results[0]
