"""
BaseGenerator — abstract base class for all LLM generation runners.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.domain.generation.models import GenerationResult
from app.domain.retrieval.models import RetrievalResult


class BaseGenerator(ABC):
    """
    Abstract base for LLM generation runners.

    Subclasses implement ``generate()`` (sync). The async ``agenerate()``
    defaults to a thread-pool wrapper, but streaming-capable subclasses
    should override it with a true async implementation.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        """
        Call the LLM with ``prompt`` and return a ``GenerationResult``.

        Parameters
        ----------
        prompt:
            The fully rendered prompt string.
        source_chunks:
            Retrieval results attached to the result for citation.
        """

    async def agenerate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        """Async wrapper — runs ``generate`` in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, source_chunks)
        )
