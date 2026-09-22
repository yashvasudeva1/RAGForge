"""
BaseLoader — abstract base class for all document loaders.

Every loader must implement:
  - ``load(source)``  — synchronous load
  - ``aload(source)`` — async load (default: runs load in a thread)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.core.logging import get_logger
from app.domain.documents.models import Document

logger = get_logger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    Subclasses implement ``load()`` (sync) and optionally override
    ``aload()`` (async).  The default async implementation runs ``load``
    in a thread pool so it never blocks the event loop.
    """

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """
        Load documents from ``source`` (file path or URL).

        Parameters
        ----------
        source:
            A file system path or URL string.

        Returns
        -------
        list[Document]
            One or more loaded documents.
        """

    async def aload(self, source: str) -> list[Document]:
        """
        Async wrapper around ``load``.

        The default implementation offloads the blocking ``load`` call to
        a thread pool executor so it does not block the async event loop.
        Subclasses that have native async I/O can override this directly.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, source)
