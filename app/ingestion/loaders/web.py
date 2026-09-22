"""
Web URL loader using Trafilatura for clean article text extraction.
"""

from __future__ import annotations

import asyncio

from app.core.constants import FileType
from app.core.exceptions import LoaderError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class WebLoader(BaseLoader):
    """
    Fetch and extract clean readable text from a web URL.

    Uses Trafilatura, which is purpose-built for extracting article content
    from web pages while removing navigation, ads, and boilerplate.

    Parameters
    ----------
    include_comments:
        Include user comment sections.
    timeout_seconds:
        HTTP request timeout per URL.
    """

    def __init__(
        self,
        include_comments: bool = False,
        timeout_seconds: int = 30,
    ) -> None:
        self.include_comments = include_comments
        self.timeout_seconds = timeout_seconds

    def load(self, source: str) -> list[Document]:
        try:
            import trafilatura  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "trafilatura is required for web loading. Run: pip install trafilatura"
            ) from exc

        try:
            downloaded = trafilatura.fetch_url(source)
            if not downloaded:
                raise LoaderError(
                    f"Could not fetch URL: {source}",
                    details={"url": source},
                )

            content = trafilatura.extract(
                downloaded,
                include_comments=self.include_comments,
                include_tables=True,
                no_fallback=False,
            )

            if not content:
                # Fall back to raw text extraction
                content = trafilatura.extract(downloaded, no_fallback=True) or ""

            metadata_obj = trafilatura.extract_metadata(downloaded)
            title = getattr(metadata_obj, "title", None) if metadata_obj else None
            author = getattr(metadata_obj, "author", None) if metadata_obj else None

            document = Document(
                content=content.strip(),
                metadata=DocumentMetadata(
                    source=source,
                    file_type=FileType.WEB_URL,
                    title=title,
                    author=author,
                    extra={"url": source},
                ),
            )
            logger.debug("Loaded URL: %s (%d chars)", source, len(content))
            return [document]

        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(
                f"Failed to load URL '{source}': {exc}",
                details={"url": source, "cause": str(exc)},
            ) from exc

    async def aload(self, source: str) -> list[Document]:
        """Run web fetch in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, source)
