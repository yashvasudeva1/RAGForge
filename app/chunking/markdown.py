"""
Markdown-aware chunker that splits at heading boundaries.
"""

from __future__ import annotations

import re

from app.chunking.base import BaseChunker
from app.chunking.recursive import RecursiveChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document

# Matches Markdown ATX-style headings: # H1, ## H2, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownChunker(BaseChunker):
    """
    Splits Markdown at heading boundaries.

    Each section (heading + body) becomes one chunk.  Sections longer
    than ``max_chunk_size`` are recursively split with ``RecursiveChunker``.

    Parameters
    ----------
    max_chunk_size:
        Hard upper limit on chunk size in characters.
    heading_levels:
        Heading level names to split at (``["h1","h2","h3"]``).
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        heading_levels: list[str] | None = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        # Convert "h1" -> 1, "h2" -> 2, etc.
        levels = heading_levels or ["h1", "h2", "h3"]
        self.split_levels: set[int] = {
            int(h[1]) for h in levels if h.startswith("h") and h[1:].isdigit()
        }
        self._fallback = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=100)

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sections = self._split_by_headings(text)
        chunks: list[Chunk] = []
        idx = 0

        for title, body in sections:
            section_text = f"{title}\n\n{body}".strip() if title else body.strip()
            if not section_text:
                continue

            if len(section_text) <= self.max_chunk_size:
                chunks.append(
                    self._make_chunk(
                        document,
                        content=section_text,
                        index=idx,
                        chunker_type="markdown",
                        section_title=title.lstrip("#").strip() if title else None,
                    )
                )
                idx += 1
            else:
                # Use a temporary Document to recurse via RecursiveChunker
                from app.domain.documents.models import Document as Doc
                tmp = Doc(content=section_text, metadata=document.metadata)
                for sub in self._fallback.split(tmp):
                    sub.metadata.chunk_index = idx
                    sub.metadata.document_id = document.id
                    sub.metadata.chunker_type = "markdown"
                    sub.metadata.section_title = title.lstrip("#").strip() if title else None
                    chunks.append(sub)
                    idx += 1

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Return list of (heading_line, body_text) pairs."""
        matches = list(_HEADING_RE.finditer(text))

        if not matches:
            return [("", text)]

        # Filter to the heading levels we care about
        filtered = [m for m in matches if len(m.group(1)) in self.split_levels]
        if not filtered:
            return [("", text)]

        sections: list[tuple[str, str]] = []

        # Text before the first heading
        if filtered[0].start() > 0:
            sections.append(("", text[: filtered[0].start()]))

        for i, match in enumerate(filtered):
            heading_line = match.group(0)
            body_start = match.end()
            body_end = filtered[i + 1].start() if i + 1 < len(filtered) else len(text)
            body = text[body_start:body_end]
            sections.append((heading_line, body))

        return sections
