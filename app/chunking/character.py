"""
Character-based fixed-size text chunker.
"""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document


class CharacterChunker(BaseChunker):
    """
    Split text into fixed-size character windows with configurable overlap.

    Tries to split at ``separator`` boundaries before hard-truncating.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Number of characters to carry over into the next chunk.
    separator:
        Preferred split character (defaults to newline).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunk_text = text[start:]
            else:
                # Try to find the separator within the window
                sep_pos = text.rfind(self.separator, start, end)
                if sep_pos > start:
                    end = sep_pos + len(self.separator)
                chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    self._make_chunk(
                        document,
                        content=chunk_text.strip(),
                        index=idx,
                        start_char=start,
                        end_char=start + len(chunk_text),
                        chunker_type="character",
                    )
                )
                idx += 1

            # Advance with overlap
            advance = max(len(chunk_text) - self.chunk_overlap, 1)
            start += advance

        return chunks
