"""
Recursive character text splitter.

Splits text by trying a hierarchy of separators in order: paragraph
breaks, newlines, sentences, words, then individual characters.  This
produces the most natural-feeling chunks for most document types.
"""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Characters to carry into the next chunk.
    separators:
        Ordered list of split strings tried from coarsest to finest.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        raw_splits = self._split_text(text, self.separators)
        merged = self._merge_splits(raw_splits)

        return [
            self._make_chunk(
                document,
                content=chunk_text,
                index=i,
                chunker_type="recursive",
            )
            for i, chunk_text in enumerate(merged)
            if chunk_text.strip()
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that works."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level split as last resort
            return list(text)

        splits = text.split(sep)

        good_splits: list[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                # Recurse into smaller separators
                good_splits.extend(self._split_text(s, remaining_seps))

        return good_splits

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """
        Merge small splits back up to ``chunk_size``, with overlap.
        """
        merged: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for s in splits:
            s = s.strip()
            if not s:
                continue
            s_len = len(s)

            if current_len + s_len + 1 > self.chunk_size and current_parts:
                merged.append(" ".join(current_parts))
                # Keep overlap: drop from the front until within overlap budget
                while current_parts and current_len > self.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + 1

            current_parts.append(s)
            current_len += s_len + 1

        if current_parts:
            merged.append(" ".join(current_parts))

        return merged
