"""
Token-based chunker using tiktoken.

Splits at token boundaries so chunks fit precisely within LLM context windows.
"""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document


class TokenChunker(BaseChunker):
    """
    Split text at token boundaries using tiktoken.

    Parameters
    ----------
    chunk_size:
        Maximum tokens per chunk.
    chunk_overlap:
        Overlapping tokens between consecutive chunks.
    encoding_name:
        Tiktoken encoding (``cl100k_base`` for GPT-4, ``o200k_base`` for GPT-4o).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self._enc = None

    def _get_encoding(self):
        if self._enc is None:
            try:
                import tiktoken  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("tiktoken is required. Run: pip install tiktoken") from exc
            self._enc = tiktoken.get_encoding(self.encoding_name)
        return self._enc

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        enc = self._get_encoding()
        token_ids: list[int] = enc.encode(text)

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            chunk_tokens = token_ids[start:end]
            chunk_text = enc.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(
                    self._make_chunk(
                        document,
                        content=chunk_text.strip(),
                        index=idx,
                        chunker_type="token",
                    )
                )
                # Set accurate token count
                chunks[-1].token_count = len(chunk_tokens)
                idx += 1

            step = max(self.chunk_size - self.chunk_overlap, 1)
            start += step

        return chunks
