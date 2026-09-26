"""
Chunking manager — factory for all chunker types.
"""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.core.constants import ChunkingStrategy
from app.core.exceptions import RAGForgeError


class ChunkingManager:
    """
    Factory that returns the correct ``BaseChunker`` subclass by name.

    Usage
    -----
        chunker = ChunkingManager.get("recursive", chunk_size=800)
        chunks = chunker.split(document)
    """

    @staticmethod
    def get(strategy: str, **kwargs) -> BaseChunker:
        """
        Return a configured chunker for ``strategy``.

        Parameters
        ----------
        strategy:
            One of: ``character``, ``recursive``, ``token``, ``sentence``,
            ``markdown``, ``semantic``, ``parent_child``.
        **kwargs:
            Passed directly to the chunker constructor.
        """
        from app.chunking.character import CharacterChunker
        from app.chunking.markdown import MarkdownChunker
        from app.chunking.parent_child import ParentChildChunker
        from app.chunking.recursive import RecursiveChunker
        from app.chunking.semantic import SemanticChunker
        from app.chunking.sentence import SentenceChunker
        from app.chunking.token import TokenChunker

        registry: dict[str, type[BaseChunker]] = {
            ChunkingStrategy.CHARACTER: CharacterChunker,
            ChunkingStrategy.RECURSIVE: RecursiveChunker,
            ChunkingStrategy.TOKEN: TokenChunker,
            ChunkingStrategy.SENTENCE: SentenceChunker,
            ChunkingStrategy.MARKDOWN: MarkdownChunker,
            ChunkingStrategy.SEMANTIC: SemanticChunker,
            ChunkingStrategy.PARENT_CHILD: ParentChildChunker,
        }

        cls = registry.get(strategy)
        if cls is None:
            available = list(registry.keys())
            raise RAGForgeError(
                f"Unknown chunking strategy '{strategy}'. Available: {available}",
                details={"strategy": strategy, "available": available},
            )
        return cls(**kwargs)
