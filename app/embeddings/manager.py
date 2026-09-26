"""
Embeddings manager — factory for all embedder types.
"""

from __future__ import annotations

from app.core.constants import EmbeddingProvider
from app.core.exceptions import RAGForgeError
from app.embeddings.base import BaseEmbedder


class EmbeddingManager:
    """
    Factory that returns the correct ``BaseEmbedder`` subclass by provider name.

    Usage
    -----
        embedder = EmbeddingManager.get("openai", model="text-embedding-3-small")
        vectors = embedder.embed_texts(["Hello world"])
    """

    @staticmethod
    def get(provider: str, **kwargs) -> BaseEmbedder:
        """
        Return a configured embedder for ``provider``.

        Parameters
        ----------
        provider:
            One of: ``openai``, ``google``, ``huggingface``, ``cohere``,
            ``voyage``, ``local``.
        **kwargs:
            Passed to the embedder constructor.
        """
        from app.embeddings.cohere import CohereEmbedder
        from app.embeddings.google import GoogleEmbedder
        from app.embeddings.huggingface import HuggingFaceEmbedder
        from app.embeddings.local import LocalEmbedder
        from app.embeddings.openai import OpenAIEmbedder
        from app.embeddings.voyage import VoyageEmbedder

        registry: dict[str, type[BaseEmbedder]] = {
            EmbeddingProvider.OPENAI: OpenAIEmbedder,
            EmbeddingProvider.GOOGLE: GoogleEmbedder,
            EmbeddingProvider.HUGGINGFACE: HuggingFaceEmbedder,
            EmbeddingProvider.COHERE: CohereEmbedder,
            EmbeddingProvider.VOYAGE: VoyageEmbedder,
            EmbeddingProvider.LOCAL: LocalEmbedder,
        }

        cls = registry.get(provider)
        if cls is None:
            available = list(registry.keys())
            raise RAGForgeError(
                f"Unknown embedding provider '{provider}'. Available: {available}",
                details={"provider": provider, "available": available},
            )
        return cls(**kwargs)
