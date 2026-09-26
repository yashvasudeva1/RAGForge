"""
VoyageAI embedder runner.
"""

from __future__ import annotations

import os

from app.core.constants import EmbeddingProvider
from app.embeddings.base import BaseEmbedder


class VoyageEmbedder(BaseEmbedder):
    """Embed texts using VoyageAI's embedding API."""

    provider = EmbeddingProvider.VOYAGE

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("VOYAGE_API_KEY")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("voyageai is required. Run: pip install voyageai") from exc

        client = voyageai.Client(api_key=self._api_key)
        result = client.embed(texts, model=self.model_name)
        return result.embeddings
