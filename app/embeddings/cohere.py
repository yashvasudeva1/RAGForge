"""
Cohere Embed v3 runner.
"""

from __future__ import annotations

import os

from app.core.constants import EmbeddingProvider
from app.embeddings.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """Embed texts using Cohere Embed v3."""

    provider = EmbeddingProvider.COHERE

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        input_type: str = "search_document",
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self.input_type = input_type

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            import cohere  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("cohere is required. Run: pip install cohere") from exc

        client = cohere.Client(api_key=self._api_key)
        response = client.embed(
            texts=texts,
            model=self.model_name,
            input_type=self.input_type,
        )
        return [list(emb) for emb in response.embeddings]
