"""
OpenAI embedder runner.
"""

from __future__ import annotations

import os

from app.core.constants import EmbeddingProvider
from app.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Embed texts using the OpenAI Embeddings API.

    Parameters
    ----------
    model:
        OpenAI embedding model ID.
    api_key:
        API key. Falls back to ``OPENAI_API_KEY`` env var.
    batch_size:
        Texts per API call.
    dimensions:
        Optional output dimensions (supported by text-embedding-3-* models).
    """

    provider = EmbeddingProvider.OPENAI

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
        dimensions: int | None = None,
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.batch_size = batch_size
        self.dimensions = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("openai is required. Run: pip install openai") from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs = {"input": batch, "model": self.model_name}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions  # type: ignore[assignment]

            response = client.embeddings.create(**kwargs)
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
