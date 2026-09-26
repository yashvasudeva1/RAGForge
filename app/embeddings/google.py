"""
Google Generative AI embedder runner.
"""

from __future__ import annotations

import os

from app.core.constants import EmbeddingProvider
from app.embeddings.base import BaseEmbedder


class GoogleEmbedder(BaseEmbedder):
    """Embed texts using Google Generative AI embedding models."""

    provider = EmbeddingProvider.GOOGLE

    def __init__(
        self,
        model: str = "models/text-embedding-004",
        api_key: str | None = None,
        task_type: str = "retrieval_document",
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.task_type = task_type

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required. Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self._api_key)
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type=self.task_type,
        )
        return result["embedding"] if len(texts) == 1 else [e for e in result["embedding"]]
