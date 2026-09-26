"""
HuggingFace Inference API embedder runner.
"""

from __future__ import annotations

import os

from app.core.constants import EmbeddingProvider
from app.embeddings.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """Embed texts via the HuggingFace Inference API."""

    provider = EmbeddingProvider.HUGGINGFACE

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from huggingface_hub import InferenceClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required. Run: pip install huggingface_hub"
            ) from exc

        client = InferenceClient(token=self._api_key)
        embeddings = client.feature_extraction(texts, model=self.model_name)
        # Returns numpy array or list of lists
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return list(embeddings)
