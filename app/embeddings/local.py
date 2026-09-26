"""
Local sentence-transformers embedder runner.

Runs entirely on the local machine — no API key required.
"""

from __future__ import annotations

from app.core.constants import EmbeddingProvider
from app.embeddings.base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Embed texts using a locally-downloaded sentence-transformers model.

    Parameters
    ----------
    model:
        sentence-transformers model name or local path.
    device:
        Torch device (``cpu``, ``cuda``, ``mps``).
    batch_size:
        Texts per forward pass.
    normalize:
        L2-normalize embeddings (recommended for cosine similarity).
    """

    provider = EmbeddingProvider.LOCAL

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. Run: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()
