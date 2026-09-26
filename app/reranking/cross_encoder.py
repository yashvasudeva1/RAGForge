"""
Cross-encoder reranker using sentence-transformers.

Runs the query and each candidate jointly through a cross-encoder model,
producing fine-grained relevance scores.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.reranking.base import BaseReranker

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Rerank using a local cross-encoder model.

    Parameters
    ----------
    model:
        sentence-transformers cross-encoder model name.
    device:
        Compute device (``cpu``, ``cuda``, ``mps``).
    batch_size:
        Pairs per forward pass.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. Run: pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self.model, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        model = self._get_model()
        pairs = [(query, r.content) for r in results]
        scores: list[float] = model.predict(pairs, batch_size=self.batch_size).tolist()

        scored = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )[:top_n]

        return [
            RetrievalResult(
                content=r.content,
                chunk_id=r.chunk_id,
                source=r.source,
                score=float(score),
                rank=rank,
                metadata=r.metadata,
            )
            for rank, (r, score) in enumerate(scored, start=1)
        ]
