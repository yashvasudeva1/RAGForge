"""
FAISS in-memory vector store connector.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.domain.embeddings.models import EmbeddedChunk
from app.domain.retrieval.models import RetrievalResult
from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS in-memory vector store with optional file persistence.

    Parameters
    ----------
    index_path:
        File path for saving/loading the FAISS index.
        ``None`` → ephemeral in-memory index.
    index_type:
        ``"Flat"`` (exact), ``"IVF"`` (approximate), or ``"HNSW"``.
    """

    def __init__(
        self,
        index_path: str | None = None,
        index_type: str = "Flat",
    ) -> None:
        self.index_path = index_path
        self.index_type = index_type
        self._index = None
        self._id_map: list[str] = []          # faiss int id -> chunk_id
        self._content_map: dict[str, str] = {}  # chunk_id -> content
        self._meta_map: dict[str, dict] = {}    # chunk_id -> metadata

    def _get_faiss(self):
        try:
            import faiss  # type: ignore[import-untyped]
            return faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu") from exc

    def _init_index(self, dim: int):
        faiss = self._get_faiss()
        if self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100)
        else:
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine (normalised vecs)
        return index

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        import numpy as np

        vectors = np.array([ec.embedding for ec in embedded_chunks], dtype="float32")
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        if self._index is None:
            self._index = self._init_index(vectors.shape[1])
            if self.index_type == "IVF":
                self._get_faiss().normalize_L2(vectors)
                self._index.train(vectors)

        self._index.add(vectors)

        for ec in embedded_chunks:
            self._id_map.append(ec.chunk.id)
            self._content_map[ec.chunk.id] = ec.chunk.content
            self._meta_map[ec.chunk.id] = {
                "source": ec.chunk.metadata.document_source or "",
                "chunk_index": ec.chunk.metadata.chunk_index,
            }

        logger.debug("FAISS: added %d vectors (total=%d)", len(embedded_chunks), len(self._id_map))

    def search(self, query_vector: list[float], k: int = 5, **kwargs) -> list[RetrievalResult]:
        if self._index is None or len(self._id_map) == 0:
            return []

        import numpy as np
        q = np.array([query_vector], dtype="float32")
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        actual_k = min(k, len(self._id_map))
        scores, indices = self._index.search(q, actual_k)

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._id_map):
                continue
            chunk_id = self._id_map[idx]
            results.append(
                self._to_retrieval_result(
                    chunk_content=self._content_map.get(chunk_id, ""),
                    chunk_id=chunk_id,
                    source=self._meta_map.get(chunk_id, {}).get("source", ""),
                    score=float(score),
                    rank=rank,
                    metadata=self._meta_map.get(chunk_id, {}),
                )
            )
        return results

    async def asearch(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        raise NotImplementedError(
            "FAISSVectorStore.asearch() requires a pre-computed query vector. "
            "Use the DenseRetriever node which embeds the query before calling search()."
        )
