"""
Qdrant vector store connector.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from app.core.logging import get_logger
from app.domain.embeddings.models import EmbeddedChunk
from app.domain.retrieval.models import RetrievalResult
from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector database connector.

    Parameters
    ----------
    collection_name:
        Qdrant collection name.
    url:
        Qdrant server URL.
    api_key:
        Qdrant Cloud API key (optional for local).
    """

    def __init__(
        self,
        collection_name: str = "ragforge",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.url = url
        self._api_key = api_key or os.getenv("QDRANT_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("qdrant-client is required. Run: pip install qdrant-client") from exc
            self._client = QdrantClient(url=self.url, api_key=self._api_key)
        return self._client

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]
        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

        dim = len(embedded_chunks[0].embedding)
        self._ensure_collection(dim)
        client = self._get_client()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=ec.embedding,
                payload={
                    "chunk_id": ec.chunk.id,
                    "content": ec.chunk.content,
                    "source": ec.chunk.metadata.document_source or "",
                    "chunk_index": ec.chunk.metadata.chunk_index,
                    "document_id": ec.chunk.metadata.document_id,
                },
            )
            for ec in embedded_chunks
        ]

        client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Qdrant: upserted %d points into '%s'", len(points), self.collection_name)

    def search(self, query_vector: list[float], k: int = 5, **kwargs) -> list[RetrievalResult]:
        client = self._get_client()
        hits = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )
        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            results.append(
                self._to_retrieval_result(
                    chunk_content=payload.get("content", ""),
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    source=payload.get("source", ""),
                    score=float(hit.score),
                    rank=rank,
                    metadata=payload,
                )
            )
        return results

    async def asearch(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search([], k))
