"""
Pinecone vector store connector.
"""

from __future__ import annotations

import asyncio
import os

from app.core.logging import get_logger
from app.domain.embeddings.models import EmbeddedChunk
from app.domain.retrieval.models import RetrievalResult
from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone managed vector database connector.

    Parameters
    ----------
    api_key:
        Pinecone API key.
    index_name:
        Pinecone index name.
    namespace:
        Namespace within the index.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str = "ragforge",
        namespace: str = "",
    ) -> None:
        self._api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.namespace = namespace
        self._index = None

    def _get_index(self):
        if self._index is None:
            try:
                from pinecone import Pinecone  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("pinecone-client is required. Run: pip install pinecone-client") from exc
            pc = Pinecone(api_key=self._api_key)
            self._index = pc.Index(self.index_name)
        return self._index

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        index = self._get_index()

        vectors = [
            {
                "id": ec.chunk.id,
                "values": ec.embedding,
                "metadata": {
                    "content": ec.chunk.content[:1000],  # Pinecone metadata size limit
                    "source": ec.chunk.metadata.document_source or "",
                    "chunk_index": ec.chunk.metadata.chunk_index,
                },
            }
            for ec in embedded_chunks
        ]
        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i+100], namespace=self.namespace)

        logger.debug("Pinecone: upserted %d vectors into '%s'", len(vectors), self.index_name)

    def search(self, query_vector: list[float], k: int = 5, **kwargs) -> list[RetrievalResult]:
        index = self._get_index()
        response = index.query(
            vector=query_vector,
            top_k=k,
            namespace=self.namespace,
            include_metadata=True,
        )

        results: list[RetrievalResult] = []
        for rank, match in enumerate(response.matches, start=1):
            meta = match.metadata or {}
            results.append(
                self._to_retrieval_result(
                    chunk_content=meta.get("content", ""),
                    chunk_id=match.id,
                    source=meta.get("source", ""),
                    score=float(match.score),
                    rank=rank,
                    metadata=meta,
                )
            )
        return results

    async def asearch(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search([], k))
