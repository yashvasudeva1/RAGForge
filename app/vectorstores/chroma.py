"""
ChromaDB vector store connector.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.logging import get_logger
from app.domain.embeddings.models import EmbeddedChunk
from app.domain.retrieval.models import RetrievalResult
from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store — local persistent or server mode.

    Parameters
    ----------
    collection_name:
        ChromaDB collection name.
    persist_dir:
        Local directory for persistence. ``None`` → in-memory.
    host / port:
        For client-server mode (overrides ``persist_dir``).
    """

    def __init__(
        self,
        collection_name: str = "ragforge",
        persist_dir: str | None = "./data/chroma",
        host: str | None = None,
        port: int = 8001,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.host = host
        self.port = port
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("chromadb is required. Run: pip install chromadb") from exc

            if self.host:
                client = chromadb.HttpClient(host=self.host, port=self.port)
            elif self.persist_dir:
                client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                client = chromadb.Client()

            self._client = client
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        coll = self._get_collection()

        ids = [ec.chunk.id for ec in embedded_chunks]
        embeddings = [ec.embedding for ec in embedded_chunks]
        documents = [ec.chunk.content for ec in embedded_chunks]
        metadatas = [
            {
                "source": ec.chunk.metadata.document_source or "",
                "chunk_index": ec.chunk.metadata.chunk_index,
                "document_id": ec.chunk.metadata.document_id,
            }
            for ec in embedded_chunks
        ]

        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("ChromaDB: upserted %d chunks into '%s'", len(ids), self.collection_name)

    def search(self, query_vector: list[float], k: int = 5, **kwargs) -> list[RetrievalResult]:
        coll = self._get_collection()
        results = coll.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: list[RetrievalResult] = []
        for rank, (doc_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            # Chroma returns L2 distance when cosine space is used it's 1-similarity
            score = max(0.0, 1.0 - dist)
            retrieval_results.append(
                self._to_retrieval_result(
                    chunk_content=doc,
                    chunk_id=doc_id,
                    source=meta.get("source", ""),
                    score=score,
                    rank=rank,
                    metadata=meta,
                )
            )
        return retrieval_results

    async def asearch(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        """Embed the query via the collection's embedding function, then search."""
        import asyncio
        loop = asyncio.get_event_loop()

        def _search():
            coll = self._get_collection()
            results = coll.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            out: list[RetrievalResult] = []
            for rank, (doc_id, doc, meta, dist) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ),
                start=1,
            ):
                score = max(0.0, 1.0 - dist)
                out.append(
                    self._to_retrieval_result(
                        chunk_content=doc,
                        chunk_id=doc_id,
                        source=meta.get("source", ""),
                        score=score,
                        rank=rank,
                        metadata=meta,
                    )
                )
            return out

        return await loop.run_in_executor(None, _search)
