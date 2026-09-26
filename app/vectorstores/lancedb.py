"""
LanceDB local columnar vector store connector.
"""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.domain.embeddings.models import EmbeddedChunk
from app.domain.retrieval.models import RetrievalResult
from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class LanceDBVectorStore(BaseVectorStore):
    """
    LanceDB local columnar vector database.

    No server required — data stored as files on disk.

    Parameters
    ----------
    uri:
        Local directory path or LanceDB Cloud URI.
    table_name:
        LanceDB table name.
    """

    def __init__(
        self,
        uri: str = "./data/lancedb",
        table_name: str = "ragforge",
    ) -> None:
        self.uri = uri
        self.table_name = table_name
        self._db = None
        self._table = None

    def _get_db(self):
        if self._db is None:
            try:
                import lancedb  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("lancedb is required. Run: pip install lancedb") from exc
            self._db = lancedb.connect(self.uri)
        return self._db

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        db = self._get_db()
        data = [
            {
                "id": ec.chunk.id,
                "vector": ec.embedding,
                "content": ec.chunk.content,
                "source": ec.chunk.metadata.document_source or "",
                "chunk_index": ec.chunk.metadata.chunk_index,
                "document_id": ec.chunk.metadata.document_id,
            }
            for ec in embedded_chunks
        ]

        if self.table_name in db.table_names():
            table = db.open_table(self.table_name)
            table.add(data)
        else:
            db.create_table(self.table_name, data=data)
            self._table = None  # Reset cached table reference

        logger.debug("LanceDB: added %d rows to '%s'", len(data), self.table_name)

    def search(self, query_vector: list[float], k: int = 5, **kwargs) -> list[RetrievalResult]:
        db = self._get_db()
        if self.table_name not in db.table_names():
            return []

        table = db.open_table(self.table_name)
        results_df = table.search(query_vector).limit(k).to_pandas()

        retrieval_results: list[RetrievalResult] = []
        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            score = 1.0 - float(row.get("_distance", 0.0))
            retrieval_results.append(
                self._to_retrieval_result(
                    chunk_content=row.get("content", ""),
                    chunk_id=str(row.get("id", "")),
                    source=str(row.get("source", "")),
                    score=score,
                    rank=rank,
                )
            )
        return retrieval_results

    async def asearch(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search([], k))
