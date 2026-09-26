"""
Dense vector similarity retriever.

Embeds the query, then searches a vector store for the nearest neighbours.
The embedder and vector store are configured by the node's fields.
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.retrieval.base import BaseRetriever

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retriever: embed query -> cosine search in a vector store.

    Parameters
    ----------
    vectorstore_type:
        One of ``chroma``, ``qdrant``, ``faiss``, ``pinecone``, ``lancedb``.
    collection_name:
        Collection / index name in the vector store.
    embedding_provider:
        Embedder to use for the query (matches the one used during indexing).
    embedding_model:
        Model name for the embedder.
    score_threshold:
        Minimum score. Results below this are discarded (0 = keep all).
    **store_kwargs:
        Extra keyword arguments passed to the vector store constructor.
    """

    def __init__(
        self,
        vectorstore_type: str = "chroma",
        collection_name: str = "ragforge",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        score_threshold: float = 0.0,
        **store_kwargs,
    ) -> None:
        self.vectorstore_type = vectorstore_type
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.score_threshold = score_threshold
        self.store_kwargs = store_kwargs

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        from app.embeddings.manager import EmbeddingManager
        from app.vectorstores.manager import VectorStoreManager

        embedder = EmbeddingManager.get(self.embedding_provider, model=self.embedding_model)
        query_vector = embedder.embed_query(query)

        store = VectorStoreManager.get(
            self.vectorstore_type,
            collection_name=self.collection_name,
            **self.store_kwargs,
        )
        results = store.search(query_vector, k=k)

        if self.score_threshold > 0:
            results = [r for r in results if r.score >= self.score_threshold]

        logger.debug("DenseRetriever: %d results for query='%s...'", len(results), query[:40])
        return results
