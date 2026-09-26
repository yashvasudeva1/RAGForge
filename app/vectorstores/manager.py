"""
Vector store manager — factory for all vector store connectors.
"""

from __future__ import annotations

from app.core.constants import VectorStoreBackend
from app.core.exceptions import RAGForgeError
from app.vectorstores.base import BaseVectorStore


class VectorStoreManager:
    """
    Factory that returns the correct vector store connector by backend name.

    Usage
    -----
        store = VectorStoreManager.get(
            "chroma",
            collection_name="my_rag",
            persist_dir="./data/chroma",
        )
        store.add(embedded_chunks)
        results = store.search(query_vector, k=5)
    """

    @staticmethod
    def get(backend: str, **kwargs) -> BaseVectorStore:
        """
        Return a configured vector store for ``backend``.

        Parameters
        ----------
        backend:
            One of: ``chroma``, ``qdrant``, ``faiss``, ``pinecone``,
            ``lancedb``, ``weaviate``, ``milvus``.
        **kwargs:
            Passed to the store constructor.
        """
        from app.vectorstores.chroma import ChromaVectorStore
        from app.vectorstores.faiss import FAISSVectorStore
        from app.vectorstores.lancedb import LanceDBVectorStore
        from app.vectorstores.pinecone import PineconeVectorStore
        from app.vectorstores.qdrant import QdrantVectorStore

        registry: dict[str, type[BaseVectorStore]] = {
            VectorStoreBackend.CHROMA: ChromaVectorStore,
            VectorStoreBackend.QDRANT: QdrantVectorStore,
            VectorStoreBackend.FAISS: FAISSVectorStore,
            VectorStoreBackend.PINECONE: PineconeVectorStore,
            VectorStoreBackend.LANCEDB: LanceDBVectorStore,
        }

        cls = registry.get(backend)
        if cls is None:
            available = list(registry.keys())
            raise RAGForgeError(
                f"Unknown vector store backend '{backend}'. Available: {available}",
                details={"backend": backend, "available": available},
            )
        return cls(**kwargs)
